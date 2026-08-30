#!/usr/bin/env python3
"""
Daedalus Supply AI - Module 5: Web Dashboard
============================================
Streamlit front end over the database built by Module 1 and enriched by
Modules 2 and 3.

This is a read-only presentation layer. It issues SELECT statements only and
writes nothing back, so it can be pointed at a live continuing-airworthiness
database without any risk to the record system. Almost every figure it shows
is already computed and stored by an upstream module; the dashboard's job is
to make the fleet picture legible at a glance, not to recompute it.

First view:
    Fleet Overview      the aircraft register and its reliability measures

A page is one function plus one entry in PAGES, drawing on the data-access
and presentation helpers below. The remaining views plug into that shell.

Visual conventions (applied consistently across every chart):
    Sequential blue ramp  encodes magnitude (one hue, more-is-darker)
    Categorical slots     encode identity, assigned in fixed order
    Status palette        reserved for state, always paired with a text label
    No dual-axis charts   two measures of different scale get two charts

Author: Evangelos Tampachaniotis
Version: 1.1.0
License: MIT

Regulatory Framework:
    - EASA Part-M (EU 1321/2014) M.A.305 - the fleet view presents the
      continuing-airworthiness record: FH, FC and maintenance events per
      airframe.
    - EASA Part-145 145.A.42 - only serviceable stock is reported as
      available; unserviceable units awaiting shop input are excluded.
    - ATA/JASC 100 - chapter numbering throughout the parts and findings views.
    - MEL - the AOG / MEL / ROUTINE criticality that drives alert ordering.

Usage:
    streamlit run dashboard.py

    Theme follows the Streamlit setting (hamburger menu -> Settings), or is
    fixed in .streamlit/config.toml. Every colour below is resolved against
    the active theme at render time; nothing is hardcoded to a light surface.
"""

import os
import sqlite3

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from config import get_config

# Database path resolved relative to this file so the app runs from any working
# directory. File name retained for backwards compatibility with earlier builds.
# No credentials appear anywhere in this module: SQLite is a file, and when the
# system is moved to PostgreSQL (v1.2) the DSN will come from the environment
# exactly as load_to_postgres.py already reads it.
DB_PATH = os.environ.get(
    "DAEDALUS_DB_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "aerosupply.db"),
)

# Directory holding the plots written by prediction_model.py. Same folder as
# this file; resolved absolutely for the same reason as the database.
ASSET_DIR = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------
# Hardware profile. config.py is the single place that knows what this machine
# can do, and the dashboard honours it for the one query that can actually hurt
# on a constrained box: the 195,000-row SDR corpus. On MINIMAL the SDR views
# aggregate over a capped sample and say so on the page - a silently sampled
# number would be worse than a slow one.
# ----------------------------------------------------------------------------
CFG = get_config()
SDR_SAMPLE_CAP = CFG["ml"]["max_sdr_records"]      # None on STANDARD and FULL

# Cache lifetime in seconds. Long enough that paging around the app costs one
# query per table, short enough that re-running the optimiser in a terminal
# shows up in the browser without a restart.
CACHE_TTL = 300


# ============================================================================
# VISUAL SYSTEM
# ============================================================================
# Colour is assigned by the job it does, never by decoration. Four jobs, four
# palettes, and they are not interchangeable: reusing a status colour for a
# data series would make a chart claim something it does not mean.

# --- Categorical: encodes IDENTITY (which series is this?) ---
# Fixed order, never cycled. The ordering itself is the colour-vision-deficiency
# safety mechanism: adjacent pairs are the ones a reader compares, and this
# sequence keeps every adjacent pair separable for CVD readers. Charts here use
# at most three slots, which is the cap for scatter-type forms where any pair
# may end up adjacent on screen.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

# --- Sequential: encodes MAGNITUDE (how much?) ---
# One hue, light to dark. This is the default for any "compare quantities"
# chart. A rainbow scale would imply category boundaries that do not exist in
# a continuous measure. Reversed on a dark surface, where "more" has to mean
# brighter rather than darker to stay legible.
SEQUENTIAL_LIGHT = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                    "#2a78d6", "#256abf", "#184f95"]
SEQUENTIAL_DARK = ["#16365f", "#1c4a83", "#2360a8", "#2a78d6",
                   "#4f93e0", "#7db0e9", "#a9cbf2"]

# --- Status: encodes STATE (is this healthy?) ---
# Reserved. Never used for a data series. Always shipped alongside a text label
# so the meaning never rests on colour alone, which matters both for CVD
# readers and because two of these steps sit below 3:1 contrast on a light
# surface by design. The dark variants are lifted in luminance to clear 4.5:1
# against a dark surface; the hues are unchanged so the two themes read as one
# system.
STATUS_LIGHT = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}
STATUS_DARK = {
    "good": "#4ac94a",
    "warning": "#ffc74d",
    "serious": "#ff9e72",
    "critical": "#ff6b6b",
}

# --- Chart chrome: everything that is not data ---
# Grid and axis are deliberately recessive. They orient the reader without
# competing with the marks for attention. Two full sets, one per theme.
CHROME_LIGHT = {
    "ink_primary": "#0b0b0b",
    "ink_secondary": "#52514e",
    "ink_muted": "#898781",     # axis ticks and labels
    "gridline": "#e1e0d9",      # hairline grid
    "baseline": "#c3c2b7",      # axis rule
    "surface": "#fcfcfb",       # chart surface
    "raised": "#f4f3ef",        # alert cards and other raised blocks
    "map_style": "carto-positron",
}
CHROME_DARK = {
    "ink_primary": "#f5f4f1",
    "ink_secondary": "#c2c0ba",
    "ink_muted": "#8b8983",
    "gridline": "#33322e",
    "baseline": "#4a4843",
    "surface": "#1c1b19",
    "raised": "#26251f",
    "map_style": "carto-darkmatter",
}

# ----------------------------------------------------------------------------
# Station coordinates. The `stations` table carries operational attributes but
# no geography, and adding a column would mean changing the Module 1 schema for
# a presentation concern. These are the published airport reference points.
# ----------------------------------------------------------------------------
STATION_COORDS = {
    "ATH": (37.9364, 23.9445),   # Athens International
    "SKG": (40.5197, 22.9709),   # Thessaloniki, Makedonia
    "HER": (35.3397, 25.1803),   # Heraklion, Nikos Kazantzakis
    "RHO": (36.4054, 28.0862),   # Rhodes, Diagoras
    "CFU": (39.6019, 19.9117),   # Corfu, Ioannis Kapodistrias
}

# Criticality ordering used wherever alerts are sorted. AOG first, because an
# AOG shortage is the only one that can stop an aircraft flying today.
CRIT_ORDER = {"AOG": 0, "MEL": 1, "ROUTINE": 2}


# Sidebar palette selector state. "Auto" follows the browser; the two explicit
# settings exist because an operations screen is often a wall-mounted display
# whose surroundings decide what is legible, not the machine's OS setting.
PALETTE_KEY = "palette_choice"


def theme():
    """
    Resolve the palette for the theme currently in force.

    Args:
        (none)

    Returns:
        dict - the chrome tokens for the active theme, plus "status" (the
        status palette) and "sequential" (the magnitude ramp) so a caller
        needs exactly one lookup to style anything.

    Notes:
        Three sources, in order. An explicit choice in the sidebar wins; that
        value is a widget key, so it is already restored into session state
        when the script reruns and the page header resolves the same way as
        the body. Otherwise st.context.theme reports what the user actually
        sees, including when they picked "Use system setting" and the OS is in
        dark mode; it is a recent addition, so st.get_option covers an older
        Streamlit. The final default is light - the safer guess, since a light
        palette on a dark surface is merely flat while the reverse is
        illegible.
    """
    mode = st.session_state.get(PALETTE_KEY)

    if mode in (None, "Auto"):
        mode = None
        try:
            mode = st.context.theme.type
        except Exception:
            pass
    if mode is None:
        mode = st.get_option("theme.base") or "light"

    dark = str(mode).lower() == "dark"
    tokens = dict(CHROME_DARK if dark else CHROME_LIGHT)
    tokens["status"] = STATUS_DARK if dark else STATUS_LIGHT
    tokens["sequential"] = SEQUENTIAL_DARK if dark else SEQUENTIAL_LIGHT
    tokens["dark"] = dark
    return tokens


# ============================================================================
# DATA ACCESS
# ============================================================================

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_table(query: str, params: tuple = ()) -> pd.DataFrame:
    """
    Run a read-only query against the database and cache the result.

    Args:
        query: str - the SELECT statement to execute. Always a literal defined
            in this module; user input never reaches it except through params.
        params: tuple - values bound to the statement's placeholders. Search
            terms and filter selections travel this way, so a part number
            typed into the search box is data to sqlite3, never SQL.

    Returns:
        pd.DataFrame - the query result.

    Notes:
        Streamlit re-executes the whole script on every widget interaction, so
        without caching a single filter change would re-read the 54,000-row
        flight log and the 195,000-row SDR corpus. The TTL means an operator
        who reruns the optimiser sees new recommendations without restarting
        the app.

        The connection is opened read-only via the URI form: any statement
        that would write fails at the driver, which is a stronger guarantee
        than a code review of the queries below.
    """
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        return pd.read_sql(query, conn, params=params)
    finally:
        # Closed in a finally block so a malformed query cannot leak the handle.
        conn.close()


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def resolve_stock_column() -> str:
    """
    Determine the serviceable-quantity column name for the active backend.

    Args:
        (none)

    Returns:
        str - "serviceable" or "quantity_serviceable".

    Notes:
        The SQLite schema written by data_pipeline.py names this column
        `serviceable`; schema_postgres.sql and schema_mysql.sql name it
        `quantity_serviceable`. Detecting it once keeps every query below free
        of backend conditionals. The value comes from the database catalogue,
        never from user input, so interpolating it is safe.
    """
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(inventory)").fetchall()]
    finally:
        conn.close()
    return "quantity_serviceable" if "quantity_serviceable" in cols else "serviceable"


def database_exists() -> bool:
    """Return True when the database file is present."""
    return os.path.exists(DB_PATH)


def stat_tile(col, label, value, caption=None, tone=None):
    """
    Render a single headline number.

    Args:
        col: Streamlit column container to render into.
        label: str - what the number measures.
        value: str - the pre-formatted value.
        caption: str or None - supporting context below the value.
        tone: str or None - a STATUS key. Colours the value when the number
            itself represents a state that needs attention.

    Returns:
        None

    Notes:
        A single current value belongs in a stat tile, not a one-bar bar chart.
        The tone colour is a redundant cue only: the label and caption always
        state the meaning in words, so a reader who cannot distinguish the
        colour loses nothing.
    """
    t = theme()
    colour = t["status"].get(tone, t["ink_primary"]) if tone else t["ink_primary"]
    with col:
        st.markdown(
            f"""
            <div style="padding:0.75rem 0;">
              <div style="font-size:0.78rem;color:{t['ink_muted']};
                          text-transform:uppercase;letter-spacing:0.04em;">{label}</div>
              <div style="font-size:2rem;font-weight:600;color:{colour};
                          line-height:1.2;">{value}</div>
              <div style="font-size:0.78rem;color:{t['ink_secondary']};">{caption or ""}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================================
# PAGE 1: FLEET OVERVIEW
# ============================================================================

def page_fleet():
    """
    Fleet utilisation and reliability, one row per airframe.

    Args:
        (none)

    Returns:
        None - renders directly to the Streamlit page.

    Notes:
        Presents the EASA Part-M M.A.305 record: accumulated flight hours and
        cycles per aircraft, with the maintenance events raised against each.
        The ratio of unscheduled to scheduled work is the headline reliability
        metric of any operator reliability programme.
    """
    st.subheader("Fleet Overview")

    fleet = load_table("SELECT * FROM fleet ORDER BY total_flight_hours DESC")

    # Work orders split by origin. UNSCHEDULED_FAILURE and PILOT_REPORT are
    # counted together because both are unplanned; they differ only in who
    # detected the defect.
    wo = load_table("""
        SELECT tail_number,
               COUNT(*) AS total_wo,
               SUM(CASE WHEN source IN ('UNSCHEDULED_FAILURE','PILOT_REPORT')
                        THEN 1 ELSE 0 END) AS unscheduled,
               SUM(CASE WHEN source = 'SCHEDULED' THEN 1 ELSE 0 END) AS scheduled
        FROM work_orders GROUP BY tail_number
    """)
    fleet = fleet.merge(wo, on="tail_number", how="left").fillna(
        {"total_wo": 0, "unscheduled": 0, "scheduled": 0})

    # --- KPI row ---
    # Five headline numbers, each a single current value. Reading them as a row
    # gives the fleet's state before any table is examined.
    c1, c2, c3, c4, c5 = st.columns(5)
    stat_tile(c1, "Aircraft", f"{len(fleet)}", "A320 family, active")
    stat_tile(c2, "Total flight hours", f"{fleet['total_flight_hours'].sum():,.0f}",
              "accumulated, all airframes")
    stat_tile(c3, "Mean age", f"{fleet['age_years'].mean():.1f}y",
              f"{fleet['age_years'].min()}-{fleet['age_years'].max()} years")
    stat_tile(c4, "Work orders", f"{int(fleet['total_wo'].sum()):,}",
              "scheduled and unscheduled")

    # Unscheduled share is a quality measure, so it carries a status tone.
    # Thresholds reflect industry practice: below 40% unscheduled is healthy
    # for a mature narrow-body fleet, above 60% suggests the maintenance
    # programme is not catching defects before they become failures.
    unsched_pct = 100 * fleet["unscheduled"].sum() / max(1, fleet["total_wo"].sum())
    tone = "good" if unsched_pct < 40 else "warning" if unsched_pct < 60 else "critical"
    stat_tile(c5, "Unscheduled share", f"{unsched_pct:.0f}%",
              "of all work orders", tone=tone)

    st.divider()

    # --- Full register ---
    # Fifteen rows that all carry meaning belong in a table. Every column
    # header sorts, so the operator can rank the fleet by whichever measure
    # the question of the moment turns on.
    st.markdown("**Aircraft register**")
    display = fleet[["tail_number", "aircraft_model", "age_years", "home_base",
                     "primary_role", "total_flight_hours", "total_flight_cycles",
                     "cycles_per_fh_ratio", "scheduled", "unscheduled"]].copy()
    display.columns = ["Tail", "Model", "Age", "Base", "Role", "FH", "FC",
                       "Cycles/FH", "Scheduled", "Unscheduled"]
    st.dataframe(
        display, width="stretch", hide_index=True,
        column_config={
            "FH": st.column_config.NumberColumn(format="%,.0f"),
            "FC": st.column_config.NumberColumn(format="%,.0f"),
            "Cycles/FH": st.column_config.NumberColumn(format="%.2f"),
        },
    )


# ============================================================================
# APPLICATION SHELL
# ============================================================================

PAGES = {
    "Fleet Overview": page_fleet,
}


def main():
    """
    Configure the page, render the sidebar and dispatch to the selected view.

    Args:
        (none)

    Returns:
        None

    Notes:
        A missing database is the most common first-run problem, so it is
        checked before any query runs and answered with the command that fixes
        it rather than a traceback.

        Layout is wide with column-based rows throughout. Streamlit stacks
        columns vertically below roughly 640px, so every row on every page
        degrades to a single column on a phone or a narrow split screen without
        a media query anywhere in this file.
    """
    st.set_page_config(page_title="Daedalus Supply AI", page_icon=None,
                       layout="wide", initial_sidebar_state="expanded")

    t = theme()
    st.markdown(
        f"""
        <div style="padding-bottom:0.5rem;">
          <div style="font-size:1.6rem;font-weight:650;color:{t['ink_primary']};">
            Daedalus Supply AI</div>
          <div style="font-size:0.9rem;color:{t['ink_secondary']};">
            Predictive maintenance and intelligent supply chain for aircraft</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not database_exists():
        st.error("Database not found.")
        st.code("python data_pipeline.py", language="bash")
        st.caption(f"Expected at: {DB_PATH}")
        return

    # Sidebar navigation. One view so far. The dict is the extension point:
    # every later page is a function registered here, and nothing else changes.
    with st.sidebar:
        st.markdown("### View")
        choice = st.radio("View", list(PAGES.keys()), label_visibility="collapsed")
        st.divider()
        st.caption(
            "Read-only view of the continuing-airworthiness database. "
            "Rebuild it with `python data_pipeline.py`, refresh the models "
            "with `python prediction_model.py`, and refresh stock "
            "recommendations with `python logistics_optimizer.py`."
        )
        st.markdown("### Palette")
        # Auto first and selected by default: the app should look like the rest
        # of the user's Streamlit until they say otherwise.
        st.segmented_control(
            "Palette", ["Auto", "Light", "Dark"], default="Auto",
            key=PALETTE_KEY, label_visibility="collapsed",
            help="Auto follows the browser and the Streamlit theme setting. "
                 "Light and Dark force the chart palette, which is what a "
                 "wall-mounted screen in a bright hangar office usually needs.",
        )
        st.divider()

        hw = CFG["hardware"]
        st.caption(
            f"Profile {CFG['profile']} - {hw['ram_gb']} GB RAM, {hw['cores']} cores, "
            f"{hw['os']} {hw['arch']}. Cache {CACHE_TTL}s."
        )
        st.caption("v1.1.0")

    PAGES[choice]()


if __name__ == "__main__":
    main()
