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
    Fleet Overview      the base network on a map, and the aircraft register

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


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fleet_risk() -> pd.DataFrame:
    """
    Score every airframe for failure risk and band the result.

    Args:
        (none)

    Returns:
        pd.DataFrame - one row per aircraft with risk_score and band
        (LOW / MEDIUM / HIGH), sorted worst first.

    Notes:
        This is the same fast screening heuristic the query agent answers
        "failure risk" with (agent.py, _failure_risk), reproduced here so the
        two surfaces cannot disagree about which aircraft is worst. It is NOT
        the calibrated model: prediction_model.py fits the Cox
        proportional-hazards model that gives defensible hazard ratios. The
        heuristic exists so an operator gets an instant ranking without a
        training run.

        Weights scale each term to a comparable magnitude given its natural
        range, mirroring the sign and relative strength of the fitted Cox
        covariates:
            age (0-18 years)      x 0.30 -> up to  5.4  cumulative fatigue
            cycle ratio (0.3-1.2) x 10   -> up to 12.0  short-sector cyclic damage
            salt exposure (0-1)   x 15   -> up to 15.0  corrosion, strongest driver
            demand count          x 0.05          demonstrated unreliability
    """
    df = load_table("""
        SELECT f.tail_number, f.aircraft_model, f.home_base, f.age_years,
               f.cycles_per_fh_ratio, f.total_flight_hours, f.total_flight_cycles,
               f.primary_role, f.daily_utilization_fh, s.salt_exposure,
               COUNT(pd.part_number) AS demands
        FROM fleet f
        JOIN stations s ON f.home_base = s.station_code
        LEFT JOIN part_demands pd ON f.tail_number = pd.tail_number
        GROUP BY f.tail_number
    """)
    df["risk_score"] = (df["age_years"] * 0.3
                        + df["cycles_per_fh_ratio"] * 10
                        + df["salt_exposure"] * 15
                        + df["demands"] * 0.05).round(1)

    # Band the continuous score. Cut points split the fleet into roughly
    # thirds, keeping the HIGH band small enough to act on: a risk list that
    # flags everything flags nothing.
    df["band"] = pd.cut(df["risk_score"], bins=[0, 8, 12, 100],
                        labels=["LOW", "MEDIUM", "HIGH"])
    return df.sort_values("risk_score", ascending=False).reset_index(drop=True)


# ============================================================================
# PRESENTATION HELPERS
# ============================================================================

def style_figure(fig, height=340, showlegend=False):
    """
    Apply the shared chart chrome to a Plotly figure.

    Args:
        fig: plotly.graph_objects.Figure - the figure to style.
        height: int - pixel height. 340 suits a two-column dashboard row.
        showlegend: bool - whether a legend is required. True whenever the
            chart carries two or more series, since identity must never rest
            on colour alone.

    Returns:
        The same figure, styled in place and returned for chaining.

    Notes:
        Centralised so every chart in the app shares one visual language, and
        so switching theme restyles all of them at once. The grid is a hairline
        and the axis rule is one step darker: both are present enough to orient
        the reader and recessive enough that the data marks stay dominant.
    """
    t = theme()
    fig.update_layout(
        height=height,
        showlegend=showlegend,
        # Transparent so the figure inherits the Streamlit surface rather than
        # painting its own rectangle over the page. This is what makes one
        # figure definition work in both themes.
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif",
                  size=12, color=t["ink_secondary"]),
        # Tight margins: dashboard real estate is scarce and Plotly's defaults
        # reserve far more than a titled panel needs.
        margin=dict(l=8, r=8, t=8, b=8),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0, title_text=""),
        hoverlabel=dict(bgcolor=t["surface"], font_size=12,
                        font_family="system-ui, sans-serif"),
    )
    fig.update_xaxes(showgrid=False, zeroline=False,
                     linecolor=t["baseline"], tickfont=dict(color=t["ink_muted"]))
    fig.update_yaxes(gridcolor=t["gridline"], zeroline=False,
                     linecolor="rgba(0,0,0,0)", tickfont=dict(color=t["ink_muted"]))
    return fig


def ramp(values):
    """
    Build a Plotly marker spec that maps a series onto the sequential ramp.

    Args:
        values: pd.Series - the magnitudes being encoded.

    Returns:
        dict - a marker specification for go.Bar.

    Notes:
        Every ranked bar chart in the app shares this, which is what keeps
        "darker (or brighter) means more" true everywhere. The colour bar is
        suppressed: the axis already quantifies the measure, so a scale beside
        it would be a second, redundant legend.
    """
    seq = theme()["sequential"]
    return dict(
        color=values,
        colorscale=[[i / (len(seq) - 1), c] for i, c in enumerate(seq)],
        showscale=False,
    )


def ranked_bar(labels, values, hover, height=260):
    """
    Render the app's standard horizontal ranked bar chart.

    Args:
        labels: pd.Series - the category axis, one entry per bar.
        values: pd.Series - the magnitude each bar encodes.
        hover: str - Plotly hovertemplate for a bar.
        height: int - pixel height of the figure.

    Returns:
        plotly.graph_objects.Figure - styled and ready for st.plotly_chart.

    Notes:
        Horizontal because the labels are words (part numbers, station codes,
        ATA chapters), and words read horizontally. Sorted ascending so the
        largest bar lands at the top of the rendered chart, which is where a
        reader looks first.
    """
    d = pd.DataFrame({"label": labels, "value": values}).sort_values("value")
    fig = go.Figure(go.Bar(
        x=d["value"], y=d["label"], orientation="h",
        marker=ramp(d["value"]),
        # 4px rounded data-end, anchored to the baseline.
        marker_cornerradius=4,
        hovertemplate=hover,
    ))
    fig.update_layout(bargap=0.35)
    return style_figure(fig, height=height)


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


def band_colours():
    """
    Map the three risk bands onto the status palette.

    Args:
        (none)

    Returns:
        dict - band name to hex colour for the active theme.

    Notes:
        Status colour is correct for risk bands because a band IS a state, not
        a series. Every use pairs it with the band name in text.
    """
    s = theme()["status"]
    return {"LOW": s["good"], "MEDIUM": s["warning"], "HIGH": s["critical"]}


def colour_band_cells(df, column):
    """
    Colour a band column green / amber / red inside a rendered table.

    Args:
        df: pd.DataFrame - the frame about to be displayed.
        column: str - the column holding LOW / MEDIUM / HIGH values.

    Returns:
        pandas.io.formats.style.Styler - ready to pass to st.dataframe.

    Notes:
        Text colour rather than a filled cell: filling every row of a 15-row
        table with saturated colour makes the table the loudest thing on the
        page, which is the opposite of what a reference table should be. The
        word is still there to be read, so the colour only accelerates the
        scan.
    """
    colours = band_colours()

    def paint(value):
        colour = colours.get(value)
        return f"color:{colour};font-weight:600" if colour else ""

    return df.style.map(paint, subset=[column])


# ============================================================================
# PAGE 1: FLEET OVERVIEW
# ============================================================================

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def station_summary() -> pd.DataFrame:
    """
    Assemble the per-station picture: geography, based aircraft and stock.

    Args:
        (none)

    Returns:
        pd.DataFrame - one row per station with coordinates, based-aircraft
        count, stock value, lines stocked and lines below minimum.

    Notes:
        Shared by the map on this page and the station table on the parts page,
        so both always report the same counts. Only serviceable units are
        valued, per Part-145 145.A.42.
    """
    stock_col = resolve_stock_column()
    stations = load_table("SELECT * FROM stations")
    inv = load_table(f"""
        SELECT i.station,
               SUM(i.{stock_col} * pc.unit_cost_eur) AS stock_value,
               SUM(CASE WHEN i.{stock_col} > 0 THEN 1 ELSE 0 END) AS lines_stocked,
               SUM(CASE WHEN pc.criticality = 'AOG' AND i.{stock_col} > 0
                        THEN 1 ELSE 0 END) AS aog_stocked,
               SUM(CASE WHEN i.{stock_col} < i.minimum_stock_level
                        THEN 1 ELSE 0 END) AS below_min
        FROM inventory i
        JOIN parts_catalog pc ON i.part_number = pc.part_number
        GROUP BY i.station
    """)
    based = load_table("SELECT home_base AS station, COUNT(*) AS aircraft "
                       "FROM fleet GROUP BY home_base")

    df = (stations.rename(columns={"station_code": "station"})
                  .merge(inv, on="station", how="left")
                  .merge(based, on="station", how="left")
                  .fillna({"aircraft": 0, "stock_value": 0, "lines_stocked": 0,
                           "aog_stocked": 0, "below_min": 0}))
    df["aircraft"] = df["aircraft"].astype(int)

    # Attach coordinates from the module-level lookup.
    df["lat"] = df["station"].map(lambda s: STATION_COORDS.get(s, (None, None))[0])
    df["lon"] = df["station"].map(lambda s: STATION_COORDS.get(s, (None, None))[1])
    return df


def station_map(df, height=460):
    """
    Draw the base network over Greece with a based-aircraft badge per station.

    Args:
        df: pd.DataFrame - the station_summary() frame.
        height: int - pixel height of the map.

    Returns:
        plotly.graph_objects.Figure

    Notes:
        Three channels, three measures, each doing one job: position is the
        airport, marker size is the number of based aircraft, and colour is
        salt exposure - the environmental covariate the Cox model found
        significant (hazard ratio 1.465, p = 0.037). Showing it geographically
        makes the reason visible: the high-exposure bases are the island
        stations whose aprons sit on open coastline.

        The badge prints the aircraft count on the marker, because reading a
        count off a size channel is guesswork and the count is the number an
        operator actually wants.
    """
    t = theme()
    seq = t["sequential"]

    fig = go.Figure(go.Scattermap(
        lat=df["lat"], lon=df["lon"],
        mode="markers+text",
        marker=dict(
            # Floor of 22px so a one-aircraft station still carries its badge
            # legibly; the increment is what encodes the count.
            size=22 + df["aircraft"] * 4,
            color=df["salt_exposure"],
            colorscale=[[i / (len(seq) - 1), c] for i, c in enumerate(seq)],
            cmin=0, cmax=1,
            colorbar=dict(title="Salt<br>exposure", thickness=10, len=0.6,
                          tickfont=dict(color=t["ink_muted"]),
                          title_font=dict(size=11, color=t["ink_muted"])),
            opacity=0.92,
        ),
        text=df["aircraft"].astype(str),
        textposition="middle center",
        # Fixed white badge text: it sits on top of a marker whose colour comes
        # from the ramp, not on the page surface, so it must not follow theme.
        textfont=dict(size=13, color="#ffffff", family="system-ui, sans-serif"),
        customdata=df[["station", "name", "aircraft", "salt_exposure",
                       "stock_value", "below_min"]],
        hovertemplate=("<b>%{customdata[0]} - %{customdata[1]}</b><br>"
                       "Based aircraft: %{customdata[2]}<br>"
                       "Salt exposure: %{customdata[3]:.1f}<br>"
                       "Stock value: EUR %{customdata[4]:,.0f}<br>"
                       "Lines below minimum: %{customdata[5]}<extra></extra>"),
    ))
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=0, b=0),
        map=dict(style=t["map_style"], zoom=4.6,
                 center=dict(lat=38.2, lon=24.0)),
        hoverlabel=dict(bgcolor=t["surface"], font_size=12),
    )
    return fig


def page_fleet():
    """
    The base network on a map, and the aircraft register beneath it.

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

    risk = fleet_risk()
    stations = station_summary()

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
    fleet = risk.merge(wo, on="tail_number", how="left").fillna(
        {"total_wo": 0, "unscheduled": 0, "scheduled": 0})

    # --- KPI row ---
    # Five headline numbers, each a single current value. Reading them as a row
    # gives the fleet's state before any chart is examined.
    c1, c2, c3, c4, c5 = st.columns(5)
    stat_tile(c1, "Aircraft", f"{len(fleet)}", "A320 family, active")
    stat_tile(c2, "Total flight hours", f"{fleet['total_flight_hours'].sum():,.0f}",
              "accumulated, all airframes")
    stat_tile(c3, "Mean age", f"{fleet['age_years'].mean():.1f}y",
              f"{fleet['age_years'].min()}-{fleet['age_years'].max()} years")

    high = int((fleet["band"] == "HIGH").sum())
    stat_tile(c4, "High risk", f"{high}", "airframes in the HIGH band",
              tone="critical" if high else "good")

    # Unscheduled share is a quality measure, so it carries a status tone.
    # Thresholds reflect industry practice: below 40% unscheduled is healthy
    # for a mature narrow-body fleet, above 60% suggests the maintenance
    # programme is not catching defects before they become failures.
    unsched_pct = 100 * fleet["unscheduled"].sum() / max(1, fleet["total_wo"].sum())
    tone = "good" if unsched_pct < 40 else "warning" if unsched_pct < 60 else "critical"
    stat_tile(c5, "Unscheduled share", f"{unsched_pct:.0f}%",
              "of all work orders", tone=tone)

    st.divider()

    left, right = st.columns([3, 2])

    with left:
        st.markdown("**Base network** - marker size is based aircraft, colour is salt exposure")
        st.plotly_chart(station_map(stations), width="stretch")

    with right:
        st.markdown("**Risk score by aircraft**")
        # Status colour is correct here because the bands ARE states, not
        # series. Every bar is additionally labelled with its band name in the
        # hover and in the table below, so the meaning never rests on colour.
        colours = band_colours()
        d = fleet.sort_values("risk_score")
        fig = go.Figure(go.Bar(
            x=d["risk_score"], y=d["tail_number"], orientation="h",
            marker=dict(color=[colours[b] for b in d["band"]]),
            marker_cornerradius=4,
            customdata=d[["band", "home_base", "salt_exposure"]],
            hovertemplate=("<b>%{y}</b> (%{customdata[1]})<br>"
                           "Score %{x:.1f} - %{customdata[0]} risk<br>"
                           "Salt exposure %{customdata[2]:.1f}<extra></extra>"),
        ))
        fig.update_layout(bargap=0.35)
        st.plotly_chart(style_figure(fig, height=460), width="stretch")

    st.divider()

    # --- Full register ---
    # More than seven rows that all carry meaning belongs in a table, not in
    # more colours. Every column header sorts, so the operator can rank the
    # fleet by whichever measure the question of the moment turns on.
    st.markdown("**Aircraft register**")
    display = fleet[["tail_number", "aircraft_model", "age_years", "home_base",
                     "primary_role", "total_flight_hours", "total_flight_cycles",
                     "cycles_per_fh_ratio", "scheduled", "unscheduled",
                     "risk_score", "band"]].copy()
    display["band"] = display["band"].astype(str)
    display.columns = ["Tail", "Model", "Age", "Base", "Role", "FH", "FC",
                       "Cycles/FH", "Scheduled", "Unscheduled", "Risk", "Band"]
    st.dataframe(
        colour_band_cells(display, "Band"),
        width="stretch", hide_index=True,
        column_config={
            "FH": st.column_config.NumberColumn(format="%,.0f"),
            "FC": st.column_config.NumberColumn(format="%,.0f"),
            "Cycles/FH": st.column_config.NumberColumn(format="%.2f"),
            "Risk": st.column_config.NumberColumn(
                format="%.1f",
                help="Screening heuristic: age, cycle ratio, salt exposure and "
                     "demonstrated consumption. Bands: LOW below 8, MEDIUM 8-12, "
                     "HIGH above 12."),
        },
    )
    st.caption(
        "Risk is the fast screening heuristic, the same one the query agent "
        "reports. The calibrated Cox proportional-hazards model lives in "
        "prediction_model.py."
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
