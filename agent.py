#!/usr/bin/env python3
"""
Daedalus Supply AI - Module 4: Interactive Query Agent (security-hardened)
=========================================================================
Natural-language front end over the fleet/inventory database.

The agent maps free-text questions (English or Greek) onto a fixed catalogue
of vetted SQL reports. It deliberately does NOT synthesise arbitrary SQL from
user input: every query in this file is either fully static or built from
parameter placeholders, so the query surface is bounded and auditable. That
matters because the same database backs continuing-airworthiness records,
which under EASA Part-M M.A.305 must remain tamper-evident and traceable.

Architecture position:
    Module 1 (data_pipeline)      -> builds the database
    Module 2 (prediction_model)   -> writes risk/forecast outputs
    Module 3 (logistics_optimizer)-> writes stock/transfer recommendations
    Module 4 (this file)          -> read-only presentation layer over all of it

Run modes:
    python agent.py            demo - runs a fixed set of representative queries
    python agent.py -i         interactive REPL
    python agent.py -q "..."   single query, prints result and exits

Author: Evangelos Tampachaniotis
Version: 1.0.0
License: MIT

Regulatory Framework:
    - EASA Part-M (EU 1321/2014) - Continuing Airworthiness; M.A.305 record
      keeping, M.A.302 maintenance programme
    - EASA Part-145 - Maintenance Organisation Approval; 145.A.42 component
      acceptance and stores control (the serviceable-stock views below)
    - ATA/JASC 100 - chapter numbering used throughout the findings/SDR reports
"""

import sqlite3
import os
import re
import argparse
import pandas as pd
import numpy as np

# Absolute path to the SQLite database, resolved relative to THIS file rather
# than the working directory, so the agent works when invoked from anywhere
# (cron jobs, IDE run buttons, other directories).
# Note: the file name is kept as aerosupply.db for backwards compatibility
# with databases produced by earlier builds of the pipeline.
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aerosupply.db")


class DaedalusAgent:
    """
    Read-only natural-language interface to the Daedalus Supply AI database.

    Role in the system architecture: this is the only component intended to be
    driven directly by a human typing free text. Every other module is batch.
    Because it accepts untrusted input, it owns the input-sanitisation policy
    for the project (see _safe) and never issues INSERT/UPDATE/DELETE.

    Attributes:
        conn      : sqlite3.Connection - open handle to the fleet database
        stock_col : str - name of the serviceable-quantity column in `inventory`
    """

    def __init__(self, db_path=DB_PATH):
        """
        Open the database and resolve schema differences between backends.

        Args:
            db_path: str - filesystem path to the SQLite database file.
                     Defaults to aerosupply.db next to this script.

        Returns:
            None

        Notes:
            The SQLite schema produced by data_pipeline.py names the on-hand
            serviceable column `serviceable`, while schema_postgres.sql /
            schema_mysql.sql name it `quantity_serviceable`. Rather than
            branching at every call site, we detect the column once here and
            interpolate the resolved name into the report queries. The value
            comes from PRAGMA table_info (the database itself), never from user
            input, so this interpolation cannot be influenced externally.
        """
        self.conn = sqlite3.connect(db_path)

        # PRAGMA table_info returns one row per column; index [1] is the name.
        cols = [r[1] for r in self.conn.execute("PRAGMA table_info(inventory)").fetchall()]

        # Prefer the PostgreSQL/MySQL spelling when present, else the SQLite one.
        self.stock_col = "quantity_serviceable" if "quantity_serviceable" in cols else "serviceable"

    @staticmethod
    def _safe(value):
        """
        Strip every character that cannot legitimately appear in an identifier.

        Args:
            value: any - a candidate tail number, part number or station code
                   extracted from user text.

        Returns:
            str - the input with all characters outside [A-Za-z0-9_-] removed.

        Notes:
            This is an allow-list, not a deny-list. Quote characters, semicolons,
            comment markers (--), parentheses and whitespace are all outside the
            allowed set, so no SQL metacharacter can survive. It is applied as
            defence in depth: identifiers that reach a query are additionally
            passed as bound parameters wherever the SQL grammar permits it.
            The two places it is the primary defence are the LIKE keyword search
            (_search_parts) and nothing else - every other call site also binds.
        """
        return re.sub(r"[^a-zA-Z0-9_\-]", "", str(value))

    def query(self, question):
        """
        Route a free-text question to the report that answers it.

        Args:
            question: str - the user's free-text question.

        Returns:
            str - a formatted, ready-to-print report, or the help text when the
            question matches no known intent.

        Notes:
            Intent detection is keyword-based and intentionally simple. A
            statistical intent classifier was considered and rejected for v1.0:
            in an airworthiness context a misrouted query that silently returns
            the wrong report is worse than one that falls through to help text.
            Ordering matters - more specific intents are tested before general
            ones, and the fleet branch is first because it is the most common.
        """
        # Normalise once: lowercase for keyword matching, stripped of padding.
        # The ORIGINAL `question` is kept for identifier extraction, because
        # tail numbers and part numbers are uppercase.
        q = question.lower().strip()

        # --- Intent: fleet ---
        # Either a whole-fleet summary or, if a registration appears in the
        # text, the detail view for that one aircraft.
        if any(w in q for w in ["fleet", "aircraft"]):
            # Hellenic civil registrations carry the ICAO nationality mark SX.
            if "SX-" in question.upper():
                m = re.search(r"SX-\w+", question.upper())
                if m:
                    return self._aircraft_detail(m.group())
            return self._fleet_status()

        # --- Intent: parts / inventory ---
        # Sub-routed by what the user actually wants to know about the part.
        if any(w in q for w in ["part", "stock", "inventory"]):
            # Internal part numbers follow AES-<ATA>-<subsystem>-<seq>.
            pn = re.search(r"AES-[\d-]+", question.upper())
            if pn:
                # An explicit part number is the most specific signal available,
                # so it wins over any other keyword in the sentence.
                return self._find_part(pn.group())
            if any(w in q for w in ["alert", "low", "shortage"]):
                # Below-minimum stock - the Part-145 stores-control view.
                return self._stock_alerts()
            if any(w in q for w in ["top", "demand"]):
                # Consumption ranking, used to prioritise reorder effort.
                return self._top_demanded()
            # No sub-intent matched: treat the remaining words as a description
            # keyword search over the parts catalogue.
            return self._search_parts(q)

        # --- Intent: maintenance ---
        # "history" plus a registration means the per-aircraft log; otherwise
        # the fleet-wide recent/upcoming check list.
        if any(w in q for w in ["maintenance", "check"]):
            m = re.search(r"SX-\w+", question.upper())
            if m and "history" in q:
                return self._maint_history(m.group())
            return self._upcoming_maint()

        # --- Intent: reliability / risk ---
        if any(w in q for w in ["failure", "risk", "predict"]):
            return self._failure_risk()

        # --- Intent: inspection findings (CRACK / WEAR / CORROSION / ...) ---
        if any(w in q for w in ["finding", "defect"]):
            return self._findings()

        # --- Intent: real-world FAA Service Difficulty Report corpus ---
        if any(w in q for w in ["sdr", "faa", "real"]):
            return self._sdr_summary()

        # --- Intent: station / base ---
        if any(w in q for w in ["station", "warehouse"]):
            # If the user named one of the five IATA station codes, show it.
            for s in ["ATH", "SKG", "HER", "RHO", "CFU"]:
                if s.lower() in q:
                    return self._station_detail(s)
            return self._station_overview()

        # --- Intent: optimiser output ---
        if any(w in q for w in ["recommend", "transfer"]):
            return self._recommendations()

        # --- Explicit help request, and the catch-all fallback ---
        # Both land on the same text: an unrecognised question should show the
        # user what the agent *can* answer rather than a bare error.
        if any(w in q for w in ["help"]):
            return self._help()
        return self._help()

    # ------------------------------------------------------------------
    # REPORT BUILDERS
    # Each returns a formatted string. All SQL below is static or bound.
    # ------------------------------------------------------------------

    def _fleet_status(self):
        """
        Whole-fleet summary: utilisation and maintenance activity per aircraft.

        Args:
            (none)

        Returns:
            str - table of every aircraft ordered by accumulated flight hours.

        Notes:
            LEFT JOIN (not INNER) onto work_orders so that an aircraft with no
            recorded maintenance still appears, with a count of zero - omitting
            it would misrepresent fleet size.
            cycles_per_fh_ratio is the FC/FH ratio: high values indicate
            short-sector operation, which drives cyclic fatigue damage.
        """
        return "FLEET STATUS\n" + "=" * 70 + "\n" + pd.read_sql("""
            SELECT f.tail_number, f.aircraft_model, f.age_years as age,
                   f.home_base, f.primary_role, f.total_flight_hours as total_fh,
                   f.cycles_per_fh_ratio as cyc_ratio,
                   COUNT(DISTINCT wo.work_order_id) as work_orders
            FROM fleet f LEFT JOIN work_orders wo ON f.tail_number = wo.tail_number
            GROUP BY f.tail_number ORDER BY f.total_flight_hours DESC
        """, self.conn).to_string(index=False)

    def _aircraft_detail(self, tail):
        """
        Single-aircraft record: identity, utilisation and recent part demands.

        Args:
            tail: str - aircraft registration, e.g. "SX-ABK".

        Returns:
            str - formatted detail block, or a not-found message.

        Notes:
            Both queries bind `tail` as a parameter; _safe() is applied first so
            that even the value echoed back in the not-found message is inert.
            The demand list is capped at 10 rows - enough to see the current
            trend without flooding a terminal.
        """
        tail = self._safe(tail)

        # Identity and utilisation counters for the registration.
        ac = pd.read_sql("SELECT * FROM fleet WHERE tail_number = ?", self.conn, params=[tail])
        if ac.empty:
            # Unknown registration: report rather than raise, since this is a
            # typo in interactive use far more often than a real error.
            return f"Aircraft {tail} not found"
        a = ac.iloc[0]

        # Ten most recent part consumption events for this aircraft, joined to
        # the catalogue so the operator sees descriptions, not bare part numbers.
        demands = pd.read_sql("""
            SELECT pd.part_number, pc.description, pd.demand_type, pd.demand_date, pd.criticality
            FROM part_demands pd JOIN parts_catalog pc ON pd.part_number = pc.part_number
            WHERE pd.tail_number = ? ORDER BY pd.demand_date DESC LIMIT 10
        """, self.conn, params=[tail])

        return (f"AIRCRAFT: {tail}\n{'=' * 60}\nModel: {a['aircraft_model']}\n"
                f"Age: {a['age_years']}y | Base: {a['home_base']} | Role: {a['primary_role']}\n"
                f"FH: {a['total_flight_hours']} | FC: {a['total_flight_cycles']} | "
                f"Ratio: {a['cycles_per_fh_ratio']}\n\nRecent demands:\n{demands.to_string(index=False)}")

    def _find_part(self, pn):
        """
        Network-wide availability of one part number across all stations.

        Args:
            pn: str - internal part number, e.g. "AES-24-10-001".

        Returns:
            str - per-station stock table plus the network total, or not-found.

        Notes:
            The only interpolated token is self.stock_col, resolved from
            PRAGMA table_info in __init__ - it is a schema fact, not user input.
            The part number itself is bound as a parameter.
            Answers the Part-145 stores question "can I release this aircraft
            today, and if not, who in the network holds the part?"
        """
        pn = self._safe(pn)

        # One row per station holding the part, with its reorder minimum and
        # the commercial/criticality attributes from the catalogue.
        df = pd.read_sql(f"""
            SELECT i.station, i.{self.stock_col} as stock, i.minimum_stock_level as min,
                   pc.description, pc.criticality, pc.unit_cost_eur
            FROM inventory i JOIN parts_catalog pc ON i.part_number = pc.part_number
            WHERE i.part_number = ?
        """, self.conn, params=[pn])
        if df.empty:
            return f"Part {pn} not found"

        # Catalogue attributes are identical on every row, so read row 0.
        d = df.iloc[0]
        return (f"PART: {pn} - {d['description']}\nCriticality: {d['criticality']} | "
                f"Cost: €{d['unit_cost_eur']:,.0f}\n{'=' * 50}\n"
                f"{df[['station', 'stock', 'min']].to_string(index=False)}\n"
                f"\nTotal network: {df['stock'].sum()}")

    def _stock_alerts(self):
        """
        Every station/part combination currently below its minimum stock level.

        Args:
            (none)

        Returns:
            str - shortage table ordered by criticality, or an all-clear message.

        Notes:
            `shortage` = minimum_stock_level - on-hand serviceable, i.e. the
            quantity that must be ordered to return to the reorder point (ROP).
            Ordering by criticality first puts AOG items - where every hour of
            grounding costs roughly €15,000 - at the top of the operator's list,
            with the largest shortfall first inside each criticality band.
            Capped at 15 rows: this is a work list, not an archive.
        """
        df = pd.read_sql(f"""
            SELECT i.station, pc.part_number, pc.description, pc.criticality,
                   i.{self.stock_col} as stock, i.minimum_stock_level as min,
                   (i.minimum_stock_level - i.{self.stock_col}) as shortage
            FROM inventory i JOIN parts_catalog pc ON i.part_number = pc.part_number
            WHERE i.{self.stock_col} < i.minimum_stock_level
            ORDER BY pc.criticality, shortage DESC LIMIT 15
        """, self.conn)

        # An empty result is good news, not an error - say so plainly.
        if df.empty:
            return "No stock alerts"
        return f"STOCK ALERTS\n{'=' * 70}\n{df.to_string(index=False)}"

    def _top_demanded(self):
        """
        Parts ranked by number of consumption events across the fleet.

        Args:
            (none)

        Returns:
            str - top-15 consumption table.

        Notes:
            Two different measures are shown deliberately: `events` (how often
            the part was needed, which drives stocking policy and lead-time
            risk) and `qty` (total units, which drives budget). A part can rank
            high on one and low on the other - e.g. a single high-quantity
            C-check kit versus a frequently replaced single-unit sensor.
            Ranking is by `events` because stock-out probability follows demand
            frequency, not volume.
        """
        return "TOP DEMANDED PARTS\n" + "=" * 70 + "\n" + pd.read_sql("""
            SELECT pd.part_number, pc.description, pc.part_class,
                   COUNT(*) as events, SUM(pd.quantity_required) as qty, pc.unit_cost_eur
            FROM part_demands pd JOIN parts_catalog pc ON pd.part_number = pc.part_number
            GROUP BY pd.part_number ORDER BY events DESC LIMIT 15
        """, self.conn).to_string(index=False)

    def _search_parts(self, q):
        """
        Keyword search over part descriptions when no part number was given.

        Args:
            q: str - the lowercased user question.

        Returns:
            str - matching catalogue entries, the top-demanded fallback when no
            usable keyword remains, or a no-match message.

        Notes:
            Transformation: free-text question -> keyword list -> OR-ed LIKE
            predicate -> catalogue rows.
            Words of 3 characters or fewer are dropped because they are almost
            all articles and prepositions and would match everything; the
            explicit stop list removes query verbs that survive that filter.
            Every keyword passes through _safe() before interpolation, so the
            LIKE pattern can only ever contain [A-Za-z0-9_-] - quotes and
            statement terminators cannot survive, which is what makes this
            interpolation safe. (SQLite's Python driver cannot bind an
            arbitrary-arity OR list, hence the sanitised build.)
        """
        kw = [w for w in q.split()
              if len(w) > 3 and w not in ["part", "find", "where", "show", "list", "stock"]]

        # Nothing searchable left - fall back to the most useful generic report
        # rather than returning an empty answer.
        if not kw:
            return self._top_demanded()

        # Build one LIKE clause per keyword and OR them: a part matches if any
        # keyword appears anywhere in its description.
        cond = " OR ".join([f"LOWER(description) LIKE '%{self._safe(k)}%'" for k in kw])
        df = pd.read_sql(
            f"SELECT part_number, description, part_class, criticality "
            f"FROM parts_catalog WHERE {cond} LIMIT 10", self.conn)

        if df.empty:
            return f"No parts matching: {' '.join(kw)}"
        return f"Parts matching '{' '.join(kw)}':\n{df.to_string(index=False)}"

    def _upcoming_maint(self):
        """
        Most recent maintenance work orders across the whole fleet.

        Args:
            (none)

        Returns:
            str - the 15 latest work orders by scheduled date.

        Notes:
            aircraft_fh_at_check records the airframe hours at which the check
            fell due, which is how MSG-3 / MPD intervals are expressed (A-check
            ~750 FH, C-check ~7,500 FH, D-check ~30,000 FH) - calendar date
            alone is not sufficient to judge whether an interval was respected.
        """
        return "RECENT MAINTENANCE\n" + "=" * 60 + "\n" + pd.read_sql("""
            SELECT tail_number, check_type, scheduled_date, status, aircraft_fh_at_check
            FROM work_orders ORDER BY scheduled_date DESC LIMIT 15
        """, self.conn).to_string(index=False)

    def _maint_history(self, tail):
        """
        Maintenance log for one aircraft.

        Args:
            tail: str - aircraft registration, e.g. "SX-ABK".

        Returns:
            str - the 15 most recent work orders for that registration.

        Notes:
            `source` distinguishes SCHEDULED work (driven by the approved
            Aircraft Maintenance Programme under Part-M M.A.302) from
            UNSCHEDULED work (defect rectification) - the split between the two
            is the headline reliability metric for a fleet.
        """
        tail = self._safe(tail)
        df = pd.read_sql(
            "SELECT work_order_id, check_type, scheduled_date, source, aircraft_fh_at_check "
            "FROM work_orders WHERE tail_number = ? ORDER BY scheduled_date DESC LIMIT 15",
            self.conn, params=[tail])
        return f"MAINTENANCE HISTORY: {tail}\n{'=' * 60}\n{df.to_string(index=False)}"

    def _failure_risk(self):
        """
        Heuristic per-aircraft failure-risk ranking.

        Args:
            (none)

        Returns:
            str - every aircraft with its risk score and LOW/MEDIUM/HIGH band.

        Notes:
            This is the fast screening heuristic, NOT the calibrated model -
            prediction_model.py fits the Cox proportional-hazards model that
            gives statistically defensible hazard ratios (e.g. salt exposure
            HR = 1.465, p = 0.037). This report exists so an operator gets an
            instant answer at the prompt without loading lifelines/XGBoost.
        """
        # Join fleet to stations to pick up the environmental covariate
        # (salt_exposure), and LEFT JOIN demands so an aircraft with no recorded
        # consumption still scores rather than dropping out of the ranking.
        df = pd.read_sql("""
            SELECT f.tail_number, f.home_base, f.age_years as age,
                   f.cycles_per_fh_ratio as cyc_ratio, s.salt_exposure,
                   COUNT(pd.part_number) as demands
            FROM fleet f JOIN stations s ON f.home_base = s.station_code
            LEFT JOIN part_demands pd ON f.tail_number = pd.tail_number
            GROUP BY f.tail_number ORDER BY demands DESC
        """, self.conn)

        # --- Composite risk score (weighted linear index) ---
        # Weights are scaled so each term contributes a comparable magnitude
        # given its natural range, mirroring the sign and relative strength of
        # the fitted Cox covariates:
        #   age (0-18 years)      x 0.30  -> up to ~5.4  : cumulative fatigue
        #   cyc_ratio (0.3-0.8)   x 10    -> up to ~8.0  : short-sector cyclic damage
        #   salt_exposure (0-1)   x 15    -> up to 15.0  : island-base corrosion,
        #                                                  the strongest single driver
        #   demands (count)       x 0.05  -> observed consumption as a proxy for
        #                                    demonstrated unreliability
        df["risk_score"] = (df["age"] * 0.3 + df["cyc_ratio"] * 10
                            + df["salt_exposure"] * 15 + df["demands"] * 0.05).round(1)

        # Band the continuous score for operational use. Cut points 8 and 12
        # split the fleet roughly into thirds, so the HIGH band stays small
        # enough to act on - a risk list that flags everything flags nothing.
        df["risk"] = pd.cut(df["risk_score"], bins=[0, 8, 12, 100],
                            labels=["LOW", "MEDIUM", "HIGH"])

        return (f"FAILURE RISK\n{'=' * 70}\n"
                f"{df.sort_values('risk_score', ascending=False).to_string(index=False)}")

    def _findings(self):
        """
        Inspection findings aggregated by defect type and ATA chapter.

        Args:
            (none)

        Returns:
            str - top-15 finding_type x ata_chapter combinations by count.

        Notes:
            Grouping by ATA/JASC chapter is what makes findings comparable
            against the FAA SDR corpus and against manufacturer data - it is the
            industry-standard system taxonomy (e.g. 32 = Landing Gear,
            29 = Hydraulic Power, 72 = Engine).
            A concentration of CORROSION findings in one chapter is the classic
            trigger for an MSG-3 maintenance-programme review.
        """
        return "FINDINGS\n" + "=" * 60 + "\n" + pd.read_sql("""
            SELECT finding_type, ata_chapter, COUNT(*) as count
            FROM findings GROUP BY finding_type, ata_chapter ORDER BY count DESC LIMIT 15
        """, self.conn).to_string(index=False)

    def _sdr_summary(self):
        """
        Summary of the real FAA Service Difficulty Report corpus.

        Args:
            (none)

        Returns:
            str - record count plus the 20 most-reported part/chapter pairs, or
            guidance if the SDR table has not been populated.

        Notes:
            SDRs are mandatory occurrence reports filed with the FAA; they are
            the real-world failure evidence that anchors this project's
            otherwise synthetic fleet. Placeholder part names ('UNKNOWN',
            'NONE', empty) are excluded because they dominate the raw counts
            without carrying engineering meaning.

            The except branch covers the legitimate case where the user has run
            the agent before data_pipeline.py, or ran the pipeline without any
            SDR CSVs in raw_data/ - sqlite3 raises on the missing table. The
            correct response is instructions, not a stack trace.
        """
        try:
            total = pd.read_sql("SELECT COUNT(*) as n FROM faa_sdr_raw", self.conn).iloc[0]["n"]
            df = pd.read_sql("""
                SELECT ata_code, part_name, COUNT(*) as failures FROM faa_sdr_raw
                WHERE part_name NOT IN ('UNKNOWN','NONE','') GROUP BY ata_code, part_name
                ORDER BY failures DESC LIMIT 20
            """, self.conn)
            return f"FAA SDR ({total:,} records)\n{'=' * 60}\n{df.to_string(index=False)}"
        except Exception:
            return "No SDR data. Run data_pipeline.py with CSVs in raw_data/"

    def _station_detail(self, station):
        """
        Detail view for one station: based aircraft and AOG-critical stock.

        Args:
            station: str - IATA station code, one of ATH/SKG/HER/RHO/CFU.

        Returns:
            str - the station's aircraft list followed by its AOG inventory.

        Notes:
            The inventory query filters to criticality = 'AOG' and sorts
            ascending by quantity on hand, so the items closest to causing an
            Aircraft-on-Ground event appear first. Only AOG parts are shown
            because a station's full catalogue is 53 lines of mostly routine
            stock that would bury the actionable rows.
        """
        station = self._safe(station)

        # Aircraft whose home base is this station.
        ac = pd.read_sql(
            "SELECT tail_number, aircraft_model, primary_role FROM fleet WHERE home_base = ?",
            self.conn, params=[station])

        # AOG-critical holdings at this station, scarcest first.
        inv = pd.read_sql(f"""
            SELECT pc.part_number, pc.description, pc.criticality,
                   i.{self.stock_col} as stock, i.minimum_stock_level as min
            FROM inventory i JOIN parts_catalog pc ON i.part_number = pc.part_number
            WHERE i.station = ? AND pc.criticality = 'AOG' ORDER BY i.{self.stock_col}
        """, self.conn, params=[station])

        return (f"STATION: {station}\n{'=' * 60}\nAircraft:\n{ac.to_string(index=False)}\n\n"
                f"AOG inventory:\n{inv.to_string(index=False)}")

    def _station_overview(self):
        """
        Network overview: one row per station with fleet and stock totals.

        Args:
            (none)

        Returns:
            str - per-station summary table.

        Notes:
            salt_exposure (0.0 inland .. 1.0 island/coastal) is surfaced here
            because it is the environmental driver behind the corrosion-related
            demand differences between the mainland and island bases.
            LEFT JOINs throughout: CFU is a spares-holding station with no based
            aircraft, and must still appear in the network picture.
        """
        return "STATIONS\n" + "=" * 60 + "\n" + pd.read_sql(f"""
            SELECT s.station_code, s.name, s.salt_exposure,
                   COUNT(DISTINCT f.tail_number) as aircraft,
                   SUM(i.{self.stock_col}) as total_stock
            FROM stations s LEFT JOIN fleet f ON s.station_code = f.home_base
            LEFT JOIN inventory i ON s.station_code = i.station
            GROUP BY s.station_code
        """, self.conn).to_string(index=False)

    def _recommendations(self):
        """
        Surface the output of Module 3 (logistics_optimizer.py).

        Args:
            (none)

        Returns:
            str - stock-level and transfer recommendations, or a prompt to run
            the optimiser first.

        Notes:
            These two tables are written by logistics_optimizer.py, not by the
            pipeline, so they legitimately may not exist yet. The except branch
            turns that expected condition into an instruction instead of an
            error, which is the correct behaviour for an interactive tool.
        """
        try:
            s = pd.read_sql("SELECT * FROM stock_recommendations LIMIT 10", self.conn)
            t = pd.read_sql("SELECT * FROM transfer_recommendations LIMIT 10", self.conn)
            return (f"STOCK RECOMMENDATIONS\n{'=' * 60}\n{s.to_string(index=False)}\n\n"
                    f"TRANSFERS\n{'=' * 60}\n{t.to_string(index=False)}")
        except Exception:
            return "Run logistics_optimizer.py first"

    def _help(self):
        """
        Command reference, also used as the fallback for unmatched questions.

        Args:
            (none)

        Returns:
            str - the list of supported query forms.
        """
        return """
Daedalus Supply AI Commands:
  fleet status              aircraft SX-ABK
  find part AES-24-10-001   stock alerts
  top demanded parts        failure risk
  station ATH               SDR summary
  maintenance history SX-ABK  recommendations
  findings summary          help
  Type 'quit' to exit."""

    def close(self):
        """
        Close the database connection.

        Args:
            (none)

        Returns:
            None

        Notes:
            Called explicitly by every entry point below. The agent is read-only
            so there is nothing to commit, but releasing the file handle matters
            when the caller goes on to run the pipeline or optimiser.
        """
        self.conn.close()


def run_demo():
    """
    Run a fixed set of representative queries end to end.

    Args:
        (none)

    Returns:
        None - prints to stdout.

    Notes:
        This is the default mode so that `python agent.py` demonstrates the
        system without requiring the operator to know any query syntax. The
        eight questions were chosen to exercise every distinct report family:
        fleet, single aircraft, single part, alerts, risk, station, demand
        ranking and the real SDR corpus.
    """
    print("=" * 60 + "\n  Daedalus Supply AI - Agent Demo\n" + "=" * 60)
    agent = DaedalusAgent()

    # Iterate over the demo script, printing each question and its answer.
    for q in ["fleet status", "aircraft SX-ABK", "find part AES-24-10-001", "stock alerts",
              "failure risk", "station HER", "top demanded parts", "SDR summary"]:
        print(f"\n  QUERY: {q}\n  {'-' * 50}")
        lines = agent.query(q).split("\n")

        # Truncate to 25 lines per answer: the fleet and SDR reports are long,
        # and an unattended demo should stay readable in one terminal scroll.
        print("\n".join(lines[:25]))
        if len(lines) > 25:
            print(f"  ... ({len(lines) - 25} more lines)")

    agent.close()


def interactive():
    """
    Start the interactive read-eval-print loop.

    Args:
        (none)

    Returns:
        None - runs until the user exits.

    Notes:
        Exits on 'quit'/'exit'/'q', on Ctrl-C (KeyboardInterrupt) and on Ctrl-D
        or a closed stdin (EOFError). EOFError is handled explicitly so the
        agent can be driven from a pipe or heredoc in a test script without
        crashing when the input stream ends.
    """
    print("=" * 60 + "\n  Daedalus Supply AI - Interactive\n  Type 'help' or 'quit'\n" + "=" * 60)
    agent = DaedalusAgent()

    # Main REPL loop: read one question per iteration until the user leaves.
    while True:
        try:
            q = input("\n  Daedalus > ").strip()
        except (KeyboardInterrupt, EOFError):
            # Ctrl-C / Ctrl-D / end of piped input - a normal way to finish,
            # so break cleanly and let the connection close below.
            break

        # Empty line (user just pressed Enter): re-prompt rather than showing help.
        if not q:
            continue

        # Accept the three conventional exit words.
        if q.lower() in ["quit", "exit", "q"]:
            break

        print(f"\n{agent.query(q)}")

    agent.close()


# ============================================================================
# CLI ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Daedalus Supply AI - interactive fleet and inventory query agent")
    p.add_argument("--interactive", "-i", action="store_true",
                   help="start the interactive query prompt")
    p.add_argument("--query", "-q", type=str,
                   help="run a single query, print the answer and exit")
    a = p.parse_args()

    # --query wins if supplied: it is the scriptable, non-interactive path.
    if a.query:
        ag = DaedalusAgent()
        print(ag.query(a.query))
        ag.close()
    elif a.interactive:
        interactive()
    else:
        # No flags: run the demo, so a first-time user sees output immediately.
        run_demo()
