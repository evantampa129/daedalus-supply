#!/usr/bin/env python3
"""
Daedalus Supply AI - Data Explorer
==================================
Nine standing analyses over the database built by data_pipeline.py. Used to
sanity-check a freshly generated dataset before any modelling: if the seasonal
peak is missing, the cycle-ratio correlation is absent or the AOG shortage list
is empty, the generated data does not reflect the operating model and the
downstream reliability figures cannot be trusted.

Runs against either backend. The SQLite schema produced by the pipeline and the
PostgreSQL/MySQL schemas differ in two places - table qualification (the 'aero'
schema) and the serviceable-quantity column name - both resolved at startup.

Author: Evangelos Tampachaniotis
Version: 1.0.0
License: MIT

Regulatory Framework:
    - EASA Part-M (EU 1321/2014) M.A.302 - the reliability programme; analyses
      4, 5, 7 and 8 are the standard views such a programme reviews.
    - EASA Part-145 - stores control; analysis 6 is the shortage report.
    - ATA/JASC 100 - chapter numbering used in the demand ranking.

Usage:
    python explore_data.py              # SQLite (default)
    python explore_data.py --postgres   # PostgreSQL
"""

import argparse
import sys
import os

import pandas as pd
import numpy as np

# SQLite database path, resolved relative to this file. Name retained for
# backwards compatibility with databases built by earlier pipeline versions.
SQLITE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aerosupply.db")


def get_connection(use_postgres=False):
    """
    Open a connection to whichever backend was requested.

    Args:
        use_postgres: bool - connect to PostgreSQL instead of SQLite.

    Returns:
        tuple (connection_or_engine, db_type_str) where db_type_str is
        "postgresql" or "sqlite".

    Notes:
        Imports are deferred into each branch so that running the default
        SQLite path does not require SQLAlchemy to be installed, and vice versa.
        PostgreSQL connection details are read from environment variables
        (POSTGRES_HOST/PORT/USER/PASSWORD/DATABASE), typically supplied by the
        gitignored .env file - no credential is ever written into source.
        A missing SQLite database is a setup problem with an obvious fix, so it
        exits with the instruction rather than raising a traceback.
    """
    if use_postgres:
        from sqlalchemy import create_engine

        # Credentials come from the environment, never from source. .env is
        # loaded when python-dotenv is available; otherwise the variables are
        # expected to be exported by the shell or the service manager.
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass

        # Assembled from discrete variables rather than one URL string so a
        # password is never concatenated into a value that might be logged.
        # Defaults describe a local trust-authenticated server with no password.
        host = os.environ.get("POSTGRES_HOST", "localhost")
        port = os.environ.get("POSTGRES_PORT", "5432")
        user = os.environ.get("POSTGRES_USER", "")
        password = os.environ.get("POSTGRES_PASSWORD", "")
        database = os.environ.get("POSTGRES_DATABASE", "daedalus_supply")

        # Only include the credential segment when a user is actually set;
        # an empty "@" prefix is not a valid SQLAlchemy URL.
        auth = f"{user}:{password}@" if user else ""
        engine = create_engine(f"postgresql://{auth}{host}:{port}/{database}")
        return engine, "postgresql"

    import sqlite3
    if not os.path.exists(SQLITE_PATH):
        print(f"Database not found: {SQLITE_PATH}")
        print("Run 'python data_pipeline.py' first")
        sys.exit(1)
    return sqlite3.connect(SQLITE_PATH), "sqlite"


def resolve_stock_column(conn, db_type, prefix):
    """
    Determine the name of the serviceable-quantity column in `inventory`.

    Args:
        conn: connection or SQLAlchemy engine.
        db_type: str - "sqlite" or "postgresql".
        prefix: str - schema qualifier for the table name ("aero." or "").

    Returns:
        str - "serviceable" or "quantity_serviceable".

    Notes:
        The pipeline's SQLite schema names this column `serviceable`; the
        PostgreSQL and MySQL schemas name it `quantity_serviceable`. Detecting
        it once here keeps the analysis queries below free of backend
        conditionals. The value comes from the database catalogue, never from
        user input, so interpolating it into SQL is safe.
    """
    try:
        # LIMIT 0 returns the column headers without reading any rows.
        cols = pd.read_sql(f"SELECT * FROM {prefix}inventory LIMIT 0", conn).columns
        return "quantity_serviceable" if "quantity_serviceable" in cols else "serviceable"
    except Exception:
        # Fall back to the SQLite spelling, which is the default backend.
        return "serviceable"


def section(title):
    """Print a numbered section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def run_query(conn, query, title=None):
    """
    Execute a query and print the result as a table.

    Args:
        conn: connection or SQLAlchemy engine.
        query: str - the SQL to run.
        title: str or None - sub-heading printed above the result.

    Returns:
        pd.DataFrame - the result, so callers can compute further statistics
        from it (analysis 8 does this).
    """
    if title:
        print(f"\n  {title}")
        print(f"  {'-' * 50}")
    df = pd.read_sql(query, conn)
    print(df.to_string(index=False))
    return df


def main():
    """Run all nine analyses in order against the selected backend."""
    parser = argparse.ArgumentParser(
        description="Daedalus Supply AI - run the standing analyses over the database")
    parser.add_argument("--postgres", action="store_true",
                        help="use PostgreSQL instead of SQLite")
    args = parser.parse_args()

    conn, db_type = get_connection(args.postgres)

    # PostgreSQL keeps every table in the 'aero' schema; SQLite has no schemas,
    # so the prefix is empty there.
    prefix = "aero." if db_type == "postgresql" else ""

    # Resolve the backend-specific column name once, up front.
    stock_col = resolve_stock_column(conn, db_type, prefix)

    # ------------------------------------------------------------------
    # 1. Fleet overview - utilisation ranking
    # ------------------------------------------------------------------
    # age_years is stored directly in SQLite but derived from manufacture_year
    # in the PostgreSQL schema, so the expression is chosen per backend rather
    # than pushed into the SQL as a CASE over a literal.
    age_expr = "age_years" if db_type == "sqlite" else "(2026 - manufacture_year)"

    section("1. FLEET OVERVIEW")
    run_query(conn, f"""
        SELECT tail_number, aircraft_model,
               {age_expr} as age,
               home_base, primary_role,
               total_flight_hours as total_fh,
               cycles_per_fh_ratio as cyc_per_fh
        FROM {prefix}fleet
        ORDER BY total_flight_hours DESC
    """, "Aircraft sorted by flight hours")

    # ------------------------------------------------------------------
    # 2. Catalogue composition
    # ------------------------------------------------------------------
    # Confirms the three part classes are populated and shows how much of each
    # class is AOG-critical - the split that drives every stocking decision.
    section("2. PARTS CATALOG BREAKDOWN")
    run_query(conn, f"""
        SELECT part_class,
               COUNT(*) as count,
               ROUND(AVG(unit_cost_eur), 0) as avg_cost_eur,
               SUM(CASE WHEN criticality = 'AOG' THEN 1 ELSE 0 END) as aog_count
        FROM {prefix}parts_catalog
        GROUP BY part_class
    """, "Parts by class")

    # ------------------------------------------------------------------
    # 3. Consumption ranking
    # ------------------------------------------------------------------
    # Ranked by event count rather than quantity or value: stock-out
    # probability follows how OFTEN a part is needed, not how many are used.
    # total_cost is carried alongside so the budget view is visible too.
    section("3. TOP 10 MOST DEMANDED PARTS")
    run_query(conn, f"""
        SELECT pd.part_number,
               pc.description,
               pc.part_class,
               pc.ata_chapter as ata,
               COUNT(*) as demand_events,
               SUM(pd.quantity_required) as total_qty,
               pc.unit_cost_eur,
               ROUND(SUM(pd.quantity_required) * pc.unit_cost_eur, 0) as total_cost
        FROM {prefix}part_demands pd
        JOIN {prefix}parts_catalog pc ON pd.part_number = pc.part_number
        GROUP BY pd.part_number, pc.description, pc.part_class, pc.ata_chapter, pc.unit_cost_eur
        ORDER BY demand_events DESC
        LIMIT 10
    """, "Highest demand parts (by event count)")

    # ------------------------------------------------------------------
    # 4. Scheduled versus unscheduled work per aircraft
    # ------------------------------------------------------------------
    # The headline reliability metric of any Part-M programme. A high
    # unscheduled ratio on an older, high-cycle-ratio, island-based airframe is
    # the pattern the whole project exists to predict.
    # UNSCHEDULED_FAILURE and PILOT_REPORT are counted together: both are
    # unplanned, they differ only in who detected the defect.
    section("4. FAILURE ANALYSIS BY AIRCRAFT AGE")
    run_query(conn, f"""
        SELECT f.tail_number,
               {age_expr.replace('age_years', 'f.age_years').replace('manufacture_year', 'f.manufacture_year')} as age,
               f.home_base,
               f.cycles_per_fh_ratio as cyc_ratio,
               COUNT(CASE WHEN wo.source IN ('UNSCHEDULED_FAILURE','PILOT_REPORT') THEN 1 END) as unscheduled,
               COUNT(CASE WHEN wo.source = 'SCHEDULED' THEN 1 END) as scheduled
        FROM {prefix}fleet f
        LEFT JOIN {prefix}work_orders wo ON f.tail_number = wo.tail_number
        GROUP BY f.tail_number, f.age_years, f.manufacture_year, f.home_base, f.cycles_per_fh_ratio
        ORDER BY unscheduled DESC
    """, "Scheduled vs unscheduled events per aircraft")

    # ------------------------------------------------------------------
    # 5. Seasonality
    # ------------------------------------------------------------------
    # substr(demand_date, 6, 2) extracts the month from an ISO 'YYYY-MM-DD'
    # string - portable across both backends without date functions.
    # Season boundaries match the pipeline's traffic model: summer Jun-Sep,
    # winter Dec-Feb. Summer should lead; if it does not, the seasonal factor
    # in the generator is not reaching the demand records.
    section("5. SEASONAL DEMAND PATTERN")
    run_query(conn, f"""
        SELECT
            CASE
                WHEN CAST(substr(demand_date, 6, 2) AS INTEGER) IN (6,7,8,9) THEN 'summer'
                WHEN CAST(substr(demand_date, 6, 2) AS INTEGER) IN (12,1,2) THEN 'winter'
                ELSE 'shoulder'
            END as season,
            COUNT(*) as demand_events,
            SUM(quantity_required) as total_qty
        FROM {prefix}part_demands
        GROUP BY season
        ORDER BY demand_events DESC
    """, "Demand by season (summer peak expected)")

    # ------------------------------------------------------------------
    # 6. AOG-critical shortages
    # ------------------------------------------------------------------
    # The only report here that demands immediate action: parts whose absence
    # grounds an aircraft, currently held below their minimum level.
    section("6. STATION STOCK ALERTS")
    run_query(conn, f"""
        SELECT i.station,
               pc.part_number,
               pc.description,
               pc.criticality,
               i.{stock_col} as on_hand,
               i.minimum_stock_level as min_level,
               (i.minimum_stock_level - i.{stock_col}) as shortage
        FROM {prefix}inventory i
        JOIN {prefix}parts_catalog pc ON i.part_number = pc.part_number
        WHERE i.{stock_col} < i.minimum_stock_level
          AND pc.criticality = 'AOG'
        ORDER BY shortage DESC
        LIMIT 10
    """, "AOG-critical parts below minimum stock")

    # ------------------------------------------------------------------
    # 7. Defect type distribution
    # ------------------------------------------------------------------
    # Percentages computed against the total finding count via a scalar
    # subquery. A rising CORROSION share is the classic trigger for an MSG-3
    # review of the Corrosion Prevention and Control Programme.
    section("7. FINDINGS BY TYPE")
    run_query(conn, f"""
        SELECT finding_type,
               COUNT(*) as count,
               ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM {prefix}findings), 1) as pct
        FROM {prefix}findings
        GROUP BY finding_type
        ORDER BY count DESC
    """, "What types of defects are found during inspections")

    # ------------------------------------------------------------------
    # 8. Does cyclic stress drive consumption?
    # ------------------------------------------------------------------
    # The central hypothesis of the whole dataset: short-sector operation
    # produces more cycles per flight hour, and therefore more part demands.
    # COUNT(pd.part_number) rather than COUNT(*) - with a LEFT JOIN, COUNT(*)
    # would score an aircraft with no demands as 1 instead of 0.
    section("8. CORRELATION: CYCLE RATIO vs FAILURES")
    df = run_query(conn, f"""
        SELECT f.tail_number,
               f.cycles_per_fh_ratio,
               COUNT(pd.part_number) as demand_count
        FROM {prefix}fleet f
        LEFT JOIN {prefix}part_demands pd ON f.tail_number = pd.tail_number
        GROUP BY f.tail_number, f.cycles_per_fh_ratio
    """, "Higher cycle ratio, more failures?")

    # Pearson correlation needs at least three points to mean anything.
    if len(df) > 2:
        corr = df["cycles_per_fh_ratio"].corr(df["demand_count"])
        print(f"\n  Pearson correlation (cycle_ratio vs demand_count): {corr:.3f}")
        # 0.3 is the conventional threshold for a moderate linear relationship.
        if corr > 0.3:
            print("  Positive correlation: short-haul operation drives more part demands")
        else:
            print("  Weak correlation: age and climate may dominate over cycle ratio")

    # ------------------------------------------------------------------
    # 9. Dataset size
    # ------------------------------------------------------------------
    section("9. DATABASE SIZE")
    if db_type == "sqlite":
        size_mb = os.path.getsize(SQLITE_PATH) / (1024 * 1024)
        print(f"  {os.path.basename(SQLITE_PATH)}: {size_mb:.1f} MB")

    # Row counts across the core tables. faa_sdr_raw is excluded because it is
    # optional and dwarfs everything else, which would obscure the synthetic
    # dataset's own proportions.
    tables = ["fleet", "flight_log", "parts_catalog", "inventory",
              "work_orders", "findings", "part_demands"]
    total = 0
    for t in tables:
        count = pd.read_sql(f"SELECT COUNT(*) as c FROM {prefix}{t}", conn).iloc[0, 0]
        total += count
        print(f"  {t:<25} {count:>8}")
    print(f"  {'TOTAL':<25} {total:>8}")

    # SQLAlchemy engines expose dispose(), raw DBAPI connections expose close().
    if hasattr(conn, "close"):
        conn.close()

    print(f"\n{'=' * 60}")
    print("  Exploration complete")
    print("  Next: python prediction_model.py")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
