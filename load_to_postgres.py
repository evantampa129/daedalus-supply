#!/usr/bin/env python3
"""
Daedalus Supply AI - Database Migration Tool
============================================
Transfers the SQLite database produced by data_pipeline.py into PostgreSQL or
MySQL, so the dataset can be served from a proper database server with foreign
keys, enumerated types and the reporting views defined in the schema files.

SQLite is the working format because the pipeline must run with no server
installed. PostgreSQL/MySQL is the deployment format: referential integrity,
concurrent access and the PL/pgSQL helpers that SQLite cannot provide.

Author: Evangelos Tampachaniotis
Version: 1.0.0
License: MIT

Regulatory Framework:
    - EASA Part-M (EU 1321/2014) M.A.305 - the continuing-airworthiness record
      system. Enforced foreign keys in the server schema are what prevent an
      orphaned work order or a finding against an unknown aircraft, which is a
      record-integrity requirement rather than a database preference.
    - EASA Part-145 145.A.55 - maintenance records retention.

Prerequisites:
    1. Target database created:
       sudo -u postgres createdb daedalus_supply

    2. Schema loaded:
       psql -d daedalus_supply -f schema_postgres.sql

    3. Dependencies:
       pip install psycopg2-binary sqlalchemy pandas

Usage:
    python load_to_postgres.py
    python load_to_postgres.py --db-url postgresql://user:pass@host/daedalus_supply
    python load_to_postgres.py --db-url mysql+pymysql://user:pass@host/daedalus_supply
"""

import argparse
import os
import sqlite3
import sys

import pandas as pd
from sqlalchemy import create_engine, text

# Load database credentials from .env when python-dotenv is installed.
# Optional by design: the --db-url argument covers every case on its own, and
# requiring a dependency just to read a file would be disproportionate.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ----------------------------------------------------------------------------
# Transfer order. This sequence is mandatory, not cosmetic: the server schemas
# declare foreign keys, so a child row inserted before its parent is rejected.
#   stations, inspection_program - reference data, no dependencies
#   fleet, parts_catalog         - depend on stations
#   inventory, flight_log        - depend on fleet / parts_catalog
#   work_orders                  - depends on fleet
#   findings, part_demands       - depend on work_orders and parts_catalog
# ----------------------------------------------------------------------------
TABLES_ORDERED = [
    "stations",
    "inspection_program",
    "fleet",
    "parts_catalog",
    "inventory",
    "flight_log",
    "work_orders",
    "findings",
    "part_demands",
]

# Source database. Name retained for backwards compatibility with databases
# produced by earlier versions of the pipeline.
SQLITE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aerosupply.db")


def load_from_sqlite(table: str) -> pd.DataFrame:
    """
    Read one table in full from the SQLite source.

    Args:
        table: str - table name, taken from TABLES_ORDERED (never user input).

    Returns:
        pd.DataFrame - the complete table.

    Notes:
        A fresh connection per table rather than one shared handle: the tables
        are transferred sequentially and the largest by far (flight_log,
        faa_sdr_raw) are better released as soon as they are written.
    """
    conn = sqlite3.connect(SQLITE_PATH)
    df = pd.read_sql(f"SELECT * FROM {table}", conn)
    conn.close()
    return df


def detect_db_type(url: str) -> str:
    """
    Classify the target backend from a SQLAlchemy URL.

    Args:
        url: str - SQLAlchemy connection URL.

    Returns:
        str - "postgresql", "mysql" or "unknown".

    Notes:
        Both "postgresql://" and the legacy "postgres://" prefix are accepted,
        since older tooling and hosted providers still emit the latter.
        The backend determines two things: whether a schema qualifier applies
        (PostgreSQL only) and which setup instructions to print on failure.
    """
    if "postgresql" in url or "postgres" in url:
        return "postgresql"
    elif "mysql" in url:
        return "mysql"
    else:
        return "unknown"


def transfer_table(engine, table: str, df: pd.DataFrame, db_type: str, schema: str = None):
    """
    Write one table into the target database, replacing its contents.

    Args:
        engine: SQLAlchemy Engine for the target database.
        table: str - target table name.
        df: pd.DataFrame - rows read from SQLite.
        db_type: str - "postgresql" or "mysql".
        schema: str or None - PostgreSQL schema qualifier ("aero").

    Returns:
        bool - True if the table was written successfully.

    Notes:
        Transformation: SQLite column names -> server schema column names.
        The pipeline writes `serviceable` / `unserviceable`; the server schemas
        use `quantity_serviceable` / `quantity_unserviceable`. Renaming here
        rather than changing the pipeline keeps the SQLite database readable by
        every existing local tool.

        DELETE rather than DROP: the server tables carry constraints, indexes
        and view dependencies defined by the schema files, all of which a
        to_sql replace would silently destroy. DELETE empties the table and
        leaves the structure intact, which makes the load re-runnable.
    """
    # SQLite -> server column name mapping.
    col_renames = {
        "serviceable": "quantity_serviceable",
        "unserviceable": "quantity_unserviceable",
    }

    # Rename only when the source name is present and the target name is not,
    # so a database already using the server spelling passes through untouched.
    for old, new in col_renames.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    # The server inventory schema has no separate quantity_on_hand column; the
    # serviceable quantity is the authoritative available figure. Fall back to
    # quantity_on_hand only if the serviceable column is genuinely absent.
    if table == "inventory" and "quantity_on_hand" in df.columns:
        if "quantity_serviceable" not in df.columns:
            df = df.rename(columns={"quantity_on_hand": "quantity_serviceable"})

    try:
        # Empty the target inside a transaction so a failure leaves the table
        # as it was rather than half-cleared.
        with engine.begin() as conn:
            if db_type == "postgresql" and schema:
                conn.execute(text(f"DELETE FROM {schema}.{table}"))
            else:
                conn.execute(text(f"DELETE FROM {table}"))

        # append (not replace) preserves the schema-defined structure.
        # method="multi" batches rows into multi-value INSERTs, and a 500-row
        # chunksize keeps each statement below the parameter limits both
        # servers impose while still cutting round trips by orders of magnitude
        # on the 54,000-row flight log.
        kwargs = {"if_exists": "append", "index": False, "method": "multi", "chunksize": 500}
        if db_type == "postgresql" and schema:
            kwargs["schema"] = schema

        df.to_sql(table, engine, **kwargs)
        print(f"  {table:<25} {len(df):>8} records loaded")
        return True

    except Exception as e:
        # One table failing (missing target table, constraint violation, type
        # mismatch) must not abort the whole migration - the remaining tables
        # are still worth loading, and the message identifies what to fix.
        print(f"  {table:<25} ERROR: {e}")
        return False


def main():
    """Connect to the target database and transfer every table in order."""
    parser = argparse.ArgumentParser(
        description="Load Daedalus Supply AI data into PostgreSQL or MySQL")
    parser.add_argument(
        "--db-url",
        default="postgresql://localhost/daedalus_supply",
        help="SQLAlchemy database URL (default: postgresql://localhost/daedalus_supply)"
    )
    parser.add_argument(
        "--schema",
        default=None,
        help="PostgreSQL schema name (default: 'aero' for PostgreSQL, none for MySQL)"
    )
    args = parser.parse_args()

    db_type = detect_db_type(args.db_url)
    schema = args.schema

    # PostgreSQL keeps every table in the 'aero' schema (see
    # schema_postgres.sql). MySQL has no equivalent, so schema stays None.
    if schema is None and db_type == "postgresql":
        schema = "aero"

    print("=" * 60)
    print("  Daedalus Supply AI - Database Loader")
    print("=" * 60)
    print(f"\n  Source:  {SQLITE_PATH}")
    print(f"  Target:  {args.db_url}")
    print(f"  Type:    {db_type}")
    if schema:
        print(f"  Schema:  {schema}")

    # Nothing to migrate without the source database.
    if not os.path.exists(SQLITE_PATH):
        print(f"\n  SQLite database not found: {SQLITE_PATH}")
        print("  Run 'python data_pipeline.py' first to generate it.")
        sys.exit(1)

    # Verify connectivity before reading anything: a failed connection after
    # loading a 54,000-row table into memory wastes the work.
    try:
        engine = create_engine(args.db_url)
        with engine.connect() as conn:
            # Minimal round trip that works on both backends.
            conn.execute(text("SELECT 1"))
        print(f"\n  Connected to {db_type}")
    except Exception as e:
        # Almost always a missing server, missing database or wrong
        # credentials, so print the exact setup commands rather than a bare error.
        print(f"\n  Connection failed: {e}")
        print()
        if db_type == "postgresql":
            print("  Setup PostgreSQL:")
            print("    sudo apt install postgresql")
            print("    sudo -u postgres createuser --interactive")
            print("    sudo -u postgres createdb daedalus_supply")
            print("    psql -d daedalus_supply -f schema_postgres.sql")
        elif db_type == "mysql":
            print("  Setup MySQL:")
            print("    sudo apt install mysql-server")
            print("    sudo mysql -e \"CREATE DATABASE daedalus_supply;\"")
            print("    mysql -u root -p daedalus_supply < schema_mysql.sql")
        sys.exit(1)

    # Set the search path so unqualified names resolve inside the aero schema.
    # Best effort: to_sql is given the schema explicitly anyway, and some
    # connection poolers reset session state between checkouts.
    if db_type == "postgresql" and schema:
        try:
            with engine.begin() as conn:
                conn.execute(text(f"SET search_path TO {schema}, public"))
        except Exception:
            pass

    # --- Transfer, parents before children ---
    print(f"\n  Loading tables:")
    success_count = 0
    for table in TABLES_ORDERED:
        try:
            df = load_from_sqlite(table)
            # An empty source table is legitimate (no findings on a very short
            # simulation run); skip it rather than issuing an empty insert.
            if df.empty:
                print(f"  {table:<25} (empty in SQLite, skipping)")
                continue
            if transfer_table(engine, table, df, db_type, schema):
                success_count += 1
        except Exception as e:
            print(f"  {table:<25} ERROR reading from SQLite: {e}")

    # --- Optional: the real FAA SDR corpus ---
    # Handled separately because it exists only when the pipeline was run with
    # CSVs in raw_data/, and it has no foreign-key relationships to order.
    try:
        sdr_df = load_from_sqlite("faa_sdr_raw")
        if not sdr_df.empty:
            transfer_table(engine, "faa_sdr_raw", sdr_df, db_type, schema)
            success_count += 1
    except Exception:
        # Table absent - expected when running without SDR data.
        pass

    print(f"\n  {'=' * 50}")
    print(f"  Loaded {success_count}/{len(TABLES_ORDERED)} tables")

    # Verification commands, so the user can confirm the load independently.
    if db_type == "postgresql":
        print(f"\n  Verify with:")
        print(f"    psql -d daedalus_supply -c 'SET search_path TO aero; SELECT * FROM v_fleet_status;'")
    elif db_type == "mysql":
        print(f"\n  Verify with:")
        print(f"    mysql -u root -p daedalus_supply -e 'SELECT * FROM v_part_reliability;'")

    print(f"\n  Or in Python:")
    print(f"    from sqlalchemy import create_engine")
    print(f"    import pandas as pd")
    print(f"    engine = create_engine(\"{args.db_url}\")")
    print(f"    pd.read_sql(\"SELECT * FROM {'aero.' if schema else ''}v_fleet_status\", engine)")
    print("=" * 60)


if __name__ == "__main__":
    main()
