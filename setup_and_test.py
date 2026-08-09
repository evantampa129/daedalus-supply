#!/usr/bin/env python3
"""
Daedalus Supply AI - Setup Verification
=======================================
Checks the environment, then runs the data pipeline and verifies the resulting
database has the expected tables and row counts.

Author: Evangelos Tampachaniotis
Version: 1.0.0
License: MIT

Usage:
    python setup_and_test.py
"""

import subprocess
import sys
import os

# Project root, resolved from this file so the script works from any cwd.
ROOT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(ROOT, "aerosupply.db")

# Minimum row counts per table. Deliberately floors rather than exact figures:
# the pipeline is seeded and reproducible, but the SDR corpus and the hardware
# profile both affect totals, so an exact match would produce false failures.
EXPECTED_MIN_ROWS = {
    "fleet": 15,
    "flight_log": 50000,
    "parts_catalog": 50,
    "inventory": 250,
    "work_orders": 1000,
    "findings": 100,
    "part_demands": 1000,
    "inspection_program": 7,
    "stations": 5,
}


def check_python_version():
    """
    Verify the interpreter is Python 3.9 or newer.

    Returns:
        bool - True if the version is sufficient.

    Notes:
        3.9 is the floor because the project uses dict merge semantics and the
        pandas 2.x / numpy 1.24 baselines in requirements.txt, none of which
        support earlier releases.
    """
    v = sys.version_info
    ok = v.major == 3 and v.minor >= 9
    print(f"  Python {v.major}.{v.minor}.{v.micro}" + ("" if ok else "  (need 3.9+)"))
    return ok


def check_dependency(name, import_name=None):
    """
    Report whether a package is importable.

    Args:
        name: str - package name as it appears on PyPI, used in the output.
        import_name: str or None - module name to import when it differs from
            the package name (scikit-learn imports as sklearn).

    Returns:
        bool - True if the import succeeded.
    """
    try:
        __import__(import_name or name)
        print(f"  {name}: installed")
        return True
    except ImportError:
        print(f"  {name}: missing")
        return False


def check_command(name, binary):
    """
    Report whether an external database client is on PATH.

    Args:
        name: str - display name, e.g. "PostgreSQL".
        binary: str - executable to probe, e.g. "psql".

    Returns:
        bool - True if the command ran successfully.

    Notes:
        Both servers are optional; the project runs entirely on SQLite. A
        missing binary raises FileNotFoundError, which is the expected outcome
        on a machine that never installed the client, not an error condition.
    """
    try:
        result = subprocess.run([binary, "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  {name}: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    print(f"  {name}: not found (optional)")
    return False


def test_pipeline():
    """
    Run data_pipeline.py and verify the database it produces.

    Returns:
        bool - True if the pipeline exited cleanly and every table meets its
        minimum row count.

    Notes:
        Runs the pipeline as a subprocess rather than importing it, so a crash
        or a sys.exit inside the pipeline is reported here instead of killing
        this verification run.
    """
    print("\n  Running data_pipeline.py")
    result = subprocess.run(
        [sys.executable, "data_pipeline.py"],
        capture_output=True, text=True, cwd=ROOT
    )

    if result.returncode != 0:
        print(f"  Pipeline failed:\n{result.stderr}")
        return False

    if not os.path.exists(DB_PATH):
        print(f"  {os.path.basename(DB_PATH)} was not created")
        return False

    # Imported here rather than at module level: it is only needed once the
    # pipeline has actually produced a database to inspect.
    import sqlite3
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    all_ok = True

    # Count rows in each expected table and compare against its floor.
    for table, min_count in EXPECTED_MIN_ROWS.items():
        try:
            count = cursor.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            ok = count >= min_count
            note = "" if ok else f"  (expected >= {min_count})"
            print(f"    {table:<22} {count:>8} rows{note}")
            if not ok:
                all_ok = False
        except Exception as e:
            # A missing table means the pipeline did not complete that stage.
            print(f"    {table:<22} error: {e}")
            all_ok = False

    conn.close()
    return all_ok


def main():
    """Run every check in order and print a single pass/fail verdict."""
    print("Daedalus Supply AI - setup verification")

    print("\n[1] Environment")
    py_ok = check_python_version()

    print("\n[2] Core dependencies")
    core_ok = all([
        check_dependency("pandas"),
        check_dependency("numpy"),
        check_dependency("sqlalchemy"),
    ])

    print("\n[3] ML dependencies")
    for name, mod in [("scikit-learn", "sklearn"), ("xgboost", None),
                      ("lifelines", None), ("matplotlib", None), ("seaborn", None)]:
        check_dependency(name, mod)

    print("\n[4] Database drivers")
    check_dependency("psycopg2")
    check_dependency("pymysql")

    print("\n[5] Database servers")
    check_command("PostgreSQL", "psql")
    check_command("MySQL", "mysql")

    # The pipeline needs the core dependencies; running it without them would
    # only reproduce the import errors already reported above.
    print("\n[6] Pipeline")
    if py_ok and core_ok:
        pipeline_ok = test_pipeline()
    else:
        print("  Skipped - fix the failures above first")
        pipeline_ok = False

    print()
    if pipeline_ok:
        print("Setup verified. Next: python explore_data.py")
    else:
        print("Setup incomplete. Install dependencies with:")
        print("  pip install -r requirements.txt")


if __name__ == "__main__":
    main()
