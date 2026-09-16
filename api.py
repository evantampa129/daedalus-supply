#!/usr/bin/env python3
"""
Daedalus Supply AI - Module 6: REST API
=======================================
FastAPI service exposing the database built by Module 1 and enriched by
Modules 2 and 3 over HTTP, with the endpoint separation that role-based
access control is built on.

The surface is split in two, and the split is structural rather than
cosmetic:

    /api/user/*     what a technician or line-maintenance engineer needs.
                    Read-only, with exactly one exception - a part request,
                    which records a demand and touches no stock.
    /api/admin/*    what a supply officer or engineering manager needs.
                    Every write in the system lives here.

Routes arrive per module across the sessions that follow; this one carries
the shell, the database layer, the access seam and the service metadata.

There is no authentication in this version. That is deliberate and it is the
whole point of shipping the separation first: v1.3 adds JWT and bcrypt, and
when it does, the only change required is the body of require_admin() and
require_user(). Every endpoint already sits behind one of them, so no route
can be forgotten. Until then the service must not be exposed - bind it to
localhost, or run it with DAEDALUS_API_READONLY=1, which removes the admin
router from the application entirely.

Design decisions:
    - sqlite3 directly, not an ORM. The queries are the vetted, bounded set
      this system has used since Module 4; an ORM would add a translation
      layer over SQL that is already known to be correct.
    - Read endpoints open the database read-only (mode=ro), so a bug in a
      GET handler cannot write. Write endpoints open a separate connection
      and commit explicitly.
    - Every value from a client reaches the database as a bound parameter.
    - Pydantic models mirror the table schemas, so the OpenAPI document is
      generated from the same definitions the handlers return.

Author: Evangelos Tampachaniotis
Version: 1.2.0
License: MIT

Regulatory Framework:
    - EASA Part-M (EU 1321/2014) M.A.305 - the fleet endpoints serve the
      continuing-airworthiness record.
    - EASA Part-145 145.A.42 - only serviceable stock is reported as
      available, and a transfer may only move serviceable units.
    - EASA Part-145 145.A.55 - record keeping. Every write endpoint returns
      the resulting state, which is what v1.4's audit log will record as the
      "after" value.
    - MEL - the AOG / MEL / ROUTINE criticality that orders every alert.

Usage:
    uvicorn api:app --reload                 # development
    python api.py                            # same, via __main__
    curl http://127.0.0.1:8000/api/meta      # service metadata
    http://127.0.0.1:8000/docs               # interactive OpenAPI browser
"""

import os
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime
from typing import Any, Dict, Generator, Generic, List, Optional, TypeVar

from fastapi import (APIRouter, Depends, FastAPI, HTTPException, Path, Query,
                     Request, status)
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field, field_validator

from config import get_config

# ============================================================================
# CONFIGURATION
# ============================================================================
# Nothing here is hardcoded to a machine. The database location comes from the
# environment or defaults to the file beside this module, which is the same
# rule dashboard.py follows, and no credential appears anywhere: SQLite is a
# file. When the service moves to PostgreSQL the DSN will come from the
# environment exactly as load_to_postgres.py already reads it.

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.environ.get("DAEDALUS_DB_PATH", os.path.join(BASE_DIR, "aerosupply.db"))

# Read the shipped version rather than repeating it in a constant, so the
# OpenAPI document cannot disagree with the VERSION file.
try:
    with open(os.path.join(BASE_DIR, "VERSION")) as _f:
        API_VERSION = _f.read().strip()
except OSError:
    API_VERSION = "0.0.0"

# Hard switch that removes the admin router from the application. Intended for
# the case where the service is reachable by anything other than localhost
# before v1.3 lands: without authentication, an exposed write endpoint is an
# open door to the maintenance record.
READ_ONLY_MODE = os.environ.get("DAEDALUS_API_READONLY", "").lower() in {"1", "true", "yes"}

# Pagination bounds. The default is small because the common case is a browser
# or a technician's tablet; the ceiling exists so one request cannot ask for
# the 54,000-row flight log in a single response.
DEFAULT_LIMIT = 50
MAX_LIMIT = 500

CFG = get_config()

# Criticality ordering, used by every endpoint that ranks alerts or sources.
# AOG first: it is the only class of shortage that stops an aircraft flying
# today.
CRIT_ORDER_SQL = "CASE criticality WHEN 'AOG' THEN 0 WHEN 'MEL' THEN 1 ELSE 2 END"


# ============================================================================
# DATABASE LAYER
# ============================================================================

def _connect(read_only: bool) -> sqlite3.Connection:
    """
    Open a connection to the operational database.

    Args:
        read_only: bool - True opens the file through the URI form with
            mode=ro, which makes any writing statement fail at the driver.

    Returns:
        sqlite3.Connection with a row factory that yields mappings.

    Notes:
        Read-only is not a convention here, it is enforced by SQLite. A GET
        handler that somehow issued an UPDATE would raise rather than modify
        a continuing-airworthiness record, which is a stronger guarantee than
        a code review of every handler.
    """
    # check_same_thread=False because a request is served across two threads:
    # FastAPI resolves a synchronous dependency in one worker thread and may
    # run the handler in another. The connection is still used by exactly one
    # request and closed when it ends - it is never shared between requests -
    # so the guarantee the flag relaxes is not one this service relies on.
    if read_only:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, check_same_thread=False)
    else:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        # Foreign keys are off by default in SQLite and have to be enabled per
        # connection. The schema declares them; without this they are inert.
        conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def read_db() -> Generator[sqlite3.Connection, None, None]:
    """
    FastAPI dependency yielding a read-only connection for one request.

    Args:
        (none)

    Returns:
        Generator yielding sqlite3.Connection.

    Notes:
        One connection per request, closed in the finally block whether the
        handler returned or raised. SQLite connections are cheap to open on a
        local file, so pooling would add machinery for no measurable gain at
        this scale.
    """
    conn = _connect(read_only=True)
    try:
        yield conn
    finally:
        conn.close()


def write_db() -> Generator[sqlite3.Connection, None, None]:
    """
    FastAPI dependency yielding a writable connection for one request.

    Args:
        (none)

    Returns:
        Generator yielding sqlite3.Connection.

    Notes:
        The handler is responsible for committing. Anything that raises before
        the commit leaves the transaction to be rolled back here, so a failed
        write never lands half-applied - a partially executed stock transfer
        would leave units in neither station.
    """
    conn = _connect(read_only=False)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def fetch_all(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
    """
    Run a query and return its rows as dictionaries.

    Args:
        conn: sqlite3.Connection - from one of the dependencies above.
        sql: str - the statement, with ? placeholders for every client value.
        params: tuple - the values bound to those placeholders.

    Returns:
        list of dict - one entry per row.

    Notes:
        Client input is never formatted into the SQL string. Where a statement
        needs a variable number of placeholders, they are generated from the
        length of the value list and the values are still bound.
    """
    return [dict(row) for row in conn.execute(sql, params).fetchall()]


def fetch_one(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> Optional[Dict[str, Any]]:
    """
    Run a query expected to match at most one row.

    Args:
        conn: sqlite3.Connection.
        sql: str - the statement.
        params: tuple - bound values.

    Returns:
        dict or None when nothing matched.
    """
    row = conn.execute(sql, params).fetchone()
    return dict(row) if row else None


def require_found(value: Optional[Any], detail: str) -> Any:
    """
    Turn a missing row into a 404 with a message that names what was missing.

    Args:
        value: the result of a lookup, or None.
        detail: str - what the client asked for, e.g. "Aircraft SX-XXX".

    Returns:
        The value, when it is not None.

    Raises:
        HTTPException 404.

    Notes:
        An unknown tail number is a client error, not a server fault, and the
        message says which identifier failed so the caller does not have to
        guess which of several path parameters was wrong.
    """
    if value is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"{detail} not found")
    return value


@contextmanager
def stock_column(conn: sqlite3.Connection):
    """
    Yield the serviceable-quantity column name for the active schema.

    Args:
        conn: sqlite3.Connection.

    Returns:
        Context manager yielding str - "serviceable" or "quantity_serviceable".

    Notes:
        The SQLite pipeline names this column `serviceable`; the PostgreSQL and
        MySQL schemas name it `quantity_serviceable`. The value comes from the
        database catalogue and never from a client, so interpolating it into a
        statement is safe - and doing it once here keeps every query in this
        module free of backend conditionals.
    """
    cols = [r[1] for r in conn.execute("PRAGMA table_info(inventory)").fetchall()]
    yield "quantity_serviceable" if "quantity_serviceable" in cols else "serviceable"


# ============================================================================
# ACCESS CONTROL SEAM
# ============================================================================
# v1.2 separates the surfaces; v1.3 authenticates them. These two dependencies
# are the entire insertion point for that work: every route in this module
# already depends on one of them, so when they start verifying a JWT and a
# role claim there is no route left to forget. Until then they exist to make
# the boundary explicit in the code and in the OpenAPI document.

def require_user() -> Dict[str, Any]:
    """
    Identify the caller as an operator-tier principal.

    Args:
        (none)

    Returns:
        dict - the principal. Anonymous in this version.

    Notes:
        v1.3 replaces the body with JWT verification and returns the decoded
        subject. The signature stays, so no handler changes.
    """
    return {"role": "operator", "authenticated": False}


def require_admin() -> Dict[str, Any]:
    """
    Identify the caller as an administrator-tier principal.

    Args:
        (none)

    Returns:
        dict - the principal. Anonymous in this version.

    Raises:
        HTTPException 503 when the service runs with writes disabled.

    Notes:
        The 503 is not authentication - it is the operator of the service
        having declared, through DAEDALUS_API_READONLY, that this deployment
        holds no write surface at all. Refusing here as well as omitting the
        router means a stale client gets an honest answer rather than a 404
        that looks like a typo.
    """
    if READ_ONLY_MODE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Administrative endpoints are disabled (DAEDALUS_API_READONLY).",
        )
    return {"role": "administrator", "authenticated": False}


# ============================================================================
# COMMON MODELS
# ============================================================================
# Pydantic models mirror the table schemas, so the OpenAPI document is
# generated from the same definitions the handlers return rather than from a
# separate specification that can drift away from the code.

ItemT = TypeVar("ItemT")


class Page(BaseModel, Generic[ItemT]):
    """
    Envelope carrying a slice of a collection and the size of the whole.

    Generic in the row type, so Page[Aircraft] and Page[StockLine] each
    publish their own item schema in the OpenAPI document. An envelope typed
    as a list of anything would generate clients that hand back untyped
    dictionaries, and it would not have caught the column-type mismatch this
    form surfaced on the first run against real data.
    """

    total: int = Field(..., description="Rows matching the filters, before paging")
    limit: int = Field(..., description="Maximum rows in this response")
    offset: int = Field(..., description="Rows skipped")
    items: List[ItemT] = Field(..., description="The rows themselves")


class ServiceMeta(BaseModel):
    """What this service is, and what it is running against."""

    name: str
    version: str
    database: str = Field(..., description="Database file in use")
    database_present: bool
    read_only_mode: bool = Field(..., description="True when the admin router is disabled")
    hardware_profile: str = Field(..., description="MINIMAL / STANDARD / FULL, from config.py")
    authentication: str = Field(..., description="Auth scheme in force")
    tables: Dict[str, int] = Field(..., description="Row counts for the principal tables")


class HealthStatus(BaseModel):
    """Liveness and readiness in one answer."""

    status: str = Field(..., description="ok when the database answered a query")
    database: str = Field(..., description="reachable / unreachable")
    detail: Optional[str] = None


class ErrorResponse(BaseModel):
    """The shape every error in this service takes."""

    detail: str


class Acknowledgement(BaseModel):
    """Result of a write, including the state the record now holds."""

    ok: bool = True
    message: str
    record: Optional[Dict[str, Any]] = Field(
        None, description="The resulting row - the 'after' value v1.4 will audit")


# ============================================================================
# APPLICATION
# ============================================================================

TAGS_METADATA = [
    {"name": "meta", "description": "Service metadata and health."},
    {"name": "fleet", "description": "Continuing-airworthiness record per airframe "
                                     "(EASA Part-M M.A.305). Operator tier, read-only."},
    {"name": "inventory", "description": "Parts catalogue and stock position. Only "
                                         "serviceable units count as available "
                                         "(Part-145 145.A.42). Operator tier, read-only."},
    {"name": "predictions", "description": "Failure risk and expendable demand, from "
                                           "Module 2. Operator tier, read-only."},
    {"name": "logistics", "description": "Stock recommendations, transfers and AOG "
                                         "routing, from Module 3. Operator tier, "
                                         "read-only."},
    {"name": "sdr", "description": "Real FAA Service Difficulty Reports. Operator tier, "
                                   "read-only."},
    {"name": "requests", "description": "Part requests raised by maintenance staff. "
                                        "Records a demand; never moves stock."},
    {"name": "admin", "description": "Administrator tier. Every write operation in the "
                                     "system. No authentication in v1.2 - see the module "
                                     "docstring before exposing this service."},
]

DESCRIPTION = """
REST interface to the Daedalus Supply AI database.

**Two surfaces.** `/api/user/*` is what a technician needs and is read-only
apart from raising a part request. `/api/admin/*` carries every write in the
system. The separation is structural: each route depends on `require_user` or
`require_admin`, which is where v1.3 will verify a JWT and a role claim.

**No authentication in this version.** Run the service on localhost, or with
`DAEDALUS_API_READONLY=1` to drop the administrative surface entirely.

**Only serviceable stock is ever reported as available**, per EASA Part-145
145.A.42. Unserviceable units awaiting shop input are reported in their own
field and are never offered as a source for a transfer or an AOG request.
"""

app = FastAPI(
    title="Daedalus Supply AI API",
    description=DESCRIPTION,
    version=API_VERSION,
    openapi_tags=TAGS_METADATA,
    contact={"name": "Evangelos Tampachaniotis"},
    license_info={"name": "MIT"},
)


@app.exception_handler(sqlite3.OperationalError)
async def database_error_handler(request: Request, exc: sqlite3.OperationalError) -> JSONResponse:
    """
    Turn a database-level failure into a 503 rather than a 500.

    Args:
        request: Request - the failing request.
        exc: sqlite3.OperationalError - what the driver raised.

    Returns:
        JSONResponse with status 503.

    Notes:
        The two cases that reach here in practice are a missing database file
        and a table written by a module that has not been run yet. Neither is
        a fault in the request, and both are fixed by running a pipeline
        command, so the response says which one rather than presenting a
        stack trace as an internal error.
    """
    missing_db = not os.path.exists(DB_PATH)
    detail = (f"Database not found at {DB_PATH}. Build it with: python data_pipeline.py"
              if missing_db else
              f"Database error: {exc}. The table may be written by a module that has "
              f"not been run yet.")
    return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        content={"detail": detail})


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Send a bare browser hit to the interactive documentation."""
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthStatus, tags=["meta"])
def health() -> HealthStatus:
    """
    Report whether the service can reach its database.

    Args:
        (none)

    Returns:
        HealthStatus - ok only when a query actually succeeded.

    Notes:
        Deliberately issues a query rather than checking that the file exists.
        A present but unreadable database is exactly the failure a health
        check is supposed to catch, and a stat() call would report it healthy.
        This endpoint takes no dependency, so it stays answerable when the
        database is the thing that is broken. Defined synchronously, like
        every handler here that touches SQLite: the driver blocks, and
        blocking inside an async handler would stall the event loop for every
        other request in flight.
    """
    try:
        conn = _connect(read_only=True)
        try:
            conn.execute("SELECT 1 FROM fleet LIMIT 1").fetchone()
        finally:
            conn.close()
        return HealthStatus(status="ok", database="reachable")
    except Exception as exc:
        return HealthStatus(status="degraded", database="unreachable", detail=str(exc))


@app.get("/api/meta", response_model=ServiceMeta, tags=["meta"])
def service_meta(conn: sqlite3.Connection = Depends(read_db)) -> ServiceMeta:
    """
    Describe the service and the dataset behind it.

    Args:
        conn: sqlite3.Connection - injected read-only connection.

    Returns:
        ServiceMeta - version, database, hardware profile and row counts.

    Notes:
        The row counts are the fastest way for a client to tell whether the
        upstream modules have been run: a database with 0 stock
        recommendations has not seen logistics_optimizer.py yet, and a client
        can say so rather than showing an empty screen with no explanation.
        Tables that do not exist report 0 rather than failing the request.
    """
    counts: Dict[str, int] = {}
    for table in ("fleet", "stations", "parts_catalog", "inventory", "part_demands",
                  "work_orders", "findings", "stock_recommendations",
                  "transfer_recommendations", "faa_sdr_raw"):
        try:
            row = conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()
            counts[table] = int(row["n"])
        except sqlite3.OperationalError:
            # Written by an optional module that has not run. Absence is a
            # legitimate state, and reporting it as 0 is more useful to a
            # client than failing the whole metadata request.
            counts[table] = 0

    return ServiceMeta(
        name="Daedalus Supply AI API",
        version=API_VERSION,
        database=os.path.basename(DB_PATH),
        database_present=os.path.exists(DB_PATH),
        read_only_mode=READ_ONLY_MODE,
        hardware_profile=CFG["profile"],
        authentication="none (v1.2 - JWT arrives in v1.3)",
        tables=counts,
    )


# ============================================================================
# ROUTER REGISTRATION
# ============================================================================
# Registered last, after every route is defined, so the OpenAPI document is
# assembled from routes that are all in place.

# No routers yet: this session carries the shell, the database layer and the
# access seam. Each module's routes are registered here as they are built.


# ============================================================================
# ENTRY POINT
# ============================================================================

def main() -> None:
    """
    Run the service with uvicorn.

    Args:
        (none)

    Returns:
        None

    Notes:
        Binds to 127.0.0.1 rather than 0.0.0.0, and that is not a placeholder.
        Until v1.3 adds authentication, the administrative surface is
        unauthenticated, so the default must be a service that nothing outside
        the machine can reach. Overriding the host is a deliberate act.
    """
    import uvicorn

    host = os.environ.get("DAEDALUS_API_HOST", "127.0.0.1")
    port = int(os.environ.get("DAEDALUS_API_PORT", "8000"))

    if host not in {"127.0.0.1", "localhost"} and not READ_ONLY_MODE:
        # Loud, once, at startup: the combination that matters is a reachable
        # host and an enabled write surface with no authentication in front.
        print("WARNING: binding to a non-local address with administrative "
              "endpoints enabled and no authentication. Set "
              "DAEDALUS_API_READONLY=1 or wait for v1.3.")

    print(f"Daedalus Supply AI API {API_VERSION} - database {DB_PATH}")
    print(f"Profile {CFG['profile']} | admin router "
          f"{'disabled' if READ_ONLY_MODE else 'enabled'}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
