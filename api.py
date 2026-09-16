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

The whole operator surface is in place; the administrative surface follows.

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
# OPERATOR TIER - FLEET
# ============================================================================
# Everything under /api/user is what a technician or a line-maintenance
# engineer needs to do their job. The router carries require_user as a
# router-level dependency, so a route added later cannot accidentally escape
# the tier by forgetting to declare it.

user_router = APIRouter(
    prefix="/api/user",
    dependencies=[Depends(require_user)],
    responses={404: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)


class Aircraft(BaseModel):
    """One airframe in the continuing-airworthiness register."""

    tail_number: str = Field(..., description="Registration, e.g. SX-ABK")
    aircraft_model: str
    manufacture_year: Optional[int] = None
    age_years: Optional[float] = None
    home_base: str = Field(..., description="IATA code of the base station")
    primary_role: Optional[str] = Field(None, description="short_haul / medium_haul")
    total_flight_hours: Optional[float] = None
    total_flight_cycles: Optional[int] = None
    cycles_per_fh_ratio: Optional[float] = Field(
        None, description="Cycles per flight hour - the cyclic-stress measure")
    daily_utilization_fh: Optional[float] = None
    status: Optional[str] = None


class MaintenanceEvent(BaseModel):
    """A single work order raised against an airframe."""

    work_order_id: str
    check_type: Optional[str] = None
    scheduled_date: Optional[str] = None
    status: Optional[str] = None
    aircraft_fh_at_check: Optional[float] = None
    source: Optional[str] = Field(
        None, description="SCHEDULED / UNSCHEDULED_FAILURE / PILOT_REPORT")


class AircraftDetail(Aircraft):
    """An airframe with the maintenance picture attached."""

    work_orders_total: int = 0
    work_orders_scheduled: int = 0
    work_orders_unscheduled: int = Field(
        0, description="Unscheduled failures and pilot reports together - both "
                       "are unplanned and differ only in who found the defect")
    part_demands: int = Field(0, description="Part demands recorded against this airframe")
    findings: int = Field(0, description="Findings raised against this airframe")
    salt_exposure: Optional[float] = Field(
        None, description="Chloride-load index of the home base, 0.0-1.0")


class Station(BaseModel):
    """A base station and the stock it holds."""

    station_code: str
    name: str
    climate: Optional[str] = None
    salt_exposure: Optional[float] = None
    based_aircraft: int = 0
    stock_lines: int = Field(0, description="Part-station lines carried at this station")
    lines_below_minimum: int = 0
    stock_value_eur: float = 0.0


def _paginate(conn: sqlite3.Connection, base_sql: str, count_sql: str,
              params: tuple, limit: int, offset: int, model: type) -> Page:
    """
    Run a filtered query twice: once for the page, once for the total.

    Args:
        conn: sqlite3.Connection.
        base_sql: str - the SELECT, without LIMIT or OFFSET.
        count_sql: str - the matching COUNT over the same FROM and WHERE.
        params: tuple - bound values shared by both statements.
        limit: int - page size, already validated by the route signature.
        offset: int - rows to skip.
        model: type - the Pydantic model each row is validated into.

    Returns:
        Page[model] - items plus the size of the full result.

    Notes:
        The count is a second query rather than a window function so the same
        code works unchanged against the MySQL 5.7-era deployments the schema
        still supports. At this data size both statements hit the same indexes
        and the extra round trip is not measurable.
    """
    total = conn.execute(count_sql, params).fetchone()[0]
    rows = fetch_all(conn, f"{base_sql} LIMIT ? OFFSET ?", params + (limit, offset))
    return Page[model](total=int(total), limit=limit, offset=offset, items=rows)


@user_router.get("/fleet", response_model=Page[Aircraft], tags=["fleet"])
def list_fleet(
    conn: sqlite3.Connection = Depends(read_db),
    home_base: Optional[str] = Query(None, description="Filter by base station code"),
    primary_role: Optional[str] = Query(None, description="short_haul / medium_haul"),
    status_filter: Optional[str] = Query(None, alias="status", description="Airframe status"),
    limit: int = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
    offset: int = Query(0, ge=0),
) -> Page[Aircraft]:
    """
    List the fleet register, filtered and paged.

    Args:
        conn: sqlite3.Connection - injected read-only connection.
        home_base: str or None - restrict to one base station.
        primary_role: str or None - restrict to one operating role.
        status_filter: str or None - restrict to one airframe status. Exposed
            as `status` to the client; renamed in Python because `status` is
            the FastAPI status-code module imported here.
        limit: int - page size, 1 to 500.
        offset: int - rows to skip.

    Returns:
        Page of Aircraft rows, ordered by accumulated flight hours descending.

    Notes:
        Ordered by flight hours because that is the axis an engineer scans a
        fleet on: the highest-time airframe is the one nearest its next heavy
        check. Filters are composed as bound parameters, never concatenated.
    """
    clauses: List[str] = []
    params: List[Any] = []
    for column, value in (("home_base", home_base), ("primary_role", primary_role),
                          ("status", status_filter)):
        if value:
            clauses.append(f"{column} = ?")
            params.append(value)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""

    return _paginate(
        conn,
        f"SELECT * FROM fleet{where} ORDER BY total_flight_hours DESC",
        f"SELECT COUNT(*) FROM fleet{where}",
        tuple(params), limit, offset, Aircraft,
    )


@user_router.get("/fleet/{tail_number}", response_model=AircraftDetail, tags=["fleet"])
def get_aircraft(
    tail_number: str = Path(..., description="Registration, e.g. SX-ABK"),
    conn: sqlite3.Connection = Depends(read_db),
) -> AircraftDetail:
    """
    Return one airframe with its maintenance and consumption summary.

    Args:
        tail_number: str - the registration.
        conn: sqlite3.Connection.

    Returns:
        AircraftDetail.

    Raises:
        HTTPException 404 when the registration is unknown.

    Notes:
        This is the EASA Part-M M.A.305 record for a single aircraft: hours,
        cycles and every maintenance event raised against it. salt_exposure is
        joined from the home base because it is the environmental covariate the
        Cox model in Module 2 found significant (hazard ratio 1.465,
        p = 0.037), and a client assessing one airframe needs it without a
        second call to the stations endpoint.

        The counts are subqueries rather than joins: joining three one-to-many
        tables at once would multiply the rows and require a DISTINCT that
        costs more than the three scalar lookups.
    """
    row = fetch_one(conn, """
        SELECT f.*, s.salt_exposure,
               (SELECT COUNT(*) FROM work_orders w
                 WHERE w.tail_number = f.tail_number) AS work_orders_total,
               (SELECT COUNT(*) FROM work_orders w
                 WHERE w.tail_number = f.tail_number
                   AND w.source = 'SCHEDULED') AS work_orders_scheduled,
               (SELECT COUNT(*) FROM work_orders w
                 WHERE w.tail_number = f.tail_number
                   AND w.source IN ('UNSCHEDULED_FAILURE','PILOT_REPORT'))
                   AS work_orders_unscheduled,
               (SELECT COUNT(*) FROM part_demands d
                 WHERE d.tail_number = f.tail_number) AS part_demands,
               (SELECT COUNT(*) FROM findings n
                 WHERE n.tail_number = f.tail_number) AS findings
        FROM fleet f
        LEFT JOIN stations s ON f.home_base = s.station_code
        WHERE f.tail_number = ?
    """, (tail_number,))
    return AircraftDetail(**require_found(row, f"Aircraft {tail_number}"))


@user_router.get("/fleet/{tail_number}/maintenance",
                 response_model=Page[MaintenanceEvent], tags=["fleet"])
def get_aircraft_maintenance(
    tail_number: str = Path(..., description="Registration, e.g. SX-ABK"),
    conn: sqlite3.Connection = Depends(read_db),
    source: Optional[str] = Query(None, description="SCHEDULED / UNSCHEDULED_FAILURE / PILOT_REPORT"),
    limit: int = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
    offset: int = Query(0, ge=0),
) -> Page[MaintenanceEvent]:
    """
    Return the work-order history of one airframe, most recent first.

    Args:
        tail_number: str - the registration.
        conn: sqlite3.Connection.
        source: str or None - restrict to one origin of work.
        limit: int - page size.
        offset: int - rows to skip.

    Returns:
        Page of MaintenanceEvent rows.

    Raises:
        HTTPException 404 when the registration is unknown.

    Notes:
        The aircraft is verified before the history is read, so an unknown
        registration returns 404 rather than an empty page that a client
        cannot tell apart from an airframe with no recorded work.
    """
    require_found(fetch_one(conn, "SELECT 1 AS ok FROM fleet WHERE tail_number = ?",
                            (tail_number,)), f"Aircraft {tail_number}")

    clauses = ["tail_number = ?"]
    params: List[Any] = [tail_number]
    if source:
        clauses.append("source = ?")
        params.append(source)
    where = " WHERE " + " AND ".join(clauses)

    return _paginate(
        conn,
        f"SELECT work_order_id, check_type, scheduled_date, status, "
        f"aircraft_fh_at_check, source FROM work_orders{where} "
        f"ORDER BY scheduled_date DESC",
        f"SELECT COUNT(*) FROM work_orders{where}",
        tuple(params), limit, offset, MaintenanceEvent,
    )


@user_router.get("/stations", response_model=List[Station], tags=["fleet"])
def list_stations(conn: sqlite3.Connection = Depends(read_db)) -> List[Station]:
    """
    List the base network with a stock summary per station.

    Args:
        conn: sqlite3.Connection.

    Returns:
        list of Station.

    Notes:
        Five stations, so this endpoint does not page - a limit parameter on a
        collection that cannot grow past a handful would be ceremony.

        Stock value counts serviceable units only, per Part-145 145.A.42: an
        unserviceable unit awaiting shop input is an asset on the books but
        not stock a technician can draw, and reporting it here would overstate
        what the station can actually dispatch.
    """
    with stock_column(conn) as qty:
        rows = fetch_all(conn, f"""
            SELECT s.station_code, s.name, s.climate, s.salt_exposure,
                   (SELECT COUNT(*) FROM fleet f
                     WHERE f.home_base = s.station_code) AS based_aircraft,
                   (SELECT COUNT(*) FROM inventory i
                     WHERE i.station = s.station_code) AS stock_lines,
                   (SELECT COUNT(*) FROM inventory i
                     WHERE i.station = s.station_code
                       AND i.{qty} < i.minimum_stock_level) AS lines_below_minimum,
                   COALESCE((SELECT SUM(i.{qty} * p.unit_cost_eur)
                               FROM inventory i
                               JOIN parts_catalog p ON i.part_number = p.part_number
                              WHERE i.station = s.station_code), 0) AS stock_value_eur
            FROM stations s
            ORDER BY s.station_code
        """)
    return [Station(**row) for row in rows]


@user_router.get("/stations/{station_code}", response_model=Station, tags=["fleet"])
def get_station(
    station_code: str = Path(..., description="IATA code, e.g. HER"),
    conn: sqlite3.Connection = Depends(read_db),
) -> Station:
    """
    Return one station with its stock summary.

    Args:
        station_code: str - the IATA code.
        conn: sqlite3.Connection.

    Returns:
        Station.

    Raises:
        HTTPException 404 when the code is unknown.

    Notes:
        Reuses the list query and filters in Python. The network is five rows;
        a second parameterised query would be more code for no gain.
    """
    code = station_code.upper()
    match = next((s for s in list_stations(conn) if s.station_code.upper() == code), None)
    return require_found(match, f"Station {station_code}")


# ============================================================================
# OPERATOR TIER - PARTS AND INVENTORY
# ============================================================================

class Part(BaseModel):
    """A catalogue entry."""

    part_number: str
    description: str
    ata_chapter: Optional[int] = Field(None, description="ATA/JASC 100 chapter")
    ata_subchapter: Optional[int] = Field(None, description="ATA subchapter, e.g. 10")
    part_class: Optional[str] = Field(None, description="ROTABLE / EXPENDABLE / CONSUMABLE")
    criticality: Optional[str] = Field(None, description="AOG / MEL / ROUTINE")
    unit_of_measure: Optional[str] = None
    unit_cost_eur: Optional[float] = None
    mtbf_flight_hours: Optional[float] = Field(
        None, description="Null for parts not tracked by MTBF, which is most "
                          "expendables and consumables")
    mtbf_flight_cycles: Optional[float] = None
    lead_time_days_normal: Optional[int] = None
    lead_time_days_aog: Optional[int] = None
    shelf_life_months: Optional[int] = None


class StockLine(BaseModel):
    """Stock held for one part at one station."""

    part_number: str
    station: str
    description: Optional[str] = None
    criticality: Optional[str] = None
    part_class: Optional[str] = None
    serviceable: int = Field(..., description="Units available to fit (Part-145 145.A.42)")
    unserviceable: int = Field(0, description="Units awaiting shop input - not available")
    quantity_on_hand: int = Field(..., description="Serviceable and unserviceable together")
    minimum_stock_level: int
    reorder_point: Optional[int] = None
    below_minimum: bool
    shortage: int = Field(..., description="Units short of the minimum, 0 when covered")
    coverage_pct: Optional[float] = Field(
        None, description="Serviceable stock as a percentage of the minimum level")
    unit_cost_eur: Optional[float] = None
    last_receipt_date: Optional[str] = None


class PartAvailability(Part):
    """A catalogue entry with its position across the network."""

    network_serviceable: int = 0
    network_unserviceable: int = 0
    stations_stocked: int = Field(0, description="Stations holding at least one serviceable unit")
    stock: List[StockLine] = Field(default_factory=list)


def _stock_select(qty: str) -> str:
    """
    Build the SELECT that every stock query in this module shares.

    Args:
        qty: str - the serviceable-quantity column for the active schema.

    Returns:
        str - the SELECT and FROM, without a WHERE clause.

    Notes:
        Shortage and coverage are computed in SQL rather than in Python so
        that filtering and ordering can use them directly. Coverage divides by
        the minimum level, so it is guarded: a line with no minimum set is
        reported as null rather than as a division by zero, and the client is
        left to decide what "no minimum" means for its own display.

        below_minimum is selected as 0 or 1 because SQLite has no boolean
        type. StockLine declares it as a bool and Pydantic coerces it, so no
        call site has to remember to convert it.
    """
    return f"""
        SELECT i.part_number, i.station, p.description, p.criticality, p.part_class,
               i.{qty} AS serviceable, i.unserviceable, i.quantity_on_hand,
               i.minimum_stock_level, i.reorder_point,
               CASE WHEN i.{qty} < i.minimum_stock_level THEN 1 ELSE 0 END AS below_minimum,
               MAX(i.minimum_stock_level - i.{qty}, 0) AS shortage,
               CASE WHEN i.minimum_stock_level > 0
                    THEN ROUND(100.0 * i.{qty} / i.minimum_stock_level, 1)
                    ELSE NULL END AS coverage_pct,
               p.unit_cost_eur, i.last_receipt_date
        FROM inventory i
        JOIN parts_catalog p ON i.part_number = p.part_number
    """


@user_router.get("/parts", response_model=Page[Part], tags=["inventory"])
def list_parts(
    conn: sqlite3.Connection = Depends(read_db),
    q: Optional[str] = Query(None, min_length=1, max_length=64,
                             description="Match against part number or description"),
    criticality: Optional[str] = Query(None, description="AOG / MEL / ROUTINE"),
    part_class: Optional[str] = Query(None, description="ROTABLE / EXPENDABLE / CONSUMABLE"),
    ata_chapter: Optional[int] = Query(None, ge=0, le=99, description="ATA/JASC chapter"),
    limit: int = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
    offset: int = Query(0, ge=0),
) -> Page[Part]:
    """
    Search the parts catalogue.

    Args:
        conn: sqlite3.Connection.
        q: str or None - free text matched against number and description.
        criticality: str or None - dispatch class.
        part_class: str or None - rotable, expendable or consumable.
        ata_chapter: int or None - ATA chapter.
        limit: int - page size.
        offset: int - rows to skip.

    Returns:
        Page of Part rows, most dispatch-critical first.

    Notes:
        One search field matched against both the number and the description,
        because an engineer holding a removed unit has the number and an
        engineer reading a defect report has the words.

        The LIKE pattern is bound as a parameter with the wildcards added to
        the value, not to the statement. A search for "50%" is therefore a
        search for the characters 5, 0 and % rather than a pattern the client
        can widen at will.
    """
    clauses: List[str] = []
    params: List[Any] = []

    if q:
        clauses.append("(part_number LIKE ? OR description LIKE ?)")
        pattern = f"%{q}%"
        params.extend([pattern, pattern])
    for column, value in (("criticality", criticality), ("part_class", part_class)):
        if value:
            clauses.append(f"{column} = ?")
            params.append(value)
    if ata_chapter is not None:
        clauses.append("ata_chapter = ?")
        params.append(ata_chapter)

    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    return _paginate(
        conn,
        f"SELECT * FROM parts_catalog{where} ORDER BY {CRIT_ORDER_SQL}, part_number",
        f"SELECT COUNT(*) FROM parts_catalog{where}",
        tuple(params), limit, offset, Part,
    )


@user_router.get("/parts/{part_number}", response_model=PartAvailability, tags=["inventory"])
def get_part(
    part_number: str = Path(..., description="Catalogue number, e.g. AES-24-10-001"),
    conn: sqlite3.Connection = Depends(read_db),
) -> PartAvailability:
    """
    Return a catalogue entry with its availability at every station.

    Args:
        part_number: str - the catalogue number.
        conn: sqlite3.Connection.

    Returns:
        PartAvailability - the part, its network totals and its stock lines.

    Raises:
        HTTPException 404 when the part is unknown.

    Notes:
        This is the call a technician makes with a part number in hand, and it
        answers the only question that matters at that moment: where in the
        network is one, and is it serviceable. Network totals count
        serviceable units alone; the unserviceable figure is reported beside
        them rather than folded in.
    """
    part = require_found(
        fetch_one(conn, "SELECT * FROM parts_catalog WHERE part_number = ?", (part_number,)),
        f"Part {part_number}")

    with stock_column(conn) as qty:
        lines = fetch_all(conn, _stock_select(qty) + " WHERE i.part_number = ? "
                                "ORDER BY i.station", (part_number,))

    return PartAvailability(
        **part,
        network_serviceable=sum(int(r["serviceable"]) for r in lines),
        network_unserviceable=sum(int(r["unserviceable"] or 0) for r in lines),
        stations_stocked=sum(1 for r in lines if int(r["serviceable"]) > 0),
        stock=[StockLine(**r) for r in lines],
    )


@user_router.get("/inventory", response_model=Page[StockLine], tags=["inventory"])
def list_inventory(
    conn: sqlite3.Connection = Depends(read_db),
    station: Optional[str] = Query(None, description="Restrict to one station"),
    criticality: Optional[str] = Query(None, description="AOG / MEL / ROUTINE"),
    below_minimum: bool = Query(False, description="Only lines below their minimum level"),
    limit: int = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
    offset: int = Query(0, ge=0),
) -> Page[StockLine]:
    """
    List stock lines across the network.

    Args:
        conn: sqlite3.Connection.
        station: str or None - restrict to one station.
        criticality: str or None - dispatch class.
        below_minimum: bool - restrict to shortages.
        limit: int - page size.
        offset: int - rows to skip.

    Returns:
        Page of StockLine rows, most critical and least covered first.

    Notes:
        Ordering is criticality, then coverage ascending: the first page is
        always the lines closest to grounding an aircraft, whatever filters
        the client applied.
    """
    with stock_column(conn) as qty:
        clauses: List[str] = []
        params: List[Any] = []
        if station:
            clauses.append("i.station = ?")
            params.append(station.upper())
        if criticality:
            clauses.append("p.criticality = ?")
            params.append(criticality.upper())
        if below_minimum:
            clauses.append(f"i.{qty} < i.minimum_stock_level")

        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        order = (f" ORDER BY CASE p.criticality WHEN 'AOG' THEN 0 WHEN 'MEL' THEN 1 "
                 f"ELSE 2 END, coverage_pct IS NULL, coverage_pct, shortage DESC")

        return _paginate(
            conn,
            _stock_select(qty) + where + order,
            f"SELECT COUNT(*) FROM inventory i "
            f"JOIN parts_catalog p ON i.part_number = p.part_number{where}",
            tuple(params), limit, offset, StockLine,
        )


@user_router.get("/inventory/alerts", response_model=List[StockLine], tags=["inventory"])
def stock_alerts(
    conn: sqlite3.Connection = Depends(read_db),
    station: Optional[str] = Query(None, description="Restrict to one station"),
    limit: int = Query(100, ge=1, le=MAX_LIMIT),
) -> List[StockLine]:
    """
    Return every line below its minimum stock level, worst first.

    Args:
        conn: sqlite3.Connection.
        station: str or None - restrict to one station, which is what a
            station's own display asks for.
        limit: int - maximum alerts returned.

    Returns:
        list of StockLine, AOG-critical first and largest shortage first.

    Notes:
        This is the endpoint a dashboard or a tablet polls, so it returns a
        plain list rather than a page: an alert feed that arrives in pages
        invites a client to render the first page and quietly drop the rest,
        and the dropped ones would be the least critical only by luck.

        An AOG-critical line below minimum is the one alert in this system
        with a direct cost attached - roughly EUR 15,000 per hour of grounding
        if it is needed before it is replenished.
    """
    with stock_column(conn) as qty:
        clauses = [f"i.{qty} < i.minimum_stock_level"]
        params: List[Any] = []
        if station:
            clauses.append("i.station = ?")
            params.append(station.upper())

        rows = fetch_all(
            conn,
            _stock_select(qty) + " WHERE " + " AND ".join(clauses) +
            " ORDER BY CASE p.criticality WHEN 'AOG' THEN 0 WHEN 'MEL' THEN 1 "
            "ELSE 2 END, shortage DESC LIMIT ?",
            tuple(params) + (limit,),
        )

    return [StockLine(**row) for row in rows]


# ============================================================================
# OPERATOR TIER - PREDICTIONS
# ============================================================================
# Module 2 fits the models; this tier serves what they produced. Nothing is
# refitted per request: training the Cox model and the booster takes minutes,
# and the answer does not change between two calls a second apart.

class RiskAssessment(BaseModel):
    """Screening risk for one airframe."""

    tail_number: str
    home_base: str
    age_years: Optional[float] = None
    cycles_per_fh_ratio: Optional[float] = None
    salt_exposure: Optional[float] = Field(None, description="Chloride load at the home base")
    recorded_demands: int = Field(0, description="Part demands already raised - observed, not predicted")
    risk_score: float
    risk_band: str = Field(..., description="LOW below 8, MEDIUM 8-12, HIGH above 12")


class DemandForecast(BaseModel):
    """Next-month expendable demand for one part at one station."""

    part_number: str
    station: str
    description: Optional[str] = None
    mean_monthly_demand: float = Field(..., description="Trailing six-month mean")
    forecast_qty: int = Field(..., description="Next month, rounded up to whole units")
    serviceable: int = Field(..., description="Stock held now")
    months_cover: Optional[float] = Field(
        None, description="Stock divided by forecast demand; null when no demand is forecast")
    under_covered: bool = Field(..., description="Stock below the forecast")


# Weights of the screening heuristic. Defined once here and documented as a
# mirror of agent.py._failure_risk, which is where they were first fitted to
# the Cox covariates. Three surfaces now answer "which airframe is worst" -
# the agent, the dashboard and this service - and they have to give the same
# answer, so the arithmetic is written out in one SQL expression rather than
# reimplemented per caller.
RISK_SQL = """
    SELECT f.tail_number, f.home_base, f.age_years, f.cycles_per_fh_ratio,
           s.salt_exposure,
           COUNT(d.part_number) AS recorded_demands,
           ROUND(f.age_years * 0.3
                 + f.cycles_per_fh_ratio * 10
                 + s.salt_exposure * 15
                 + COUNT(d.part_number) * 0.05, 1) AS risk_score
    FROM fleet f
    JOIN stations s ON f.home_base = s.station_code
    LEFT JOIN part_demands d ON f.tail_number = d.tail_number
"""


def _risk_band(score: float) -> str:
    """
    Band a continuous risk score for operational use.

    Args:
        score: float - the weighted index.

    Returns:
        str - LOW, MEDIUM or HIGH.

    Notes:
        Cut points 8 and 12 split the fleet roughly into thirds and keep the
        HIGH band small enough to act on. A risk list that flags everything
        flags nothing.
    """
    if score > 12:
        return "HIGH"
    return "MEDIUM" if score > 8 else "LOW"


@user_router.get("/predictions/risk", response_model=List[RiskAssessment], tags=["predictions"])
def failure_risk(
    conn: sqlite3.Connection = Depends(read_db),
    band: Optional[str] = Query(None, description="Filter to LOW / MEDIUM / HIGH"),
) -> List[RiskAssessment]:
    """
    Rank the fleet by screening failure risk, worst first.

    Args:
        conn: sqlite3.Connection.
        band: str or None - return only one band.

    Returns:
        list of RiskAssessment.

    Notes:
        This is the screening heuristic, not the calibrated model, and the
        field names say so: risk_score is an index, not a probability and not
        a remaining-life estimate. The Cox proportional-hazards model that
        does produce defensible hazard ratios lives in prediction_model.py and
        needs lifelines to evaluate; exposing it over HTTP means serialising a
        fitted model, which is v1.3 work at the earliest.

        The whole fleet is 15 rows, so the band filter is applied after
        scoring rather than in SQL. Scoring first also means the caller gets
        the same scores whichever band they asked for.
    """
    rows = fetch_all(conn, RISK_SQL + " GROUP BY f.tail_number ORDER BY risk_score DESC")
    out = [RiskAssessment(**row, risk_band=_risk_band(row["risk_score"])) for row in rows]
    if band:
        out = [r for r in out if r.risk_band == band.upper()]
    return out


@user_router.get("/predictions/risk/{tail_number}", response_model=RiskAssessment,
                 tags=["predictions"])
def failure_risk_for_aircraft(
    tail_number: str = Path(..., description="Registration, e.g. SX-ABK"),
    conn: sqlite3.Connection = Depends(read_db),
) -> RiskAssessment:
    """
    Return the screening risk for one airframe.

    Args:
        tail_number: str - the registration.
        conn: sqlite3.Connection.

    Returns:
        RiskAssessment.

    Raises:
        HTTPException 404 when the registration is unknown.

    Notes:
        Exists so a client displaying one aircraft does not have to fetch and
        filter the whole fleet, which is the call a tablet on the ramp makes.
    """
    row = fetch_one(conn, RISK_SQL + " WHERE f.tail_number = ? GROUP BY f.tail_number",
                    (tail_number,))
    row = require_found(row, f"Aircraft {tail_number}")
    return RiskAssessment(**row, risk_band=_risk_band(row["risk_score"]))


@user_router.get("/predictions/demand", response_model=List[DemandForecast],
                 tags=["predictions"])
def expendable_demand(
    conn: sqlite3.Connection = Depends(read_db),
    station: Optional[str] = Query(None, description="Restrict to one station"),
    under_covered: bool = Query(False, description="Only lines whose stock is below the forecast"),
    limit: int = Query(100, ge=1, le=MAX_LIMIT),
) -> List[DemandForecast]:
    """
    Forecast next-month expendable consumption per part and station.

    Args:
        conn: sqlite3.Connection.
        station: str or None - restrict to one station.
        under_covered: bool - only lines that will not cover the forecast.
        limit: int - maximum rows.

    Returns:
        list of DemandForecast, least covered first.

    Notes:
        The trailing six-month mean, not the XGBoost regressor. The regressor
        is trained in prediction_model.py and is not persisted, so serving it
        here would mean either a second copy of the feature pipeline or
        training on request - and on the current data its held-out R-squared
        is negative, so it does not beat this baseline. An API that promises a
        model and returns a worse number than the mean is worse than an API
        that says which number it is returning.

        Months with no consumption count as real zeros. The window is anchored
        on the latest month present in the data rather than on the clock, so
        the endpoint behaves identically against live records and against the
        generated dataset.

        Six months of history at 15 airframes is a small sample, and the
        forecast is a planning aid rather than a commitment. v1.5 replaces the
        baseline with per-series models.
    """
    window = fetch_one(conn, """
        SELECT MIN(month) AS first_month, MAX(month) AS last_month FROM (
            SELECT DISTINCT substr(demand_date, 1, 7) AS month
            FROM part_demands ORDER BY month DESC LIMIT 6)
    """)
    if not window or not window["last_month"]:
        return []

    with stock_column(conn) as qty:
        params: List[Any] = [window["first_month"], window["last_month"]]
        station_clause = ""
        if station:
            station_clause = " AND d.station = ?"
            params.append(station.upper())

        # Six is the divisor rather than COUNT(DISTINCT month): a part that
        # was consumed in two of the six months has a mean of two months of
        # demand spread over six, and dividing by two would report it as a
        # line that moves every month.
        rows = fetch_all(conn, f"""
            SELECT d.part_number, d.station, p.description,
                   ROUND(SUM(d.quantity_required) / 6.0, 2) AS mean_monthly_demand,
                   CAST(CASE WHEN SUM(d.quantity_required) % 6 = 0
                             THEN SUM(d.quantity_required) / 6
                             ELSE SUM(d.quantity_required) / 6 + 1 END AS INTEGER)
                        AS forecast_qty,
                   COALESCE(i.{qty}, 0) AS serviceable
            FROM part_demands d
            JOIN parts_catalog p ON d.part_number = p.part_number
            LEFT JOIN inventory i ON i.part_number = d.part_number
                                 AND i.station = d.station
            WHERE p.part_class = 'EXPENDABLE'
              AND substr(d.demand_date, 1, 7) BETWEEN ? AND ?{station_clause}
            GROUP BY d.part_number, d.station
            ORDER BY mean_monthly_demand DESC
        """, tuple(params))

    out: List[DemandForecast] = []
    for row in rows:
        forecast = int(row["forecast_qty"])
        held = int(row["serviceable"])
        # Months of cover is the number a planner acts on. A line with no
        # forecast demand has unbounded cover, reported as null rather than as
        # a large number a client might sort into the wrong end of a list.
        cover = round(held / forecast, 1) if forecast > 0 else None
        entry = DemandForecast(**row, months_cover=cover, under_covered=held < forecast)
        if not under_covered or entry.under_covered:
            out.append(entry)

    out.sort(key=lambda e: (e.months_cover if e.months_cover is not None else 99.0))
    return out[:limit]


# ============================================================================
# OPERATOR TIER - LOGISTICS
# ============================================================================

class StockRecommendation(BaseModel):
    """Module 3's recommended levels for one part at one station."""

    part_number: str
    station: str
    description: Optional[str] = None
    criticality: Optional[str] = None
    part_class: Optional[str] = None
    mean_monthly_demand: Optional[float] = None
    std_monthly_demand: Optional[float] = None
    current_stock: int = Field(0, description="Serviceable units held now")
    optimal_min_stock: int
    optimal_reorder_point: int
    optimal_max_stock: int
    delta_to_optimal: int = Field(
        ..., description="Optimal minimum less stock held; positive means short of target")
    annual_holding_cost_eur: Optional[float] = None


class TransferRecommendation(BaseModel):
    """A pre-positioning move Module 3 recommends."""

    part_number: str
    description: Optional[str] = None
    criticality: Optional[str] = None
    from_station: str
    to_station: str
    quantity: int
    transfer_hours: float = Field(..., description="Door-to-door transit time")
    reason: Optional[str] = None


class AogOption(BaseModel):
    """One way of getting a part to a grounded aircraft."""

    option: str = Field(..., description="LOCAL_STOCK / STATION_TRANSFER / EMERGENCY_ORDER")
    source: str = Field(..., description="Station code, or SUPPLIER")
    eta_hours: float
    quantity_available: int
    cost_eur: float = Field(..., description="Logistics cost of this option")
    aog_cost_eur: float = Field(..., description="Grounding cost accrued while waiting")
    total_cost_eur: float = Field(..., description="Logistics plus grounding")
    description: str


class AogRoute(BaseModel):
    """The ranked answer to an AOG request."""

    tail_number: str
    station: str = Field(..., description="Where the aircraft is")
    part_number: str
    part_description: str
    criticality: str
    recommended: AogOption
    options: List[AogOption]
    aog_cost_per_hour_eur: float = Field(
        ..., description="The grounding rate every option is priced against")


@user_router.get("/logistics/stock-recommendations", response_model=Page[StockRecommendation],
                 tags=["logistics"])
def stock_recommendations(
    conn: sqlite3.Connection = Depends(read_db),
    station: Optional[str] = Query(None, description="Restrict to one station"),
    criticality: Optional[str] = Query(None, description="AOG / MEL / ROUTINE"),
    under_target: bool = Query(False, description="Only lines below their optimal minimum"),
    limit: int = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
    offset: int = Query(0, ge=0),
) -> Page[StockRecommendation]:
    """
    Serve the stock levels Module 3 derived from demand history.

    Args:
        conn: sqlite3.Connection.
        station: str or None - restrict to one station.
        criticality: str or None - dispatch class.
        under_target: bool - only lines short of the recommended minimum.
        limit: int - page size.
        offset: int - rows to skip.

    Returns:
        Page[StockRecommendation].

    Raises:
        HTTPException 503 when logistics_optimizer.py has not been run.

    Notes:
        Every row is joined against the stock actually held, because a
        recommendation alone does not tell a planner whether to raise a
        purchase order. The delta is computed in SQL so the caller can filter
        and order on it.

        Service levels behind these numbers are set by dispatch criticality
        rather than by cost: 99.5% for AOG-critical items, 95% for MEL, 85%
        for routine. The safety stock is the z-score for that level times the
        demand standard deviation over the lead time.
    """
    if not fetch_one(conn, "SELECT 1 AS ok FROM sqlite_master WHERE type='table' "
                           "AND name='stock_recommendations'"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No stock recommendations yet. Run: python logistics_optimizer.py")

    with stock_column(conn) as qty:
        clauses: List[str] = []
        params: List[Any] = []
        if station:
            clauses.append("r.station = ?")
            params.append(station.upper())
        if criticality:
            clauses.append("r.criticality = ?")
            params.append(criticality.upper())
        if under_target:
            clauses.append(f"r.optimal_min_stock > COALESCE(i.{qty}, 0)")
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""

        base_from = f"""
            FROM stock_recommendations r
            JOIN parts_catalog p ON r.part_number = p.part_number
            LEFT JOIN inventory i ON i.part_number = r.part_number
                                 AND i.station = r.station
        """
        return _paginate(
            conn,
            f"""SELECT r.part_number, r.station, p.description, r.criticality,
                       r.part_class, r.mean_monthly_demand, r.std_monthly_demand,
                       COALESCE(i.{qty}, 0) AS current_stock,
                       r.optimal_min_stock, r.optimal_reorder_point, r.optimal_max_stock,
                       r.optimal_min_stock - COALESCE(i.{qty}, 0) AS delta_to_optimal,
                       r.annual_holding_cost_eur
                {base_from}{where}
                ORDER BY CASE r.criticality WHEN 'AOG' THEN 0 WHEN 'MEL' THEN 1 ELSE 2 END,
                         delta_to_optimal DESC""",
            f"SELECT COUNT(*) {base_from}{where}",
            tuple(params), limit, offset, StockRecommendation,
        )


@user_router.get("/logistics/transfers", response_model=List[TransferRecommendation],
                 tags=["logistics"])
def transfer_recommendations(
    conn: sqlite3.Connection = Depends(read_db),
    from_station: Optional[str] = Query(None, description="Moves out of this station"),
    to_station: Optional[str] = Query(None, description="Moves into this station"),
) -> List[TransferRecommendation]:
    """
    Serve the pre-positioning moves Module 3 recommends.

    Args:
        conn: sqlite3.Connection.
        from_station: str or None - origin filter.
        to_station: str or None - destination filter.

    Returns:
        list of TransferRecommendation, most critical and fastest first.

    Raises:
        HTTPException 503 when logistics_optimizer.py has not been run.

    Notes:
        Ordered by criticality and then transit time. An AOG-critical part
        that takes 14 hours to move outranks a routine part arriving in 4,
        because the consequence of not moving it is three orders of magnitude
        larger.

        These are recommendations, not instructions: executing one is an
        administrator action and lives in the admin tier, where it moves
        stock and is recorded.
    """
    if not fetch_one(conn, "SELECT 1 AS ok FROM sqlite_master WHERE type='table' "
                           "AND name='transfer_recommendations'"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No transfer recommendations yet. Run: python logistics_optimizer.py")

    clauses: List[str] = []
    params: List[Any] = []
    if from_station:
        clauses.append("from_station = ?")
        params.append(from_station.upper())
    if to_station:
        clauses.append("to_station = ?")
        params.append(to_station.upper())
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""

    rows = fetch_all(conn, f"""
        SELECT part_number, description, criticality, from_station, to_station,
               quantity, transfer_hours, reason
        FROM transfer_recommendations{where}
        ORDER BY crit_rank, transfer_hours
    """, tuple(params))
    return [TransferRecommendation(**row) for row in rows]


@user_router.get("/logistics/aog-route", response_model=AogRoute, tags=["logistics"])
def aog_route(
    tail_number: str = Query(..., description="Registration of the grounded aircraft"),
    part_number: str = Query(..., description="Part required"),
) -> AogRoute:
    """
    Rank every source that can get a part to a grounded aircraft.

    Args:
        tail_number: str - the grounded airframe.
        part_number: str - the part it needs.

    Returns:
        AogRoute - the recommendation and every option considered.

    Raises:
        HTTPException 404 when the aircraft or the part is unknown.
        HTTPException 503 when SciPy is not installed.

    Notes:
        A GET, because it computes an answer and changes nothing. Raising an
        actual request against stock is a different act with a different verb,
        and it lives in the admin tier.

        The routing itself is logistics_optimizer.route_aog_request, called
        directly rather than reimplemented. The rules it applies - serviceable
        stock only, door-to-door transit times, EUR 15,000 per hour of
        grounding, a 3x premium on an expedited order - are Module 3
        decisions, and a second copy here would drift from the terminal output
        the first time either changed.

        Options come back ranked by time to availability rather than by cash
        price. At EUR 15,000 per hour on the ground, an hour saved outweighs
        any realistic difference in freight or supplier premium, and the full
        cost breakdown is returned so the choice can be justified afterwards.
    """
    try:
        import logistics_optimizer as opt
    except ImportError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Routing engine unavailable: {exc}. Install with: "
                   f"pip install -r requirements.txt") from exc

    result = opt.route_aog_request(opt.load_data(), tail_number, part_number)
    if "error" in result:
        # The engine reports an unknown aircraft or part as data so a batch of
        # scenarios can continue. Over HTTP the same condition is a 404.
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=result["error"])

    return AogRoute(
        tail_number=result["aircraft"],
        station=result["station"],
        part_number=result["part_number"],
        part_description=result["part_description"],
        criticality=result["criticality"],
        recommended=AogOption(**result["recommended"]),
        options=[AogOption(**o) for o in result["all_options"]],
        aog_cost_per_hour_eur=float(opt.AOG_COST_PER_HOUR),
    )


# ============================================================================
# OPERATOR TIER - SERVICE DIFFICULTY REPORTS
# ============================================================================
# The only genuinely observed failure evidence in the system. The fleet, its
# flight log and its consumption history are all generated; this corpus is
# what anchors the synthetic data to reality.

# Part-name placeholders used by FAA filers when the field does not apply.
# They dominate the raw counts without carrying engineering meaning, so every
# ranking excludes them - the same exclusion agent.py makes.
SDR_PLACEHOLDERS = ("UNKNOWN", "NONE", "")

# Row cap for constrained machines, from config.py. None on STANDARD and FULL.
SDR_SAMPLE_CAP = CFG["ml"]["max_sdr_records"]


class SdrCount(BaseModel):
    """One row of an SDR ranking."""

    label: str = Field(..., description="Part name, or ATA chapter")
    reports: int
    share_pct: float = Field(..., description="Share of the filtered corpus")


class SdrSummary(BaseModel):
    """Aggregate view of the SDR corpus under a filter."""

    records: int
    distinct_parts: int
    airframes: int
    first_report: Optional[str] = None
    last_report: Optional[str] = None
    sampled: bool = Field(
        ..., description="True when a hardware profile capped the rows aggregated")
    sample_cap: Optional[int] = None
    top_parts: List[SdrCount]
    by_ata_chapter: List[SdrCount]


def _sdr_source() -> str:
    """
    Return the SQL source expression for the SDR corpus.

    Args:
        (none)

    Returns:
        str - the table name, or a capped subquery on a MINIMAL machine.

    Notes:
        195,801 rows is nothing for SQLite to aggregate on a workstation, but
        the MINIMAL profile exists for 2 GB machines where this query competes
        with everything else running. The cap is the same max_sdr_records the
        training pipeline honours, and the response carries `sampled` so a
        client is never given a partial statistic that looks complete.
    """
    if SDR_SAMPLE_CAP:
        return f"(SELECT * FROM faa_sdr_raw LIMIT {int(SDR_SAMPLE_CAP)})"
    return "faa_sdr_raw"


@user_router.get("/sdr/summary", response_model=SdrSummary, tags=["sdr"])
def sdr_summary(
    conn: sqlite3.Connection = Depends(read_db),
    ata_chapter: Optional[int] = Query(None, ge=0, le=99, description="Restrict to one chapter"),
    manufacturer: Optional[str] = Query(None, description="Airframe manufacturer, e.g. AIRBUS"),
    top: int = Query(15, ge=1, le=100, description="Rows in each ranking"),
) -> SdrSummary:
    """
    Summarise the FAA Service Difficulty Report corpus under a filter.

    Args:
        conn: sqlite3.Connection.
        ata_chapter: int or None - restrict to one ATA chapter.
        manufacturer: str or None - restrict to one airframe manufacturer.
        top: int - how many rows each ranking returns.

    Returns:
        SdrSummary - totals, the most-reported parts and the chapter
        distribution.

    Raises:
        HTTPException 503 when the corpus has not been loaded.

    Notes:
        Three aggregates over the same filter are served in one response
        rather than three endpoints: a client showing this data shows all
        three together, and splitting them would mean three scans of a
        195,801-row table where one does.

        Shares are computed against the filtered total, so a genuinely
        dominant component can be told apart from the top of a long flat tail.
    """
    if not fetch_one(conn, "SELECT 1 AS ok FROM sqlite_master WHERE type='table' "
                           "AND name='faa_sdr_raw'"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No SDR data. Run data_pipeline.py with the FAA CSV files in raw_data/.")

    src = _sdr_source()
    clauses = ["part_name NOT IN (?,?,?)"]
    params: List[Any] = list(SDR_PLACEHOLDERS)
    if ata_chapter is not None:
        clauses.append("ata_chapter = ?")
        params.append(ata_chapter)
    if manufacturer:
        clauses.append("acft_make = ?")
        params.append(manufacturer.upper())
    where = " WHERE " + " AND ".join(clauses)

    totals = fetch_one(conn, f"""
        SELECT COUNT(*) AS records, COUNT(DISTINCT part_name) AS distinct_parts,
               COUNT(DISTINCT registration) AS airframes,
               MIN(report_date) AS first_report, MAX(report_date) AS last_report
        FROM {src}{where}
    """, tuple(params)) or {}

    records = int(totals.get("records") or 0)
    if not records:
        return SdrSummary(records=0, distinct_parts=0, airframes=0,
                          sampled=bool(SDR_SAMPLE_CAP), sample_cap=SDR_SAMPLE_CAP,
                          top_parts=[], by_ata_chapter=[])

    def ranking(column: str, label_prefix: str = "") -> List[SdrCount]:
        """Aggregate one dimension and express each row as a share."""
        rows = fetch_all(conn, f"""
            SELECT {column} AS label, COUNT(*) AS reports
            FROM {src}{where} AND {column} IS NOT NULL
            GROUP BY {column} ORDER BY reports DESC LIMIT ?
        """, tuple(params) + (top,))
        return [SdrCount(label=f"{label_prefix}{row['label']}",
                         reports=int(row["reports"]),
                         share_pct=round(100.0 * int(row["reports"]) / records, 2))
                for row in rows]

    return SdrSummary(
        records=records,
        distinct_parts=int(totals.get("distinct_parts") or 0),
        airframes=int(totals.get("airframes") or 0),
        first_report=totals.get("first_report"),
        last_report=totals.get("last_report"),
        sampled=bool(SDR_SAMPLE_CAP),
        sample_cap=SDR_SAMPLE_CAP,
        # Column names here are literals in this module, never client input.
        top_parts=ranking("part_name"),
        by_ata_chapter=ranking("ata_chapter", "ATA "),
    )


# ============================================================================
# ROUTER REGISTRATION
# ============================================================================
# Registered last, after every route is defined, so the OpenAPI document is
# assembled from routes that are all in place.

app.include_router(user_router)


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
