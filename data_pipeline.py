"""
Daedalus Supply AI - Module 1: Data Pipeline
============================================
Builds the project database from two complementary sources:

  1. REAL data - FAA Service Difficulty Reports (SDR). Mandatory occurrence
     reports filed by US operators and repair stations. They provide genuine
     part names, ATA/JASC chapters and failure conditions, and are the
     empirical anchor for everything the prediction module later claims.

  2. SYNTHETIC data - fleet register, flight log, parts catalogue, inventory
     and maintenance history for a fictitious Greek narrow-body operator.
     Synthetic because no airline publishes its maintenance records; generated
     from documented reliability models so the statistics remain defensible.

The synthetic layer is not random noise: rotable failures are drawn from a
Weibull process with a wear-out shape parameter, expendables from a Poisson
process, and consumables deterministically per check type. Operational stress
(sector length, airframe age, salt exposure at the home base) modulates all
three, which is what makes the downstream survival model find real signal.

Output: aerosupply.db (SQLite). The file name is retained from earlier builds
for backwards compatibility with existing local databases.

Author: Evangelos Tampachaniotis
Version: 1.0.0
License: MIT

Regulatory Framework:
    - EASA Part-M (EU 1321/2014) - Continuing Airworthiness. M.A.305 aircraft
      continuing-airworthiness record system; M.A.302 approved Aircraft
      Maintenance Programme (AMP) driving the scheduled-check generator.
    - EASA Part-145 - Maintenance Organisation Approval. 145.A.42 component
      classification and acceptance; 145.A.55 records; stores traceability
      (batch/serial) reflected in the inventory table.
    - ICAO Annex 6 - Operation of Aircraft (operator maintenance obligations).
    - ICAO Annex 8 - Airworthiness of Aircraft (continued airworthiness basis).
    - MSG-3 - Maintenance Steering Group methodology; the check hierarchy and
      task-to-ATA-chapter mapping below follow MSG-3 logic.
    - ATA/JASC 100 - chapter numbering for every part and finding.
    - MPD - A320 Maintenance Planning Document; check intervals taken from
      publicly available excerpts.

Usage:
    1. Download SDR CSV(s) from https://www.faa.gov/av-info/download_SDR
    2. Place them in ./raw_data/
    3. Run: python data_pipeline.py
"""

import pandas as pd
import numpy as np
import sqlite3
import os
import json
from datetime import datetime, timedelta
from pathlib import Path

# Hardware-adaptive settings (chunk sizes, record caps). Imported so this
# pipeline runs unchanged on a 2 GB machine - see config.py.
from config import get_config

# ============================================================================
# CONFIGURATION
# ============================================================================

# SQLite output file. Kept as aerosupply.db so databases produced by earlier
# versions of this pipeline remain readable by the current modules.
DB_PATH = "aerosupply.db"

# Directory scanned for FAA SDR CSV exports. Created on first run if absent.
RAW_DATA_DIR = "raw_data"

# Regular expression matched (case-insensitively) against the SDR
# AircraftModel column. Restricting the corpus to comparable airframes matters:
# a hazard rate learned across piston singles and widebodies is meaningless.
# Covers Airbus A320 family, Boeing 737, Embraer regional jets, Gulfstream and
# Dassault Falcon - i.e. transport-category types with similar system
# architectures. Narrow to a single type (e.g. "A320") for a tighter model.
AIRCRAFT_TYPE_FILTER = "A320|737|ERJ|EMB1|FALCON|GV|GULFSTREAM"

# ----------------------------------------------------------------------------
# Fleet definition - synthetic but dimensioned like a real Greek carrier:
# 15 A320-family aircraft, a mix of CEO and NEO variants, ages 4-18 years,
# distributed across a main hub (ATH) and three island/regional bases.
# Tuple layout: (tail_number, model, manufacture_year, home_base, primary_role)
# `primary_role` is the single most influential field in the whole file - it
# sets sector length, which sets the flight-cycle-to-flight-hour ratio, which
# drives cyclic fatigue on landing gear, brakes, doors and pressurisation.
# ----------------------------------------------------------------------------
FLEET = [
    ("SX-ABC", "A320-214", 2008, "ATH", "short_haul"),
    ("SX-ABD", "A320-214", 2010, "ATH", "short_haul"),
    ("SX-ABE", "A320-232", 2012, "ATH", "medium_haul"),
    ("SX-ABF", "A320-214", 2014, "ATH", "short_haul"),
    ("SX-ABG", "A320-232", 2015, "SKG", "short_haul"),
    ("SX-ABH", "A320-214", 2016, "SKG", "short_haul"),
    ("SX-ABI", "A320-271N", 2018, "ATH", "medium_haul"),  # NEO - LEAP-1A powered
    ("SX-ABJ", "A320-271N", 2019, "ATH", "medium_haul"),  # NEO
    ("SX-ABK", "A320-214", 2009, "HER", "short_haul"),
    ("SX-ABL", "A320-214", 2011, "HER", "short_haul"),
    ("SX-ABM", "A320-232", 2013, "RHO", "short_haul"),
    ("SX-ABN", "A320-214", 2007, "ATH", "short_haul"),
    ("SX-ABO", "A320-271N", 2020, "ATH", "medium_haul"),  # NEO
    ("SX-ABP", "A320-214", 2006, "SKG", "short_haul"),
    ("SX-ABQ", "A320-232", 2017, "ATH", "medium_haul"),
]

# ----------------------------------------------------------------------------
# Station (line-station / stores) network.
# `salt_exposure` is a dimensionless 0.0-1.0 index of airborne chloride load:
#   0.0 = fully inland, 1.0 = exposed island coastal site.
# It is the environmental covariate that the Cox model in Module 2 later finds
# significant (hazard ratio 1.465, p = 0.037). Values reflect distance from
# open sea and prevailing winds: RHO and HER are island stations with aprons
# metres from the shoreline; SKG is the most sheltered.
# Chloride-accelerated corrosion is the reason MSG-3 mandates the Corrosion
# Prevention and Control Programme (CPCP) tasks folded into the C-check below.
# ----------------------------------------------------------------------------
STATIONS = {
    "ATH": {"name": "Athens International", "climate": "mediterranean", "salt_exposure": 0.3},
    "SKG": {"name": "Thessaloniki", "climate": "continental", "salt_exposure": 0.2},
    "HER": {"name": "Heraklion", "climate": "mediterranean", "salt_exposure": 0.8},
    "RHO": {"name": "Rhodes", "climate": "mediterranean", "salt_exposure": 0.9},
    "CFU": {"name": "Corfu", "climate": "mediterranean", "salt_exposure": 0.7},
}

# ----------------------------------------------------------------------------
# Route network. Tuple layout: (origin, destination, flight_hours, flight_cycles)
# Every sector is one cycle (one takeoff + one landing) regardless of length,
# which is exactly why the two role groups diverge so sharply:
#   short_haul  ~0.9 FH/sector -> ~1.1 cycles per flight hour
#   medium_haul ~3.0 FH/sector -> ~0.35 cycles per flight hour
# A short-haul airframe therefore accumulates roughly three times the cyclic
# damage per flight hour, which is the physical basis of the stress factor
# applied in generate_maintenance_events().
# ----------------------------------------------------------------------------
ROUTES = {
    "short_haul": [
        ("ATH", "SKG", 0.9, 1),
        ("ATH", "HER", 0.8, 1),
        ("ATH", "RHO", 0.9, 1),
        ("ATH", "CFU", 1.0, 1),
        ("ATH", "SOF", 1.1, 1),
        ("ATH", "SKP", 0.9, 1),
    ],
    "medium_haul": [
        ("ATH", "CDG", 3.2, 1),
        ("ATH", "FRA", 2.8, 1),
        ("ATH", "LHR", 3.5, 1),
        ("ATH", "FCO", 2.0, 1),
        ("ATH", "AMS", 3.3, 1),
        ("ATH", "BRU", 3.0, 1),
    ],
}

# ============================================================================
# PART 1: PARTS CATALOGUE (ATA/JASC 100 structured, Part-145 classified)
# ============================================================================
# Structured like an Illustrated Parts Catalog (IPC): every line item is filed
# under an ATA chapter and subchapter. Part numbers are synthetic but follow
# the industry AES-<chapter>-<subchapter>-<sequence> convention so they sort
# and group the same way real ones do.
#
# Tuple layout:
#   (ata, sub, part_number, description, part_class, unit,
#    mtbf_fh, mtbf_fc, cost_eur, criticality)
#
# part_class - drives which failure model applies downstream:
#   ROTABLE     expensive, repairable, serial-number tracked, returned to a
#               Part-145 shop and re-installed. Wears out => Weibull, and it is
#               these parts that survival analysis can actually predict.
#   EXPENDABLE  cheap, scrapped on removal (lamps, seals, filters, tyres).
#               Fails by random external cause => Poisson.
#   CONSUMABLE  bulk material issued by quantity (oil, sealant, solvent,
#               lockwire). Demand is deterministic per check type.
#
# mtbf_fh / mtbf_fc - Mean Time Between Failures, expressed in flight HOURS or
# flight CYCLES. Exactly one is populated per part, and which one is chosen is
# an engineering statement about the failure mechanism: cycle-limited items
# (wheels, tyres, brakes, gear actuators, doors) fail from repeated load
# application, not from elapsed running time. Values are representative ranges
# from published reliability-engineering literature, not manufacturer data.
#
# criticality - dispatch consequence of not having the part on hand:
#   AOG      Aircraft on Ground. No dispatch. Roughly €15,000 per hour.
#   MEL      deferrable under the Minimum Equipment List, with conditions and
#            a repair interval (typically category A-D, 3-120 days).
#   ROUTINE  no dispatch impact.
# ============================================================================

PARTS_CATALOG = [
    # --- ATA 21 - Air Conditioning ---
    (21, 50, "AES-21-50-001", "Pack Flow Control Valve", "ROTABLE", "EA", 15000, None, 28000, "AOG"),
    (21, 50, "AES-21-50-010", "Pack Valve O-Ring Seal", "EXPENDABLE", "EA", None, None, 12, "ROUTINE"),
    (21, 60, "AES-21-60-001", "Temperature Controller", "ROTABLE", "EA", 20000, None, 8500, "AOG"),

    # --- ATA 24 - Electrical Power ---
    # The IDG has the lowest MTBF of any rotable here (8,000 FH) and the second
    # highest unit cost - it is the classic pre-positioning candidate.
    (24, 10, "AES-24-10-001", "Integrated Drive Generator (IDG)", "ROTABLE", "EA", 8000, None, 95000, "AOG"),
    (24, 30, "AES-24-30-001", "Generator Control Unit (GCU)", "ROTABLE", "EA", 18000, None, 15000, "AOG"),
    (24, 30, "AES-24-30-010", "GCU Connector Gasket", "EXPENDABLE", "EA", None, None, 8, "ROUTINE"),

    # --- ATA 26 - Fire Protection ---
    (26, 10, "AES-26-10-001", "Fire Detection Loop", "ROTABLE", "EA", 25000, None, 4200, "AOG"),
    (26, 10, "AES-26-10-010", "Fire Loop Grommet Seal", "EXPENDABLE", "EA", None, None, 5, "ROUTINE"),

    # --- ATA 27 - Flight Controls ---
    (27, 10, "AES-27-10-001", "Aileron PCU (Power Control Unit)", "ROTABLE", "EA", 12000, None, 45000, "AOG"),
    (27, 50, "AES-27-50-001", "Flap Actuator", "ROTABLE", "EA", 10000, None, 38000, "AOG"),
    (27, 50, "AES-27-50-010", "Flap Track Roller Bearing", "EXPENDABLE", "EA", None, None, 180, "MEL"),
    (27, 00, "AES-27-00-050", "Flight Control Cable Turnbuckle", "EXPENDABLE", "EA", None, None, 45, "ROUTINE"),

    # --- ATA 28 - Fuel System ---
    (28, 20, "AES-28-20-001", "Fuel Boost Pump", "ROTABLE", "EA", 12000, None, 18000, "AOG"),
    (28, 40, "AES-28-40-001", "Fuel Quantity Indicator", "ROTABLE", "EA", 22000, None, 5500, "MEL"),
    (28, 20, "AES-28-20-010", "Fuel Pump Inlet Filter Screen", "EXPENDABLE", "EA", None, None, 35, "ROUTINE"),

    # --- ATA 29 - Hydraulic Power ---
    (29, 10, "AES-29-10-001", "Hydraulic Pump (Engine Driven)", "ROTABLE", "EA", 10000, None, 42000, "AOG"),
    (29, 10, "AES-29-10-002", "Hydraulic Pump (Electric)", "ROTABLE", "EA", 8000, None, 55000, "AOG"),
    (29, 10, "AES-29-10-010", "Hydraulic Pump O-Ring Kit", "EXPENDABLE", "KIT", None, None, 65, "ROUTINE"),
    (29, 00, "AES-29-00-050", "Hydraulic Fluid MIL-PRF-83282", "CONSUMABLE", "LTR", None, None, 18, "ROUTINE"),
    (29, 30, "AES-29-30-001", "Hydraulic Accumulator", "ROTABLE", "EA", 15000, None, 12000, "AOG"),

    # --- ATA 30 - Ice & Rain Protection ---
    (30, 10, "AES-30-10-001", "Bleed Air Valve (Anti-Ice)", "ROTABLE", "EA", 14000, None, 8500, "MEL"),

    # --- ATA 32 - Landing Gear ---
    # Note the MTBF column shift: every load-bearing item here is CYCLE limited.
    # Main wheels at 1,500 FC and tyres at 350 FC are the highest-turnover
    # AOG-critical items in the network and dominate transfer traffic.
    (32, 10, "AES-32-10-001", "Nose Gear Steering Actuator", "ROTABLE", "EA", None, 15000, 32000, "AOG"),
    (32, 40, "AES-32-40-001", "Main Wheel Assembly", "ROTABLE", "EA", None, 1500, 4800, "AOG"),
    (32, 40, "AES-32-40-002", "Nose Wheel Assembly", "ROTABLE", "EA", None, 2000, 3200, "AOG"),
    (32, 40, "AES-32-40-010", "Brake Wear Pin Indicator", "EXPENDABLE", "EA", None, None, 22, "ROUTINE"),
    (32, 40, "AES-32-40-020", "Tire Main Gear", "EXPENDABLE", "EA", None, 350, 2800, "AOG"),
    (32, 40, "AES-32-40-021", "Tire Nose Gear", "EXPENDABLE", "EA", None, 400, 1900, "AOG"),
    (32, 50, "AES-32-50-001", "Brake Assembly (Carbon)", "ROTABLE", "EA", None, 2000, 28000, "AOG"),
    (32, 50, "AES-32-50-010", "Brake Lining Segment", "EXPENDABLE", "EA", None, None, 350, "ROUTINE"),

    # --- ATA 33 - Lights ---
    # Lamps are the archetypal Poisson item: high frequency, trivial cost,
    # no dispatch impact individually, but they dominate transaction volume.
    (33, 40, "AES-33-40-001", "Landing Light Assembly", "ROTABLE", "EA", 4000, None, 1200, "MEL"),
    (33, 40, "AES-33-40-010", "Landing Light Bulb", "EXPENDABLE", "EA", None, None, 85, "ROUTINE"),
    (33, 40, "AES-33-40-011", "Navigation Light Bulb", "EXPENDABLE", "EA", None, None, 45, "ROUTINE"),
    (33, 10, "AES-33-10-010", "Cockpit Indicator Lamp", "EXPENDABLE", "EA", None, None, 18, "ROUTINE"),

    # --- ATA 34 - Navigation ---
    (34, 10, "AES-34-10-001", "IRS (Inertial Reference System)", "ROTABLE", "EA", 30000, None, 85000, "AOG"),
    (34, 50, "AES-34-50-001", "Radio Altimeter Transceiver", "ROTABLE", "EA", 25000, None, 22000, "MEL"),

    # --- ATA 36 - Pneumatic ---
    (36, 10, "AES-36-10-001", "Bleed Air Pre-Cooler", "ROTABLE", "EA", 18000, None, 16000, "AOG"),
    (36, 10, "AES-36-10-010", "Bleed Duct Gasket", "EXPENDABLE", "EA", None, None, 35, "ROUTINE"),

    # --- ATA 38 - Water/Waste ---
    (38, 10, "AES-38-10-001", "Potable Water Heater", "ROTABLE", "EA", 12000, None, 3500, "ROUTINE"),

    # --- ATA 49 - Auxiliary Power Unit ---
    (49, 10, "AES-49-10-001", "APU Starter Motor", "ROTABLE", "EA", 6000, None, 35000, "MEL"),
    (49, 10, "AES-49-10-010", "APU Oil Filter Element", "EXPENDABLE", "EA", None, None, 120, "ROUTINE"),
    (49, 10, "AES-49-10-050", "APU Oil", "CONSUMABLE", "LTR", None, None, 25, "ROUTINE"),

    # --- ATA 52 - Doors ---
    (52, 10, "AES-52-10-001", "Door Actuator Assembly", "ROTABLE", "EA", None, 20000, 18000, "AOG"),
    (52, 10, "AES-52-10-010", "Door Seal (Main Cabin)", "EXPENDABLE", "EA", None, None, 280, "MEL"),

    # --- ATA 72 - Engine (CFM56-5B on CEO, LEAP-1A on NEO) ---
    (72, 00, "AES-72-00-001", "Engine Oil Filter", "EXPENDABLE", "EA", None, None, 220, "ROUTINE"),
    (72, 50, "AES-72-50-001", "Fuel Control Unit (FCU/FADEC)", "ROTABLE", "EA", 15000, None, 120000, "AOG"),
    (72, 60, "AES-72-60-001", "Fan Blade", "ROTABLE", "EA", 20000, None, 45000, "AOG"),
    (72, 00, "AES-72-00-050", "Engine Oil (per liter)", "CONSUMABLE", "LTR", None, None, 32, "ROUTINE"),

    # --- ATA 73 - Engine Fuel and Control ---
    (73, 10, "AES-73-10-001", "Engine Fuel Pump", "ROTABLE", "EA", 12000, None, 28000, "AOG"),

    # --- ATA 78 - Exhaust ---
    (78, 10, "AES-78-10-001", "Exhaust Gas Temperature Probe (EGT)", "ROTABLE", "EA", 10000, None, 6500, "MEL"),

    # --- ATA 79 - Engine Oil ---
    (79, 20, "AES-79-20-001", "Oil Cooler", "ROTABLE", "EA", 14000, None, 9500, "AOG"),

    # --- ATA 12 / 20 - Standard practices and servicing materials ---
    # Chapter 20 covers standard practices (lockwire, sealants) and chapter 12
    # servicing; these are consumed on essentially every check.
    (20, 00, "AES-20-00-050", "Lockwire (Safety Wire) Roll", "CONSUMABLE", "ROLL", None, None, 8, "ROUTINE"),
    (20, 00, "AES-20-00-051", "Sealant PR-1776 Cartridge", "CONSUMABLE", "EA", None, None, 45, "ROUTINE"),
    (12, 00, "AES-12-00-050", "Cleaning Solvent MEK (liter)", "CONSUMABLE", "LTR", None, None, 12, "ROUTINE"),
]

# ============================================================================
# PART 2: INSPECTION PROGRAM (MSG-3 / A320 MPD derived)
# ============================================================================
# The operator's Aircraft Maintenance Programme (AMP) approved under EASA
# Part-M M.A.302. Intervals follow the published A320 MPD structure.
#
# Tuple layout:
#   (check_type, interval_fh, interval_fc, interval_months, duration_days, desc)
#
# A check becomes due at whichever limit is reached FIRST - hours, cycles or
# calendar months - which is why three independent interval columns exist and
# why most rows leave two of them None. Landing gear overhaul, for example, is
# limited by cycles and calendar only; elapsed flight hours are irrelevant to
# its failure mechanism.
#
# duration_days is aircraft downtime, and it is the number that turns a
# maintenance plan into a fleet-availability problem: a 60-day D-check removes
# 6.7% of a 15-aircraft fleet from service for two months.
# ============================================================================

INSPECTION_PROGRAM = [
    # Line checks - no hour/cycle interval, performed on a calendar rhythm.
    ("DAILY", None, None, None, 0.1, "Daily/Preflight Check - visual inspection, fluid levels"),
    ("WEEKLY", None, None, None, 0.2, "Weekly Check - detailed walk-around, tire pressure, fluid checks"),
    # Base checks - hour driven.
    ("A-CHECK", 750, None, None, 1, "A-Check - systems checks, lubrication, minor inspections"),
    # C-check is dual-limited: 7,500 FH or 24 months, whichever comes first, so
    # a low-utilisation airframe still gets its structural inspection on time.
    ("C-CHECK", 7500, None, 24, 18, "C-Check - detailed structural/systems inspection, CPCP tasks"),
    ("D-CHECK", 30000, None, 120, 60, "D-Check - heavy maintenance, complete strip-down inspection"),
    # Shop visits - component life limits rather than airframe checks.
    ("ENGINE_SHOP", 20000, None, None, 45, "Engine shop visit - overhaul, module replacement"),
    ("LANDING_GEAR_OH", None, 18000, 120, 14, "Landing gear overhaul"),
]

# ----------------------------------------------------------------------------
# Task-to-ATA-chapter mapping per check type. Each entry is (ata_chapter, task
# description). This drives finding generation: a finding can only be raised
# against a system that was actually inspected during that check - a C-check
# opens structure and finds corrosion, a daily walk-around does not.
# The chapter coverage widens with check depth, following MSG-3 task escalation:
#   DAILY   3 chapters  (what is visible from the ground)
#   WEEKLY  4 chapters  (plus functional checks)
#   A-CHECK 10 chapters (systems operational checks, engine borescope)
#   C-CHECK 17 chapters (structure opened, components removed and bench tested)
# ----------------------------------------------------------------------------
CHECK_TASKS = {
    "DAILY": [(33, "Check all lights"), (32, "Visual tire/brake check"), (29, "Hydraulic level check")],
    "WEEKLY": [
        (33, "Test all external/internal lights"),
        (32, "Tire pressure check, brake wear pin"),
        (29, "Hydraulic system functional test"),
        (28, "Fuel system leak check"),
    ],
    "A-CHECK": [
        (21, "Air conditioning system operational check"),
        (24, "Electrical generation functional test"),
        (27, "Flight control system operational check"),
        (28, "Fuel system inspection"),
        (29, "Hydraulic system inspection and filter check"),
        (32, "Landing gear visual and functional inspection"),
        (33, "All lights check and replacement"),
        (36, "Pneumatic duct leak check"),
        (49, "APU inspection and oil check"),
        (72, "Engine borescope inspection, oil analysis"),
    ],
    "C-CHECK": [
        (21, "Pack valve overhaul/replacement assessment"),
        (24, "IDG performance test, wiring inspection"),
        (26, "Fire detection loop resistance check"),
        (27, "Flight control actuator inspection, cable tensions"),
        (28, "Fuel tank internal inspection, boost pump check"),
        (29, "Hydraulic pump performance test, accumulator service"),
        (30, "Anti-ice valve functional test"),
        (32, "Landing gear detailed inspection, bearing checks"),
        (34, "Navigation equipment calibration"),
        (36, "Bleed air system overhaul inspection"),
        (38, "Water/waste system servicing"),
        (49, "APU major inspection"),
        (52, "Door mechanism and seal inspection"),
        (72, "Engine major inspection, FCU test"),
        (73, "Engine fuel system check"),
        (78, "EGT probe calibration check"),
        (79, "Oil cooler inspection"),
    ],
}


# ============================================================================
# PART 3: FAA SDR DATA PARSER
# ============================================================================

def parse_sdr_data(raw_data_dir: str, aircraft_filter: str = "A320") -> pd.DataFrame:
    """
    Load, normalise and type-filter the FAA Service Difficulty Report corpus.

    Args:
        raw_data_dir: str - directory holding the SDR CSV exports. Created if
            it does not exist, in which case an empty frame is returned.
        aircraft_filter: str - regular expression matched case-insensitively
            against the AircraftModel column, e.g. "A320|737".

    Returns:
        pd.DataFrame - the filtered SDR records with normalised (snake_case)
        column names. Empty frame when no CSVs are present, which is a
        supported configuration: the pipeline then builds a synthetic-only
        database.

    Notes:
        Transformation: N raw FAA CSVs (mixed CamelCase schema, latin-1
        encoded) -> one concatenated frame -> renamed to snake_case ->
        filtered to comparable aircraft types.

        CSVs are read in chunks sized by the active hardware profile so the
        full ~413,000-record corpus can be ingested on a 2 GB machine, where
        loading a whole file at once would exhaust memory.
    """
    cfg = get_config()
    chunk_size = cfg["ml"]["chunk_size"]        # rows per read_csv chunk
    max_records = cfg["ml"]["max_sdr_records"]  # None on STANDARD/FULL

    raw_path = Path(raw_data_dir)

    # First run: create the directory so the user has somewhere to drop CSVs,
    # then return empty - there is nothing to parse yet.
    if not raw_path.exists():
        raw_path.mkdir(parents=True)
        return pd.DataFrame()

    csv_files = list(raw_path.glob("*.csv"))

    # Directory exists but is empty. Not an error: the synthetic layer alone is
    # a complete, runnable dataset. Tell the user where the files should go.
    if not csv_files:
        print(f"  No SDR CSV files found in {raw_path.absolute()}")
        return pd.DataFrame()

    print(f"\n  Found {len(csv_files)} SDR CSV file(s)")

    frames = []

    # Iterate over every CSV in raw_data/. The FAA publishes one file per
    # calendar year, so this accumulates the full multi-year history.
    for f in csv_files:
        try:
            # latin-1 rather than utf-8: SDR free-text discrepancy fields
            # contain legacy single-byte characters that break a strict utf-8
            # decode. latin-1 maps every byte, so ingestion never fails.
            # low_memory=False forces a single type-inference pass per chunk,
            # avoiding pandas' mixed-dtype warnings on the sparse part columns.
            chunks = pd.read_csv(f, low_memory=False, encoding="latin-1",
                                 chunksize=chunk_size)

            # Accumulate this file's chunks; counted afterwards for the log line.
            file_rows = 0
            for chunk in chunks:
                frames.append(chunk)
                file_rows += len(chunk)
            print(f"    {f.name}: {file_rows} records")
        except Exception as e:
            # One malformed or unreadable file must not abort ingestion of the
            # other years - report it and carry on with what is readable.
            print(f"    {f.name}: {e}")

    # Every file failed to parse, or every file was empty.
    if not frames:
        return pd.DataFrame()

    all_sdr = pd.concat(frames, ignore_index=True)
    print(f"\n  Total SDR records: {len(all_sdr)}")

    # --- Column normalisation ---
    # The FAA export uses CamelCase names that vary slightly between yearly
    # releases. Map the fields this project uses onto stable snake_case names
    # so downstream SQL and the ML feature builder never see the raw schema.
    rename_map = {
        "AircraftMake": "acft_make",
        "AircraftModel": "acft_model",
        "AircraftSerialNumber": "acft_serial",
        "AircraftTotalTime": "total_time",          # airframe FH at occurrence
        "AircraftTotalCycles": "total_cycles",      # airframe FC at occurrence
        "JASCCode": "ata_code",                     # JASC == ATA 100 chapter code
        "PartName": "part_name",
        "PartNumber": "part_number",
        "PartCondition": "part_condition",
        "NatureOfConditionA": "nature_condition",
        "StageOfOperationCode": "stage_of_operation",
        "DifficultyDate": "report_date",
        "Discrepancy": "remarks",
        "RegistryNNumber": "registration",
        "PartTotalTime": "part_total_time",         # component FH - survival input
        "PartTotalCycles": "part_total_cycles",     # component FC - survival input
    }

    # Rename only the columns actually present, so a year whose export omits a
    # field still loads instead of raising KeyError.
    actual_renames = {k: v for k, v in rename_map.items() if k in all_sdr.columns}
    all_sdr = all_sdr.rename(columns=actual_renames)

    # Drop the repeated NatureOfConditionB..E and PrecautionaryProcedureA..E
    # columns. They are sparse alternates of the single field already kept and
    # roughly double the memory footprint of the frame for no analytical gain.
    cols_to_drop = [c for c in all_sdr.columns
                    if c.startswith("NatureOfCondition") or c.startswith("PrecautionaryProcedure")]
    all_sdr = all_sdr.drop(columns=cols_to_drop, errors="ignore")

    # --- Aircraft type filter ---
    # Restrict to comparable transport-category types; see AIRCRAFT_TYPE_FILTER.
    if "acft_model" in all_sdr.columns:
        mask = all_sdr["acft_model"].astype(str).str.contains(aircraft_filter, case=False, na=False)
        filtered = all_sdr[mask].copy()
        print(f"  Filtered for '{aircraft_filter}': {len(filtered)} records")
    else:
        # Schema without a model column - keep everything rather than silently
        # returning nothing, but make the loss of filtering visible.
        print("  Warning: no acft_model column found - type filter not applied")
        filtered = all_sdr.copy()

    # --- Low-memory cap (MINIMAL profile only) ---
    # On constrained hardware, retain a bounded sample. Taken as a head slice
    # of the type-filtered corpus, which is already chronologically ordered by
    # source file, so the sample stays internally consistent.
    if max_records is not None and len(filtered) > max_records:
        print(f"  Profile {cfg['profile']}: capping to {max_records} records for low-memory operation")
        filtered = filtered.head(max_records).copy()

    return filtered


def extract_parts_from_sdr(sdr_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank real-world components by how often they appear in SDR occurrences.

    Args:
        sdr_df: pd.DataFrame - the filtered SDR frame from parse_sdr_data().

    Returns:
        pd.DataFrame - one row per distinct part name, with occurrence count,
        modal ATA/JASC code, sample part numbers and observed failure
        conditions, sorted by count descending. Empty frame if there is no
        usable SDR input.

    Notes:
        Transformation: one row per occurrence -> one row per part name.
        This is the empirical failure-frequency picture that justifies the
        synthetic catalogue's MTBF assignments: if a component type dominates
        real occurrence reports, its synthetic counterpart should carry a
        correspondingly low MTBF.
    """
    # No SDR data, or a schema without the part-name column - nothing to rank.
    if sdr_df.empty or 'part_name' not in sdr_df.columns:
        return pd.DataFrame()

    # --- Derive the ATA chapter from the JASC code ---
    # JASC codes are 4 digits, chapter-subchapter (e.g. 3240 = ATA 32-40,
    # Landing Gear / Wheels & Brakes). The first two digits are the chapter.
    if 'ata_code' in sdr_df.columns:
        sdr_df['ata_chapter'] = sdr_df['ata_code'].astype(str).str[:2]
        # errors="coerce" turns non-numeric codes (blanks, text placeholders)
        # into NaN rather than raising - the raw corpus is not fully clean.
        sdr_df['ata_chapter'] = pd.to_numeric(sdr_df['ata_chapter'], errors='coerce')

    # Aggregate to one row per part name.
    parts_analysis = sdr_df.groupby(['part_name']).agg(
        # How many occurrences named this part - the reliability signal.
        failure_count=('part_name', 'count'),
        # Modal (most frequent) JASC code. Mode rather than first, because the
        # same part name is occasionally filed under a neighbouring chapter;
        # the guard handles the case where mode() returns nothing at all.
        ata_codes=('ata_code', lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else None),
        # Up to 3 example manufacturer part numbers - enough to identify the
        # component without storing thousands of variants per row.
        part_numbers=('part_number', lambda x: list(x.dropna().unique())[:3]),
        # Up to 5 distinct observed failure conditions (cracked, leaking, ...).
        nature_conditions=('nature_condition', lambda x: list(x.dropna().unique())[:5]),
    ).reset_index().sort_values('failure_count', ascending=False)

    # --- Console report: the 20 most-reported components ---
    print(f"\n  Top 20 most frequent failures ({AIRCRAFT_TYPE_FILTER}):")
    print(f"  {'Part Name':<40} {'Count':>6}  ATA")
    print(f"  {'-' * 55}")

    # Print one line per ranked part, truncating long names to keep the table
    # aligned in an 80-column terminal.
    for _, row in parts_analysis.head(20).iterrows():
        print(f"  {str(row['part_name'])[:40]:<40} {row['failure_count']:>6}  {row['ata_codes']}")

    return parts_analysis


# ============================================================================
# PART 4: SYNTHETIC DATA GENERATION
# ============================================================================

class AeroDataGenerator:
    """
    Generator for the synthetic operational dataset.

    Role in the system architecture: produces every table that a real operator
    would hold in its continuing-airworthiness management system but that no
    operator publishes - fleet register, flight log, inventory and maintenance
    history. Everything Modules 2 and 3 learn from originates here, so the
    fidelity of these models determines whether the downstream analysis is
    meaningful or circular.

    Failure modelling by part class:
        ROTABLE     Weibull renewal process, shape beta = 2.5 (wear-out),
                    scale eta = stress-adjusted MTBF.
        EXPENDABLE  Homogeneous Poisson process with a per-flight-hour rate
                    modulated by environment and part family.
        CONSUMABLE  Deterministic issue per check type.

    Attributes:
        rng             : numpy.random.Generator - seeded PRNG
        reference_date  : datetime - simulation start (2024-01-01)
        simulation_days : int - length of the simulated period in days
    """

    def __init__(self, seed=42):
        """
        Initialise the generator with a fixed random seed.

        Args:
            seed: int - PRNG seed. Fixed by default so the database is
                reproducible run to run.

        Returns:
            None

        Notes:
            Reproducibility is a regulatory concern, not just a convenience:
            a reliability programme under Part-M M.A.302 must be auditable, and
            an auditor has to be able to regenerate the exact dataset behind a
            stated conclusion. default_rng (PCG64) is used rather than the
            legacy np.random global state so the stream is isolated from any
            other library that also draws random numbers.
        """
        self.rng = np.random.default_rng(seed)

        # Simulation epoch. All dates in the database are relative to this.
        self.reference_date = datetime(2024, 1, 1)

        # 3 years x 365 days = 1,095 days. Three years is the shortest window
        # that contains a full C-check cycle (24 months) while still producing
        # enough A-checks (~every 750 FH, so roughly every 2-3 months) for the
        # survival model to have repeated events per aircraft.
        self.simulation_days = 365 * 3

    def generate_fleet_table(self) -> pd.DataFrame:
        """
        Build the aircraft register.

        Args:
            (none)

        Returns:
            pd.DataFrame - one row per aircraft with identity, utilisation
            counters and the derived cycles-per-flight-hour ratio.

        Notes:
            This is the EASA Part-M M.A.305 continuing-airworthiness record:
            the authoritative statement of each airframe's accumulated flight
            hours and flight cycles, against which every maintenance interval
            is measured.
        """
        records = []

        # Iterate over the static fleet definition, deriving utilisation
        # figures from age and role.
        for tail, model, year, base, role in FLEET:
            # Age relative to the simulation epoch year.
            age_years = 2024 - year

            # --- Annual utilisation by role (flight hours per year) ---
            # Short-haul aircraft fly more sectors but shorter ones, so they
            # accumulate fewer hours per year than medium-haul aircraft even
            # though they work harder in cyclic terms. 3,200 and 3,800 FH/year
            # are typical European narrow-body figures.
            annual_fh = 3200 if role == "short_haul" else 3800

            # Lifetime hours, with +/-200 FH of noise so no two same-age
            # aircraft have identical counters (they never do in reality).
            total_fh = annual_fh * age_years + self.rng.integers(-200, 200)

            # --- Cycles per flight hour: the key stress indicator ---
            # Derived from mean sector length: 1 cycle / sector.
            if role == "short_haul":
                # ~0.9 FH sectors => ~1.1 cycles/FH. Every cycle is one
                # pressurisation, one gear retraction and one brake application.
                cycles_per_fh = 1.1 + self.rng.uniform(-0.1, 0.1)
            else:
                # ~3.0 FH sectors => ~0.35 cycles/FH.
                cycles_per_fh = 0.35 + self.rng.uniform(-0.05, 0.05)

            total_fc = int(total_fh * cycles_per_fh)

            records.append({
                "tail_number": tail,
                "aircraft_model": model,
                "manufacture_year": year,
                "age_years": age_years,
                "home_base": base,
                "primary_role": role,
                "total_flight_hours": int(total_fh),
                "total_flight_cycles": total_fc,
                "cycles_per_fh_ratio": round(cycles_per_fh, 3),
                # Mean daily utilisation - used later to convert an accumulated
                # flight-hour figure back into a calendar date.
                "daily_utilization_fh": round(annual_fh / 365, 1),
                "status": "ACTIVE",
            })

        return pd.DataFrame(records)

    def generate_flight_log(self, fleet_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate three years of sector-by-sector flight records.

        Args:
            fleet_df: pd.DataFrame - the fleet register from
                generate_fleet_table().

        Returns:
            pd.DataFrame - one row per flown sector (~54,500 rows), carrying
            cumulative FH/FC counters and a season label.

        Notes:
            This table drives everything else. Maintenance intervals are
            triggered off cumulative_fh, failure exposure is proportional to
            accumulated hours, and the seasonal demand pattern that Module 2's
            forecaster learns originates in the season factor applied here.
        """
        print("  Generating flight logs...")
        logs = []

        # Iterate over each aircraft; every airframe flies its own independent
        # three-year schedule from its own home base.
        for _, ac in fleet_df.iterrows():
            role = ac["primary_role"]
            base = ac["home_base"]
            route_list = ROUTES[role]

            # --- Daily sector count by role ---
            # A short-haul narrow-body typically flies 5 sectors/day (roughly
            # 2-3 out-and-back rotations); a medium-haul aircraft flies 2.
            if role == "short_haul":
                flights_per_day_base = 5
            else:
                flights_per_day_base = 2

            current_date = self.reference_date
            cumulative_fh = 0
            cumulative_fc = 0

            # Iterate over every calendar day in the simulation window.
            for day in range(self.simulation_days):
                current_date = self.reference_date + timedelta(days=day)
                month = current_date.month

                # --- Seasonal utilisation factor ---
                # Greek traffic is strongly seasonal: island tourism peaks in
                # summer and collapses in winter. This single factor is what
                # produces the seasonal demand signal the forecaster relies on.
                if month in [6, 7, 8, 9]:
                    season_factor = 1.2   # summer peak: +20% sectors
                elif month in [12, 1, 2]:
                    season_factor = 0.7   # winter trough: -30% sectors
                else:
                    season_factor = 1.0   # shoulder season: baseline

                # --- Maintenance downtime ---
                # ~5% of days the aircraft is unavailable (check, defect
                # rectification, spare). Skipping the day means no hours are
                # accumulated, which is what makes fleet availability finite.
                if self.rng.random() < 0.05:
                    continue

                # Daily sector count: seasonal expectation plus Gaussian noise
                # (sigma = 0.5 sectors) for day-to-day schedule variation.
                # max(1, ...) guarantees an available aircraft flies at least
                # one sector rather than idling with no explanation.
                n_flights = max(1, int(flights_per_day_base * season_factor +
                                       self.rng.normal(0, 0.5)))

                # Iterate over the sectors flown on this day.
                for f in range(n_flights):
                    # Pick a route uniformly from the role's network.
                    route = route_list[self.rng.integers(0, len(route_list))]
                    origin, dest, fh, fc = route

                    # Actual block time varies with routing, holding and wind;
                    # sigma = 0.05 h (3 minutes) reproduces that scatter.
                    actual_fh = round(fh + self.rng.normal(0, 0.05), 2)
                    cumulative_fh += actual_fh
                    cumulative_fc += fc

                    logs.append({
                        "tail_number": ac["tail_number"],
                        "flight_date": current_date.strftime("%Y-%m-%d"),
                        "origin": origin,
                        "destination": dest,
                        "flight_hours": actual_fh,
                        "flight_cycles": fc,
                        # Running totals - the interval-tracking counters.
                        "cumulative_fh": round(cumulative_fh, 1),
                        "cumulative_fc": cumulative_fc,
                        "month": month,
                        # Denormalised season label so seasonal aggregation in
                        # SQL needs no date arithmetic.
                        "season": "summer" if month in [6, 7, 8, 9] else
                                  "winter" if month in [12, 1, 2] else "shoulder",
                    })

        print(f"    Generated {len(logs)} flight records")
        return pd.DataFrame(logs)

    def generate_parts_catalog_table(self) -> pd.DataFrame:
        """
        Expand PARTS_CATALOG into the database table, adding logistics fields.

        Args:
            (none)

        Returns:
            pd.DataFrame - one row per part number with the engineering
            attributes from PARTS_CATALOG plus generated lead times and shelf
            life.

        Notes:
            Transformation: static tuple list -> table with the supply-chain
            fields EASA Part-145 stores control requires (traceable class,
            unit of measure, shelf life for life-limited materials).
        """
        records = []

        # Iterate over the static catalogue, unpacking each tuple into a row.
        for ata, sub, pn, desc, pclass, unit, mtbf_fh, mtbf_fc, cost, crit in PARTS_CATALOG:
            records.append({
                "part_number": pn,
                "description": desc,
                "ata_chapter": ata,
                "ata_subchapter": sub,
                "part_class": pclass,       # ROTABLE / EXPENDABLE / CONSUMABLE
                "unit_of_measure": unit,    # EA / KIT / LTR / ROLL
                "mtbf_flight_hours": mtbf_fh,
                "mtbf_flight_cycles": mtbf_fc,
                "unit_cost_eur": cost,
                "criticality": crit,        # AOG / MEL / ROUTINE

                # --- Routine procurement lead time (days) ---
                # Rotables need 5-30 days: they come from an overhaul shop with
                # a queue and an EASA Form 1 release. Expendables and
                # consumables are 1-10 days off a distributor's shelf.
                "lead_time_days_normal": (self.rng.integers(5, 30) if pclass == "ROTABLE"
                                          else self.rng.integers(1, 10)),

                # --- AOG lead time (days) ---
                # Only AOG-critical parts have an expedited channel (1-2 days
                # by courier/hand-carry). Anything else is None: paying AOG
                # freight for a routine item is not a real option.
                "lead_time_days_aog": self.rng.integers(1, 3) if crit == "AOG" else None,

                # Shelf life applies only to consumables - sealants, oils and
                # solvents cure or degrade in storage (12-60 months). Rotables
                # and expendables have no calendar storage limit here.
                "shelf_life_months": None if pclass != "CONSUMABLE" else self.rng.integers(12, 60),
            })

        return pd.DataFrame(records)

    def generate_inventory(self, parts_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate opening stock positions for every part at every station.

        Args:
            parts_df: pd.DataFrame - the parts catalogue table.

        Returns:
            pd.DataFrame - one row per (part, station) pair; 53 parts x 5
            stations = 265 rows.

        Notes:
            EASA Part-145 145.A.42 requires components to be segregated by
            condition, which is why serviceable and unserviceable quantities
            are tracked separately: an unserviceable unit awaiting shop input
            is physically present but cannot be fitted, and must never be
            counted as available stock.
        """
        records = []

        # Iterate over the full cartesian product: each part is stocked (or
        # deliberately not stocked) at each station.
        for _, part in parts_df.iterrows():
            for station, info in STATIONS.items():
                # --- Station stocking weight ---
                # Classic hub-and-spoke inventory policy: hold depth at the
                # main base where the heavy maintenance capability sits, and
                # keep outstations lean, relying on transfers from the hub.
                if station == "ATH":
                    stock_factor = 3.0   # main hub / main store
                elif station == "SKG":
                    stock_factor = 2.0   # secondary base
                else:
                    stock_factor = 1.0   # island outstations

                # --- Stocking policy by part class ---
                if part["part_class"] == "ROTABLE":
                    # Rotables are capital assets - 0-2 units per station even
                    # at the hub. Minimum of 1 only for AOG-critical items,
                    # where a stock-out grounds an aircraft; everything else
                    # is ordered on demand rather than held.
                    qty = max(0, int(stock_factor * self.rng.integers(0, 3)))
                    min_stock = 1 if part["criticality"] == "AOG" else 0
                elif part["part_class"] == "EXPENDABLE":
                    # Cheap and consumed steadily - held in tens.
                    qty = max(0, int(stock_factor * self.rng.integers(5, 30)))
                    min_stock = int(stock_factor * 5)
                else:
                    # CONSUMABLE - bulk material, held in the largest quantities
                    # and the cheapest class to over-stock.
                    qty = max(0, int(stock_factor * self.rng.integers(10, 50)))
                    min_stock = int(stock_factor * 10)

                records.append({
                    "part_number": part["part_number"],
                    "station": station,
                    "quantity_on_hand": qty,
                    "minimum_stock_level": min_stock,
                    # Reorder Point (ROP) set 50% above the minimum, so the
                    # replenishment order is raised before the floor is hit
                    # rather than at it - the buffer covers the lead time.
                    "reorder_point": int(min_stock * 1.5),
                    # Serviceable quantity - units with a valid EASA Form 1,
                    # fit to install. This is the column every availability
                    # query uses.
                    "serviceable": qty,
                    # Unserviceable - removed units awaiting shop or scrap,
                    # modelled as up to 20% of the on-hand quantity.
                    "unserviceable": self.rng.integers(0, max(1, qty // 5)),
                    # Last goods-receipt date within the previous 6 months,
                    # part of the Part-145 stores traceability record.
                    "last_receipt_date": (self.reference_date -
                                          timedelta(days=int(self.rng.integers(1, 180)))).strftime("%Y-%m-%d"),
                })

        return pd.DataFrame(records)

    def generate_maintenance_events(self, fleet_df: pd.DataFrame,
                                     flight_log_df: pd.DataFrame,
                                     parts_df: pd.DataFrame) -> tuple:
        """
        Generate work orders, inspection findings and part demands.

        Args:
            fleet_df: pd.DataFrame - aircraft register.
            flight_log_df: pd.DataFrame - sector-level flight log; supplies the
                cumulative flight hours that trigger scheduled checks.
            parts_df: pd.DataFrame - parts catalogue; supplies MTBF and class.

        Returns:
            tuple of three DataFrames:
                work_orders_df - every maintenance event, scheduled or not
                findings_df    - defects raised during scheduled inspections
                part_demands_df- the material consumption record; this is the
                                 training dataset for Module 2

        Notes:
            Two independent generating processes run per aircraft:

            1. SCHEDULED - walk the flight log day by day and raise a check
               whenever the accumulated flight hours cross an MPD interval
               (A-check 750 FH, C-check 7,500 FH). Findings are then drawn per
               inspected ATA chapter, with probability rising with airframe age
               and salt exposure.

            2. UNSCHEDULED - failures between checks. Expendables follow a
               Poisson process over accumulated flight hours; rotables follow a
               Weibull renewal process with a wear-out shape parameter.

            Both write into part_demands, tagged by demand_type, because the
            forecaster must learn the combined signal an operator actually
            sees - planned consumption plus random failures.
        """
        print("  Generating maintenance events...")
        work_orders = []
        findings = []
        part_demands = []

        # Human-readable running identifiers. Offsets (1000, 5000) keep work
        # order and finding numbers visually distinct in reports and logs.
        wo_counter = 1000
        finding_counter = 5000

        # Split the catalogue once, outside the per-aircraft loop - the two
        # classes are driven by completely different stochastic processes.
        rotables = parts_df[parts_df["part_class"] == "ROTABLE"]
        expendables = parts_df[parts_df["part_class"] == "EXPENDABLE"]

        # Iterate over the fleet: each aircraft has its own independent
        # maintenance history driven by its own utilisation.
        for _, ac in fleet_df.iterrows():
            tail = ac["tail_number"]
            ac_flights = flight_log_df[flight_log_df["tail_number"] == tail]

            # An aircraft with no flights (should not occur, but guard anyway)
            # has no exposure and therefore no maintenance history.
            if ac_flights.empty:
                continue

            # ================================================================
            # SCHEDULED MAINTENANCE - MPD interval tracking
            # ================================================================
            # Interval compliance is measured from the hours at which the last
            # check of that type was performed, not from a fixed grid; that is
            # how Part-M M.A.302 programmes actually work.
            cumulative_fh = 0
            last_a_check = 0
            last_c_check = 0

            # Walk the aircraft's operating days in chronological order,
            # accumulating flight hours until a check falls due.
            for date_str in sorted(ac_flights["flight_date"].unique()):
                day_flights = ac_flights[ac_flights["flight_date"] == date_str]
                daily_fh = day_flights["flight_hours"].sum()
                daily_fc = day_flights["flight_cycles"].sum()
                cumulative_fh += daily_fh

                # --- A-CHECK due? (MPD interval: 750 FH) ---
                if cumulative_fh - last_a_check >= 750:
                    wo_counter += 1
                    wo_id = f"WO-{wo_counter}"
                    work_orders.append({
                        "work_order_id": wo_id,
                        "tail_number": tail,
                        "check_type": "A-CHECK",
                        "scheduled_date": date_str,
                        "status": "COMPLETED",
                        # Airframe hours at the moment of the check - the
                        # figure an auditor uses to verify interval compliance.
                        "aircraft_fh_at_check": round(cumulative_fh, 1),
                        "source": "SCHEDULED",
                    })
                    # Reset the interval datum for the next A-check.
                    last_a_check = cumulative_fh

                    # --- Findings raised during this A-check ---
                    # One draw per inspected ATA chapter: an inspection can only
                    # find a defect in a system it actually looked at.
                    for ata, task_desc in CHECK_TASKS.get("A-CHECK", []):
                        # Age factor: divide by 20 so an 18-year airframe carries
                        # ~1.9x the finding rate of a brand-new one, reflecting
                        # cumulative fatigue and wear.
                        age_factor = 1 + (ac["age_years"] / 20)

                        # Environment factor: salt exposure 0.0-1.0 adds up to
                        # 100% to the finding rate. Chloride-driven corrosion is
                        # the dominant defect mechanism at island stations.
                        climate_factor = 1 + STATIONS.get(ac["home_base"], {}).get("salt_exposure", 0)

                        # 15% base probability per inspected chapter per A-check
                        # - an A-check is a light, largely operational inspection.
                        finding_prob = 0.15 * age_factor * climate_factor

                        # Bernoulli trial: was a defect found in this system?
                        if self.rng.random() < finding_prob:
                            finding_counter += 1

                            # Attribute the finding to a real part in that ATA
                            # chapter, so the resulting demand is coherent.
                            ata_parts = parts_df[parts_df["ata_chapter"] == ata]
                            if ata_parts.empty:
                                # Chapter inspected but no catalogued part in it
                                # (e.g. a purely structural task) - no demand.
                                continue
                            part = ata_parts.iloc[self.rng.integers(0, len(ata_parts))]

                            # Defect type distribution for a LIGHT check. WEAR
                            # dominates (30%) because that is what an
                            # operational inspection surfaces; CRACK is rare
                            # (10%) since structure is not opened at A-check.
                            finding_type = self.rng.choice(
                                ["CRACK", "WEAR", "CORROSION", "LEAK", "MALFUNCTION"],
                                p=[0.1, 0.3, 0.2, 0.2, 0.2]
                            )

                            findings.append({
                                "finding_id": f"FND-{finding_counter}",
                                "work_order_id": wo_id,
                                "tail_number": tail,
                                "ata_chapter": ata,
                                "finding_date": date_str,
                                "finding_type": finding_type,
                                "description": f"{finding_type} found on {part['description']} during {task_desc}",
                                # Rotables and expendables are replaced;
                                # consumables are serviced/replenished in place.
                                "action_taken": "REPLACED" if part["part_class"] in ["ROTABLE", "EXPENDABLE"] else "SERVICED",
                                "part_number_required": part["part_number"],
                                "aircraft_fh": round(cumulative_fh, 1),
                            })

                            # Every finding generates a material demand - this
                            # is the link between engineering and supply chain.
                            part_demands.append({
                                "finding_id": f"FND-{finding_counter}",
                                "work_order_id": wo_id,
                                "tail_number": tail,
                                "part_number": part["part_number"],
                                "quantity_required": 1,
                                "demand_type": "SCHEDULED_FINDING",
                                "demand_date": date_str,
                                "station": ac["home_base"],
                                "criticality": part["criticality"],
                            })

                # --- C-CHECK due? (MPD interval: 7,500 FH) ---
                if cumulative_fh - last_c_check >= 7500:
                    wo_counter += 1
                    wo_id = f"WO-{wo_counter}"
                    work_orders.append({
                        "work_order_id": wo_id,
                        "tail_number": tail,
                        "check_type": "C-CHECK",
                        "scheduled_date": date_str,
                        "status": "COMPLETED",
                        "aircraft_fh_at_check": round(cumulative_fh, 1),
                        "source": "SCHEDULED",
                    })
                    last_c_check = cumulative_fh

                    # --- Findings raised during this C-check ---
                    # A C-check opens structure and bench-tests components, so
                    # every factor is stronger than at A-check.
                    for ata, task_desc in CHECK_TASKS.get("C-CHECK", []):
                        # Steeper age response (divide by 15, not 20): deep
                        # inspection exposes accumulated damage that a light
                        # check never reaches.
                        age_factor = 1 + (ac["age_years"] / 15)

                        # Salt exposure weighted 1.5x: the CPCP corrosion tasks
                        # performed at C-check are exactly what detects
                        # chloride attack, so the environment shows up far more
                        # strongly here than in an operational check.
                        climate_factor = 1 + STATIONS.get(ac["home_base"], {}).get("salt_exposure", 0) * 1.5

                        # 35% base probability per chapter - more than double
                        # the A-check rate, reflecting inspection depth.
                        finding_prob = 0.35 * age_factor * climate_factor

                        if self.rng.random() < finding_prob:
                            finding_counter += 1
                            ata_parts = parts_df[parts_df["ata_chapter"] == ata]
                            if ata_parts.empty:
                                continue
                            part = ata_parts.iloc[self.rng.integers(0, len(ata_parts))]

                            # Defect distribution for a HEAVY check: CRACK rises
                            # to 15% and CORROSION to 25% because structure is
                            # opened and CPCP tasks are performed; simple WEAR
                            # falls to 25% as it is no longer the main finding.
                            finding_type = self.rng.choice(
                                ["CRACK", "WEAR", "CORROSION", "LEAK", "MALFUNCTION"],
                                p=[0.15, 0.25, 0.25, 0.15, 0.2]
                            )

                            findings.append({
                                "finding_id": f"FND-{finding_counter}",
                                "work_order_id": wo_id,
                                "tail_number": tail,
                                "ata_chapter": ata,
                                "finding_date": date_str,
                                "finding_type": finding_type,
                                "description": f"{finding_type} on {part['description']} during C-CHECK {task_desc}",
                                "action_taken": "REPLACED",
                                "part_number_required": part["part_number"],
                                "aircraft_fh": round(cumulative_fh, 1),
                            })

                            part_demands.append({
                                "finding_id": f"FND-{finding_counter}",
                                "work_order_id": wo_id,
                                "tail_number": tail,
                                "part_number": part["part_number"],
                                # Rotables are replaced one at a time; cheaper
                                # classes are consumed in batches of 1-3 during
                                # heavy maintenance (multiple seals, fasteners).
                                "quantity_required": 1 if part["part_class"] == "ROTABLE" else self.rng.integers(1, 4),
                                "demand_type": "SCHEDULED_FINDING",
                                "demand_date": date_str,
                                "station": ac["home_base"],
                                "criticality": part["criticality"],
                            })

            # ================================================================
            # UNSCHEDULED FAILURES - EXPENDABLES (Poisson process)
            # ================================================================
            # Expendables fail from random external causes (vibration, thermal
            # shock, filament burnout), not progressive wear. The appropriate
            # model is a homogeneous Poisson process: constant rate per flight
            # hour, memoryless, so N failures over exposure T is Poisson(lambda*T).
            for _, part in expendables.iterrows():
                # Base rate: 5e-4 failures per flight hour = 1 failure per
                # 2,000 FH per aircraft for a typical expendable.
                # Salt exposure adds up to 100%: corrosion attacks the
                # connectors, seals and contacts these parts are made of.
                lambda_per_fh = 0.0005 * (1 + STATIONS.get(ac["home_base"], {}).get("salt_exposure", 0))

                # --- Part-family rate multipliers ---
                # Different expendable families have genuinely different
                # baseline rates; these multipliers encode that ordering.
                if "LAMP" in part["description"].upper() or "LIGHT" in part["description"].upper():
                    # 3x - filament and LED assemblies are the single highest
                    # turnover expendable class in any fleet.
                    lambda_per_fh *= 3
                elif "SEAL" in part["description"].upper() or "O-RING" in part["description"].upper():
                    # 2x - elastomers harden and take a compression set with
                    # thermal cycling.
                    lambda_per_fh *= 2
                elif "FILTER" in part["description"].upper():
                    # 1.5x - filters clog progressively; replacement is often
                    # condition-driven rather than a true failure.
                    lambda_per_fh *= 1.5

                # Poisson mean over the whole simulated exposure:
                #   E[N] = lambda * total_flight_hours
                total_expected_failures = lambda_per_fh * cumulative_fh
                n_failures = self.rng.poisson(total_expected_failures)

                if n_failures > 0:
                    # In a homogeneous Poisson process the event times are
                    # uniformly distributed over the interval, conditional on
                    # the count - so draw uniform days and sort them.
                    # Capped at 20 events per part per aircraft: beyond that the
                    # row count explodes without adding modelling information.
                    failure_days = sorted(self.rng.integers(0, self.simulation_days,
                                                            size=min(n_failures, 20)))

                    # One work order and one material demand per failure event.
                    for fd in failure_days:
                        failure_date = (self.reference_date + timedelta(days=int(fd))).strftime("%Y-%m-%d")
                        wo_counter += 1
                        finding_counter += 1

                        work_orders.append({
                            "work_order_id": f"WO-{wo_counter}",
                            "tail_number": tail,
                            "check_type": "UNSCHEDULED",
                            "scheduled_date": failure_date,
                            "status": "COMPLETED",
                            # Linear interpolation of airframe hours at the
                            # failure date: total_fh * (day / total_days).
                            "aircraft_fh_at_check": round(cumulative_fh * fd / self.simulation_days, 1),
                            # Expendable failures surface as crew-reported
                            # defects (a lamp out, a leak seen on walk-around).
                            "source": "PILOT_REPORT",
                        })

                        part_demands.append({
                            "finding_id": f"FND-{finding_counter}",
                            "work_order_id": f"WO-{wo_counter}",
                            "tail_number": tail,
                            "part_number": part["part_number"],
                            "quantity_required": 1,
                            "demand_type": "UNSCHEDULED",
                            "demand_date": failure_date,
                            "station": ac["home_base"],
                            "criticality": part["criticality"],
                        })

            # ================================================================
            # UNSCHEDULED FAILURES - ROTABLES (Weibull renewal process)
            # ================================================================
            # Rotable parts (expensive, repairable) fail by progressive wear-out.
            # The Weibull distribution models this: shape beta > 1 means the
            # hazard rate increases with age, which is the defining
            # characteristic of mechanical wear as opposed to random failure.
            # Reference: EASA Part-M M.A.302 - reliability programme data analysis.
            for _, part in rotables.iterrows():
                # --- Establish the baseline MTBF in flight hours ---
                # Prefer the hour-based figure. For cycle-limited parts (wheels,
                # brakes, gear and door actuators) convert cycles to hours using
                # THIS aircraft's cycle ratio - the same wheel lasts far fewer
                # calendar hours on a short-haul airframe than a medium-haul one.
                #   MTBF_fh = MTBF_fc / (cycles per flight hour)
                mtbf = part["mtbf_flight_hours"] or (
                    part["mtbf_flight_cycles"] / ac["cycles_per_fh_ratio"]
                    if part["mtbf_flight_cycles"] else None
                )
                if mtbf is None:
                    # No reliability data for this part - cannot model it, and
                    # inventing a figure would corrupt the training set.
                    continue

                # --- Multi-factor stress adjustment ---
                # A published MTBF assumes average operating conditions. Real
                # failure rates depend on how the aircraft is actually used.
                stress_factor = 1.0

                # Factor 1 - short-sector penalty (+30%).
                # More takeoff/landing cycles per flight hour means more thermal
                # cycling, more pressurisation cycles, more gear and brake
                # applications per hour of running time.
                if ac["primary_role"] == "short_haul":
                    stress_factor *= 1.3

                # Factor 2 - airframe age degradation (+2% per year).
                # An ageing structure distributes loads differently as fatigue
                # accumulates, and installed systems inherit that environment.
                stress_factor *= 1 + (ac["age_years"] * 0.02)

                # Factor 3 - corrosive environment (up to +15%).
                # Weighted lower than the finding-rate effect because a sealed,
                # installed rotable is far better protected from chloride attack
                # than the airframe structure around it.
                stress_factor *= 1 + STATIONS.get(ac["home_base"], {}).get("salt_exposure", 0) * 0.15

                # Higher stress compresses the characteristic life.
                adjusted_mtbf = mtbf / stress_factor

                # --- Weibull parameters ---
                # beta (shape) = 2.5 - the standard value for mechanical
                # wear-out in reliability engineering. beta = 1 would be a
                # constant hazard (random failure, i.e. exponential), beta < 1
                # infant mortality; 2.5 places the failure mode firmly in
                # wear-out, matching bearings, seals, pumps and actuators.
                beta = 2.5
                # eta (scale) = characteristic life, the age by which ~63.2% of
                # the population has failed. Approximated by the adjusted MTBF.
                eta = adjusted_mtbf

                # Total flight-hour exposure over the simulated period.
                sim_fh = ac["daily_utilization_fh"] * self.simulation_days
                current_life = 0

                # Renewal process: each failure is repaired/replaced and the
                # clock restarts, so keep drawing successive times-to-failure
                # until the accumulated life exceeds the simulation window.
                while current_life < sim_fh:
                    # Draw a time-to-failure. numpy's weibull(beta) returns a
                    # standardised variate (eta = 1); multiplying by eta scales
                    # it to the part's characteristic life.
                    ttf = eta * self.rng.weibull(beta)
                    current_life += ttf

                    # Only record failures that fall inside the window; the
                    # final draw normally overshoots the end of the period.
                    if current_life < sim_fh:
                        # Convert accumulated flight hours back to a calendar
                        # day using mean daily utilisation.
                        failure_day = int(current_life / ac["daily_utilization_fh"])
                        if failure_day >= self.simulation_days:
                            # Rounding pushed it past the last simulated day.
                            break
                        failure_date = (self.reference_date + timedelta(days=failure_day)).strftime("%Y-%m-%d")

                        wo_counter += 1
                        finding_counter += 1

                        work_orders.append({
                            "work_order_id": f"WO-{wo_counter}",
                            "tail_number": tail,
                            "check_type": "UNSCHEDULED",
                            "scheduled_date": failure_date,
                            "status": "COMPLETED",
                            # Exact accumulated life at failure - this is the
                            # survival time Module 2's Cox model consumes.
                            "aircraft_fh_at_check": round(current_life, 1),
                            "source": "UNSCHEDULED_FAILURE",
                        })

                        part_demands.append({
                            "finding_id": f"FND-{finding_counter}",
                            "work_order_id": f"WO-{wo_counter}",
                            "tail_number": tail,
                            "part_number": part["part_number"],
                            "quantity_required": 1,
                            "demand_type": "UNSCHEDULED",
                            "demand_date": failure_date,
                            "station": ac["home_base"],
                            "criticality": part["criticality"],
                        })

        wo_df = pd.DataFrame(work_orders)
        find_df = pd.DataFrame(findings)
        demand_df = pd.DataFrame(part_demands)

        print(f"    Work orders: {len(wo_df)}")
        print(f"    Findings: {len(find_df)}")
        print(f"    Part demands: {len(demand_df)}")

        return wo_df, find_df, demand_df


# ============================================================================
# PART 5: DATABASE BUILDER
# ============================================================================

def build_database(fleet_df, flight_log_df, parts_df, inventory_df,
                   work_orders_df, findings_df, part_demands_df,
                   sdr_df=None):
    """
    Persist every generated table to SQLite and create the reporting views.

    Args:
        fleet_df: pd.DataFrame - aircraft register
        flight_log_df: pd.DataFrame - sector-level flight log
        parts_df: pd.DataFrame - parts catalogue
        inventory_df: pd.DataFrame - per-station stock positions
        work_orders_df: pd.DataFrame - maintenance events
        findings_df: pd.DataFrame - inspection findings
        part_demands_df: pd.DataFrame - material consumption records
        sdr_df: pd.DataFrame or None - real FAA SDR records, when available

    Returns:
        None - writes DB_PATH as a side effect.

    Notes:
        Tables are written with if_exists="replace" so the pipeline is
        idempotent: re-running it rebuilds a clean database rather than
        appending duplicates. The schema mirrors the EASA Part-M / Part-145
        data requirements implemented fully in schema_postgres.sql.
    """
    print(f"\n  Building database: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)

    # --- Core tables ---
    fleet_df.to_sql("fleet", conn, if_exists="replace", index=False)
    flight_log_df.to_sql("flight_log", conn, if_exists="replace", index=False)
    parts_df.to_sql("parts_catalog", conn, if_exists="replace", index=False)
    inventory_df.to_sql("inventory", conn, if_exists="replace", index=False)
    work_orders_df.to_sql("work_orders", conn, if_exists="replace", index=False)

    # Findings and demands can legitimately be empty on a very short simulation
    # run; to_sql on an empty frame would create a column-less table that later
    # queries could not read, so guard both writes.
    if not findings_df.empty:
        findings_df.to_sql("findings", conn, if_exists="replace", index=False)
    if not part_demands_df.empty:
        part_demands_df.to_sql("part_demands", conn, if_exists="replace", index=False)

    # --- Reference table: the approved maintenance programme ---
    # Persisted so the check intervals used to generate the data are auditable
    # from the database itself, not only from this source file.
    insp_records = []
    for ct, ifh, ifc, imo, dur, desc in INSPECTION_PROGRAM:
        insp_records.append({
            "check_type": ct,
            "interval_flight_hours": ifh,
            "interval_flight_cycles": ifc,
            "interval_months": imo,
            "duration_days": dur,
            "description": desc,
        })
    pd.DataFrame(insp_records).to_sql("inspection_program", conn, if_exists="replace", index=False)

    # --- Reference table: stations ---
    # Flatten {code: {attrs}} into rows so salt_exposure can be joined as a
    # covariate in the survival model.
    station_records = [{"station_code": k, **v} for k, v in STATIONS.items()]
    pd.DataFrame(station_records).to_sql("stations", conn, if_exists="replace", index=False)

    # --- Real FAA SDR corpus, when CSVs were supplied ---
    if sdr_df is not None and not sdr_df.empty:
        sdr_df.to_sql("faa_sdr_raw", conn, if_exists="replace", index=False)
        print(f"    FAA SDR data: {len(sdr_df)} records")

    # ------------------------------------------------------------------
    # REPORTING VIEWS
    # Pre-defined so the agent, the explorer and any ad-hoc SQL client all
    # compute the same figures the same way.
    # ------------------------------------------------------------------

    # Monthly demand per part / type / station - the aggregation the XGBoost
    # demand forecaster trains on. substr(demand_date,1,7) yields 'YYYY-MM',
    # SQLite's cheapest way to bucket an ISO date string by month.
    conn.execute("""
        CREATE VIEW IF NOT EXISTS v_demand_by_part_month AS
        SELECT
            part_number,
            substr(demand_date, 1, 7) as month,
            demand_type,
            station,
            COUNT(*) as demand_count,
            SUM(quantity_required) as total_qty
        FROM part_demands
        GROUP BY part_number, substr(demand_date, 1, 7), demand_type, station
    """)

    # Fleet status with a reliability metric: total work orders versus the
    # subset that were genuine unscheduled failures. The ratio between the two
    # is the headline figure of any operator reliability report.
    conn.execute("""
        CREATE VIEW IF NOT EXISTS v_fleet_status AS
        SELECT
            f.tail_number,
            f.aircraft_model,
            f.age_years,
            f.home_base,
            f.primary_role,
            f.cycles_per_fh_ratio,
            COUNT(DISTINCT wo.work_order_id) as total_work_orders,
            SUM(CASE WHEN wo.source = 'UNSCHEDULED_FAILURE' THEN 1 ELSE 0 END) as unscheduled_failures
        FROM fleet f
        LEFT JOIN work_orders wo ON f.tail_number = wo.tail_number
        GROUP BY f.tail_number
    """)

    # Per-part reliability: observed demand count against the MTBF that was
    # assumed. Divergence between the two is exactly what a Part-M reliability
    # programme looks for when deciding to revise the maintenance programme.
    conn.execute("""
        CREATE VIEW IF NOT EXISTS v_part_reliability AS
        SELECT
            pc.part_number,
            pc.description,
            pc.ata_chapter,
            pc.part_class,
            pc.mtbf_flight_hours,
            COUNT(pd.part_number) as total_demands,
            pc.unit_cost_eur,
            pc.criticality
        FROM parts_catalog pc
        LEFT JOIN part_demands pd ON pc.part_number = pd.part_number
        GROUP BY pc.part_number
        ORDER BY total_demands DESC
    """)

    conn.commit()

    # --- Summary report ---
    cursor = conn.cursor()
    print(f"\n  {'=' * 50}")
    print(f"  DATABASE SUMMARY")
    print(f"  {'=' * 50}")

    # Count rows in each core table so the user can confirm the build produced
    # what they expected before moving on to the ML modules.
    for table in ["fleet", "flight_log", "parts_catalog", "inventory",
                  "work_orders", "findings", "part_demands", "inspection_program", "stations"]:
        try:
            count = cursor.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"    {table:<25} {count:>8} records")
        except sqlite3.Error:
            # A table can legitimately be absent (empty findings/demands on a
            # very short run). Skip it rather than aborting the summary.
            pass

    conn.close()
    print(f"\n  Database saved: {os.path.abspath(DB_PATH)}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """
    Run the full pipeline: parse real data, generate synthetic data, build DB.

    Args:
        (none)

    Returns:
        None

    Notes:
        Ordering matters. SDR parsing runs first so that its failure-frequency
        report is visible before any synthetic data is produced - the real data
        is the reference against which the synthetic catalogue is judged.
        Within generation, each step consumes the previous one's output: fleet
        -> flights -> catalogue/inventory -> maintenance events.
    """
    # Report the active hardware profile up front: it determines chunk sizes
    # and any record cap, and explains any difference in run time or row counts.
    cfg = get_config()

    print("=" * 60)
    print("  Daedalus Supply AI - Module 1: Data Pipeline")
    print("  EASA Part-M / Part-145 / ICAO Annex 6 & 8 / MSG-3")
    print(f"  Hardware profile: {cfg['profile']} "
          f"({cfg['hardware']['ram_gb']}GB RAM, {cfg['hardware']['cores']} cores)")
    print("=" * 60)

    # --- Step 1: real data ---
    print("\n[1/5] Loading FAA SDR data...")
    sdr_df = parse_sdr_data(RAW_DATA_DIR, AIRCRAFT_TYPE_FILTER)

    if not sdr_df.empty:
        parts_from_sdr = extract_parts_from_sdr(sdr_df)
        print(f"\n  Extracted {len(parts_from_sdr)} unique parts from SDR data")
    else:
        # Running without SDR files is fully supported - the synthetic layer is
        # self-contained. Say so plainly and point at the download source.
        print("  No SDR data found. Using synthetic catalog only.")
        print("  Download SDR CSVs from https://www.faa.gov/av-info/download_SDR")

    # --- Step 2-5: synthetic generation ---
    # Fixed seed: the database must be reproducible for audit purposes.
    gen = AeroDataGenerator(seed=42)

    print("\n[2/5] Generating fleet register...")
    fleet_df = gen.generate_fleet_table()
    print(f"  {len(fleet_df)} aircraft")

    print("\n[3/5] Generating flight logs (3 years)...")
    flight_log_df = gen.generate_flight_log(fleet_df)

    print("\n[4/5] Generating parts catalog & inventory...")
    parts_df = gen.generate_parts_catalog_table()
    inventory_df = gen.generate_inventory(parts_df)
    print(f"  {len(parts_df)} part numbers")
    print(f"  {len(inventory_df)} inventory records "
          f"({len(parts_df)} parts x {len(STATIONS)} stations)")

    print("\n[5/5] Generating maintenance events...")
    wo_df, find_df, demand_df = gen.generate_maintenance_events(fleet_df, flight_log_df, parts_df)

    # --- Persist everything ---
    build_database(fleet_df, flight_log_df, parts_df, inventory_df,
                   wo_df, find_df, demand_df,
                   sdr_df if not sdr_df.empty else None)

    # Closing status line: where the data is and what to run next.
    print("\n" + "=" * 60)
    print(f"  DONE - database ready at {DB_PATH}")
    print("  Run queries in the database, or continue with:")
    print("    python prediction_model.py     (Module 2 - train the models)")
    print("=" * 60)


if __name__ == "__main__":
    main()
