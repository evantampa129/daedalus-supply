-- ============================================================================
-- Daedalus Supply AI - PostgreSQL Schema
-- EASA Part-M / Part-145 / ICAO Annex 6 & 8 Compliant Data Model
-- ============================================================================
-- Run: psql -U your_user -d daedalus_supply -f schema_postgres.sql
-- ============================================================================

-- Clean start
DROP SCHEMA IF EXISTS aero CASCADE;
CREATE SCHEMA aero;
SET search_path TO aero;

-- Custom ENUM types
CREATE TYPE part_class_enum AS ENUM ('ROTABLE', 'EXPENDABLE', 'CONSUMABLE');
CREATE TYPE criticality_enum AS ENUM ('AOG', 'MEL', 'ROUTINE');
CREATE TYPE aircraft_status_enum AS ENUM ('ACTIVE', 'STORED', 'GROUNDED', 'RETIRED');
CREATE TYPE wo_status_enum AS ENUM ('OPEN', 'IN_PROGRESS', 'COMPLETED', 'DEFERRED', 'CANCELLED');
CREATE TYPE wo_source_enum AS ENUM ('SCHEDULED', 'PILOT_REPORT', 'UNSCHEDULED_FAILURE', 'AD_COMPLIANCE', 'SB_COMPLIANCE');
CREATE TYPE finding_type_enum AS ENUM ('CRACK', 'WEAR', 'CORROSION', 'LEAK', 'MALFUNCTION', 'DAMAGE', 'CONTAMINATION', 'OTHER');
CREATE TYPE demand_type_enum AS ENUM ('SCHEDULED_FINDING', 'UNSCHEDULED', 'PROACTIVE_PREPOSITION', 'STOCK_REPLENISHMENT');
CREATE TYPE season_enum AS ENUM ('summer', 'winter', 'shoulder');

-- ============================================================================
-- STATIONS / WAREHOUSES
-- ============================================================================
CREATE TABLE stations (
    station_code    VARCHAR(5) PRIMARY KEY,
    name            VARCHAR(100) NOT NULL,
    city            VARCHAR(100),
    country         VARCHAR(3) DEFAULT 'GRC',
    climate         VARCHAR(30),
    salt_exposure   NUMERIC(3,2) DEFAULT 0.0 CHECK (salt_exposure BETWEEN 0 AND 1),
    is_main_base    BOOLEAN DEFAULT FALSE,
    has_hangar      BOOLEAN DEFAULT FALSE,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

COMMENT ON TABLE stations IS 'Stations/warehouses - EASA Part-145 approved locations';
COMMENT ON COLUMN stations.salt_exposure IS 'Corrosion risk factor 0-1 (island bases higher)';

-- ============================================================================
-- FLEET REGISTER - EASA Part-M M.A.305
-- ============================================================================
CREATE TABLE fleet (
    tail_number         VARCHAR(10) PRIMARY KEY,
    aircraft_model      VARCHAR(30) NOT NULL,
    manufacturer        VARCHAR(30) DEFAULT 'AIRBUS',
    serial_number       VARCHAR(20),
    manufacture_year    INTEGER NOT NULL CHECK (manufacture_year BETWEEN 1970 AND 2030),
    home_base           VARCHAR(5) NOT NULL REFERENCES stations(station_code),
    primary_role        VARCHAR(20) NOT NULL,
    total_flight_hours  NUMERIC(10,1) DEFAULT 0,
    total_flight_cycles INTEGER DEFAULT 0,
    cycles_per_fh_ratio NUMERIC(5,3),
    daily_utilization_fh NUMERIC(4,1),
    status              aircraft_status_enum DEFAULT 'ACTIVE',
    last_c_check_date   DATE,
    last_c_check_fh     NUMERIC(10,1),
    next_c_check_due_fh NUMERIC(10,1),
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    updated_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_fleet_base ON fleet(home_base);
CREATE INDEX idx_fleet_status ON fleet(status);

COMMENT ON TABLE fleet IS 'Aircraft register per EASA Part-M M.A.305 continuing airworthiness records';

-- ============================================================================
-- PARTS CATALOG - IPC structure, ATA/JASC coded
-- ============================================================================
CREATE TABLE parts_catalog (
    part_number         VARCHAR(30) PRIMARY KEY,
    description         VARCHAR(200) NOT NULL,
    ata_chapter         SMALLINT NOT NULL CHECK (ata_chapter BETWEEN 1 AND 99),
    ata_subchapter      SMALLINT DEFAULT 0,
    part_class          part_class_enum NOT NULL,
    unit_of_measure     VARCHAR(10) DEFAULT 'EA',
    mtbf_flight_hours   NUMERIC(10,1),
    mtbf_flight_cycles  NUMERIC(10,1),
    unit_cost_eur       NUMERIC(12,2),
    criticality         criticality_enum NOT NULL DEFAULT 'ROUTINE',
    lead_time_days_normal INTEGER,
    lead_time_days_aog  INTEGER,
    shelf_life_months   INTEGER,
    is_serialized       BOOLEAN DEFAULT FALSE,    -- rotables are typically serialized
    requires_form1      BOOLEAN DEFAULT TRUE,     -- EASA Form 1 / FAA 8130-3
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    updated_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_parts_ata ON parts_catalog(ata_chapter, ata_subchapter);
CREATE INDEX idx_parts_class ON parts_catalog(part_class);
CREATE INDEX idx_parts_criticality ON parts_catalog(criticality);

COMMENT ON TABLE parts_catalog IS 'Parts catalog per IPC structure - ATA/JASC coded';
COMMENT ON COLUMN parts_catalog.requires_form1 IS 'EASA Form 1 required for installation (Part-M M.A.501)';

-- ============================================================================
-- ALTERNATE PART NUMBERS (interchangeability)
-- ============================================================================
CREATE TABLE part_alternates (
    id                  SERIAL PRIMARY KEY,
    part_number         VARCHAR(30) NOT NULL REFERENCES parts_catalog(part_number),
    alternate_pn        VARCHAR(30) NOT NULL,
    alternate_source    VARCHAR(20) DEFAULT 'OEM',  -- OEM, PMA, TSO
    notes               TEXT,
    UNIQUE(part_number, alternate_pn)
);

COMMENT ON TABLE part_alternates IS 'Interchangeable part numbers - IPC cross-reference';

-- ============================================================================
-- INVENTORY - stock per station, EASA Part-145 stores
-- ============================================================================
CREATE TABLE inventory (
    id                  SERIAL PRIMARY KEY,
    part_number         VARCHAR(30) NOT NULL REFERENCES parts_catalog(part_number),
    station             VARCHAR(5) NOT NULL REFERENCES stations(station_code),
    quantity_serviceable INTEGER DEFAULT 0 CHECK (quantity_serviceable >= 0),
    quantity_unserviceable INTEGER DEFAULT 0 CHECK (quantity_unserviceable >= 0),
    quantity_in_repair  INTEGER DEFAULT 0,
    minimum_stock_level INTEGER DEFAULT 0,
    reorder_point       INTEGER DEFAULT 0,
    maximum_stock_level INTEGER,
    last_receipt_date   DATE,
    last_issue_date     DATE,
    updated_at          TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(part_number, station)
);

CREATE INDEX idx_inv_station ON inventory(station);
CREATE INDEX idx_inv_part ON inventory(part_number);
CREATE INDEX idx_inv_low_stock ON inventory(quantity_serviceable) 
    WHERE quantity_serviceable <= 2;  -- partial index for alerts

COMMENT ON TABLE inventory IS 'Stock levels per station - EASA Part-145 stores management';

-- ============================================================================
-- FLIGHT LOG - operational data driving maintenance intervals
-- ============================================================================
CREATE TABLE flight_log (
    id                  BIGSERIAL PRIMARY KEY,
    tail_number         VARCHAR(10) NOT NULL REFERENCES fleet(tail_number),
    flight_date         DATE NOT NULL,
    flight_number       VARCHAR(10),
    origin              VARCHAR(5) NOT NULL,
    destination         VARCHAR(5) NOT NULL,
    flight_hours        NUMERIC(5,2) NOT NULL CHECK (flight_hours > 0),
    flight_cycles       SMALLINT NOT NULL DEFAULT 1,
    cumulative_fh       NUMERIC(10,1),
    cumulative_fc       INTEGER,
    month               SMALLINT,
    season              season_enum
);

CREATE INDEX idx_flight_tail ON flight_log(tail_number);
CREATE INDEX idx_flight_date ON flight_log(flight_date);
CREATE INDEX idx_flight_tail_date ON flight_log(tail_number, flight_date);

-- Partition by year for performance on large datasets
-- CREATE TABLE flight_log_2024 PARTITION OF flight_log FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');
-- CREATE TABLE flight_log_2025 PARTITION OF flight_log FOR VALUES FROM ('2025-01-01') TO ('2026-01-01');

COMMENT ON TABLE flight_log IS 'Flight log - drives FH/FC accumulation and interval tracking';

-- ============================================================================
-- INSPECTION PROGRAM - MSG-3 / AMP based
-- ============================================================================
CREATE TABLE inspection_program (
    check_type              VARCHAR(20) PRIMARY KEY,
    interval_flight_hours   NUMERIC(10,1),
    interval_flight_cycles  INTEGER,
    interval_months         INTEGER,
    duration_days           NUMERIC(5,1),
    description             TEXT,
    source_document         VARCHAR(50) DEFAULT 'MPD',
    msg3_category           VARCHAR(30)
);

COMMENT ON TABLE inspection_program IS 'Check intervals per MSG-3/MPD - EASA Part-M M.A.302';

-- ============================================================================
-- WORK ORDERS - maintenance actions
-- ============================================================================
CREATE TABLE work_orders (
    work_order_id       VARCHAR(20) PRIMARY KEY,
    tail_number         VARCHAR(10) NOT NULL REFERENCES fleet(tail_number),
    check_type          VARCHAR(20) NOT NULL,
    scheduled_date      DATE NOT NULL,
    completion_date     DATE,
    status              wo_status_enum DEFAULT 'OPEN',
    aircraft_fh_at_check NUMERIC(10,1),
    aircraft_fc_at_check INTEGER,
    source              wo_source_enum NOT NULL,
    station             VARCHAR(5) REFERENCES stations(station_code),
    assigned_to         VARCHAR(50),           -- technician / team
    release_to_service  BOOLEAN DEFAULT FALSE, -- CRS issued
    crs_date            TIMESTAMPTZ,
    crs_signatory       VARCHAR(50),           -- Part-145 certifying staff
    notes               TEXT,
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    updated_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_wo_tail ON work_orders(tail_number);
CREATE INDEX idx_wo_date ON work_orders(scheduled_date);
CREATE INDEX idx_wo_status ON work_orders(status);
CREATE INDEX idx_wo_source ON work_orders(source);
CREATE INDEX idx_wo_check ON work_orders(check_type);

COMMENT ON TABLE work_orders IS 'Maintenance work orders - Part-145 M.A.801';
COMMENT ON COLUMN work_orders.release_to_service IS 'Certificate of Release to Service per Part-145 145.A.50';

-- ============================================================================
-- FINDINGS - inspection results
-- ============================================================================
CREATE TABLE findings (
    finding_id          VARCHAR(20) PRIMARY KEY,
    work_order_id       VARCHAR(20) NOT NULL REFERENCES work_orders(work_order_id),
    tail_number         VARCHAR(10) NOT NULL REFERENCES fleet(tail_number),
    ata_chapter         SMALLINT NOT NULL,
    finding_date        DATE NOT NULL,
    finding_type        finding_type_enum NOT NULL,
    description         TEXT NOT NULL,
    action_taken        VARCHAR(30),
    part_number_required VARCHAR(30) REFERENCES parts_catalog(part_number),
    aircraft_fh         NUMERIC(10,1),
    aircraft_fc         INTEGER,
    is_repetitive       BOOLEAN DEFAULT FALSE,
    deferred            BOOLEAN DEFAULT FALSE,  -- MEL deferral
    mel_reference       VARCHAR(20),
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_find_wo ON findings(work_order_id);
CREATE INDEX idx_find_tail ON findings(tail_number);
CREATE INDEX idx_find_ata ON findings(ata_chapter);
CREATE INDEX idx_find_type ON findings(finding_type);
CREATE INDEX idx_find_date ON findings(finding_date);

COMMENT ON TABLE findings IS 'Inspection findings - non-routine items per Part-M M.A.403';

-- ============================================================================
-- PART DEMANDS - THE ML TRAINING TABLE
-- ============================================================================
CREATE TABLE part_demands (
    id                  BIGSERIAL PRIMARY KEY,
    finding_id          VARCHAR(20),
    work_order_id       VARCHAR(20) REFERENCES work_orders(work_order_id),
    tail_number         VARCHAR(10) NOT NULL REFERENCES fleet(tail_number),
    part_number         VARCHAR(30) NOT NULL REFERENCES parts_catalog(part_number),
    quantity_required   INTEGER NOT NULL DEFAULT 1 CHECK (quantity_required > 0),
    demand_type         demand_type_enum NOT NULL,
    demand_date         DATE NOT NULL,
    station             VARCHAR(5) NOT NULL REFERENCES stations(station_code),
    criticality         criticality_enum NOT NULL,
    fulfilled           BOOLEAN DEFAULT FALSE,
    fulfilled_from      VARCHAR(5) REFERENCES stations(station_code),  -- which station supplied
    fulfillment_hours   NUMERIC(6,1),  -- time to fulfill (for logistics KPI)
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_demand_part ON part_demands(part_number);
CREATE INDEX idx_demand_date ON part_demands(demand_date);
CREATE INDEX idx_demand_station ON part_demands(station);
CREATE INDEX idx_demand_tail ON part_demands(tail_number);
CREATE INDEX idx_demand_type ON part_demands(demand_type);
CREATE INDEX idx_demand_unfulfilled ON part_demands(fulfilled) WHERE fulfilled = FALSE;

COMMENT ON TABLE part_demands IS 'Part demand history - primary ML training dataset';

-- ============================================================================
-- FAA SDR RAW DATA (optional - loaded from CSV)
-- ============================================================================
CREATE TABLE faa_sdr_raw (
    id                  BIGSERIAL PRIMARY KEY,
    report_date         DATE,
    acft_make           VARCHAR(50),
    acft_model          VARCHAR(50),
    acft_serial         VARCHAR(30),
    total_time          NUMERIC(10,1),
    ata_code            VARCHAR(10),
    part_name           VARCHAR(200),
    part_number         VARCHAR(50),
    part_condition      VARCHAR(50),
    nature_condition    VARCHAR(100),
    stage_of_operation  VARCHAR(50),
    precautionary_procedure VARCHAR(100),
    remarks             TEXT,
    loaded_at           TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_sdr_model ON faa_sdr_raw(acft_model);
CREATE INDEX idx_sdr_ata ON faa_sdr_raw(ata_code);
CREATE INDEX idx_sdr_part ON faa_sdr_raw(part_name);

-- ============================================================================
-- VIEWS - ready-made analytical queries
-- ============================================================================

-- Part reliability ranking
CREATE OR REPLACE VIEW v_part_reliability AS
SELECT 
    pc.part_number,
    pc.description,
    pc.ata_chapter,
    pc.part_class,
    pc.mtbf_flight_hours,
    pc.unit_cost_eur,
    pc.criticality,
    COALESCE(COUNT(pd.id), 0) AS total_demands,
    COALESCE(SUM(pd.quantity_required), 0) AS total_qty_demanded
FROM parts_catalog pc
LEFT JOIN part_demands pd ON pc.part_number = pd.part_number
GROUP BY pc.part_number, pc.description, pc.ata_chapter, 
         pc.part_class, pc.mtbf_flight_hours, pc.unit_cost_eur, pc.criticality
ORDER BY total_demands DESC;

-- Monthly demand per part per station
CREATE OR REPLACE VIEW v_demand_by_part_month AS
SELECT 
    part_number,
    TO_CHAR(demand_date, 'YYYY-MM') AS month,
    demand_type::TEXT,
    station,
    COUNT(*) AS demand_count,
    SUM(quantity_required) AS total_qty
FROM part_demands
GROUP BY part_number, TO_CHAR(demand_date, 'YYYY-MM'), demand_type, station;

-- Fleet status with failure counts
CREATE OR REPLACE VIEW v_fleet_status AS
SELECT 
    f.tail_number,
    f.aircraft_model,
    f.manufacture_year,
    EXTRACT(YEAR FROM NOW())::INT - f.manufacture_year AS age_years,
    f.home_base,
    f.primary_role,
    f.cycles_per_fh_ratio,
    f.status,
    COUNT(DISTINCT wo.work_order_id) AS total_work_orders,
    COUNT(DISTINCT wo.work_order_id) FILTER (WHERE wo.source = 'UNSCHEDULED_FAILURE') AS unscheduled_failures,
    COUNT(DISTINCT wo.work_order_id) FILTER (WHERE wo.source = 'SCHEDULED') AS scheduled_checks
FROM fleet f
LEFT JOIN work_orders wo ON f.tail_number = wo.tail_number
GROUP BY f.tail_number, f.aircraft_model, f.manufacture_year,
         f.home_base, f.primary_role, f.cycles_per_fh_ratio, f.status;

-- Stock alerts - parts below minimum
CREATE OR REPLACE VIEW v_stock_alerts AS
SELECT 
    i.part_number,
    pc.description,
    pc.criticality,
    i.station,
    i.quantity_serviceable,
    i.minimum_stock_level,
    i.minimum_stock_level - i.quantity_serviceable AS shortage,
    pc.lead_time_days_normal,
    pc.lead_time_days_aog
FROM inventory i
JOIN parts_catalog pc ON i.part_number = pc.part_number
WHERE i.quantity_serviceable < i.minimum_stock_level
ORDER BY pc.criticality, shortage DESC;

-- Aircraft next maintenance due
CREATE OR REPLACE VIEW v_next_maintenance AS
SELECT 
    f.tail_number,
    f.aircraft_model,
    f.total_flight_hours,
    ip.check_type,
    ip.interval_flight_hours,
    f.total_flight_hours - COALESCE(
        (SELECT MAX(wo.aircraft_fh_at_check) 
         FROM work_orders wo 
         WHERE wo.tail_number = f.tail_number 
           AND wo.check_type = ip.check_type), 0
    ) AS fh_since_last,
    ip.interval_flight_hours - (
        f.total_flight_hours - COALESCE(
            (SELECT MAX(wo.aircraft_fh_at_check) 
             FROM work_orders wo 
             WHERE wo.tail_number = f.tail_number 
               AND wo.check_type = ip.check_type), 0
        )
    ) AS fh_remaining
FROM fleet f
CROSS JOIN inspection_program ip
WHERE ip.interval_flight_hours IS NOT NULL
  AND f.status = 'ACTIVE'
ORDER BY fh_remaining ASC;

-- ============================================================================
-- FUNCTIONS
-- ============================================================================

-- Calculate equivalent flight hours (EFH) for a given aircraft
-- Adjusts raw FH by stress factors
CREATE OR REPLACE FUNCTION calc_equivalent_fh(
    p_tail VARCHAR,
    p_raw_fh NUMERIC
) RETURNS NUMERIC AS $$
DECLARE
    v_ratio NUMERIC;
    v_salt NUMERIC;
    v_age INTEGER;
    v_efh NUMERIC;
BEGIN
    SELECT f.cycles_per_fh_ratio, s.salt_exposure,
           EXTRACT(YEAR FROM NOW())::INT - f.manufacture_year
    INTO v_ratio, v_salt, v_age
    FROM fleet f
    JOIN stations s ON f.home_base = s.station_code
    WHERE f.tail_number = p_tail;
    
    -- EFH = raw_FH x cycle_stress x climate x age
    v_efh := p_raw_fh 
             * (1 + (v_ratio - 0.5) * 0.3)   -- cycle stress adjustment
             * (1 + v_salt * 0.15)             -- corrosion factor
             * (1 + v_age * 0.01);             -- age degradation
    
    RETURN ROUND(v_efh, 1);
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION calc_equivalent_fh IS 'Calculates stress-adjusted flight hours for failure prediction';

-- ============================================================================
-- INITIAL DATA: Stations
-- ============================================================================
INSERT INTO stations (station_code, name, city, climate, salt_exposure, is_main_base, has_hangar) VALUES
('ATH', 'Athens International', 'Athens', 'mediterranean', 0.30, TRUE, TRUE),
('SKG', 'Thessaloniki', 'Thessaloniki', 'continental', 0.20, TRUE, TRUE),
('HER', 'Heraklion', 'Heraklion', 'mediterranean', 0.80, FALSE, FALSE),
('RHO', 'Rhodes', 'Rhodes', 'mediterranean', 0.90, FALSE, FALSE),
('CFU', 'Corfu', 'Corfu', 'mediterranean', 0.70, FALSE, FALSE);

-- ============================================================================
-- INITIAL DATA: Inspection Program (MSG-3 / MPD based)
-- ============================================================================
INSERT INTO inspection_program (check_type, interval_flight_hours, interval_flight_cycles, interval_months, duration_days, description, source_document) VALUES
('DAILY',           NULL, NULL, NULL,  0.1, 'Daily/Preflight Check', 'AMP'),
('WEEKLY',          NULL, NULL, NULL,  0.2, 'Weekly Check', 'AMP'),
('A-CHECK',          750, NULL, NULL,  1.0, 'A-Check - systems, lubrication, minor inspections', 'MPD'),
('C-CHECK',         7500, NULL,   24, 18.0, 'C-Check - detailed structural/systems inspection', 'MPD'),
('D-CHECK',        30000, NULL,  120, 60.0, 'D-Check - heavy maintenance, complete strip-down', 'MPD'),
('ENGINE_SHOP',    20000, NULL, NULL, 45.0, 'Engine shop visit - overhaul', 'ENGINE_MAN'),
('LANDING_GEAR_OH', NULL, 18000, 120, 14.0, 'Landing gear overhaul', 'MPD');
