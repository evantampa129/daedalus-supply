-- ============================================================================
-- Daedalus Supply AI - MySQL Schema
-- EASA Part-M / Part-145 / ICAO Annex 6 & 8 Compliant Data Model
-- ============================================================================
-- Run: mysql -u your_user -p daedalus_supply < schema_mysql.sql
-- ============================================================================

CREATE DATABASE IF NOT EXISTS daedalus_supply CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE daedalus_supply;

-- ============================================================================
-- STATIONS
-- ============================================================================
CREATE TABLE stations (
    station_code    VARCHAR(5) PRIMARY KEY,
    name            VARCHAR(100) NOT NULL,
    city            VARCHAR(100),
    country         VARCHAR(3) DEFAULT 'GRC',
    climate         VARCHAR(30),
    salt_exposure   DECIMAL(3,2) DEFAULT 0.00,
    is_main_base    BOOLEAN DEFAULT FALSE,
    has_hangar      BOOLEAN DEFAULT FALSE,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;

-- ============================================================================
-- FLEET REGISTER
-- ============================================================================
CREATE TABLE fleet (
    tail_number         VARCHAR(10) PRIMARY KEY,
    aircraft_model      VARCHAR(30) NOT NULL,
    manufacturer        VARCHAR(30) DEFAULT 'AIRBUS',
    serial_number       VARCHAR(20),
    manufacture_year    INT NOT NULL,
    home_base           VARCHAR(5) NOT NULL,
    primary_role        VARCHAR(20) NOT NULL,
    total_flight_hours  DECIMAL(10,1) DEFAULT 0,
    total_flight_cycles INT DEFAULT 0,
    cycles_per_fh_ratio DECIMAL(5,3),
    daily_utilization_fh DECIMAL(4,1),
    status              ENUM('ACTIVE','STORED','GROUNDED','RETIRED') DEFAULT 'ACTIVE',
    last_c_check_date   DATE,
    last_c_check_fh     DECIMAL(10,1),
    next_c_check_due_fh DECIMAL(10,1),
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (home_base) REFERENCES stations(station_code),
    INDEX idx_fleet_base (home_base),
    INDEX idx_fleet_status (status)
) ENGINE=InnoDB;

-- ============================================================================
-- PARTS CATALOG
-- ============================================================================
CREATE TABLE parts_catalog (
    part_number         VARCHAR(30) PRIMARY KEY,
    description         VARCHAR(200) NOT NULL,
    ata_chapter         SMALLINT NOT NULL,
    ata_subchapter      SMALLINT DEFAULT 0,
    part_class          ENUM('ROTABLE','EXPENDABLE','CONSUMABLE') NOT NULL,
    unit_of_measure     VARCHAR(10) DEFAULT 'EA',
    mtbf_flight_hours   DECIMAL(10,1),
    mtbf_flight_cycles  DECIMAL(10,1),
    unit_cost_eur       DECIMAL(12,2),
    criticality         ENUM('AOG','MEL','ROUTINE') NOT NULL DEFAULT 'ROUTINE',
    lead_time_days_normal INT,
    lead_time_days_aog  INT,
    shelf_life_months   INT,
    is_serialized       BOOLEAN DEFAULT FALSE,
    requires_form1      BOOLEAN DEFAULT TRUE,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_parts_ata (ata_chapter, ata_subchapter),
    INDEX idx_parts_class (part_class),
    INDEX idx_parts_criticality (criticality)
) ENGINE=InnoDB;

-- ============================================================================
-- ALTERNATE PART NUMBERS
-- ============================================================================
CREATE TABLE part_alternates (
    id                  INT AUTO_INCREMENT PRIMARY KEY,
    part_number         VARCHAR(30) NOT NULL,
    alternate_pn        VARCHAR(30) NOT NULL,
    alternate_source    VARCHAR(20) DEFAULT 'OEM',
    notes               TEXT,
    UNIQUE KEY uk_alternate (part_number, alternate_pn),
    FOREIGN KEY (part_number) REFERENCES parts_catalog(part_number)
) ENGINE=InnoDB;

-- ============================================================================
-- INVENTORY
-- ============================================================================
CREATE TABLE inventory (
    id                  INT AUTO_INCREMENT PRIMARY KEY,
    part_number         VARCHAR(30) NOT NULL,
    station             VARCHAR(5) NOT NULL,
    quantity_serviceable INT DEFAULT 0,
    quantity_unserviceable INT DEFAULT 0,
    quantity_in_repair  INT DEFAULT 0,
    minimum_stock_level INT DEFAULT 0,
    reorder_point       INT DEFAULT 0,
    maximum_stock_level INT,
    last_receipt_date   DATE,
    last_issue_date     DATE,
    updated_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uk_inv (part_number, station),
    FOREIGN KEY (part_number) REFERENCES parts_catalog(part_number),
    FOREIGN KEY (station) REFERENCES stations(station_code),
    INDEX idx_inv_station (station),
    INDEX idx_inv_part (part_number)
) ENGINE=InnoDB;

-- ============================================================================
-- FLIGHT LOG
-- ============================================================================
CREATE TABLE flight_log (
    id                  BIGINT AUTO_INCREMENT PRIMARY KEY,
    tail_number         VARCHAR(10) NOT NULL,
    flight_date         DATE NOT NULL,
    flight_number       VARCHAR(10),
    origin              VARCHAR(5) NOT NULL,
    destination         VARCHAR(5) NOT NULL,
    flight_hours        DECIMAL(5,2) NOT NULL,
    flight_cycles       SMALLINT NOT NULL DEFAULT 1,
    cumulative_fh       DECIMAL(10,1),
    cumulative_fc       INT,
    month               SMALLINT,
    season              ENUM('summer','winter','shoulder'),
    FOREIGN KEY (tail_number) REFERENCES fleet(tail_number),
    INDEX idx_flight_tail (tail_number),
    INDEX idx_flight_date (flight_date),
    INDEX idx_flight_tail_date (tail_number, flight_date)
) ENGINE=InnoDB;

-- ============================================================================
-- INSPECTION PROGRAM
-- ============================================================================
CREATE TABLE inspection_program (
    check_type              VARCHAR(20) PRIMARY KEY,
    interval_flight_hours   DECIMAL(10,1),
    interval_flight_cycles  INT,
    interval_months         INT,
    duration_days           DECIMAL(5,1),
    description             TEXT,
    source_document         VARCHAR(50) DEFAULT 'MPD',
    msg3_category           VARCHAR(30)
) ENGINE=InnoDB;

-- ============================================================================
-- WORK ORDERS
-- ============================================================================
CREATE TABLE work_orders (
    work_order_id       VARCHAR(20) PRIMARY KEY,
    tail_number         VARCHAR(10) NOT NULL,
    check_type          VARCHAR(20) NOT NULL,
    scheduled_date      DATE NOT NULL,
    completion_date     DATE,
    status              ENUM('OPEN','IN_PROGRESS','COMPLETED','DEFERRED','CANCELLED') DEFAULT 'OPEN',
    aircraft_fh_at_check DECIMAL(10,1),
    aircraft_fc_at_check INT,
    source              ENUM('SCHEDULED','PILOT_REPORT','UNSCHEDULED_FAILURE','AD_COMPLIANCE','SB_COMPLIANCE') NOT NULL,
    station             VARCHAR(5),
    assigned_to         VARCHAR(50),
    release_to_service  BOOLEAN DEFAULT FALSE,
    crs_date            DATETIME,
    crs_signatory       VARCHAR(50),
    notes               TEXT,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (tail_number) REFERENCES fleet(tail_number),
    FOREIGN KEY (station) REFERENCES stations(station_code),
    INDEX idx_wo_tail (tail_number),
    INDEX idx_wo_date (scheduled_date),
    INDEX idx_wo_status (status),
    INDEX idx_wo_source (source)
) ENGINE=InnoDB;

-- ============================================================================
-- FINDINGS
-- ============================================================================
CREATE TABLE findings (
    finding_id          VARCHAR(20) PRIMARY KEY,
    work_order_id       VARCHAR(20) NOT NULL,
    tail_number         VARCHAR(10) NOT NULL,
    ata_chapter         SMALLINT NOT NULL,
    finding_date        DATE NOT NULL,
    finding_type        ENUM('CRACK','WEAR','CORROSION','LEAK','MALFUNCTION','DAMAGE','CONTAMINATION','OTHER') NOT NULL,
    description         TEXT NOT NULL,
    action_taken        VARCHAR(30),
    part_number_required VARCHAR(30),
    aircraft_fh         DECIMAL(10,1),
    aircraft_fc         INT,
    is_repetitive       BOOLEAN DEFAULT FALSE,
    deferred            BOOLEAN DEFAULT FALSE,
    mel_reference       VARCHAR(20),
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (work_order_id) REFERENCES work_orders(work_order_id),
    FOREIGN KEY (tail_number) REFERENCES fleet(tail_number),
    FOREIGN KEY (part_number_required) REFERENCES parts_catalog(part_number),
    INDEX idx_find_wo (work_order_id),
    INDEX idx_find_ata (ata_chapter),
    INDEX idx_find_type (finding_type),
    INDEX idx_find_date (finding_date)
) ENGINE=InnoDB;

-- ============================================================================
-- PART DEMANDS - ML TRAINING TABLE
-- ============================================================================
CREATE TABLE part_demands (
    id                  BIGINT AUTO_INCREMENT PRIMARY KEY,
    finding_id          VARCHAR(20),
    work_order_id       VARCHAR(20),
    tail_number         VARCHAR(10) NOT NULL,
    part_number         VARCHAR(30) NOT NULL,
    quantity_required   INT NOT NULL DEFAULT 1,
    demand_type         ENUM('SCHEDULED_FINDING','UNSCHEDULED','PROACTIVE_PREPOSITION','STOCK_REPLENISHMENT') NOT NULL,
    demand_date         DATE NOT NULL,
    station             VARCHAR(5) NOT NULL,
    criticality         ENUM('AOG','MEL','ROUTINE') NOT NULL,
    fulfilled           BOOLEAN DEFAULT FALSE,
    fulfilled_from      VARCHAR(5),
    fulfillment_hours   DECIMAL(6,1),
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (work_order_id) REFERENCES work_orders(work_order_id),
    FOREIGN KEY (tail_number) REFERENCES fleet(tail_number),
    FOREIGN KEY (part_number) REFERENCES parts_catalog(part_number),
    FOREIGN KEY (station) REFERENCES stations(station_code),
    INDEX idx_demand_part (part_number),
    INDEX idx_demand_date (demand_date),
    INDEX idx_demand_station (station),
    INDEX idx_demand_type (demand_type)
) ENGINE=InnoDB;

-- ============================================================================
-- FAA SDR RAW
-- ============================================================================
CREATE TABLE faa_sdr_raw (
    id                  BIGINT AUTO_INCREMENT PRIMARY KEY,
    report_date         DATE,
    acft_make           VARCHAR(50),
    acft_model          VARCHAR(50),
    acft_serial         VARCHAR(30),
    total_time          DECIMAL(10,1),
    ata_code            VARCHAR(10),
    part_name           VARCHAR(200),
    part_number         VARCHAR(50),
    part_condition      VARCHAR(50),
    nature_condition    VARCHAR(100),
    stage_of_operation  VARCHAR(50),
    precautionary_procedure VARCHAR(100),
    remarks             TEXT,
    loaded_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_sdr_model (acft_model),
    INDEX idx_sdr_ata (ata_code),
    INDEX idx_sdr_part (part_name)
) ENGINE=InnoDB;

-- ============================================================================
-- VIEWS
-- ============================================================================

CREATE OR REPLACE VIEW v_part_reliability AS
SELECT 
    pc.part_number,
    pc.description,
    pc.ata_chapter,
    pc.part_class,
    pc.mtbf_flight_hours,
    pc.unit_cost_eur,
    pc.criticality,
    COUNT(pd.id) AS total_demands,
    COALESCE(SUM(pd.quantity_required), 0) AS total_qty_demanded
FROM parts_catalog pc
LEFT JOIN part_demands pd ON pc.part_number = pd.part_number
GROUP BY pc.part_number;

CREATE OR REPLACE VIEW v_stock_alerts AS
SELECT 
    i.part_number,
    pc.description,
    pc.criticality,
    i.station,
    i.quantity_serviceable,
    i.minimum_stock_level,
    (i.minimum_stock_level - i.quantity_serviceable) AS shortage
FROM inventory i
JOIN parts_catalog pc ON i.part_number = pc.part_number
WHERE i.quantity_serviceable < i.minimum_stock_level
ORDER BY pc.criticality, shortage DESC;

-- ============================================================================
-- INITIAL DATA
-- ============================================================================
INSERT INTO stations (station_code, name, city, climate, salt_exposure, is_main_base, has_hangar) VALUES
('ATH', 'Athens International', 'Athens', 'mediterranean', 0.30, TRUE, TRUE),
('SKG', 'Thessaloniki', 'Thessaloniki', 'continental', 0.20, TRUE, TRUE),
('HER', 'Heraklion', 'Heraklion', 'mediterranean', 0.80, FALSE, FALSE),
('RHO', 'Rhodes', 'Rhodes', 'mediterranean', 0.90, FALSE, FALSE),
('CFU', 'Corfu', 'Corfu', 'mediterranean', 0.70, FALSE, FALSE);

INSERT INTO inspection_program (check_type, interval_flight_hours, interval_flight_cycles, interval_months, duration_days, description) VALUES
('DAILY',           NULL, NULL, NULL,  0.1, 'Daily/Preflight Check'),
('WEEKLY',          NULL, NULL, NULL,  0.2, 'Weekly Check'),
('A-CHECK',          750, NULL, NULL,  1.0, 'A-Check'),
('C-CHECK',         7500, NULL,   24, 18.0, 'C-Check'),
('D-CHECK',        30000, NULL,  120, 60.0, 'D-Check'),
('ENGINE_SHOP',    20000, NULL, NULL, 45.0, 'Engine shop visit'),
('LANDING_GEAR_OH', NULL, 18000, 120, 14.0, 'Landing gear overhaul');
