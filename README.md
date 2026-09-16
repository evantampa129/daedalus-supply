# Daedalus Supply AI

**Predictive Maintenance & Intelligent Supply Chain for Aircraft**

Daedalus Supply AI predicts when aircraft components will fail and positions spare parts
across a station network before they are needed. Where a conventional inventory system records
what was consumed, Daedalus forecasts what will be consumed, where, and when - turning
maintenance history into a stocking and distribution plan.

Built on EASA/ICAO regulatory structure, trained on real FAA Service Difficulty Report data.

Author: Evangelos Tampachaniotis
Version: 1.1.0
License: MIT

---

## Architecture

```
                          raw_data/
                    FAA SDR CSV (413K rows)
                             |
                             v
   +---------------------------------------------------------------+
   |  MODULE 1 - data_pipeline.py                                   |
   |                                                                 |
   |   SDR parser  ---->  type filter  ---->  195,801 real records   |
   |   Fleet generator                                               |
   |   Flight simulator (3 years, seasonal)                          |
   |   Failure model:  Weibull (rotable)                             |
   |                   Poisson (expendable)                          |
   |                   Deterministic (consumable)                    |
   +---------------------------------------------------------------+
                             |
                             v
                    +------------------+
                    |  aerosupply.db   |   SQLite, 12 tables, ~106 MB
                    |  59,205 records  |   (or PostgreSQL / MySQL)
                    +------------------+
                       |      |      |
        +--------------+      |      +--------------+
        |                     |                     |
        v                     v                     v
+------------------+  +------------------+  +------------------+
| MODULE 2         |  | MODULE 3         |  | MODULE 4         |
| prediction_model |  | logistics_       |  | agent.py         |
|                  |  | optimizer.py     |  |                  |
| Cox PH +         |  |                  |  | NL query router  |
| Weibull AFT      |  | Stock levels     |  | 10 report types  |
| (rotables)       |  | (s,S policy)     |  | keyword-routed   |
|                  |  |                  |  | SQL-injection    |
| XGBoost regress. |  | Pre-positioning  |  | hardened         |
| (expendables)    |  | (30 transfers)   |  |                  |
|                  |  |                  |  | demo / -i / -q   |
| XGBoost classif. |  | AOG router       |  |                  |
| (real SDR, 87.6%)|  | (EUR 15K/hour)   |  |                  |
+------------------+  +------------------+  +------------------+
        |                     |                     |
        v                     v                     v
   3 PNG plots      stock_recommendations      terminal reports
        |           transfer_recommendations
        |                    |
        |                    +--> read back by Module 4
        |                    |
        +--------+-----------+
                 |
                 v
   +---------------------------------------------------------------+
   |  MODULE 5 - dashboard.py                    streamlit run      |
   |                                                                 |
   |   Fleet Overview     station map, risk bands, register          |
   |   Parts & Inventory  search, coverage bars, AOG alerts          |
   |   Predictions        Module 2 plots, risk table, demand         |
   |   Logistics          Module 3 tables, AOG response simulator    |
   |   SDR Analysis       195,801 FAA records, filtered and ranked   |
   |                                                                 |
   |   Reads aerosupply.db directly - SELECT only, connection ro     |
   +---------------------------------------------------------------+

   config.py - hardware auto-detection, imported by modules 1-3 and 5
               MINIMAL / STANDARD / FULL
```

---

## Features

### Module 1 - Data Pipeline (`data_pipeline.py`)

Builds the entire database from two sources.

- Parses 7 years of FAA Service Difficulty Reports (413K raw rows), filters to
  transport-category types, normalizes the schema, yields 195,801 usable records
- Generates a 15-aircraft A320-family fleet across 4 Greek bases, ages 4-18 years
- Simulates 54,532 flight sectors over 3 years with seasonal traffic variation
- Builds a 53-part catalog structured on ATA/JASC chapters, classified per Part-145
  into ROTABLE (29) / EXPENDABLE (18) / CONSUMABLE (6)
- Multi-factor failure modelling: Weibull renewal (shape 2.5, wear-out) for rotables,
  Poisson for expendables, deterministic issue for consumables - all modulated by
  sector length, airframe age and salt exposure
- Generates the MSG-3 / MPD inspection program and derives scheduled checks by
  interval tracking against accumulated flight hours
- Chunked CSV reading so the full corpus ingests on a 2 GB machine

### Module 2 - Prediction Engine (`prediction_model.py`)

Three models, one per failure mechanism.

| Model | Method | Target | Result |
|---|---|---|---|
| Rotable failure | Kaplan-Meier, Cox PH, Weibull AFT | time to removal | salt_exposure HR = 1.465 (p = 0.037), C-index 0.563 |
| Expendable demand | XGBoost regression | monthly qty per part per station | MAE 0.85 parts/month |
| SDR patterns | XGBoost classification | reported failure condition | 87.6% accuracy on 193,768 real records |

The SDR classifier's key finding: ATA chapter alone carries 66.7% of feature importance.
The *system* a component belongs to determines how it fails far more than airframe age or
utilisation - which is why the whole inventory strategy is organised by ATA chapter.

Outputs `survival_analysis.png`, `demand_forecast.png`, `sdr_analysis.png`.

### Module 3 - Logistics Optimizer (`logistics_optimizer.py`)

| Engine | Method |
|---|---|
| Stock levels | Continuous-review (s,S) policy. Safety stock = z x sigma x sqrt(lead_time), with z set by dispatch criticality: AOG 99.5%, MEL 95%, ROUTINE 90% |
| Pre-positioning | Pairs surplus stations with deficit stations by transit time, prioritised by criticality. 30 transfers recommended on the reference dataset |
| AOG router | Ranks local stock / station transfer / expedited order by total cost, where total = logistics cost + (ETA x EUR 15,000/hour grounding cost) |

Writes `stock_recommendations` and `transfer_recommendations` back to the database.

### Module 4 - Query Agent (`agent.py`)

Natural-language interface over the database. Keyword-routed to a fixed catalog of vetted
reports - it does not synthesize arbitrary SQL, so the query surface stays bounded and
auditable.

Supported queries: fleet status, aircraft detail, part availability, stock alerts, failure
risk, station detail, SDR summary, maintenance history, findings summary, recommendations.

```
python agent.py            # demo, 8 representative queries
python agent.py -i         # interactive prompt
python agent.py -q "..."   # single query, scriptable
```

### Module 5 - Web Dashboard (`dashboard.py`)

Browser front end over the same database. A presentation layer only: it issues SELECT
statements on a read-only connection and recomputes nothing that an upstream module has
already written.

| View | Contents |
|---|---|
| Fleet Overview | The five Greek stations on a map, marker size by based aircraft and colour by salt exposure. Risk score per airframe, banded LOW / MEDIUM / HIGH, and the full register with sortable columns |
| Parts & Inventory | AOG-critical shortages as alerts above the fold, search across part number and description, per-station coverage bars against the minimum stock level |
| Predictions | The three Module 2 plots with their interpretation, per-aircraft risk assessment, and next-month expendable demand against stock on hand |
| Logistics | Module 3 stock recommendations against stock currently held, recommended transfers with origin and destination, and an AOG response simulator ranking every source by ETA and total cost |
| SDR Analysis | The FAA corpus filtered by ATA chapter and airframe manufacturer, most-reported parts and the failure distribution |

```
streamlit run dashboard.py       # http://localhost:8501
```

Notes on what the dashboard does and does not do:

- The risk score is the screening heuristic `agent.py` reports, reproduced so the two
  surfaces cannot disagree. The calibrated Cox model stays in Module 2; the dashboard
  displays its plots rather than refitting it.
- The expendable forecast is the trailing six-month mean, counting months with no
  consumption as zero. The XGBoost regressor is not persisted and does not yet beat that
  baseline on held-out data, so the screen reports the number that is defensible.
- The AOG simulator calls `logistics_optimizer.route_aog_request` directly. The routing
  economics live in Module 3 and are not duplicated here.
- The hardware profile is honoured: on MINIMAL the SDR views aggregate over the capped
  sample `config.py` allows, and the page says so rather than sampling silently.

Palette follows the browser theme, with a Light / Dark override in the sidebar.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.9+ |
| Data | pandas 2.x, numpy 1.24+ |
| Survival analysis | lifelines 0.28+ (Kaplan-Meier, Cox PH, Weibull AFT) |
| Machine learning | XGBoost 2.x, scikit-learn 1.3+ |
| Statistics | scipy 1.11+ |
| Plotting | matplotlib 3.7+, seaborn 0.12+ (Agg backend, headless-safe) |
| Dashboard | Streamlit 1.62+, Plotly 6+ (MapLibre station map) |
| Storage | SQLite (default), PostgreSQL 14+, MySQL 8+ |
| ORM / migration | SQLAlchemy 2.x, psycopg2-binary, PyMySQL |

---

## Hardware Profiles

`config.py` detects available RAM and cores at startup and selects a profile. Modules 1-3
read their tuning parameters from it, so the same codebase runs on an old office terminal
and on a dedicated server without modification.

| Profile | Trigger | XGBoost | Chunk size | SDR cap | Jobs | Plot DPI |
|---|---|---|---|---|---|---|
| MINIMAL | < 4 GB RAM or <= 2 cores | 50 trees, depth 4 | 5,000 | 50,000 | 1 | 72 |
| STANDARD | < 12 GB RAM or <= 4 cores | 200 trees, depth 6 | 50,000 | none | 2 | 150 |
| FULL | 16 GB+ RAM, 8+ cores | 500 trees, depth 8 | 100,000 | none | -1 | 200 |

Override auto-detection with the `DAEDALUS_PROFILE` environment variable:

```bash
python config.py                                   # show what was detected
DAEDALUS_PROFILE=MINIMAL python prediction_model.py
```

---

## Installation

```bash
git clone git@github.com:<username>/daedalus_supply_ai.git
cd daedalus_supply_ai

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt

python setup_and_test.py          # verify environment and build the database
```

`setup_and_test.py` checks the Python version, dependencies and optional database servers,
then runs the pipeline and validates the resulting row counts.

### Adding real failure data

The pipeline runs without it, but the SDR classifier needs it.

1. Download CSVs from https://www.faa.gov/av-info/download_SDR
2. Place them in `raw_data/`
3. Re-run `python data_pipeline.py`

The parser filters by aircraft type, normalizes the varying yearly schemas and stores the
result in `faa_sdr_raw`.

### Optional: PostgreSQL or MySQL

```bash
# PostgreSQL
sudo -u postgres createdb daedalus_supply
psql -d daedalus_supply -f schema_postgres.sql
python load_to_postgres.py

# MySQL
sudo mysql -e "CREATE DATABASE daedalus_supply;"
mysql -u root -p daedalus_supply < schema_mysql.sql
python load_to_postgres.py --db-url mysql+pymysql://$MYSQL_USER:$MYSQL_PASSWORD@localhost/daedalus_supply
```

---

## Usage

Run the modules in order; each consumes the previous one's output.

```bash
python config.py                # 1. show the detected hardware profile
python data_pipeline.py         # 2. build the database
python explore_data.py          # 3. nine sanity-check analyses
python prediction_model.py      # 4. train the three models, write plots
python logistics_optimizer.py   # 5. stock levels, transfers, AOG scenarios
python agent.py                 # 6. query the results
streamlit run dashboard.py      # 7. the same results in a browser
```

The dashboard reads whatever is in the database at the time. Steps 4 and 5 are optional for
it: without them the prediction plots and the optimiser tables are simply reported as not yet
generated, with the command that produces them.

### Agent examples

```
Daedalus > fleet status
Daedalus > aircraft SX-ABK
Daedalus > find part AES-24-10-001
Daedalus > stock alerts
Daedalus > failure risk
Daedalus > station HER
Daedalus > top demanded parts
Daedalus > SDR summary
Daedalus > maintenance history SX-ABK
Daedalus > recommendations
Daedalus > quit
```

### Direct SQL

```python
import sqlite3
import pandas as pd

conn = sqlite3.connect("aerosupply.db")

# Which aircraft generate the most unscheduled work?
pd.read_sql("""
    SELECT tail_number, home_base, cycles_per_fh_ratio, unscheduled_failures
    FROM v_fleet_status
    ORDER BY unscheduled_failures DESC
""", conn)

# Monthly demand for one part
pd.read_sql("""
    SELECT month, station, SUM(total_qty) AS qty
    FROM v_demand_by_part_month
    WHERE part_number = 'AES-33-40-010'
    GROUP BY month, station
    ORDER BY month
""", conn)
```

---

## Database Schema

| Table | Rows | Contents |
|---|---|---|
| `fleet` | 15 | Aircraft register - registration, model, age, base, role, FH/FC counters (Part-M M.A.305) |
| `flight_log` | 54,532 | Sector-level log over 3 years with cumulative FH/FC and season label |
| `parts_catalog` | 53 | IPC-structured catalog - ATA chapter, class, MTBF, cost, criticality, lead times |
| `inventory` | 265 | Stock per part per station - serviceable, unserviceable, minimum, reorder point |
| `work_orders` | 1,977 | Scheduled checks and unscheduled failures, with airframe FH at the event |
| `findings` | 233 | Inspection findings - CRACK, WEAR, CORROSION, LEAK, MALFUNCTION, by ATA chapter |
| `part_demands` | 2,130 | Material consumption record. **This is the ML training dataset** |
| `inspection_program` | 7 | Approved maintenance program - DAILY through D-CHECK plus shop visits |
| `stations` | 5 | Base network with climate and salt_exposure covariate |
| `faa_sdr_raw` | 195,801 | Real FAA Service Difficulty Reports |
| `stock_recommendations` | generated | Module 3 output - optimal min / ROP / max per part per station |
| `transfer_recommendations` | generated | Module 3 output - recommended inter-station movements |

**Views:** `v_fleet_status`, `v_part_reliability`, `v_demand_by_part_month`, plus
`v_stock_alerts` and `v_next_maintenance` in the PostgreSQL/MySQL schemas.

---

## Regulatory Framework

| Regulation | Scope | Where it appears in the project |
|---|---|---|
| EASA Part-M (EU 1321/2014) | Continuing airworthiness management | M.A.305 record system = `fleet` + `work_orders`; M.A.302 approved maintenance program = `inspection_program`; the reliability program M.A.302 requires is what Module 2 produces |
| EASA Part-145 | Maintenance organisation approval | 145.A.42 component classification drives ROTABLE/EXPENDABLE/CONSUMABLE; stores control is why serviceable and unserviceable quantities are tracked separately; CRS and EASA Form 1 traceability |
| ICAO Annex 6 | Operation of aircraft | Operator responsibility for maintaining airworthiness; basis for the maintenance obligations modelled |
| ICAO Annex 8 | Airworthiness of aircraft | Continued airworthiness to type design standard; why failure prediction has regulatory value, not just economic value |
| MSG-3 | Maintenance program development methodology | Check hierarchy and task-to-ATA-chapter mapping; CPCP corrosion tasks at C-check; task escalation with check depth |
| ATA/JASC 100 | Aircraft system numbering | Every part, finding and SDR record is filed by chapter - the taxonomy that makes synthetic and real data comparable |
| MPD | Manufacturer maintenance planning document | A-check 750 FH, C-check 7,500 FH / 24 months, D-check 30,000 FH intervals |
| MEL | Minimum equipment list | AOG / MEL / ROUTINE criticality, which sets the service-level targets in Module 3 |

---

## Data Sources

| Source | Type | Reference |
|---|---|---|
| FAA Service Difficulty Reports | Real mandatory occurrence reports, 2020-2026, 413K rows | https://www.faa.gov/av-info/download_SDR |
| ATA/JASC 100 code list | System numbering standard | `raw_data/JASC_Code.pdf` |
| A320 MPD excerpts | Published maintenance intervals | Public manufacturer documentation |
| Reliability engineering literature | MTBF ranges, Weibull shape parameters | Representative published ranges, not manufacturer data |
| Fleet, flights, inventory, maintenance history | Synthetic, seeded and reproducible | Generated by `data_pipeline.py` (seed 42) |

Operational data is synthetic because no airline publishes its maintenance records. It is
generated from documented reliability models rather than invented, and the seed is fixed so
any figure quoted from the dataset can be regenerated and audited.

---

## Security

### Current implementation (v1.1)

- SQL injection prevention via allow-list input sanitization (`_safe()`) and parameterized
  queries throughout the agent module
- Database credentials stored in `.env` file, excluded from version control via `.gitignore`
- `.env` file permissions set to owner-read-only (`chmod 600`)
- Read-only query agent: the interactive interface issues SELECT statements only, no
  INSERT/UPDATE/DELETE
- Input validation: all user-supplied identifiers (tail numbers, part numbers, station codes)
  pass through a strict alphanumeric filter before reaching any query
- Database backups stored in `backups/`, excluded from version control
- Read-only dashboard: the Streamlit layer opens SQLite with `mode=ro`, so a write is
  refused by the driver rather than by code review. Every filter and search term reaches
  the database as a bound parameter, never as concatenated SQL
- No credentials in the dashboard: the database location comes from `DAEDALUS_DB_PATH` or
  defaults to the file beside the module

### Encryption strategy (deployment guide)

- **At rest** - SQLite databases should be encrypted with SQLCipher in sensitive environments;
  MySQL/PostgreSQL deployments should enable TDE or filesystem-level encryption (LUKS)
- **In transit** - all client-server database connections must use TLS; any future web
  interface must enforce HTTPS with a valid certificate
- **Application layer** - credentials managed through environment variables (`.env`) with
  restricted file permissions (`chmod 600`); production deployments should migrate to a
  secrets manager (HashiCorp Vault, AWS Secrets Manager)

### Deployment recommendations

- Use PostgreSQL or MySQL with authentication and TLS for production deployments instead of
  SQLite
- Run regular database backups: `cp aerosupply.db backups/aerosupply_$(date +%Y%m%d).db`
- When deploying a web dashboard or API layer, enforce HTTPS, rate limiting and CORS policies
- Restrict network access to the database server to authorized hosts only

### Data classification

- **FAA SDR data** - publicly available federal records, no access restrictions
- **Synthetic fleet data (SX-xxx registrations)** - fictitious, no confidentiality concerns
- **Real airline maintenance records** - if integrated in future versions, tail numbers and
  operator identifiers must be anonymized before storage

### Planned security features

- Role-based access control (see [Access Control Architecture](#access-control-architecture-planned-for-v12) below)
- Audit logging for all database queries, per EASA Part-145 145.A.55 record-keeping
  requirements
- Model versioning: serialized ML models with hash verification to detect training data changes
- Database encryption at rest (SQLCipher for SQLite, TDE for MySQL/PostgreSQL)
- Data anonymization pipeline for real operator maintenance records

---

## Roadmap

### Access Control Architecture (planned for v1.2+)

The system is designed around two access tiers.

**Administrator (Supply Officer / Engineering Manager)**

- Full read/write access to all data
- Add/remove aircraft from the fleet register
- Update inventory levels and stock parameters
- Import new FAA SDR datasets
- Run prediction models and approve optimization recommendations
- Manage user accounts and access permissions
- Approve and execute part transfers between stations

**Operator (Technician / Line Maintenance)**

- Read-only access to fleet status, part availability and maintenance schedules
- Submit part requests through the system (creates a demand record, does not modify inventory
  directly)
- View predictions and risk assessments
- Receive stock alerts and transfer notifications for their assigned station
- Cannot modify inventory, the fleet register or system configuration

**Implementation plan**

- **v1.2** - FastAPI REST layer with endpoint separation (`/api/admin/*` and `/api/user/*`)
- **v1.3** - JWT-based authentication with bcrypt password hashing
- **v1.4** - Audit logging: every write operation recorded with user ID, timestamp and
  before/after values, per EASA Part-145 145.A.55

**Client-server architecture**

```
            [Web Browser]
                 |
              HTTPS/TLS
                 |
           [FastAPI Server]
            |           |
      /api/admin    /api/user
            |           |
       [Auth Layer - JWT]
                 |
          [MySQL/PostgreSQL]
```

### v1.2 - FastAPI microservices and RBAC endpoint separation

REST API per module with an OpenAPI schema, split into `/api/admin/*` (write operations) and
`/api/user/*` (read operations) as the structural basis for role-based access control.

### v1.3 - JWT authentication with bcrypt password hashing

Token-based authentication enforcing the administrator and operator tiers at the auth layer.

### v1.4 - Docker Compose deployment and audit logging

One-command setup with database, API and dashboard. Every write operation recorded with user
ID, timestamp and before/after values, satisfying EASA Part-145 145.A.55.

### v1.5 - Prophet/LSTM time-series demand forecasting

Dedicated time-series models to replace the gradient-boosted regressor, targeting the negative
R-squared noted below by modelling each part-station series directly rather than as a panel.

### v2.0 - Digital twin simulator with reinforcement learning

Full fleet simulation with what-if scenarios and Monte Carlo runs, with reinforcement learning
optimization of stocking and pre-positioning policy.

### Known limitations in v1.1

- Demand forecast R-squared is negative. With three years of synthetic history the per-series
  signal is weak and the model does not beat the test-set mean. MAE (0.85 parts/month) is the
  metric to read for stocking decisions, since those depend on absolute error in units.
- Concordance index is 0.563. The survival dataset contains no censored observations - only
  observed removals - and the first interval of each series is left-truncated. Adding censored
  records is the highest-value improvement available and needs real maintenance history.
- Transfer costs in the AOG router are a flat rate per transit hour. At EUR 15,000/hour of
  grounding this never changes the ranking, so a detailed freight model was not warranted.
- The dashboard has no authentication and talks to SQLite directly. It is safe to run on a
  workstation or behind a trusted network, not to expose. The REST layer in v1.2 and the
  role enforcement in v1.3 are what make a deployment defensible.

---

## Project Structure

```
daedalus_supply_ai/
├── config.py                 hardware detection, ML parameter profiles
├── data_pipeline.py          Module 1 - build the database
├── prediction_model.py       Module 2 - three prediction engines
├── logistics_optimizer.py    Module 3 - three optimization engines
├── agent.py                  Module 4 - query interface
├── dashboard.py              Module 5 - Streamlit web dashboard
├── explore_data.py           nine standing analyses
├── setup_and_test.py         environment verification
├── load_to_postgres.py       SQLite -> PostgreSQL / MySQL migration
├── schema_postgres.sql       PostgreSQL schema (aero namespace, 5 views)
├── schema_mysql.sql          MySQL schema (MySQL Workbench)
├── requirements.txt
├── .env.example              credential template
├── raw_data/                 FAA SDR CSVs and the JASC code list (gitignored)
├── LICENSE                   MIT
├── VERSION
└── README.md
```

---

## Author

**Evangelos Tampachaniotis**

Built as a study of how EASA/ICAO continuing-airworthiness structure maps onto a predictive
supply chain.

---

## License

MIT. See [LICENSE](LICENSE).

Copyright (c) 2026 Evangelos Tampachaniotis
