# Daedalus Supply AI

Predicting when aircraft parts fail, and putting the spares where they will be
needed before anyone asks for them.

A normal inventory system tells you what you used last month. This one tries to
tell you what you will use next month, at which station, and whether you should
move something there now. It is built on top of a synthetic A320 fleet, but the
failure statistics come from real FAA Service Difficulty Reports, and the
maintenance structure follows EASA Part-M / Part-145 and MSG-3 rather than
something I made up.

I started this to understand how continuing-airworthiness regulation actually
constrains a supply chain. It turned into five modules and about 8,000 lines of
Python.

## What is in here

Five scripts, run in order. Each one writes to the same SQLite database and the
next one reads it.

**`data_pipeline.py`** builds everything. It parses the FAA SDR CSVs (413K raw
rows), filters to transport-category types and keeps around 195,800 usable
records. Then it generates a fleet: 15 A320-family aircraft aged 4-18 years,
based at four of five Greek stations, flying three years of sectors with
seasonal traffic. Parts are classified per Part-145 into rotable, expendable and
consumable, and each class gets its own failure model - Weibull renewal for
rotables, Poisson for expendables, fixed issue rate for consumables. All of it
modulated by sector length, airframe age and how much salt the station gets.

**`prediction_model.py`** trains three models, one per failure mechanism:

- Rotables: Kaplan-Meier, Cox PH and Weibull AFT from `lifelines`. The useful
  result is that `salt_exposure` comes out significant (HR 1.465, p = 0.037) -
  island stations really do eat components faster. C-index is 0.563, which is
  not good, see the limitations below.
- Expendables: XGBoost regression on monthly quantity per part per station.
  MAE 0.85 parts/month.
- Real SDR data: XGBoost classification of the reported failure condition,
  87.6% accurate over 193,768 records.

The SDR model is the one that changed how I built the rest. ATA chapter alone
carries 66.7% of the feature importance - the system a part belongs to predicts
how it fails far better than how old the aircraft is or how hard it is flown.
That is why the whole inventory strategy is organised by ATA chapter.

**`logistics_optimizer.py`** does the supply side. Continuous-review (s,S) stock
levels with safety stock `z * sigma * sqrt(lead_time)`, where z depends on
dispatch criticality (AOG 99.5%, MEL 95%, routine 90%). Then it pairs surplus
stations with deficit stations by transit time and suggests transfers - 30 of
them on the reference dataset. The AOG router ranks local stock vs. station
transfer vs. expedited order on total cost, counting grounding at EUR 15,000 an
hour, which dominates everything else.

**`agent.py`** is a query interface in plain English. It is keyword-routed to a
fixed set of about ten report types, not an LLM writing SQL - the query surface
stays small enough to audit, and everything user-supplied goes through an
alphanumeric filter before it touches the database.

**`dashboard.py`** is a Streamlit front end over the same data: fleet map, parts
and inventory with AOG alerts, the prediction plots, the optimiser output with an
AOG response simulator, and the SDR corpus filtered by ATA chapter. It opens
SQLite read-only and recomputes nothing - if a number differs from what `agent.py`
says, that is a bug, not a different method.

`config.py` checks RAM and core count at startup and picks MINIMAL, STANDARD or
FULL, which sets XGBoost depth, CSV chunk size and how much SDR data gets loaded.
This exists because I wanted the thing to run on my laptop and on a real machine
without editing anything. Override with `DAEDALUS_PROFILE`.

## Running it

```bash
git clone git@github.com:evantampa129/daedalus-supply.git
cd daedalus-supply

python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python setup_and_test.py      # checks the environment, builds the database
```

Then, in order:

```bash
python config.py                # what hardware profile you got
python data_pipeline.py         # build the database
python explore_data.py          # nine sanity-check analyses
python prediction_model.py      # train the models, write the plots
python logistics_optimizer.py   # stock levels, transfers, AOG scenarios
python agent.py                 # ask it things
streamlit run dashboard.py      # same results in a browser
```

The dashboard works on whatever is in the database at the time. Skip steps 4 and
5 and it just tells you the plots have not been generated yet, and which command
makes them.

### Real SDR data

The pipeline runs without it, but the SDR classifier needs it. Download the CSVs
from https://www.faa.gov/av-info/download_SDR, drop them in `raw_data/`, re-run
`data_pipeline.py`. The parser handles the schema changing between years.

### PostgreSQL or MySQL instead of SQLite

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

### Asking the agent things

```
Daedalus > fleet status
Daedalus > aircraft SX-ABK
Daedalus > find part AES-24-10-001
Daedalus > stock alerts
Daedalus > failure risk
Daedalus > station HER
Daedalus > SDR summary
Daedalus > recommendations
```

`python agent.py -i` for the prompt, `-q "fleet status"` for one query, plain
`python agent.py` for a demo run.

Or just query it yourself:

```python
import sqlite3, pandas as pd
conn = sqlite3.connect("aerosupply.db")

pd.read_sql("""
    SELECT tail_number, home_base, cycles_per_fh_ratio, unscheduled_failures
    FROM v_fleet_status ORDER BY unscheduled_failures DESC
""", conn)
```

## The database

Twelve tables, roughly 106 MB once built.

| Table | Rows | What it is |
|---|---|---|
| `fleet` | 15 | Aircraft register - registration, model, age, base, FH/FC counters (M.A.305) |
| `flight_log` | 54,532 | Every sector over three years, with cumulative hours and cycles |
| `parts_catalog` | 53 | ATA-structured catalog: chapter, class, MTBF, cost, criticality, lead time |
| `inventory` | 265 | Stock per part per station, serviceable and unserviceable tracked apart |
| `work_orders` | 1,977 | Scheduled checks and unscheduled failures |
| `findings` | 233 | Inspection findings by type and ATA chapter |
| `part_demands` | 2,130 | Material consumption. This is the ML training set |
| `inspection_program` | 7 | DAILY through D-CHECK plus shop visits |
| `stations` | 5 | ATH, SKG, HER, RHO, CFU, with climate and salt exposure |
| `faa_sdr_raw` | 195,801 | The real FAA reports |
| `stock_recommendations` | generated | Module 3 output |
| `transfer_recommendations` | generated | Module 3 output |

Views: `v_fleet_status`, `v_part_reliability`, `v_demand_by_part_month`, plus
`v_stock_alerts` and `v_next_maintenance` in the Postgres and MySQL schemas.

## Where the regulation shows up

This is not decoration - the structure of the database comes out of these.

- **Part-M M.A.305** is why `fleet` and `work_orders` look the way they do.
  **M.A.302** is the approved maintenance program, which is `inspection_program`,
  and the reliability program it requires is what Module 2 produces.
- **Part-145 145.A.42** is the rotable/expendable/consumable split. Stores control
  is why serviceable and unserviceable quantities are counted separately.
- **MSG-3** gives the check hierarchy and the task-to-ATA mapping, including the
  CPCP corrosion tasks that land at C-check - which is what makes salt exposure
  worth modelling at all.
- **ATA/JASC 100** is the taxonomy. It is the only reason synthetic parts and real
  SDR records can be compared.
- **MPD intervals**: A-check 750 FH, C-check 7,500 FH or 24 months, D-check
  30,000 FH.
- **MEL** criticality (AOG / MEL / routine) sets the service levels in Module 3.
- **ICAO Annex 6 and 8** are the reason any of this has regulatory weight and not
  only economic weight.

## Data

Real: the FAA SDRs (2020-2026, public mandatory occurrence reports) and the
ATA/JASC 100 code list. MTBF ranges and Weibull shapes come from published
reliability literature, not from a manufacturer.

Synthetic: the fleet, the flights, the inventory and the maintenance history. No
airline publishes its maintenance records, so there was no alternative. It is
generated from documented models rather than invented, and the seed is fixed at
42, so any number quoted here can be regenerated and checked.

## Known problems

I would rather write these down than have someone find them.

- **The demand forecast has a negative R-squared.** Three years of synthetic
  history is not enough per-series signal and the model does not beat the test
  mean. Read the MAE (0.85 parts/month) instead, since stocking decisions depend
  on absolute error in units. Replacing this with a proper time-series model is
  the obvious next job.
- **C-index 0.563 is barely better than a coin flip.** The survival dataset has
  no censored observations, only observed removals, and the first interval of
  each series is left-truncated. Fixing this needs real maintenance history, and
  it is the single highest-value improvement available.
- **Transfer costs are a flat rate per transit hour.** At EUR 15,000/hour of
  grounding this never changes the ranking, so a detailed freight model was not
  worth building.
- **The dashboard has no authentication** and talks to SQLite directly. Fine on a
  workstation or a trusted network, not something to expose.

## Security

What is actually implemented: the agent is read-only and parameterises every
query, all user-supplied identifiers pass an allow-list filter (`_safe()`) first,
the dashboard opens SQLite with `mode=ro` so a write is refused by the driver
rather than by me remembering, credentials live in `.env` at mode 600 and are
gitignored along with `backups/`.

What is not implemented, and would be needed for a real deployment: any
authentication at all, audit logging (Part-145 145.A.55 requires it), encryption
at rest (SQLCipher or TDE), TLS on database connections, and a secrets manager
instead of a dotfile. If real operator records were ever loaded, tail numbers
would have to be anonymised first.

## Where this would go next

Roughly in order of how much they matter:

1. A FastAPI layer with `/api/admin/*` and `/api/user/*` separated, so there is
   somewhere to put access control. Two roles: supply officer (read/write, can
   approve transfers) and technician (read-only, can raise a part request).
2. JWT auth and audit logging on top of that, which is what makes a deployment
   defensible under 145.A.55.
3. Docker Compose so setup is one command.
4. A real time-series model per part-station series to replace the boosted
   regressor.
5. A digital-twin simulator with what-if scenarios, and eventually RL over the
   stocking policy. That one is ambition, not a plan.

## License

MIT, see [LICENSE](LICENSE).
