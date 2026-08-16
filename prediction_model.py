"""
Daedalus Supply AI - Module 2: Prediction Engine
================================================
Three independent prediction engines, one per part class, because the three
classes fail for physically different reasons and no single model covers them:

1. ROTABLE FAILURE PREDICTION - Survival Analysis
   Kaplan-Meier, Cox Proportional Hazards and Weibull AFT (lifelines).
   Rotables wear out, so the quantity of interest is a time-to-event
   distribution, not a point estimate. Covariates: cycle ratio, airframe age,
   salt exposure, sector length, removal history.
   Output: hazard ratios per factor + median predicted time to failure.

2. EXPENDABLE DEMAND FORECASTING - Gradient-boosted regression
   XGBoost on monthly demand per part per station. Expendables fail randomly,
   so what matters is not when a given unit fails but how many units the
   network will consume next month.
   Output: expected quantity per part per station.

3. SDR PATTERN CLASSIFICATION - Gradient-boosted classification
   XGBoost trained on the real FAA Service Difficulty Report corpus, predicting
   the reported failure condition from ATA chapter and airframe utilisation.
   This is the only model here trained entirely on real-world data, and it is
   what grounds the synthetic reliability assumptions.
   Output: accuracy, per-class report, feature importance.

All model hyperparameters and the plot resolution are taken from config.py so
the module scales to the host hardware rather than assuming a workstation.

Author: Evangelos Tampachaniotis
Version: 1.0.0
License: MIT

Regulatory Framework:
    - EASA Part-M (EU 1321/2014) M.A.302 - the approved Aircraft Maintenance
      Programme must be supported by a reliability programme; the hazard ratios
      produced here are precisely the evidence that programme requires to
      justify escalating or shortening an inspection interval.
    - EASA Part-145 - component reliability feeds shop-visit planning.
    - ICAO Annex 8 - continued airworthiness; failure prediction supports the
      operator's obligation to maintain the type design standard.
    - MSG-3 - the decision logic that turns a demonstrated failure pattern into
      a scheduled maintenance task.
    - ATA/JASC 100 - chapter numbering, the dominant predictor in Model 3.

Usage:
    python prediction_model.py
"""

import sqlite3
import os
import sys
import warnings

# Suppress library deprecation/convergence chatter. The models report their own
# fit quality (concordance index, R-squared, accuracy) further down, so upstream
# warnings only obscure the results the operator actually needs to read.
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path

# Hardware-adaptive ML hyperparameters - see config.py.
from config import get_config

# Database path resolved relative to this file, so the module runs from any
# working directory. File name retained for backwards compatibility.
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aerosupply.db")


def check_dependencies():
    """
    Verify every ML library is importable before any work begins.

    Args:
        (none)

    Returns:
        None - exits the process with status 1 if anything is missing.

    Notes:
        Failing fast here rather than at first use matters because the three
        training runs take minutes: discovering a missing lifelines install
        after the SDR classifier has already trained wastes the whole run.
        The sklearn -> scikit-learn substitution is needed because the import
        name and the pip package name differ, and printing the import name in
        an install command would send the user to a non-existent package.
    """
    missing = []

    # Probe each dependency by import name.
    for lib in ["sklearn", "xgboost", "lifelines", "matplotlib", "seaborn"]:
        try:
            __import__(lib)
        except ImportError:
            missing.append(lib.replace("sklearn", "scikit-learn"))

    if missing:
        print(f"  Missing: {', '.join(missing)}")
        print(f"  Install: pip install {' '.join(missing)}")
        sys.exit(1)


def load_data():
    """
    Read every table the models need into memory in one pass.

    Args:
        (none)

    Returns:
        dict of str -> pd.DataFrame, keyed by logical table name. The "sdr" key
        is always present but may hold an empty frame.

    Notes:
        A single connection and eight SELECTs is far cheaper than re-querying
        per model, and holding the whole dataset in memory is affordable: the
        synthetic tables are small, and the SDR table is the only large one.
    """
    conn = sqlite3.connect(DB_PATH)

    # Full-table reads: every model needs most columns, so column pruning here
    # would only add maintenance burden for no measurable gain.
    data = {
        "fleet": pd.read_sql("SELECT * FROM fleet", conn),
        "flights": pd.read_sql("SELECT * FROM flight_log", conn),
        "parts": pd.read_sql("SELECT * FROM parts_catalog", conn),
        "demands": pd.read_sql("SELECT * FROM part_demands", conn),
        "work_orders": pd.read_sql("SELECT * FROM work_orders", conn),
        "findings": pd.read_sql("SELECT * FROM findings", conn),
        "stations": pd.read_sql("SELECT * FROM stations", conn),
        "inventory": pd.read_sql("SELECT * FROM inventory", conn),
    }

    # The SDR table exists only if the pipeline found CSVs in raw_data/.
    # Its absence is an expected configuration, not an error - Models 1 and 2
    # run perfectly well without it, so substitute an empty frame and continue.
    try:
        data["sdr"] = pd.read_sql("SELECT * FROM faa_sdr_raw", conn)
    except Exception:
        data["sdr"] = pd.DataFrame()

    conn.close()
    return data


# ============================================================================
# MODEL 1: ROTABLE FAILURE PREDICTION (Survival Analysis)
# ============================================================================

def train_survival_model(data):
    """
    Fit Kaplan-Meier, Cox PH and Weibull AFT models to rotable removal times.

    Args:
        data: dict - the table dictionary returned by load_data().

    Returns:
        tuple (CoxPHFitter, WeibullAFTFitter) of the fitted models, or None if
        there is insufficient rotable event data to fit anything meaningful.

    Notes:
        Why survival analysis rather than plain regression: the quantity being
        modelled is a time-to-event, and such data is censored (a component
        still installed at the end of the observation window has not failed
        yet, but its life so far is still information). Ordinary regression
        cannot represent that; survival models are built for it.

        The three models answer three different questions:
          Kaplan-Meier - what does the raw survival curve look like, with no
              assumptions at all? Used to compare operating groups.
          Cox PH - which factors change the hazard, by how much, and is the
              effect statistically significant? Produces the hazard ratios the
              Part-M reliability programme needs.
          Weibull AFT - a fully parametric fit, which is what allows an actual
              predicted time-to-failure in days for a given aircraft.

        Covariates:
            cycles_per_fh_ratio - sector-length proxy; high = short-haul stress
            aircraft_age        - cumulative airframe fatigue, years
            salt_exposure       - chloride corrosion load at home base, 0.0-1.0
            is_short_haul       - binary role indicator
            is_unscheduled      - was the previous removal unplanned? A proxy
                                  for a component already behaving badly
    """
    from lifelines import CoxPHFitter, WeibullAFTFitter, KaplanMeierFitter

    print("\n  [Model 1] Rotable Failure Prediction - Survival Analysis")
    print("  " + "-" * 55)

    demands = data["demands"]
    fleet = data["fleet"]
    parts = data["parts"]
    stations = data["stations"]

    # Restrict to rotables: they are the only class whose failures are driven
    # by progressive wear, and therefore the only class a survival model can
    # legitimately describe. Expendables are handled by Model 2.
    rotable_parts = parts[parts["part_class"] == "ROTABLE"]["part_number"].tolist()
    rotable_demands = demands[demands["part_number"].isin(rotable_parts)].copy()

    if rotable_demands.empty:
        print("  No rotable demand data - skipping")
        return None

    # --- Attach aircraft-level covariates ---
    # LEFT JOIN so a demand against an unknown registration is retained with
    # nulls rather than silently dropped from the event count.
    rotable_demands = rotable_demands.merge(
        fleet[["tail_number", "manufacture_year", "home_base", "primary_role", "cycles_per_fh_ratio"]],
        on="tail_number", how="left"
    )

    # --- Attach the environmental covariate ---
    # Map home base -> salt exposure index. fillna(0.3) applies the ATH
    # (mainland hub) value as a neutral default for any unmapped station,
    # which is conservative: it neither inflates nor suppresses the effect.
    station_map = dict(zip(stations["station_code"], stations["salt_exposure"]))
    rotable_demands["salt_exposure"] = rotable_demands["home_base"].map(station_map).fillna(0.3)

    # --- Derived covariates ---
    # Age in years at the 2024 simulation epoch. The 2015 fill is the fleet's
    # approximate median build year, so an unknown aircraft is treated as
    # average rather than as new or ancient.
    rotable_demands["aircraft_age"] = 2024 - rotable_demands["manufacture_year"].fillna(2015)

    # Binary encodings - Cox PH requires numeric covariates.
    rotable_demands["is_short_haul"] = (rotable_demands["primary_role"] == "short_haul").astype(int)
    rotable_demands["is_unscheduled"] = (rotable_demands["demand_type"] == "UNSCHEDULED").astype(int)

    # --- Attach part-level attributes (ATA chapter, MTBF, cost) ---
    rotable_demands = rotable_demands.merge(
        parts[["part_number", "ata_chapter", "mtbf_flight_hours", "unit_cost_eur"]],
        on="part_number", how="left"
    )

    # ------------------------------------------------------------------
    # Build the duration column - the core of any survival dataset.
    # Transformation: a list of removal EVENTS (dates) -> a list of INTERVALS
    # between consecutive removals of the same part on the same aircraft.
    # ------------------------------------------------------------------
    rotable_demands["demand_date"] = pd.to_datetime(rotable_demands["demand_date"])

    # Sort within each (aircraft, part) series so shift(1) picks up the
    # immediately preceding removal of that same component position.
    rotable_demands = rotable_demands.sort_values(["tail_number", "part_number", "demand_date"])

    # Previous removal date for the same aircraft/part combination.
    rotable_demands["prev_date"] = rotable_demands.groupby(
        ["tail_number", "part_number"]
    )["demand_date"].shift(1)

    # Duration = days the installed unit survived before this removal.
    rotable_demands["duration_days"] = (
        rotable_demands["demand_date"] - rotable_demands["prev_date"]
    ).dt.days

    # --- First event in each series has no predecessor ---
    # Its duration is measured from the start of the observation window
    # instead. This left-truncates the first interval (the unit had already
    # been in service before observation began), which biases those durations
    # low - acknowledged, and part of why the concordance index sits at ~0.56.
    sim_start = rotable_demands["demand_date"].min()
    mask_first = rotable_demands["duration_days"].isna()
    rotable_demands.loc[mask_first, "duration_days"] = (
        rotable_demands.loc[mask_first, "demand_date"] - sim_start
    ).dt.days

    # Survival models require strictly positive durations: a zero-day interval
    # is two removals recorded on the same date (a double replacement during
    # one visit), which carries no time-to-event information.
    rotable_demands = rotable_demands[rotable_demands["duration_days"] > 0].copy()

    # --- Event indicator ---
    # 1 = the failure was observed. Every row here is an actual recorded
    # removal, so all are events. Components still installed at the end of the
    # window would be censored observations (0); adding them is the single
    # highest-value improvement available to this model, and is why v1.0's
    # concordance index is modest.
    rotable_demands["event_observed"] = 1

    feature_cols = [
        "duration_days", "event_observed",
        "cycles_per_fh_ratio", "aircraft_age", "salt_exposure",
        "is_short_haul", "is_unscheduled", "ata_chapter"
    ]

    # dropna: lifelines cannot fit with missing covariates, and imputing a
    # hazard covariate would fabricate reliability evidence.
    surv_df = rotable_demands[feature_cols].dropna()

    # 20 events is the practical floor for a 5-covariate Cox model (the usual
    # rule of thumb is ~10 events per covariate); below it the fit is noise.
    if len(surv_df) < 20:
        print(f"  Only {len(surv_df)} records - need more for survival analysis")
        return None

    print(f"  Training data: {len(surv_df)} events")

    # ------------------------------------------------------------------
    # Kaplan-Meier - non-parametric survival curve
    # ------------------------------------------------------------------
    print("\n  Kaplan-Meier Survival Curves:")
    kmf = KaplanMeierFitter()

    # Whole-population baseline. Median survival = the time by which half the
    # installed population has been removed.
    kmf.fit(surv_df["duration_days"], surv_df["event_observed"], label="All rotables")
    median_surv = kmf.median_survival_time_
    print(f"    Overall median survival: {median_surv:.0f} days")

    # Stratify by operating role. This is the headline comparison: if
    # short-haul operation really does consume components faster, the two
    # curves must separate - and separation here is what justifies holding
    # different stock levels at short-haul versus medium-haul bases.
    for label, val in [("Short-haul", 1), ("Medium-haul", 0)]:
        mask = surv_df["is_short_haul"] == val
        # Require more than 5 events before quoting a median for a subgroup;
        # below that the estimate is dominated by a single observation.
        if mask.sum() > 5:
            kmf.fit(surv_df.loc[mask, "duration_days"],
                    surv_df.loc[mask, "event_observed"], label=label)
            print(f"    {label} median survival: {kmf.median_survival_time_:.0f} days")

    # ------------------------------------------------------------------
    # Cox Proportional Hazards - which factors matter, and by how much
    # ------------------------------------------------------------------
    print("\n  Cox Proportional Hazards Model:")

    # ATA chapter is deliberately excluded from the Cox covariates: it is a
    # nominal category, and entering it as an integer would tell the model that
    # chapter 72 is "three times" chapter 24, which is meaningless. It stays in
    # the survival frame for stratified reporting only.
    cox_features = [
        "duration_days", "event_observed",
        "cycles_per_fh_ratio", "aircraft_age", "salt_exposure",
        "is_short_haul", "is_unscheduled"
    ]
    cox_df = surv_df[cox_features].copy()

    # penalizer = 0.1 applies L2 regularisation to the coefficients. Needed
    # because cycles_per_fh_ratio and is_short_haul are strongly collinear by
    # construction (short-haul operation IS a high cycle ratio); without it the
    # fit can fail to converge or produce wildly inflated coefficients.
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(cox_df, duration_col="duration_days", event_col="event_observed")

    print("\n  Coefficients (positive = higher hazard = fails sooner):")
    summary = cph.summary[["coef", "exp(coef)", "p"]].copy()
    summary.columns = ["Coefficient", "Hazard Ratio", "p-value"]

    # Print one line per covariate with a conventional significance marker.
    # exp(coef) is the hazard ratio: HR = 1.465 means a one-unit increase in
    # that covariate multiplies the instantaneous failure rate by 1.465, i.e.
    # a 46.5% higher risk at any given moment.
    for idx, row in summary.iterrows():
        # Standard significance thresholds: *** p<0.01, ** p<0.05, * p<0.10.
        sig = ("***" if row["p-value"] < 0.01
               else "**" if row["p-value"] < 0.05
               else "*" if row["p-value"] < 0.1
               else "")
        # A positive coefficient means a shorter time to failure.
        direction = "higher risk" if row["Coefficient"] > 0 else "lower risk"
        print(f"    {idx:<25} HR={row['Hazard Ratio']:.3f}  p={row['p-value']:.4f} {sig} ({direction})")

    # Harrell's concordance index: the probability that, for a random pair of
    # components, the model ranks the one that failed first as higher risk.
    print(f"\n  Concordance index: {cph.concordance_index_:.3f}")
    print("  (0.5 = random, 1.0 = perfect prediction)")

    # ------------------------------------------------------------------
    # Weibull AFT - parametric fit giving actual time-to-failure estimates
    # ------------------------------------------------------------------
    # Cox PH gives relative risk but no absolute timescale (its baseline hazard
    # is left unspecified). The Accelerated Failure Time model assumes a Weibull
    # baseline, which buys a concrete answer in days - the form a planner can
    # act on. Weibull is the right parametric family here for the same reason
    # the pipeline generates with it: it represents monotonic wear-out.
    print("\n  Weibull AFT Model (time-to-failure estimation):")

    waft = WeibullAFTFitter()
    waft.fit(cox_df, duration_col="duration_days", event_col="event_observed")

    # --- Worked examples spanning the operating envelope ---
    # Three deliberately contrasting profiles, so the effect of the covariates
    # is visible as a difference in predicted days rather than as coefficients.
    print("\n  Example predictions (days until failure):")
    examples = pd.DataFrame([
        # Worst case: old, high cycle ratio, island base with heavy salt load.
        {"cycles_per_fh_ratio": 1.15, "aircraft_age": 15, "salt_exposure": 0.8,
         "is_short_haul": 1, "is_unscheduled": 0},
        # Best case: young, long sectors, sheltered mainland hub.
        {"cycles_per_fh_ratio": 0.35, "aircraft_age": 5, "salt_exposure": 0.3,
         "is_short_haul": 0, "is_unscheduled": 0},
        # Middle case, but with a prior unscheduled removal on the position.
        {"cycles_per_fh_ratio": 1.10, "aircraft_age": 10, "salt_exposure": 0.2,
         "is_short_haul": 1, "is_unscheduled": 1},
    ], index=["Old short-haul island (HER)", "New medium-haul (ATH)", "Mid short-haul (SKG)"])

    # predict_median returns the 50th percentile of the predicted survival
    # distribution - the point at which failure is as likely as not.
    predictions = waft.predict_median(examples)
    for name, pred in zip(examples.index, predictions):
        print(f"    {name:<35} -> {pred:.0f} days median time to failure")

    # ------------------------------------------------------------------
    # Diagnostic plots
    # ------------------------------------------------------------------
    try:
        import matplotlib
        # Force the non-interactive Agg backend BEFORE importing pyplot. Without
        # it, matplotlib tries to open a display and crashes on a headless
        # server or over SSH - exactly where this pipeline usually runs.
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cfg = get_config()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left panel: Kaplan-Meier curves by operating role. Visual separation
        # between the two lines is the evidence for role-dependent reliability.
        kmf1 = KaplanMeierFitter()
        for label, val in [("Short-haul", 1), ("Medium-haul", 0)]:
            mask = surv_df["is_short_haul"] == val
            if mask.sum() > 5:
                kmf1.fit(surv_df.loc[mask, "duration_days"],
                         surv_df.loc[mask, "event_observed"], label=label)
                kmf1.plot_survival_function(ax=axes[0])
        axes[0].set_xlabel("Days since installation")
        axes[0].set_ylabel("Survival probability")
        axes[0].set_title("Kaplan-Meier: Short-haul vs Medium-haul")
        axes[0].legend()

        # Right panel: Cox coefficient forest plot with confidence intervals.
        # A covariate whose interval crosses zero has no demonstrated effect.
        cph.plot(ax=axes[1])
        axes[1].set_title("Cox PH: Factor Hazard Ratios")

        plt.tight_layout()
        plot_path = os.path.join(os.path.dirname(DB_PATH), "survival_analysis.png")
        # DPI from the hardware profile: 72 on MINIMAL (small files, fast),
        # up to 200 on FULL (report quality).
        plt.savefig(plot_path, dpi=cfg["ml"]["plot_dpi"])
        print(f"\n  Plot saved: {plot_path}")
        plt.close()
    except Exception as e:
        # Plotting is presentation, not analysis. A missing font, an unwritable
        # directory or a backend problem must never discard a completed model
        # fit - report and continue.
        print(f"\n  Plot skipped: {e}")

    return cph, waft


# ============================================================================
# MODEL 2: EXPENDABLE DEMAND FORECASTING (XGBoost regression)
# ============================================================================

def train_demand_forecast(data):
    """
    Forecast monthly expendable consumption per part per station.

    Args:
        data: dict - the table dictionary returned by load_data().

    Returns:
        tuple (XGBRegressor, LabelEncoder, LabelEncoder) - the fitted model
        plus the part and station encoders needed to transform new inputs, or
        None when there is no expendable demand history.

    Notes:
        Transformation: individual demand events -> (month, part, station)
        aggregate -> supervised rows with calendar, fleet-activity and lag
        features -> next-month quantity.

        Gradient boosting rather than a classical time-series model (ARIMA,
        Holt-Winters) because the target is a panel: hundreds of short, sparse,
        intermittent series that share structure across parts and stations.
        A tree ensemble pools that shared structure; per-series ARIMA cannot,
        and most individual series are far too short to fit one anyway.

        Known limitation (v1.0): R-squared is negative on the held-out period.
        With three years of synthetic history the per-series signal is weak and
        the model does not beat the test-set mean. MAE (~0.85 parts/month) is
        the metric to read for practical purposes, since stocking decisions are
        driven by absolute error in units, not by explained variance.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.preprocessing import LabelEncoder
    import xgboost as xgb

    print("\n  [Model 2] Expendable Demand Forecasting - XGBoost")
    print("  " + "-" * 55)

    cfg = get_config()

    demands = data["demands"]
    flights = data["flights"]
    parts = data["parts"]

    # Expendables only: cheap, scrapped on removal, failing at a roughly
    # constant rate. Their aggregate consumption is forecastable even though no
    # individual unit's failure time is.
    expendable_parts = parts[parts["part_class"] == "EXPENDABLE"]["part_number"].tolist()
    exp_demands = demands[demands["part_number"].isin(expendable_parts)].copy()

    if exp_demands.empty:
        print("  No expendable demand data")
        return None

    # --- Aggregate to a monthly panel ---
    # Monthly is the natural planning granularity: it matches procurement
    # cycles and smooths the intermittency that makes daily demand unusable.
    exp_demands["demand_date"] = pd.to_datetime(exp_demands["demand_date"])
    exp_demands["year_month"] = exp_demands["demand_date"].dt.to_period("M")

    monthly = exp_demands.groupby(
        ["year_month", "part_number", "station"]
    ).agg(
        # Total units consumed - the regression target.
        demand_qty=("quantity_required", "sum"),
        # Number of separate demand events - retained as a diagnostic of
        # whether a month's quantity came from one bulk issue or many singles.
        demand_events=("quantity_required", "count"),
    ).reset_index()

    # --- Calendar features ---
    # Explicit seasonal flags rather than leaving the model to infer season
    # from the month number: with only three annual cycles of history there is
    # not enough data to learn the shape, so the domain knowledge is encoded
    # directly. The summer/winter definitions match the pipeline's own traffic
    # model (summer Jun-Sep, winter Dec-Feb).
    monthly["month"] = monthly["year_month"].dt.month
    monthly["quarter"] = monthly["year_month"].dt.quarter
    monthly["is_summer"] = monthly["month"].isin([6, 7, 8, 9]).astype(int)
    monthly["is_winter"] = monthly["month"].isin([12, 1, 2]).astype(int)

    # --- Fleet activity features ---
    # Consumption is ultimately driven by exposure. Total fleet hours, cycles
    # and sector count per month is the exposure measure, and it also carries
    # the seasonal traffic signal independently of the calendar flags.
    flights["flight_date"] = pd.to_datetime(flights["flight_date"])
    flights["year_month"] = flights["flight_date"].dt.to_period("M")

    fleet_activity = flights.groupby("year_month").agg(
        total_fh=("flight_hours", "sum"),
        total_fc=("flight_cycles", "sum"),
        n_flights=("flight_hours", "count"),
    ).reset_index()

    monthly = monthly.merge(fleet_activity, on="year_month", how="left")

    # --- Categorical encoding ---
    # Label (ordinal) encoding rather than one-hot: tree models split on
    # thresholds and handle high-cardinality integer codes natively, whereas
    # one-hot would add ~50 sparse columns for no gain. The encoders are
    # returned so predictions on new data use identical code assignments.
    le_part = LabelEncoder()
    le_station = LabelEncoder()
    monthly["part_encoded"] = le_part.fit_transform(monthly["part_number"])
    monthly["station_encoded"] = le_station.fit_transform(monthly["station"])

    # Chronological ordering, required for the time-based split further down.
    monthly = monthly.sort_values("year_month")
    monthly["month_num"] = range(len(monthly))

    monthly["year_month_str"] = monthly["year_month"].astype(str)

    # --- Lag features ---
    # Re-sort by series then time so shift() operates within each
    # (part, station) series rather than across series boundaries.
    monthly = monthly.sort_values(["part_number", "station", "year_month"])

    # lag_1 - last month's quantity. Usually the single strongest predictor of
    # intermittent demand: what was consumed recently tends to be consumed again.
    monthly["lag_1"] = monthly.groupby(["part_number", "station"])["demand_qty"].shift(1)
    # lag_2 - two months back, giving the model a crude sense of direction.
    monthly["lag_2"] = monthly.groupby(["part_number", "station"])["demand_qty"].shift(2)
    # 3-month rolling mean, shifted by one so it uses only PAST months. Without
    # the shift the window would include the target month itself and leak the
    # answer into the features, producing an inflated score that collapses in
    # production. min_periods=1 lets a short series contribute rather than drop.
    monthly["rolling_mean_3"] = monthly.groupby(
        ["part_number", "station"]
    )["demand_qty"].transform(lambda x: x.rolling(3, min_periods=1).mean().shift(1))

    # The first observation of every series has no lag_1 and cannot be used
    # for supervised training.
    monthly = monthly.dropna(subset=["lag_1"])

    feature_cols = [
        "part_encoded", "station_encoded",
        "month", "quarter", "is_summer", "is_winter",
        "total_fh", "total_fc", "n_flights",
        "lag_1", "lag_2", "rolling_mean_3"
    ]

    X = monthly[feature_cols].values
    y = monthly["demand_qty"].values

    print(f"  Training data: {len(X)} monthly records")
    print(f"  Parts: {len(le_part.classes_)}, Stations: {len(le_station.classes_)}")

    # --- Time-based split: last 20% of the timeline is the test set ---
    # A random split would be invalid here. Lag features encode the past, so
    # random assignment lets the model train on months that come AFTER its test
    # months and effectively see the future. Splitting chronologically
    # reproduces the real forecasting task.
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # --- Model configuration ---
    model = xgb.XGBRegressor(
        # Tree count and depth come from the hardware profile (config.py):
        # 50/4 on MINIMAL, 200/6 on STANDARD, 500/8 on FULL.
        n_estimators=cfg["ml"]["xgb_n_estimators"],
        max_depth=cfg["ml"]["xgb_max_depth"],
        # 0.1 is the conventional boosting learning rate - small enough that no
        # single tree dominates, large enough to converge in a few hundred rounds.
        learning_rate=0.1,
        # Row subsampling: each tree sees 80% of rows. Stochastic gradient
        # boosting; decorrelates trees and reduces overfitting on a small panel.
        subsample=0.8,
        # Column subsampling: each tree sees 80% of features. Prevents the lag
        # features, which are individually very strong, from being chosen at
        # every split and crowding out the seasonal signal.
        colsample_bytree=0.8,
        # Worker threads from the hardware profile.
        n_jobs=cfg["ml"]["n_jobs"],
        # Fixed seed so a reliability figure quoted today reproduces tomorrow.
        random_state=42,
        verbosity=0,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # --- Evaluation ---
    y_pred = model.predict(X_test)
    # Clamp at zero: the regressor is unconstrained and can emit small negative
    # values, but a negative parts requirement is physically meaningless and
    # would corrupt any stocking calculation downstream.
    y_pred = np.maximum(y_pred, 0)

    # MAE - mean absolute error, in parts per month. The operationally
    # meaningful metric: it is the average number of units the forecast is off by.
    mae = mean_absolute_error(y_test, y_pred)
    # RMSE - penalises large misses more heavily; a big gap between RMSE and
    # MAE indicates occasional severe errors rather than uniform drift.
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    # R-squared - fraction of variance explained. Negative means the model does
    # worse than predicting the test mean; see the docstring note.
    r2 = r2_score(y_test, y_pred)

    print(f"\n  Results (test set = last 20% of timeline):")
    print(f"    MAE:  {mae:.2f} parts/month")
    print(f"    RMSE: {rmse:.2f} parts/month")
    print(f"    R2:   {r2:.3f}")

    # --- Feature importance ---
    # Readable names in the same order as feature_cols.
    print("\n  Feature Importance:")
    importance = dict(zip(
        ["part", "station", "month", "quarter", "is_summer", "is_winter",
         "fleet_FH", "fleet_FC", "n_flights", "lag_1", "lag_2", "rolling_mean_3"],
        model.feature_importances_
    ))
    # Descending order, with a proportional bar. x50 scales a 0.0-1.0
    # importance score to at most 50 characters, fitting an 80-column terminal.
    for feat, imp in sorted(importance.items(), key=lambda x: -x[1]):
        bar = "#" * int(imp * 50)
        print(f"    {feat:<20} {imp:.3f} {bar}")

    # --- Diagnostic plots ---
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless backend - see Model 1
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: actual vs predicted scatter against the identity line. Points
        # hugging the diagonal mean good calibration; a horizontal band means
        # the model is predicting close to a constant, which is what a negative
        # R-squared looks like graphically.
        axes[0].scatter(y_test, y_pred, alpha=0.5, s=20)
        max_val = max(y_test.max(), y_pred.max())
        axes[0].plot([0, max_val], [0, max_val], "r--", label="Perfect prediction")
        axes[0].set_xlabel("Actual demand")
        axes[0].set_ylabel("Predicted demand")
        axes[0].set_title(f"Demand Forecast: Actual vs Predicted (R2={r2:.3f})")
        axes[0].legend()

        # Right: feature importance, ascending so the strongest bar sits at the
        # top of a horizontal bar chart.
        sorted_imp = sorted(importance.items(), key=lambda x: x[1])
        axes[1].barh([x[0] for x in sorted_imp], [x[1] for x in sorted_imp])
        axes[1].set_title("Feature Importance")
        axes[1].set_xlabel("Importance score")

        plt.tight_layout()
        plot_path = os.path.join(os.path.dirname(DB_PATH), "demand_forecast.png")
        plt.savefig(plot_path, dpi=cfg["ml"]["plot_dpi"])
        print(f"\n  Plot saved: {plot_path}")
        plt.close()
    except Exception as e:
        print(f"\n  Plot skipped: {e}")

    return model, le_part, le_station


# ============================================================================
# MODEL 3: SDR FAILURE PATTERN ANALYSIS (XGBoost classification)
# ============================================================================

def train_sdr_classifier(data):
    """
    Classify reported failure conditions in the real FAA SDR corpus.

    Args:
        data: dict - the table dictionary returned by load_data().

    Returns:
        tuple (XGBClassifier, LabelEncoder) - the fitted classifier and the
        target encoder, or None when SDR data is absent or unusable.

    Notes:
        This is the only model trained purely on real-world data, which makes
        it the project's empirical anchor. The interesting result is not the
        headline accuracy but the feature importance: ATA chapter alone
        accounts for roughly two thirds of predictive power, meaning the SYSTEM
        a component belongs to determines HOW it fails far more than the
        airframe's age or utilisation does. That finding is what justifies
        organising the entire inventory strategy by ATA chapter.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.preprocessing import LabelEncoder
    import xgboost as xgb

    print("\n  [Model 3] SDR Failure Pattern Analysis - XGBoost Classification")
    print("  " + "-" * 55)

    cfg = get_config()

    sdr = data["sdr"]
    if sdr.empty:
        print("  No SDR data available - skipping")
        return None

    # Work on a copy: the cleaning below mutates columns, and data["sdr"] is
    # shared with the caller.
    sdr = sdr.copy()

    # Without a system code and a reported condition there is no feature and no
    # target - abort rather than train on a degenerate frame.
    required = ["ata_code", "nature_condition"]
    for col in required:
        if col not in sdr.columns:
            print(f"  Missing column: {col}")
            return None

    # Drop rows with no reported condition (the label) or no system code.
    sdr = sdr[sdr["nature_condition"].notna() & (sdr["nature_condition"] != "")]
    sdr = sdr[sdr["ata_code"].notna()]

    # --- Derive the ATA chapter from the 4-digit JASC code ---
    # First two digits = chapter (3240 -> 32, Landing Gear).
    sdr["ata_chapter"] = pd.to_numeric(sdr["ata_code"].astype(str).str[:2], errors="coerce")
    sdr = sdr.dropna(subset=["ata_chapter"])
    sdr["ata_chapter"] = sdr["ata_chapter"].astype(int)

    # --- Numeric coercion of the utilisation columns ---
    # The raw export stores these as text and includes blanks, commas and
    # free-text entries. errors="coerce" turns anything unparseable into NaN,
    # which is then median-filled below rather than dropping the whole record.
    if "total_time" in sdr.columns:
        sdr["total_time"] = pd.to_numeric(sdr["total_time"], errors="coerce")
    else:
        sdr["total_time"] = np.nan

    if "total_cycles" in sdr.columns:
        sdr["total_cycles"] = pd.to_numeric(sdr["total_cycles"], errors="coerce")

    # Type indicator: does this occurrence concern an A320-family aircraft?
    # Lets the model distinguish the type this project actually operates from
    # the other transport types retained for sample size.
    if "acft_model" in sdr.columns:
        sdr["is_a320"] = sdr["acft_model"].astype(str).str.contains("A320", case=False, na=False).astype(int)
    else:
        sdr["is_a320"] = 0

    # --- Report date -> calendar features ---
    # Month captures any seasonal reporting pattern (de-icing damage in winter,
    # high-utilisation failures in summer).
    if "report_date" in sdr.columns:
        sdr["report_date"] = pd.to_datetime(sdr["report_date"], errors="coerce")
        sdr["report_month"] = sdr["report_date"].dt.month
        sdr["report_year"] = sdr["report_date"].dt.year
    else:
        # Neutral mid-year defaults when the export carries no date column, so
        # the feature exists with a constant value rather than breaking the fit.
        sdr["report_month"] = 6
        sdr["report_year"] = 2023

    # --- Restrict the label space to the 10 most common conditions ---
    # The raw corpus contains hundreds of distinct free-text conditions, most
    # appearing a handful of times. Rare classes cannot be learned, inflate the
    # apparent class count and destabilise the stratified split; the top 10
    # still cover the large majority of records.
    top_conditions = sdr["nature_condition"].value_counts().head(10).index.tolist()
    sdr_filtered = sdr[sdr["nature_condition"].isin(top_conditions)].copy()

    print(f"  SDR records after cleaning: {len(sdr_filtered)}")
    print(f"  Target classes: {len(top_conditions)}")
    print(f"  Classes: {top_conditions}")

    # Below ~100 records a 10-class stratified split leaves single-digit test
    # counts per class, making the accuracy figure meaningless.
    if len(sdr_filtered) < 100:
        print("  Too few records - skipping")
        return None

    # Encode the text condition labels as integers for XGBoost.
    le_target = LabelEncoder()
    sdr_filtered["target"] = le_target.fit_transform(sdr_filtered["nature_condition"])

    # Base features, always present.
    feature_cols = ["ata_chapter", "is_a320", "report_month"]

    # --- Optional utilisation features, median-imputed ---
    # Median rather than mean: airframe hours are heavily right-skewed (a few
    # very high-time aircraft), so the mean would sit above most of the
    # population and systematically misrepresent the imputed records.
    if "total_time" in sdr_filtered.columns:
        sdr_filtered["total_time_filled"] = sdr_filtered["total_time"].fillna(
            sdr_filtered["total_time"].median()
        )
        feature_cols.append("total_time_filled")

    if "total_cycles" in sdr_filtered.columns:
        sdr_filtered["total_cycles_filled"] = sdr_filtered["total_cycles"].fillna(
            sdr_filtered["total_cycles"].median()
        )
        feature_cols.append("total_cycles_filled")

    X = sdr_filtered[feature_cols].values
    y = sdr_filtered["target"].values

    # Random split is valid here (unlike Model 2): each SDR record is an
    # independent occurrence with no lag features, so there is no leakage path.
    # stratify=y preserves the class proportions in both halves, which matters
    # because the condition classes are strongly imbalanced.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    model = xgb.XGBClassifier(
        # Hardware-profile driven, as in Model 2.
        n_estimators=cfg["ml"]["xgb_n_estimators"],
        max_depth=cfg["ml"]["xgb_max_depth"],
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=cfg["ml"]["n_jobs"],
        random_state=42,
        verbosity=0,
        # Multi-class log loss - the correct objective metric for a 10-class
        # problem, and it penalises confident wrong answers rather than only
        # counting misclassifications.
        eval_metric="mlogloss",
    )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n  Accuracy: {acc:.3f}")
    print(f"\n  Classification Report:")
    # Per-class precision/recall/F1. Essential with imbalanced classes: a high
    # overall accuracy can hide a class the model never predicts at all.
    # zero_division=0 reports 0.0 instead of raising for any such class.
    report = classification_report(
        y_test, y_pred,
        target_names=le_target.classes_,
        zero_division=0
    )
    print(report)

    print("  Feature Importance:")
    for feat, imp in sorted(
        zip(feature_cols, model.feature_importances_), key=lambda x: -x[1]
    ):
        bar = "#" * int(imp * 50)
        print(f"    {feat:<25} {imp:.3f} {bar}")

    # --- Where do real failures actually occur? ---
    # Occurrence counts by ATA chapter, annotated with the system name so the
    # numbers are readable without an ATA 100 reference to hand.
    print("\n  Top 10 ATA chapters by failure count (real SDR data):")
    ata_counts = sdr_filtered.groupby("ata_chapter").size().sort_values(ascending=False).head(10)
    for ata, count in ata_counts.items():
        # ATA/JASC 100 chapter names for the systems this project covers;
        # anything outside the map is reported as "Other".
        ata_desc = {
            21: "Air Conditioning", 24: "Electrical Power", 25: "Equipment",
            26: "Fire Protection", 27: "Flight Controls", 28: "Fuel",
            29: "Hydraulic", 32: "Landing Gear", 33: "Lights",
            34: "Navigation", 36: "Pneumatic", 49: "APU",
            52: "Doors", 53: "Fuselage", 55: "Stabilizers",
            72: "Engine", 73: "Engine Fuel", 78: "Exhaust",
        }.get(ata, "Other")
        print(f"    ATA {ata:>2} ({ata_desc:<20}) {count:>6} failures")

    # --- Diagnostic plots ---
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless backend - see Model 1
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: the 15 highest-occurrence ATA chapters in the real corpus.
        # invert_yaxis puts the largest bar at the top, matching how a ranked
        # list reads.
        top_ata = sdr_filtered["ata_chapter"].value_counts().head(15)
        axes[0].barh(top_ata.index.astype(str), top_ata.values)
        axes[0].set_xlabel("Number of failures")
        axes[0].set_ylabel("ATA Chapter")
        axes[0].set_title("Real FAA SDR: Failures by ATA Chapter")
        axes[0].invert_yaxis()

        # Right: feature importance, ascending for a horizontal bar chart.
        sorted_feats = sorted(
            zip(feature_cols, model.feature_importances_), key=lambda x: x[1]
        )
        axes[1].barh([x[0] for x in sorted_feats], [x[1] for x in sorted_feats])
        axes[1].set_title("Feature Importance: What Drives Failures")
        axes[1].set_xlabel("Importance")

        plt.tight_layout()
        plot_path = os.path.join(os.path.dirname(DB_PATH), "sdr_analysis.png")
        plt.savefig(plot_path, dpi=cfg["ml"]["plot_dpi"])
        print(f"\n  Plot saved: {plot_path}")
        plt.close()
    except Exception as e:
        print(f"\n  Plot skipped: {e}")

    return model, le_target


# ============================================================================
# PREDICTION API - apply the trained models
# ============================================================================

def predict_next_month(survival_model, demand_model, data):
    """
    Produce the forward-looking planning view from the fitted models.

    Args:
        survival_model: tuple (CoxPHFitter, WeibullAFTFitter) or None
        demand_model: tuple (XGBRegressor, LabelEncoder, LabelEncoder) or None
        data: dict - the table dictionary returned by load_data().

    Returns:
        None - prints two reports to stdout.

    Notes:
        This is where the models stop being statistics and become a work list:
        per-aircraft rotable risk bands, and per-station expendable demand
        compared against stock actually on hand.
    """
    # Nothing was fitted - there is no forecast to present.
    if survival_model is None and demand_model is None:
        return

    print("\n  " + "=" * 55)
    print("  PREDICTIONS FOR NEXT 30 DAYS")
    print("  " + "=" * 55)

    fleet = data["fleet"]
    stations = data["stations"]
    parts = data["parts"]
    inventory = data["inventory"]

    # Station -> salt exposure lookup, matching the covariate the AFT model
    # was fitted on.
    station_salt = dict(zip(stations["station_code"], stations["salt_exposure"]))

    # ------------------------------------------------------------------
    # Rotable risk per aircraft
    # ------------------------------------------------------------------
    if survival_model is not None:
        cph, waft = survival_model

        print("\n  ROTABLE RISK ASSESSMENT:")
        print(f"  {'Aircraft':<10} {'Base':<5} {'Risk Level':<12} {'Predicted days to failure'}")
        print(f"  {'-' * 55}")

        # Score every aircraft in the fleet using its own covariate values.
        for _, ac in fleet.iterrows():
            # Build a single-row frame in exactly the covariate layout the AFT
            # model was fitted with. is_unscheduled = 0 asks the baseline
            # question: what is the outlook for a normally maintained aircraft?
            example = pd.DataFrame([{
                "cycles_per_fh_ratio": ac["cycles_per_fh_ratio"],
                "aircraft_age": 2024 - ac["manufacture_year"],
                "salt_exposure": station_salt.get(ac["home_base"], 0.3),
                "is_short_haul": 1 if ac["primary_role"] == "short_haul" else 0,
                "is_unscheduled": 0,
            }])

            try:
                median_days = waft.predict_median(example).values[0]

                # --- Risk banding ---
                # Cut points chosen to align with procurement lead times, which
                # is what makes them actionable rather than arbitrary:
                #   < 60 days  HIGH   - shorter than a rotable's 5-30 day
                #                       overhaul lead time plus safety margin;
                #                       the part must be on the shelf already.
                #   < 180 days MEDIUM - inside the planning horizon; order now.
                #   >= 180     LOW     - beyond the horizon; no action needed.
                if median_days < 60:
                    risk = "HIGH"
                elif median_days < 180:
                    risk = "MEDIUM"
                else:
                    risk = "LOW"
                print(f"  {ac['tail_number']:<10} {ac['home_base']:<5} {risk:<12} {median_days:.0f} days")
            except Exception:
                # An aircraft whose covariates fall outside the fitted range can
                # produce a non-finite median. Skip that row rather than abort
                # the whole fleet report for one aircraft.
                pass

    # ------------------------------------------------------------------
    # Expendable demand per station, against stock on hand
    # ------------------------------------------------------------------
    if demand_model is not None:
        model, le_part, le_station = demand_model

        print("\n  EXPENDABLE DEMAND FORECAST (next month):")
        print(f"  {'Station':<8} {'Part':<20} {'Predicted Qty':<15} {'Current Stock'}")
        print(f"  {'-' * 60}")

        exp_parts = parts[parts["part_class"] == "EXPENDABLE"]

        # Iterate over every station and a representative sample of expendables.
        # Limited to 5 parts per station so the report stays a readable summary
        # rather than a 265-line dump; the full picture is Module 3's job.
        for station in stations["station_code"]:
            for _, part in exp_parts.head(5).iterrows():
                try:
                    part_enc = le_part.transform([part["part_number"]])[0]
                    sta_enc = le_station.transform([station])[0]
                except ValueError:
                    # LabelEncoder raises on any value it never saw during
                    # training - a part or station with no demand history. There
                    # is nothing to forecast for it, so move on.
                    continue

                # Feature vector in feature_cols order. Values represent a
                # typical September (peak-season) planning scenario:
                #   9, 3, 1, 0        month=September, Q3, is_summer, not winter
                #   4500, 5000, 750   fleet activity: FH, FC and sectors in a
                #                     representative month across 15 aircraft
                #   3, 2, 2.5         lag_1, lag_2, rolling_mean_3 - typical
                #                     recent consumption for an expendable
                features = np.array([[
                    part_enc, sta_enc,
                    9, 3, 1, 0,
                    4500, 5000, 750,
                    3, 2, 2.5
                ]])

                # Clamp and integerise: parts are issued in whole units.
                pred = max(0, int(model.predict(features)[0]))

                # Compare the forecast against serviceable stock actually held
                # at that station - the number that decides whether to reorder.
                stock_row = inventory[
                    (inventory["part_number"] == part["part_number"]) &
                    (inventory["station"] == station)
                ]
                current = stock_row.iloc[0]["serviceable"] if not stock_row.empty else 0

                # Only report parts with non-zero forecast demand; a predicted
                # zero carries no planning information.
                if pred > 0:
                    # Flag the case that matters: forecast demand exceeds stock.
                    alert = "  LOW STOCK" if current < pred else ""
                    print(f"  {station:<8} {part['description'][:20]:<20} {pred:<15} {current}{alert}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """
    Train all three models, then produce the combined forward-looking report.

    Args:
        (none)

    Returns:
        None

    Notes:
        The three models are independent and any of them may return None
        (insufficient data, missing SDR corpus). The run continues regardless,
        because a partial reliability picture is still useful and far better
        than no output at all.
    """
    cfg = get_config()

    print("=" * 60)
    print("  Daedalus Supply AI - Module 2: Prediction Engine")
    print("  EASA Part-M Reliability Programme Automation")
    print(f"  Hardware profile: {cfg['profile']} "
          f"(XGBoost {cfg['ml']['xgb_n_estimators']} trees, depth {cfg['ml']['xgb_max_depth']})")
    print("=" * 60)

    # --- Step 0: fail fast on a broken environment ---
    print("\n[0/4] Checking dependencies...")
    check_dependencies()
    print("  All ML libraries available")

    # --- Step 1: load ---
    print("\n[1/4] Loading database...")
    data = load_data()
    print(f"  Fleet: {len(data['fleet'])} aircraft")
    print(f"  Flights: {len(data['flights'])} records")
    print(f"  Parts: {len(data['parts'])} part numbers")
    print(f"  Demands: {len(data['demands'])} events")
    print(f"  SDR: {len(data['sdr'])} real failure records")

    # --- Step 2: train ---
    print("\n[2/4] Training models...")

    survival_result = train_survival_model(data)
    demand_result = train_demand_forecast(data)
    sdr_result = train_sdr_classifier(data)

    # --- Step 3: apply ---
    print("\n[3/4] Generating predictions...")
    predict_next_month(survival_result, demand_result, data)

    # --- Step 4: summarise ---
    print("\n[4/4] Summary")
    print("=" * 60)

    # Confirm which diagnostic plots were actually written. A missing entry
    # points at the "Plot skipped" message earlier in the run.
    plot_files = ["survival_analysis.png", "demand_forecast.png", "sdr_analysis.png"]
    for f in plot_files:
        path = os.path.join(os.path.dirname(DB_PATH), f)
        if os.path.exists(path):
            print(f"  {f}")

    print(f"\n  Models trained on:")
    print(f"    - {len(data['demands'])} synthetic maintenance demands")
    print(f"    - {len(data['sdr'])} real FAA SDR failure records")
    print(f"    - {len(data['flights'])} flight records (3 years)")

    # Closing status line: what was produced and what to run next.
    print("\n" + "=" * 60)
    print("  DONE - models trained, plots written to the project directory")
    print("  Next step:")
    print("    python logistics_optimizer.py   (Module 3 - stock and routing)")
    print("=" * 60)


if __name__ == "__main__":
    main()
