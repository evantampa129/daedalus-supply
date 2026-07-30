"""
Daedalus Supply AI - Runtime Configuration
==========================================
Hardware auto-detection and adaptive tuning layer.

This module is imported by every compute-heavy module in the system
(data_pipeline, prediction_model, logistics_optimizer) so that a single
codebase runs unchanged on hardware ranging from a 2 GB / 2-core office
terminal up to a dedicated 16 GB+ server. Instead of hardcoding model
hyperparameters and I/O chunk sizes, each module asks get_config() for the
values appropriate to the machine it is currently running on.

Design rationale: continuing-airworthiness tooling frequently has to run on
whatever equipment the operator already owns - often long-lived, low-spec
workstations inside a restricted network. Degrading gracefully (fewer trees,
smaller chunks, sampled data) is preferable to failing to run at all.

Profiles:
    MINIMAL  - 2 GB RAM,  2 cores  (old office PCs, embedded/edge systems)
    STANDARD - 8 GB RAM,  4 cores  (typical engineering workstation)
    FULL     - 16 GB+ RAM, 8+ cores (dedicated server)

Author: Evangelos Tampachaniotis
Version: 1.0.0
License: MIT

Regulatory Framework:
    - EASA Part-M (EU 1321/2014) - Continuing Airworthiness. The reliability
      programme required by M.A.302 must be reproducible; profiles are
      therefore explicit and overridable rather than silently auto-tuned.
"""

import os
import platform
import multiprocessing


def detect_hardware():
    """
    Inspect the host machine and classify it into a capability profile.

    Args:
        (none)

    Returns:
        dict with keys:
            profile : str - one of "MINIMAL" / "STANDARD" / "FULL"
            ram_gb  : float - total physical RAM in gigabytes (1 decimal)
            cores   : int - logical CPU count reported by the OS
            os      : str - platform name, e.g. "Linux" / "Windows" / "Darwin"
            arch    : str - machine architecture, e.g. "x86_64" / "aarch64"

    Notes:
        RAM is read from /proc/meminfo rather than via psutil to avoid adding
        a third-party dependency purely for one number - keeping the install
        footprint small matters on locked-down operator networks.
    """
    # --- Physical memory detection ---
    # /proc/meminfo line 1 is "MemTotal:  <kB> kB" on every Linux kernel.
    # Field [1] is the value in kilobytes; divide by 1024^2 to reach GB.
    try:
        with open("/proc/meminfo") as f:
            mem_kb = int(f.readline().split()[1])
            ram_gb = mem_kb / (1024 * 1024)  # kB -> GB
    except (FileNotFoundError, ValueError):
        # FileNotFoundError: non-Linux host (Windows/macOS have no /proc).
        # ValueError: unexpected /proc format (containers, exotic kernels).
        # Either way we must not crash - the whole point of this module is to
        # keep the system runnable. Assume a mid-range machine (8 GB), which
        # maps to STANDARD and is safe on almost any modern desktop.
        ram_gb = 8

    # Logical core count - includes SMT/hyperthreads. Used both for profile
    # selection and to derive the n_jobs parallelism setting below.
    cores = multiprocessing.cpu_count()

    # --- Profile classification ---
    # Thresholds are deliberately conservative: a machine is downgraded to the
    # lower profile if EITHER RAM or core count is constrained, because the
    # binding constraint in this pipeline alternates between the two (SDR CSV
    # parsing is memory-bound, XGBoost training is CPU-bound).
    if ram_gb < 4 or cores <= 2:
        # Below 4 GB the 195k-row SDR frame plus a pandas copy will thrash swap.
        profile = "MINIMAL"
    elif ram_gb < 12 or cores <= 4:
        # Comfortable for the full dataset, but not for 500-tree ensembles.
        profile = "STANDARD"
    else:
        # Headroom for full-depth models and unrestricted parallelism.
        profile = "FULL"

    return {
        "profile": profile,
        "ram_gb": round(ram_gb, 1),
        "cores": cores,
        "os": platform.system(),
        "arch": platform.machine(),
    }


# ============================================================================
# ML / I-O PARAMETER TABLE
# ============================================================================
# One entry per hardware profile. Every downstream module reads its tuning
# knobs from here rather than hardcoding them, so behaviour on a constrained
# machine is a single-table change instead of a code change.
ML_PROFILES = {
    "MINIMAL": {
        # 50 trees is roughly the point where XGBoost validation error on the
        # SDR classifier plateaus for a shallow model - more trees cost time
        # without accuracy on 2 cores.
        "xgb_n_estimators": 50,
        # depth 4 => at most 2^4 = 16 leaves per tree. Keeps the booster small
        # in memory and strongly regularised given the reduced sample.
        "xgb_max_depth": 4,
        # 5,000 rows per CSV chunk - approximately 5 MB resident per chunk for
        # the SDR schema, which fits alongside pandas overhead inside 2 GB.
        "chunk_size": 5000,
        # Hard cap on SDR rows retained for training. 50k rows preserves the
        # ATA-chapter class distribution while cutting peak RAM ~4x.
        "max_sdr_records": 50000,
        # Single-threaded: on a 2-core box the second core must stay free for
        # the OS, and thread contention costs more than it gains.
        "n_jobs": 1,
        # 72 DPI - screen-resolution plots only; keeps PNG files small.
        "plot_dpi": 72,
    },
    "STANDARD": {
        # 200 trees - the default operating point; matches the accuracy quoted
        # in the README (87.6% on the SDR classifier).
        "xgb_n_estimators": 200,
        # depth 6 is XGBoost's own default and captures ATA-chapter x
        # part-name interactions without overfitting 195k rows.
        "xgb_max_depth": 6,
        # 50,000 rows per chunk - ~10 chunks for a 500k-row SDR file.
        "chunk_size": 50000,
        # None = no cap; use the full SDR corpus.
        "max_sdr_records": None,
        # Two worker threads, leaving cores free for pandas and the OS.
        "n_jobs": 2,
        # 150 DPI - legible when embedded in a report or printed A4.
        "plot_dpi": 150,
    },
    "FULL": {
        # 500 trees with a low learning rate for the best achievable fit.
        "xgb_n_estimators": 500,
        # depth 8 => up to 256 leaves; only safe with the full record count.
        "xgb_max_depth": 8,
        # 100,000 rows per chunk - fewest passes over the CSV files.
        "chunk_size": 100000,
        "max_sdr_records": None,
        # -1 tells XGBoost/scikit-learn to use every available core.
        "n_jobs": -1,
        # 200 DPI - publication quality for reliability reports.
        "plot_dpi": 200,
    },
}


def get_config():
    """
    Build the effective runtime configuration for this process.

    Args:
        (none)

    Returns:
        dict with keys:
            hardware : dict - raw detection result from detect_hardware()
            profile  : str  - the profile actually in force after any override
            ml       : dict - the ML_PROFILES entry for that profile

    Notes:
        The DAEDALUS_PROFILE environment variable overrides auto-detection.
        This exists for two reasons: (1) reproducing a colleague's results on
        different hardware, and (2) deliberately stress-testing the MINIMAL
        path on a large machine before deploying to a small one.

        Example: DAEDALUS_PROFILE=MINIMAL python prediction_model.py
    """
    hw = detect_hardware()

    # Environment override takes precedence over detection; falls back to the
    # detected profile when the variable is unset.
    profile = os.environ.get("DAEDALUS_PROFILE", hw["profile"])

    # Look up the parameter block. A typo in DAEDALUS_PROFILE raises KeyError
    # here - deliberately loud, because silently running with wrong-sized
    # models would invalidate any reliability figures produced downstream.
    params = ML_PROFILES[profile]

    return {"hardware": hw, "profile": profile, "ml": params}


# ============================================================================
# CLI ENTRY POINT
# ============================================================================
# Running `python config.py` prints what the system detected. This is the
# first diagnostic step when a module runs unexpectedly slowly or runs out of
# memory on an unfamiliar machine.
if __name__ == "__main__":
    cfg = get_config()
    print(f"Hardware: {cfg['hardware']['ram_gb']}GB RAM, {cfg['hardware']['cores']} cores, "
          f"{cfg['hardware']['os']} {cfg['hardware']['arch']}")
    print(f"Profile: {cfg['profile']}")
    print(f"XGBoost: {cfg['ml']['xgb_n_estimators']} trees, depth {cfg['ml']['xgb_max_depth']}")
    # `or 'all'` renders the None sentinel as a human-readable word.
    print(f"SDR limit: {cfg['ml']['max_sdr_records'] or 'all'}")
    print(f"Parallel jobs: {cfg['ml']['n_jobs']}")
    print(f"\nOverride with: DAEDALUS_PROFILE=MINIMAL python prediction_model.py")
