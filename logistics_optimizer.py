"""
Daedalus Supply AI - Module 3: Logistics Optimizer
==================================================
Turns the reliability picture from Module 2 into supply-chain decisions.

Three optimisation engines:

1. STOCK LEVEL OPTIMIZATION
   Computes min / reorder point / max per part per station from observed demand
   statistics and procurement lead times, with the service-level target set by
   dispatch criticality. Balances stockout risk against holding cost.

2. PRE-POSITIONING ENGINE
   Rebalances the network before a shortage bites: identifies stations holding
   surplus and stations sitting below minimum, and pairs them by transit time,
   prioritised by criticality. Moving a part between stations is always cheaper
   than an AOG, so the transfer is worth making pre-emptively.

3. AOG RESPONSE ROUTER
   Given a grounded aircraft and a required part, ranks every source - local
   stores, transfer from another station, expedited supplier order - by total
   cost, where total cost is the part logistics cost PLUS the grounding cost
   accumulated while waiting.

The governing economics: an Aircraft on Ground costs roughly €15,000 per hour
in lost revenue, disruption and passenger re-accommodation. That figure dwarfs
every logistics cost in this module, which is why the optimiser consistently
prefers speed over freight economy for AOG-critical items - and why holding
inventory that looks expensive is usually the cheaper policy.

Author: Evangelos Tampachaniotis
Version: 1.0.0
License: MIT

Regulatory Framework:
    - EASA Part-145 - Maintenance Organisation Approval. 145.A.42 component
      acceptance and classification, and the stores requirements that make
      "serviceable stock" the only quantity an optimiser may count as available.
    - EASA Part-M (EU 1321/2014) M.A.301 - continuing-airworthiness tasks;
      material availability is a precondition for completing them on time.
    - MEL (Minimum Equipment List) - defines what may be deferred and for how
      long, which is exactly what the criticality-driven service levels encode.
    - ICAO Annex 6 - operator responsibility for maintaining airworthiness.

Usage:
    python logistics_optimizer.py
"""

import sqlite3
import os
import sys
import warnings

# Suppress library chatter; this module reports its own diagnostics.
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from scipy import stats

# Database path resolved relative to this file. Name retained for backwards
# compatibility with databases built by earlier versions of the pipeline.
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aerosupply.db")

# ----------------------------------------------------------------------------
# Station-to-station transit times, in HOURS, door to door.
# These are not flight times: they include collection, documentation, the next
# available flight or ferry, and delivery to the receiving store. That is why
# the island pairs are long - RHO to CFU at 14 hours has no direct connection
# and routes via the mainland.
# Only one direction of each pair is listed; the loop below mirrors them.
# ----------------------------------------------------------------------------
TRANSFER_TIMES = {
    ("ATH", "SKG"): 6, ("ATH", "HER"): 5, ("ATH", "RHO"): 8,
    ("ATH", "CFU"): 6, ("SKG", "HER"): 10, ("SKG", "RHO"): 12,
    ("SKG", "CFU"): 8, ("HER", "RHO"): 4, ("HER", "CFU"): 12,
    ("RHO", "CFU"): 14,
}

# Complete the matrix: transit time is symmetric, and moving a part within the
# station it already sits in costs zero hours. Building the full matrix here
# means every lookup downstream is a single dict access with no special cases.
# list() is taken first because the dict is mutated inside the loop.
for (a, b), t in list(TRANSFER_TIMES.items()):
    TRANSFER_TIMES[(b, a)] = t
    TRANSFER_TIMES[(a, a)] = 0
    TRANSFER_TIMES[(b, b)] = 0

# ----------------------------------------------------------------------------
# Cost model parameters
# ----------------------------------------------------------------------------

# EUR per hour of Aircraft on Ground. Industry figures for a narrow-body range
# from roughly €10,000 to €20,000 per hour once lost revenue, crew, passenger
# re-accommodation and downstream schedule disruption are counted. €15,000 is a
# mid-range planning value and the single most influential constant here: it is
# what makes a €500 expedited shipment obviously worth paying.
AOG_COST_PER_HOUR = 15000

# EUR per kilogram for inter-station shipping. Retained for weight-based
# costing; the current routing model prices transfers by transit time instead,
# since handling and connection availability dominate over mass for the small,
# dense components involved.
TRANSFER_COST_PER_KG = 2.5

# Annual inventory holding cost as a fraction of part value. 15% is the
# standard planning figure covering cost of capital, storage, insurance,
# obsolescence and periodic serviceability checks on stored rotables.
HOLDING_COST_RATE = 0.15

# Multiplier applied to a part's unit cost for an expedited AOG order.
# 3x reflects premium freight, out-of-hours handling and supplier surcharges.
EMERGENCY_ORDER_MULTIPLIER = 3.0


def load_data():
    """
    Load every table the optimisation engines need.

    Args:
        (none)

    Returns:
        dict of str -> pd.DataFrame, keyed by logical table name.

    Notes:
        All three engines read the same tables, so a single load pass avoids
        repeated queries and guarantees they operate on a consistent snapshot
        of stock levels.
    """
    conn = sqlite3.connect(DB_PATH)
    data = {
        "fleet": pd.read_sql("SELECT * FROM fleet", conn),
        "parts": pd.read_sql("SELECT * FROM parts_catalog", conn),
        "demands": pd.read_sql("SELECT * FROM part_demands", conn),
        "inventory": pd.read_sql("SELECT * FROM inventory", conn),
        "stations": pd.read_sql("SELECT * FROM stations", conn),
        "work_orders": pd.read_sql("SELECT * FROM work_orders", conn),
    }
    conn.close()
    return data


# ============================================================================
# ENGINE 1: STOCK LEVEL OPTIMIZATION
# ============================================================================

def optimize_stock_levels(data):
    """
    Derive optimal min / reorder point / max stock per part per station.

    Args:
        data: dict - the table dictionary returned by load_data().

    Returns:
        pd.DataFrame - one row per (part, station) with the recommended levels,
        the demand statistics they were derived from, and the resulting annual
        holding cost.

    Notes:
        Standard continuous-review (s, S) inventory policy:

            safety_stock  = z * sigma * sqrt(lead_time)
            reorder_point = mean_demand * lead_time + safety_stock
            max_stock     = reorder_point + EOQ-equivalent order quantity

        The safety-stock formula is the classical result for normally
        distributed demand over a fixed lead time: the sqrt(lead_time) term
        appears because the VARIANCE of demand accumulates linearly with time,
        so the standard deviation grows with its square root. Holding a full
        lead_time * sigma buffer would over-stock every line item in the network.

        z is the service-level factor - the number of standard deviations of
        buffer needed to achieve a given probability of not stocking out during
        the lead time. It is set by dispatch criticality, not by cost, which is
        the airworthiness-driven part of the model.
    """
    print("\n  [Engine 1] Stock Level Optimization")
    print("  " + "-" * 55)

    demands = data["demands"]
    parts = data["parts"]
    inventory = data["inventory"]
    stations = data["stations"]

    # --- Service-level targets by dispatch criticality ---
    # These encode the MEL consequence of a stockout:
    #   AOG     99.5% - a stockout grounds the aircraft. Effectively no
    #                   tolerance; 1 stockout per 200 replenishment cycles.
    #   MEL     95%   - dispatch may continue under MEL conditions with a
    #                   repair interval, so a short stockout is survivable.
    #   ROUTINE 90%   - no dispatch impact; an occasional stockout is the
    #                   economically correct outcome rather than a failure.
    service_levels = {
        "AOG": 0.995,
        "MEL": 0.95,
        "ROUTINE": 0.90,
    }

    # Convert each service level into a z-score via the inverse normal CDF
    # (percent point function). 99.5% -> z = 2.576, 95% -> 1.645, 90% -> 1.282.
    z_scores = {k: stats.norm.ppf(v) for k, v in service_levels.items()}

    # --- Build monthly demand statistics per part per station ---
    # Monthly buckets match the procurement planning cycle and smooth the
    # intermittency that makes daily demand statistics useless.
    demands["demand_date"] = pd.to_datetime(demands["demand_date"])
    demands["year_month"] = demands["demand_date"].dt.to_period("M")

    monthly_demand = demands.groupby(
        ["part_number", "station", "year_month"]
    )["quantity_required"].sum().reset_index()

    # Aggregate across months to get the mean and variability each line item
    # actually exhibits - the two inputs the (s, S) policy needs.
    demand_stats = monthly_demand.groupby(
        ["part_number", "station"]
    )["quantity_required"].agg(
        mean_monthly="mean",
        std_monthly="std",
        max_monthly="max",
        total="sum",
        n_months="count",
    ).reset_index()

    # A part with demand in exactly one month has undefined std (pandas returns
    # NaN). Treat it as zero variability: with a single observation there is no
    # evidence of variation, so the buffer reduces to the min(1) floor applied
    # below rather than propagating NaN through the arithmetic.
    demand_stats["std_monthly"] = demand_stats["std_monthly"].fillna(0)

    # Attach lead times, cost and criticality from the catalogue.
    demand_stats = demand_stats.merge(
        parts[["part_number", "criticality", "lead_time_days_normal",
               "lead_time_days_aog", "unit_cost_eur", "part_class"]],
        on="part_number", how="left"
    )

    results = []

    # Compute the policy for each (part, station) line item independently:
    # the same part legitimately warrants different levels at the hub and at an
    # outstation, because demand rate and variability differ.
    for _, row in demand_stats.iterrows():
        crit = row["criticality"]

        # Default z = 1.28 (90%) for any unrecognised criticality - the
        # conservative choice is the LOWEST service level, so an unclassified
        # part cannot silently consume AOG-grade stock.
        z = z_scores.get(crit, 1.28)

        # Convert lead time from days to months to match the demand statistics.
        # The 14-day fallback applies when the catalogue has no lead time; it is
        # the midpoint of the rotable 5-30 day range.
        lead_time_months = (row["lead_time_days_normal"] or 14) / 30

        # --- Safety stock ---
        #   SS = z * sigma_monthly * sqrt(lead_time_months)
        # ceil() because parts are issued in whole units and rounding down
        # would systematically under-protect every line item.
        # max(1, ...) enforces a floor: even a perfectly steady-demand part
        # keeps one unit of buffer, since zero buffer means the first
        # unexpected demand is an immediate stockout.
        safety_stock = max(1, int(np.ceil(
            z * row["std_monthly"] * np.sqrt(lead_time_months)
        )))

        # --- Reorder point (ROP) ---
        #   ROP = expected demand during the lead time + safety stock
        # Ordering when stock falls to this level means the replenishment
        # arrives just as the safety stock would otherwise be consumed.
        reorder_point = max(1, int(np.ceil(
            row["mean_monthly"] * lead_time_months + safety_stock
        )))

        # --- Maximum stock level ---
        # ROP plus two months of mean demand. The two-month order quantity is
        # the EOQ-equivalent used here: it keeps ordering frequency sensible
        # without accumulating capital in slow-moving items.
        max_stock = reorder_point + max(1, int(np.ceil(row["mean_monthly"] * 2)))

        # Minimum stock is the safety stock - the level below which the line
        # item is formally in shortage.
        min_stock = safety_stock

        # --- Annual holding cost ---
        #   cost = max_stock * unit_cost * HOLDING_COST_RATE
        # Priced at the maximum level (the worst case just after replenishment)
        # so the figure is an upper bound on the capital tied up.
        holding_cost = max_stock * (row["unit_cost_eur"] or 0) * HOLDING_COST_RATE

        results.append({
            "part_number": row["part_number"],
            "station": row["station"],
            "criticality": crit,
            "part_class": row["part_class"],
            "mean_monthly_demand": round(row["mean_monthly"], 1),
            "std_monthly_demand": round(row["std_monthly"], 1),
            "optimal_min_stock": min_stock,
            "optimal_reorder_point": reorder_point,
            "optimal_max_stock": max_stock,
            "annual_holding_cost_eur": round(holding_cost, 0),
        })

    results_df = pd.DataFrame(results)

    # --- Compare the recommendation against what is actually held ---
    inv = data["inventory"].copy()
    inv_cols = ["part_number", "station"]

    # Schema compatibility: SQLite names the column `serviceable`, the
    # PostgreSQL/MySQL schemas name it `quantity_serviceable`. Detect which is
    # present so this module works against either backend unchanged.
    stock_col = "serviceable" if "serviceable" in inv.columns else "quantity_serviceable"
    min_col = "minimum_stock_level"

    comparison = results_df.merge(
        inv[inv_cols + [stock_col, min_col]],
        on=inv_cols, how="left"
    )
    comparison = comparison.rename(columns={stock_col: "current_stock", min_col: "current_min"})

    # Gap between what is held and what the policy calls for. Negative means
    # short. fillna(0) treats "no inventory record at this station" as zero
    # stock, which is the correct interpretation for availability purposes.
    comparison["stock_delta"] = comparison["current_stock"].fillna(0) - comparison["optimal_reorder_point"]

    # Understocked: any negative gap, most severe first.
    understocked = comparison[comparison["stock_delta"] < 0].sort_values("stock_delta")

    # Overstocked: threshold at +10 units rather than +1, because small
    # positive gaps are normal and healthy just after a replenishment. Only a
    # sustained excess represents capital worth releasing.
    overstocked = comparison[comparison["stock_delta"] > 10].sort_values("stock_delta", ascending=False)

    print(f"\n  Analyzed {len(results_df)} part-station combinations")
    print(f"\n  TOP 10 UNDERSTOCKED (need more):")
    print(f"  {'Part':<20} {'Station':<8} {'Crit':<8} {'Current':>8} {'Optimal':>8} {'Shortage':>8}")
    print(f"  {'-' * 65}")

    # Print the ten worst shortages - the immediate procurement work list.
    for _, r in understocked.head(10).iterrows():
        print(f"  {str(r['part_number'])[:20]:<20} {r['station']:<8} {r['criticality']:<8} "
              f"{r['current_stock']:>8.0f} {r['optimal_reorder_point']:>8} {r['stock_delta']:>8.0f}")

    # Overstock is only reported when it exists; an empty section would be noise.
    if len(overstocked) > 0:
        print(f"\n  TOP 5 OVERSTOCKED (can reduce):")
        print(f"  {'Part':<20} {'Station':<8} {'Current':>8} {'Optimal':>8} {'Excess':>8} {'Saving EUR':>12}")
        print(f"  {'-' * 65}")
        for _, r in overstocked.head(5).iterrows():
            # Annual saving from releasing the excess:
            #   excess_units * (annual holding cost per unit)
            # where per-unit cost is the line's total holding cost divided by
            # its max level. max(1, ...) guards against division by zero for a
            # line whose optimal maximum computed to zero.
            excess_cost = r["stock_delta"] * (r.get("annual_holding_cost_eur", 0) / max(1, r["optimal_max_stock"]))
            print(f"  {str(r['part_number'])[:20]:<20} {r['station']:<8} "
                  f"{r['current_stock']:>8.0f} {r['optimal_reorder_point']:>8} "
                  f"{r['stock_delta']:>8.0f} {excess_cost:>12,.0f}")

    # Total cost of carrying the recommended network-wide inventory.
    total_holding = results_df["annual_holding_cost_eur"].sum()
    print(f"\n  Network annual holding cost: EUR {total_holding:,.0f}")

    return results_df


# ============================================================================
# ENGINE 2: PRE-POSITIONING ENGINE
# ============================================================================

def preposition_parts(data, stock_recommendations):
    """
    Recommend proactive inter-station transfers to close shortages.

    Args:
        data: dict - the table dictionary returned by load_data().
        stock_recommendations: pd.DataFrame - output of optimize_stock_levels().
            Retained for interface symmetry and planned use in v1.1, where
            transfers will be driven by predicted rather than current shortfall.

    Returns:
        pd.DataFrame - one row per recommended transfer, ordered by criticality
        then transit time. Empty frame when the network is already balanced.

    Notes:
        Matching logic, per part:
            - find each station holding less than its minimum (a deficit)
            - find every station holding more than its minimum (a surplus)
            - pair the deficit with the FASTEST reachable surplus

        Speed is the objective rather than cost because the alternative to a
        transfer is a potential AOG at €15,000/hour, against which every
        plausible freight cost is negligible. Optimising for cheapest routing
        would be optimising the wrong variable by three orders of magnitude.

        Surplus is measured strictly above the source station's OWN minimum, so
        a transfer can never solve one shortage by creating another.
    """
    print("\n  [Engine 2] Pre-Positioning Engine")
    print("  " + "-" * 55)

    parts = data["parts"]
    inventory = data["inventory"]
    fleet = data["fleet"]

    # Backend schema compatibility - see Engine 1.
    stock_col = "serviceable" if "serviceable" in inventory.columns else "quantity_serviceable"

    transfers = []

    # Evaluate each part independently: stock is only fungible within a part
    # number, so there is no cross-part optimisation to perform.
    for part_number in parts["part_number"].unique():
        part_info = parts[parts["part_number"] == part_number].iloc[0]
        part_inv = inventory[inventory["part_number"] == part_number].copy()

        # No inventory records for this part anywhere in the network.
        if part_inv.empty:
            continue

        # Examine every station as a potential DESTINATION.
        for _, deficit_row in part_inv.iterrows():
            deficit_station = deficit_row["station"]
            current = deficit_row[stock_col]
            minimum = deficit_row["minimum_stock_level"]

            # At or above minimum - no shortage, nothing to fix here.
            if current >= minimum:
                continue

            # Units required to restore the station to its minimum level.
            shortage = minimum - current

            # Search for the fastest viable SOURCE station.
            best_source = None
            best_time = float("inf")

            for _, surplus_row in part_inv.iterrows():
                surplus_station = surplus_row["station"]

                # A station cannot supply itself.
                if surplus_station == deficit_station:
                    continue

                # Only stock ABOVE the source's own minimum is transferable -
                # anything below that is the source station's own protection.
                surplus = surplus_row[stock_col] - surplus_row["minimum_stock_level"]
                if surplus <= 0:
                    continue

                # 24-hour default for any station pair not in the matrix: a
                # pessimistic full-day assumption, so an unknown route is never
                # preferred over a known one.
                transfer_time = TRANSFER_TIMES.get(
                    (surplus_station, deficit_station), 24
                )

                # Keep the fastest source found so far.
                if transfer_time < best_time:
                    best_time = transfer_time
                    best_source = {
                        "part_number": part_number,
                        "description": part_info["description"],
                        "criticality": part_info["criticality"],
                        "from_station": surplus_station,
                        "to_station": deficit_station,
                        # Move the smaller of what is needed and what is
                        # genuinely spare - never strip the source below its
                        # own minimum to satisfy the destination.
                        "quantity": min(shortage, int(surplus)),
                        "transfer_hours": transfer_time,
                        "reason": "BELOW_MINIMUM",
                    }

            # A shortage with no surplus anywhere in the network cannot be
            # solved by a transfer; it requires a supplier order instead, which
            # Engine 1's understocked list already surfaces.
            if best_source:
                transfers.append(best_source)

    transfers_df = pd.DataFrame(transfers)

    if transfers_df.empty:
        print("  No transfers needed - stock is balanced")
        return transfers_df

    # --- Prioritisation ---
    # Criticality first, transit time second. An AOG-critical part that takes
    # 14 hours to move outranks a routine part that arrives in 4, because the
    # consequence of NOT moving it is three orders of magnitude larger.
    crit_order = {"AOG": 0, "MEL": 1, "ROUTINE": 2}
    transfers_df["crit_rank"] = transfers_df["criticality"].map(crit_order)
    transfers_df = transfers_df.sort_values(["crit_rank", "transfer_hours"])

    print(f"\n  Recommended transfers: {len(transfers_df)}")
    print(f"\n  {'Part':<25} {'Crit':<6} {'From':<5} {'To':<5} {'Qty':>4} {'ETA hrs':>8}")
    print(f"  {'-' * 60}")

    # Top 15 - the actionable movements list for the current planning cycle.
    for _, t in transfers_df.head(15).iterrows():
        print(f"  {str(t['description'])[:25]:<25} {t['criticality']:<6} "
              f"{t['from_station']:<5} {t['to_station']:<5} {t['quantity']:>4} "
              f"{t['transfer_hours']:>8}h")

    # --- Exposure view ---
    # Translate stock shortages into the aircraft they actually threaten: a
    # shortage only matters if there is an airframe based where it exists.
    shortage_stations = transfers_df["to_station"].unique()
    at_risk = fleet[fleet["home_base"].isin(shortage_stations)]
    if not at_risk.empty:
        print(f"\n  Aircraft at stations with shortages:")
        for _, ac in at_risk.iterrows():
            print(f"    {ac['tail_number']} - {ac['home_base']} ({ac['primary_role']})")

    return transfers_df


# ============================================================================
# ENGINE 3: AOG RESPONSE ROUTER
# ============================================================================

def route_aog_request(data, tail_number, part_number):
    """
    Rank every available source for a part needed by a grounded aircraft.

    Args:
        data: dict - the table dictionary returned by load_data().
        tail_number: str - registration of the grounded aircraft, e.g. "SX-ABK".
        part_number: str - the required part, e.g. "AES-24-10-001".

    Returns:
        dict describing the request, the recommended option and every option
        considered; or {"error": ...} when the aircraft or part is unknown.

    Notes:
        Three source types are evaluated:
            1. LOCAL_STOCK      - already at the aircraft's station
            2. STATION_TRANSFER - held elsewhere in the network
            3. EMERGENCY_ORDER  - expedited supplier order, always available

        Options are ranked by ETA, not by cash cost, because grounding cost
        dominates: at €15,000/hour, one hour saved is worth more than any
        realistic difference in freight or premium. The full cost breakdown is
        still computed and returned so the recommendation can be justified:

            total_cost = logistics_cost + (eta_hours * AOG_COST_PER_HOUR)
    """
    fleet = data["fleet"]
    parts = data["parts"]
    inventory = data["inventory"]

    # Backend schema compatibility - see Engine 1.
    stock_col = "serviceable" if "serviceable" in inventory.columns else "quantity_serviceable"

    # Resolve the aircraft. An unknown registration is a caller error, returned
    # as data rather than raised so a scenario batch can continue.
    ac = fleet[fleet["tail_number"] == tail_number]
    if ac.empty:
        return {"error": f"Aircraft {tail_number} not found"}
    ac = ac.iloc[0]

    # Resolve the part.
    part = parts[parts["part_number"] == part_number]
    if part.empty:
        return {"error": f"Part {part_number} not found"}
    part = part.iloc[0]

    ac_station = ac["home_base"]
    options = []

    # --- Option 1: local stores ---
    # Only SERVICEABLE stock counts. Under Part-145 145.A.42 an unserviceable
    # unit awaiting shop input is physically present but may not be fitted, so
    # counting it here would produce a recommendation that cannot be executed.
    local_inv = inventory[
        (inventory["part_number"] == part_number) &
        (inventory["station"] == ac_station)
    ]
    if not local_inv.empty and local_inv.iloc[0][stock_col] > 0:
        options.append({
            "option": "LOCAL_STOCK",
            "source": ac_station,
            # 0.5 h - not zero: the part still has to be drawn from stores,
            # its EASA Form 1 verified and it moved to the aircraft.
            "eta_hours": 0.5,
            # No incremental logistics cost; the part is already paid for and
            # already here.
            "cost_eur": 0,
            "quantity_available": int(local_inv.iloc[0][stock_col]),
            "description": f"Available in {ac_station} warehouse",
        })

    # --- Option 2: transfer from any other station holding the part ---
    for _, inv_row in inventory[inventory["part_number"] == part_number].iterrows():
        station = inv_row["station"]

        # Skip the aircraft's own station (already handled as Option 1) and any
        # station with no serviceable stock.
        if station == ac_station or inv_row[stock_col] <= 0:
            continue

        # 24-hour pessimistic default for unlisted station pairs.
        transfer_time = TRANSFER_TIMES.get((station, ac_station), 24)

        # Simplified transfer cost: EUR 50 per transit hour, covering courier
        # charges, handling and documentation. Deliberately coarse - at
        # €15,000/hour of grounding, transfer cost never changes the ranking,
        # and a precise freight model would add complexity that cannot alter
        # any decision this router makes.
        transfer_cost = transfer_time * 50

        options.append({
            "option": "STATION_TRANSFER",
            "source": station,
            "eta_hours": transfer_time,
            "cost_eur": transfer_cost,
            "quantity_available": int(inv_row[stock_col]),
            "description": f"Transfer from {station} ({transfer_time}h)",
        })

    # --- Option 3: expedited supplier order ---
    # Always appended, so the router can never return an empty option list: a
    # supplier order is available even when the whole network is out of stock.
    # The `or` guards cover both a missing key and a stored NULL.
    aog_lead = part.get("lead_time_days_aog", 2) or 2       # days, default 2
    normal_cost = part.get("unit_cost_eur", 1000) or 1000   # EUR, default 1000

    options.append({
        "option": "EMERGENCY_ORDER",
        "source": "SUPPLIER",
        # Convert the supplier's AOG lead time from days to hours.
        "eta_hours": aog_lead * 24,
        # Premium pricing for expedited supply - see EMERGENCY_ORDER_MULTIPLIER.
        "cost_eur": normal_cost * EMERGENCY_ORDER_MULTIPLIER,
        # Sentinel: supplier availability is effectively unbounded, unlike a
        # station's finite shelf stock.
        "quantity_available": 999,
        "description": f"AOG order from supplier ({aog_lead} days)",
    })

    # Rank by time to availability - the variable that actually drives cost.
    options.sort(key=lambda x: x["eta_hours"])

    # Attach the economics to every option so the recommendation is auditable.
    for opt in options:
        # Grounding cost accumulated while waiting for this option.
        opt["aog_cost_eur"] = opt["eta_hours"] * AOG_COST_PER_HOUR
        # Full cost of choosing it: logistics plus grounding.
        opt["total_cost_eur"] = opt["cost_eur"] + opt["aog_cost_eur"]

    return {
        "aircraft": tail_number,
        "station": ac_station,
        "part_number": part_number,
        "part_description": part["description"],
        "criticality": part["criticality"],
        # Fastest option, i.e. the cheapest once grounding cost is counted.
        "recommended": options[0],
        "all_options": options,
    }


def run_aog_scenarios(data):
    """
    Demonstrate the AOG router across a representative set of scenarios.

    Args:
        data: dict - the table dictionary returned by load_data().

    Returns:
        None - prints each scenario's recommendation and alternatives.

    Notes:
        The five scenarios are chosen to span the decision space: an expensive
        rotable at a remote island base, a critical system failure at a station
        with no local stock, a heavy component at a secondary base, a very
        high-value item at the hub, and a trivial expendable - which should
        resolve from local stock and cost effectively nothing.
    """
    print("\n  [Engine 3] AOG Response Router")
    print("  " + "-" * 55)

    scenarios = [
        ("SX-ABK", "AES-24-10-001", "IDG failure in Heraklion"),
        ("SX-ABM", "AES-29-10-001", "Hydraulic pump failure in Rhodes"),
        ("SX-ABG", "AES-32-50-001", "Brake assembly needed in Thessaloniki"),
        ("SX-ABO", "AES-72-50-001", "FADEC failure in Athens"),
        ("SX-ABL", "AES-33-40-010", "Landing light in Heraklion"),
    ]

    # Run each scenario through the router and print the decision.
    for tail, part, scenario in scenarios:
        print(f"\n  SCENARIO: {scenario}")
        print(f"  Aircraft: {tail}, Part: {part}")

        result = route_aog_request(data, tail, part)

        # A scenario referencing an aircraft or part missing from the database
        # is reported and skipped, so one bad row cannot abort the batch.
        if "error" in result:
            print(f"  Error: {result['error']}")
            continue

        rec = result["recommended"]
        print(f"  RECOMMENDED: {rec['description']}")
        print(f"    ETA: {rec['eta_hours']}h | Part cost: EUR {rec['cost_eur']:,.0f} | "
              f"AOG cost: EUR {rec['aog_cost_eur']:,.0f} | Total: EUR {rec['total_cost_eur']:,.0f}")

        # Show the next two alternatives so the cost of the runner-up is
        # visible - that gap is the value the recommendation delivers.
        if len(result["all_options"]) > 1:
            print(f"    Alternatives:")
            for opt in result["all_options"][1:3]:
                print(f"      {opt['description']} - ETA: {opt['eta_hours']}h, "
                      f"Total: EUR {opt['total_cost_eur']:,.0f}")


# ============================================================================
# NETWORK SUMMARY
# ============================================================================

def network_summary(data, stock_recommendations):
    """
    Report overall supply-chain health across the station network.

    Args:
        data: dict - the table dictionary returned by load_data().
        stock_recommendations: pd.DataFrame - output of optimize_stock_levels(),
            available for cross-referencing in future revisions.

    Returns:
        None - prints the summary.

    Notes:
        Four views, each answering a different management question:
            inventory value per station - where is the capital?
            stockout risk               - how exposed are we right now?
            demand statistics           - what does the network consume?
            inventory per aircraft      - is coverage proportionate to fleet?
    """
    print("\n  " + "=" * 55)
    print("  SUPPLY CHAIN NETWORK SUMMARY")
    print("  " + "=" * 55)

    inventory = data["inventory"]
    parts = data["parts"]
    demands = data["demands"]

    # Backend schema compatibility - see Engine 1.
    stock_col = "serviceable" if "serviceable" in inventory.columns else "quantity_serviceable"

    # --- Value the inventory ---
    # Join unit cost onto stock positions. fillna(0) values an uncosted part at
    # zero, which understates rather than inflates the total - the safer
    # direction for a figure that may be quoted to management.
    inv_with_cost = inventory.merge(
        parts[["part_number", "unit_cost_eur", "criticality", "part_class"]],
        on="part_number", how="left"
    )
    inv_with_cost["stock_value"] = inv_with_cost[stock_col] * inv_with_cost["unit_cost_eur"].fillna(0)

    print("\n  Inventory Value by Station:")

    # One line per station: capital held, breadth of coverage, and how much of
    # that coverage is on AOG-critical items.
    for station in data["stations"]["station_code"]:
        station_inv = inv_with_cost[inv_with_cost["station"] == station]
        total_value = station_inv["stock_value"].sum()
        # Count only lines with stock actually on hand - a catalogued part with
        # zero quantity provides no coverage.
        n_parts = len(station_inv[station_inv[stock_col] > 0])
        # AOG-critical lines with stock: the dispatch-protection measure.
        aog_parts = len(station_inv[
            (station_inv["criticality"] == "AOG") & (station_inv[stock_col] > 0)
        ])
        print(f"    {station}: EUR {total_value:>12,.0f} | {n_parts:>3} parts stocked | {aog_parts:>2} AOG-critical")

    total_network = inv_with_cost["stock_value"].sum()
    print(f"    {'TOTAL':>4}: EUR {total_network:>12,.0f}")

    # --- Current exposure ---
    below_min = inventory[inventory[stock_col] < inventory["minimum_stock_level"]]

    # Inner join against AOG parts isolates the shortages that can actually
    # ground an aircraft - the only number on this report that demands action.
    below_min_aog = below_min.merge(
        parts[parts["criticality"] == "AOG"][["part_number"]],
        on="part_number"
    )

    print(f"\n  Stockout Risk:")
    print(f"    Parts below minimum stock: {len(below_min)}")
    print(f"    AOG-critical below minimum: {len(below_min_aog)}")

    # --- Consumption profile ---
    # Mean sizes the steady-state supply chain; peak sizes the surge capacity;
    # the standard deviation is what drives the safety stock in Engine 1.
    demands["demand_date"] = pd.to_datetime(demands["demand_date"])
    monthly_total = demands.groupby(
        demands["demand_date"].dt.to_period("M")
    )["quantity_required"].sum()

    print(f"\n  Demand Statistics (monthly):")
    print(f"    Average: {monthly_total.mean():.0f} parts/month")
    print(f"    Peak: {monthly_total.max():.0f} parts/month")
    print(f"    Std dev: {monthly_total.std():.0f}")

    # --- Coverage proportionality ---
    # Inventory value per based aircraft. A station far below the network norm
    # is under-protected; far above suggests capital that could be released.
    fleet = data["fleet"]
    print(f"\n  Fleet Coverage:")
    for station in data["stations"]["station_code"]:
        n_ac = len(fleet[fleet["home_base"] == station])
        # Skip pure spares-holding stations with no based aircraft (CFU) - the
        # per-aircraft ratio is undefined there, not zero.
        if n_ac > 0:
            station_value = inv_with_cost[inv_with_cost["station"] == station]["stock_value"].sum()
            per_ac = station_value / n_ac
            print(f"    {station}: {n_ac} aircraft, EUR {per_ac:,.0f} inventory per aircraft")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """
    Run all three optimisation engines and persist their recommendations.

    Args:
        (none)

    Returns:
        None

    Notes:
        The recommendation tables are written back to the database so Module 4
        (agent.py) can answer "what should we do?" without re-running the
        optimisation. if_exists="replace" makes the module idempotent: each run
        supersedes the previous recommendation set rather than appending to it,
        which is correct because a stale recommendation is worse than none.
    """
    print("=" * 60)
    print("  Daedalus Supply AI - Module 3: Logistics Optimizer")
    print("  EASA Part-145 Supply Chain Optimization")
    print("=" * 60)

    print("\n[1/4] Loading data...")
    data = load_data()
    print(f"  {len(data['parts'])} parts, {len(data['inventory'])} inventory records")
    print(f"  {len(data['demands'])} demand events")

    print("\n[2/4] Optimizing stock levels...")
    stock_rec = optimize_stock_levels(data)

    # Pre-positioning runs after stock optimisation because it operates on the
    # same shortage picture that Engine 1 has just quantified.
    print("\n[3/4] Pre-positioning analysis...")
    transfers = preposition_parts(data, stock_rec)

    print("\n[4/4] AOG response scenarios...")
    run_aog_scenarios(data)

    network_summary(data, stock_rec)

    # --- Persist recommendations for Module 4 ---
    conn = sqlite3.connect(DB_PATH)
    # Guard both writes: to_sql on an empty frame would create a column-less
    # table that the agent could not read.
    if not stock_rec.empty:
        stock_rec.to_sql("stock_recommendations", conn, if_exists="replace", index=False)
        print(f"\n  Stock recommendations saved to database")
    if not transfers.empty:
        transfers.to_sql("transfer_recommendations", conn, if_exists="replace", index=False)
        print(f"  Transfer recommendations saved to database")
    conn.close()

    # Closing status line: what was produced and what to run next.
    print("\n" + "=" * 60)
    print("  DONE - recommendations written to the database")
    print("  Next step:")
    print("    python agent.py       (Module 4 - query demo)")
    print("    python agent.py -i    (Module 4 - interactive prompt)")
    print("=" * 60)


if __name__ == "__main__":
    main()
