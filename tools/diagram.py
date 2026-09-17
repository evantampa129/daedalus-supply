#!/usr/bin/env python3
"""
Daedalus Supply AI - Architecture Diagram Generator
===================================================
Draws the ASCII figure in the Architecture section of README.md.

The figure is generated rather than hand-aligned for one reason: an ASCII
diagram maintained by hand rots. Every edit shifts a border by a column,
nobody notices until the figure is unreadable, and by then correcting it means
re-counting several hundred characters. Here the box widths are declared, text
that would not fit raises instead of overflowing silently, and the connectors
are computed from the box centres - so adding a module is an edit to a list of
strings, not an exercise in character counting.

The drawing convention the figure follows:

    Everything that is a component gets a box.
    Everything that flows between components is a line.

That distinction is the whole point. A diagram where some components are boxed
and others are bare labels makes the reader work out which is which from
context, and the answer usually turns out to be "whichever had more text".

Usage:
    python tools/diagram.py            # print the figure
    python tools/diagram.py --write    # replace the block in README.md
    python tools/diagram.py --check    # verify README matches (exit 1 if not)

Author: Evangelos Tampachaniotis
Version: 1.2.0
License: MIT
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README = os.path.join(REPO_ROOT, "README.md")

# Left margin and the gap between boxes placed side by side. Both are in
# columns, and every centre calculation below is derived from them.
MARGIN = 2
GAP = 1


# ============================================================================
# PRIMITIVES
# ============================================================================

def box(lines, width):
    """
    Frame a block of text.

    Args:
        lines: list of str - the contents, one entry per line. An empty string
            renders as a blank line inside the frame.
        width: int - total width of the box including both borders.

    Returns:
        list of str - the framed box.

    Raises:
        ValueError when a line does not fit. This is deliberately fatal: a
        silently truncated or overflowing line is exactly the failure this
        module exists to prevent, and the message names the line and both
        measurements so the fix is obvious.
    """
    area = width - 4                      # two borders, two-space inner indent
    for line in lines:
        if len(line) > area:
            raise ValueError(
                f"{line!r} needs {len(line)} columns, this box holds {area}")
    border = "+" + "-" * (width - 2) + "+"
    body = ["|" + ("  " + line).ljust(width - 2) + "|" for line in lines]
    return [border] + body + [border]


def row(boxes, gap=GAP, margin=MARGIN):
    """
    Place boxes side by side.

    Args:
        boxes: list of list of str - boxes as returned by box().
        gap: int - columns between adjacent boxes.
        margin: int - columns before the first box.

    Returns:
        tuple (list of str, list of int) - the rendered lines, and the centre
        column of each box. The centres are what every connector below is
        drawn from, which is why they are returned rather than recomputed.

    Notes:
        Shorter boxes are padded with blank frame lines so the row closes at
        the same height, and boxes may have different widths - the middle
        column of the engine row is wider because "stock_recommendations" is
        a table name and abbreviating it in a diagram would be a small lie.
    """
    height = max(len(b) for b in boxes)
    padded = []
    for b in boxes:
        filler = "|" + " " * (len(b[0]) - 2) + "|"
        padded.append(b[:-1] + [filler] * (height - len(b)) + [b[-1]])
    lines = [" " * margin + (" " * gap).join(parts) for parts in zip(*padded)]

    centres, cursor = [], margin
    for b in boxes:
        centres.append(cursor + len(b[0]) // 2)
        cursor += len(b[0]) + gap
    return lines, centres


def at(centres, char="|"):
    """
    Draw one line carrying a character at each given column.

    Args:
        centres: list of int - the columns to mark.
        char: str - "|" for a stem, "v" for an arrowhead.

    Returns:
        str - the rendered line.
    """
    line = [" "] * (max(centres) + 1)
    for c in centres:
        line[c] = char
    return "".join(line)


def gather(centres, target):
    """
    Draw a bracket collecting several columns into one.

    Args:
        centres: list of int - the columns being collected.
        target: int - the column the flow continues down.

    Returns:
        str - the rendered bracket.
    """
    line = [" "] * (max(max(centres), target) + 1)
    for i in range(min(centres), max(centres) + 1):
        line[i] = "-"
    for c in list(centres) + [target]:
        line[c] = "+"
    return "".join(line)


def spread(source, targets):
    """
    Draw a bracket fanning one column out to several.

    Args:
        source: int - the column the flow arrives on.
        targets: list of int - the columns it fans out to.

    Returns:
        str - the rendered bracket.
    """
    line = [" "] * (max(max(targets), source) + 1)
    for i in range(min(targets), max(targets) + 1):
        line[i] = "-"
    for t in list(targets) + [source]:
        line[t] = "+"
    return "".join(line)


# ============================================================================
# THE FIGURE
# ============================================================================

def build():
    """
    Assemble the architecture figure.

    Args:
        (none)

    Returns:
        str - the complete figure, without the surrounding code fence.

    Notes:
        Box widths are chosen so that Module 1, the database and the middle
        engine share a centre column: the figure then has one vertical spine
        running from the pipeline down to the consumers, and a reader's eye
        follows the data rather than hunting for the next arrow.
    """
    out = []

    # --- what the system is given ---------------------------------------
    out += [
        "        raw_data/                                   config.py",
        "     FAA SDR CSV, 413K rows                  hardware auto-detection",
        "     JASC / ATA code list                    MINIMAL / STANDARD / FULL",
        "             |                                          |",
        "             |                                          |  imported by",
        "             v                                          v  modules 1-3, 5, 6",
    ]

    # --- Module 1 --------------------------------------------------------
    pipeline, spine = row([box([
        "MODULE 1 - data_pipeline.py",
        "",
        "SDR parser        413K filed reports -> filter -> 195,801 kept",
        "Fleet generator   15 airframes, A320 family, 5 Greek stations",
        "Flight simulator  3 years, seasonal schedule, 54,532 sectors",
        "Failure model     Weibull (rotable)  Poisson (expendable)",
        "                  deterministic (consumable)",
    ], 74)])
    out += pipeline

    # --- the database ----------------------------------------------------
    # Margin chosen so this box centres on the same column as Module 1.
    database, db_centre = row([box([
        "aerosupply.db",
        "",
        "12 tables, 3 views, 106 MB",
        "SQLite / PostgreSQL / MySQL",
    ], 33)], margin=23)
    out += [at(spine), at(spine, "v")] + database

    # --- the three engines ------------------------------------------------
    engines, engine_centres = row([
        box(["MODULE 2", "prediction_model.py", "",
             "Cox PH + Weibull AFT", "rotable survival", "",
             "XGBoost regression", "expendable demand", "",
             "XGBoost classifier", "real SDR, 87.6%"], 24),
        box(["MODULE 3", "logistics_optimizer", "",
             "Stock levels (s,S)", "z set by MEL class", "",
             "Pre-positioning", "30 transfers", "",
             "AOG router", "EUR 15,000 / hour"], 25),
        box(["MODULE 4", "agent.py", "",
             "NL query router", "10 vetted reports", "",
             "keyword-routed,", "no synthesised SQL", "",
             "demo / -i / -q", ""], 24),
    ])
    out += [at(db_centre), spread(db_centre[0], engine_centres),
            at(engine_centres, "v")] + engines

    # --- what each engine leaves behind ----------------------------------
    # One box per engine, which is what makes it visible that Module 4 writes
    # nothing back: it reads the same tables the other two populate.
    artefacts, _ = row([
        box(["3 PNG plots", "survival, demand,", "SDR analysis"], 24),
        box(["written back to db", "stock_recommendations", "transfer_recommend."], 25),
        box(["terminal reports", "read back from the", "same tables"], 24),
    ])
    out += [at(engine_centres), at(engine_centres, "v")] + artefacts

    # --- the two consumers ------------------------------------------------
    consumers, consumer_centres = row([
        box(["MODULE 5 - dashboard.py", "streamlit run dashboard.py", "",
             "Fleet Overview     map, register", "Parts & Inventory  search, bars",
             "Predictions        plots, risk", "Logistics          AOG simulator",
             "SDR Analysis       FAA corpus"], 37),
        box(["MODULE 6 - api.py", "uvicorn api:app", "",
             "/api/user/*    17 routes, read", "/api/admin/*    5 routes, write", "",
             "require_user / require_admin", "the seam v1.3 fills with JWT"], 37),
    ])
    hub = (min(engine_centres) + max(engine_centres)) // 2
    out += [at(engine_centres),
            gather(engine_centres, hub),
            at([hub]),
            spread(hub, consumer_centres),
            at(consumer_centres, "v")] + consumers

    # --- what holds everywhere -------------------------------------------
    # Stated once at the foot rather than repeated in every box, because these
    # are properties of every path through the system.
    out += [
        "",
        "  Read discipline, the same on every path:",
        "    mode=ro connection   a read handler cannot write - the driver refuses",
        "    bound parameters     no client value is concatenated into SQL",
        "    serviceable only     unserviceable stock is never offered as available",
        "                         (EASA Part-145 145.A.42)",
    ]

    return "\n".join(line.rstrip() for line in out)


# ============================================================================
# README INTEGRATION
# ============================================================================

def readme_block(text):
    """
    Locate the fenced block under the Architecture heading.

    Args:
        text: str - the whole README.

    Returns:
        tuple (int, int) - line indices of the opening and closing fences.

    Raises:
        ValueError when the section or its fence cannot be found, rather than
        writing the figure into the wrong part of the document.
    """
    lines = text.split("\n")
    try:
        heading = next(i for i, l in enumerate(lines) if l.startswith("## Architecture"))
        opening = next(i for i in range(heading, len(lines)) if lines[i].strip() == "```")
        closing = next(i for i in range(opening + 1, len(lines)) if lines[i].strip() == "```")
    except StopIteration:
        raise ValueError("No fenced block found under '## Architecture' in README.md")
    return opening, closing


def main():
    """
    Print the figure, write it into the README, or check the two agree.

    Args:
        (none - reads sys.argv)

    Returns:
        None. Exits 1 when --check finds the README out of date, so the same
        command works as a pre-commit or CI guard.
    """
    figure = build()

    if "--write" in sys.argv or "--check" in sys.argv:
        with open(README) as f:
            text = f.read()
        lines = text.split("\n")
        opening, closing = readme_block(text)
        current = "\n".join(lines[opening + 1:closing])

        if "--check" in sys.argv:
            if current == figure:
                print("README.md architecture figure is up to date")
                return
            print("README.md architecture figure differs from the generator")
            sys.exit(1)

        updated = lines[:opening + 1] + figure.split("\n") + lines[closing:]
        with open(README, "w") as f:
            f.write("\n".join(updated))
        print(f"README.md updated: {closing - opening - 1} lines replaced "
              f"by {len(figure.splitlines())}")
        return

    print(figure)


if __name__ == "__main__":
    main()
