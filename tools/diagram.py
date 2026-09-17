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
# Figures are drawn with box-drawing characters rather than +-|, so a border
# reads as a line instead of as punctuation. Junctions are derived from the
# directions a column needs rather than picked by hand, which is what keeps
# them correct when a box moves or changes width.

GLYPH = {
    (0, 0, 1, 1): "\u2500", (1, 1, 0, 0): "\u2502",
    (0, 1, 0, 1): "\u250c", (0, 1, 1, 0): "\u2510",
    (1, 0, 0, 1): "\u2514", (1, 0, 1, 0): "\u2518",
    (1, 1, 0, 1): "\u251c", (1, 1, 1, 0): "\u2524",
    (0, 1, 1, 1): "\u252c", (1, 0, 1, 1): "\u2534",
    (1, 1, 1, 1): "\u253c",
    (1, 0, 0, 0): "\u2502", (0, 1, 0, 0): "\u2502",
}


def glyph(up=0, down=0, left=0, right=0):
    """
    Return the box-drawing character that joins the given directions.

    Args:
        up, down, left, right: truthy when the character must connect that way.

    Returns:
        str - one character.

    Raises:
        KeyError for a combination with no glyph, which means the figure asked
        for a junction that cannot be drawn - a bug worth failing on.
    """
    return GLYPH[(int(bool(up)), int(bool(down)), int(bool(left)), int(bool(right)))]


def box(lines, width, pad=2):
    """
    Frame a block of text.

    Args:
        lines: list of str - the contents, one per line; "" renders blank.
        width: int - total width including both borders.
        pad: int - columns of indent inside the left border.

    Returns:
        list of str - the framed box.

    Raises:
        ValueError when a line does not fit. Deliberately fatal: a silently
        overflowing line is the failure this module exists to prevent, and the
        message names the line and both measurements.
    """
    area = width - 2 - pad
    for line in lines:
        if len(line) > area:
            raise ValueError(
                f"{line!r} needs {len(line)} columns, this box holds {area}")
    top = "\u250c" + "\u2500" * (width - 2) + "\u2510"
    bottom = "\u2514" + "\u2500" * (width - 2) + "\u2518"
    body = ["\u2502" + (" " * pad + line).ljust(width - 2) + "\u2502" for line in lines]
    return [top] + body + [bottom]


def row(boxes, gap=GAP, margin=MARGIN):
    """
    Place boxes side by side.

    Args:
        boxes: list of list of str - boxes as returned by box().
        gap: int - columns between adjacent boxes.
        margin: int - columns before the first box.

    Returns:
        tuple (list of str, list of int) - the rendered lines, and the centre
        column of each box. Every connector below is drawn from those centres.

    Notes:
        Shorter boxes are padded with blank frame lines so the row closes at
        one height, and boxes may differ in width - the middle column of the
        engine row is wider because "stock_recommendations" is a table name
        and abbreviating it in a diagram would be a small lie.
    """
    height = max(len(b) for b in boxes)
    padded = []
    for b in boxes:
        filler = "\u2502" + " " * (len(b[0]) - 2) + "\u2502"
        padded.append(b[:-1] + [filler] * (height - len(b)) + [b[-1]])
    lines = [" " * margin + (" " * gap).join(parts) for parts in zip(*padded)]

    centres, cursor = [], margin
    for b in boxes:
        centres.append(cursor + len(b[0]) // 2)
        cursor += len(b[0]) + gap
    return lines, centres


def tap(lines, index, columns, downward):
    """
    Open a box border where a connector meets it.

    Args:
        lines: list of str - the rendered rows; modified in place.
        index: int - which row holds the border, usually 0 or -1.
        columns: list of int - where the connector touches.
        downward: bool - True for a line leaving a bottom edge, False for one
            arriving at a top edge.

    Returns:
        None.

    Notes:
        This is what makes the figure look drawn rather than assembled: a line
        enters the border instead of stopping one row above it.
    """
    chars = list(lines[index])
    for column in columns:
        chars[column] = glyph(up=not downward, down=downward, left=1, right=1)
    lines[index] = "".join(chars)


def stem(columns):
    """
    Draw one row carrying a vertical stroke at each given column.

    Args:
        columns: list of int - the columns to mark.

    Returns:
        str - the rendered row.
    """
    line = [" "] * (max(columns) + 1)
    for column in columns:
        line[column] = "\u2502"
    return "".join(line)


def junction(up, down):
    """
    Draw the horizontal run joining a set of columns above to a set below.

    Args:
        up: list of int - columns where a line arrives from above.
        down: list of int - columns where a line continues below.

    Returns:
        str - the rendered row.

    Notes:
        One primitive serves both fans: one column above and three below is a
        distribution, three above and one below is a collection, and a column
        that appears in both renders as a crossing rather than as two
        characters fighting over one cell.
    """
    columns = list(up) + list(down)
    lo, hi = min(columns), max(columns)
    marks = {c: {"left": c > lo, "right": c < hi} for c in range(lo, hi + 1)}
    for column in down:
        marks[column]["down"] = True
    for column in up:
        marks[column]["up"] = True

    line = [" "] * (hi + 1)
    for column, directions in marks.items():
        line[column] = glyph(**directions)
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
        Box widths are chosen so Module 1, the database and the middle engine
        share a centre column: the figure then has one vertical spine from the
        pipeline down to the consumers, and a reader's eye follows the data
        rather than hunting for the next connector.
    """
    out = []

    # --- what the system is given ----------------------------------------
    # config.py is drawn as an input because that is what it is: every
    # compute-heavy module imports it for its tuning parameters. Only its edge
    # into Module 1 is drawn - six edges to the same box would bury the data
    # flow, so the box states the rest in words.
    inputs, input_centres = row([
        box(["raw_data/", "",
             "FAA SDR CSV, 413K filed reports",
             "JASC / ATA code list"], 35),
        box(["config.py", "",
             "hardware auto-detection",
             "MINIMAL / STANDARD / FULL",
             "imported by modules 1-3, 5, 6"], 35),
    ])
    tap(inputs, -1, input_centres, downward=True)
    out += inputs + [stem(input_centres)]

    # --- Module 1 ---------------------------------------------------------
    pipeline, spine = row([box([
        "MODULE 1 - data_pipeline.py",
        "",
        "SDR parser        413K filed reports \u2192 filter \u2192 195,801 kept",
        "Fleet generator   15 airframes, A320 family, 5 Greek stations",
        "Flight simulator  3 years, seasonal schedule, 54,532 sectors",
        "Failure model     Weibull (rotable)  Poisson (expendable)",
        "                  deterministic (consumable)",
    ], 74)])
    tap(pipeline, 0, input_centres, downward=False)
    tap(pipeline, -1, spine, downward=True)
    out += pipeline + [stem(spine)]

    # --- the database -----------------------------------------------------
    # Margin chosen so this box centres on the same column as Module 1.
    database, db_centre = row([box([
        "aerosupply.db",
        "",
        "12 tables, 3 views, 106 MB",
        "SQLite / PostgreSQL / MySQL",
    ], 33)], margin=23)
    tap(database, 0, db_centre, downward=False)
    tap(database, -1, db_centre, downward=True)
    out += database

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
    tap(engines, 0, engine_centres, downward=False)
    tap(engines, -1, engine_centres, downward=True)
    out += [junction(up=db_centre, down=engine_centres)] + engines

    # --- what each engine leaves behind ------------------------------------
    # One box per engine, which is what makes it visible that Module 4 writes
    # nothing back: it reads the same tables the other two populate.
    artefacts, artefact_centres = row([
        box(["3 PNG plots", "survival, demand,", "SDR analysis"], 24),
        box(["written back to db", "stock_recommendations", "transfer_recommend."], 25),
        box(["terminal reports", "read back from the", "same tables"], 24),
    ])
    tap(artefacts, 0, artefact_centres, downward=False)
    # Module 4 produces terminal output and nothing the other modules consume,
    # so its column ends here rather than continuing into the consumers.
    tap(artefacts, -1, artefact_centres[:2], downward=True)
    out += [stem(engine_centres)] + artefacts

    # --- the two consumers -------------------------------------------------
    consumers, consumer_centres = row([
        box(["MODULE 5 - dashboard.py", "streamlit run dashboard.py", "",
             "Fleet Overview     map, register", "Parts & Inventory  search, bars",
             "Predictions        plots, risk", "Logistics          AOG simulator",
             "SDR Analysis       FAA corpus"], 37),
        box(["MODULE 6 - api.py", "uvicorn api:app", "",
             "/api/user/*    17 routes, read", "/api/admin/*    5 routes, write", "",
             "require_user / require_admin", "the seam v1.3 fills with JWT"], 37),
    ])
    tap(consumers, 0, consumer_centres, downward=False)
    out += [junction(up=artefact_centres[:2], down=consumer_centres)] + consumers

    # --- what holds everywhere ---------------------------------------------
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
