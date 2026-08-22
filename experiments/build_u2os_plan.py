"""Build the U2OS heterogeneity screen.

Probe A found that HPA's variable annotation explains 2% of the between-gene variance in
measured heterogeneity while cell line explains 24%. Holding cell line fixed removes the
larger term by construction and lets the annotation be tested at five times the sample size.
U2OS has 1652 genes with at least 40 cells, so the screen is limited by compute, not data.
"""
from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

# HPA carries three separate single-cell columns and they mean different things. Spatial marks
# proteins that sit in different *places* between cells, which is the construct this project
# measures; intensity marks different *levels* and is twenty times more common. OR-ing them
# produces a group that is mostly intensity genes, so each flag is recorded on its own and the
# comparison is made at analysis time.
FLAGS = (("S", "Single-cell variation spatial"),
         ("I", "Single-cell variation intensity"),
         ("C", "Cell cycle dependency"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=Path("/tmp/all_cells.csv"))
    parser.add_argument("--hpa", type=Path,
                        default=Path("/home/claude/data/hpa/subcellular_location.tsv"))
    parser.add_argument("--line", default="U2OS")
    parser.add_argument("--cells", type=int, default=40)
    parser.add_argument("--genes", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("/home/claude/data/subset/u2os.txt"))
    args = parser.parse_args()

    annotation = {}
    with args.hpa.open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            marks = "".join(letter if row.get(field, "").strip() else "-"
                            for letter, field in FLAGS)
            annotation[row["Gene name"]] = (
                marks,
                row.get("Main location", "") or row.get("Additional location", ""),
                row.get("Reliability", ""),
            )

    cells = defaultdict(list)
    with args.index.open() as handle:
        for row in csv.reader(handle):
            if len(row) < 13 or row[8] != args.line:
                continue
            gene = row[11]
            if not gene or ("," in gene) or gene not in annotation:
                continue
            plate, position, sample, cell = row[5], row[6], row[7], row[0]
            cells[gene].append(f"hpa-processed/cell_crops/{plate}/{plate}_{position}_{sample}_{cell}")

    eligible = {g: s for g, s in cells.items() if len(s) >= args.cells
                and annotation[g][2] in {"Enhanced", "Supported", "Approved"}}
    print(f"{args.line}: {len(cells)} genes indexed, {len(eligible)} with >={args.cells} cells "
          f"and a reliable annotation")

    # Spatial and cell-cycle marks are rare enough that a uniform sample of 600 would carry
    # about nine spatial genes, too few to test. Every gene carrying a rare mark is taken, then
    # the rest is filled at random, so the screen produces the ranking and tests all three
    # columns from the same run. The ranking is unaffected: it is a within-gene measurement.
    rng = random.Random(args.seed)
    rare = sorted(g for g in eligible if "S" in annotation[g][0] or "C" in annotation[g][0])
    rest = sorted(set(eligible) - set(rare))
    rng.shuffle(rest)
    chosen = rare + rest[: max(args.genes - len(rare), 0)]
    print(f"  taking all {len(rare)} genes marked spatial or cell-cycle, "
          f"filling to {len(chosen)} at random")

    counts: defaultdict[str, int] = defaultdict(int)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        for gene in chosen:
            marks, location, _ = annotation[gene]
            for letter in marks.replace("-", ""):
                counts[letter] += 1
            stems = eligible[gene]
            rng.shuffle(stems)
            for stem in stems[: args.cells]:
                handle.write(f"{gene}\t{marks}\t{location}\t{args.line}\t{stem}\n")
    total = len(chosen) * args.cells
    print(f"wrote {len(chosen)} genes -> {args.out}")
    print(f"  spatial {counts['S']}   intensity {counts['I']}   cell cycle {counts['C']}"
          f"   of {len(chosen)},  {total} cells")
    print(f"at 1 cell/s that is {total / 3600:.1f} hours of streaming inference")


if __name__ == "__main__":
    main()
