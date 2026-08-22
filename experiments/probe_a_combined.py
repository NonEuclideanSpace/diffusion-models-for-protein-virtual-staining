"""Analyse probe A across the discovery set and the expanded set.

The expanded run shares 39 of its 116 genes with the discovery run, so it is not an
independent replication of that result. The 77 genes it does not share are, and they are
reported separately and first — an effect that survives on genes never previously looked at is
worth more than the same effect measured again on a larger sample containing the original.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def mann_whitney(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    ranks = torch.cat([a, b]).argsort().argsort().float() + 1
    n, m = len(a), len(b)
    u = ranks[:n].sum().item() - n * (n + 1) / 2
    return u / (n * m), (u - n * m / 2) / max((n * m * (n + m + 1) / 12) ** 0.5, 1e-9)


def bootstrap(a: torch.Tensor, b: torch.Tensor, draws: int = 4000, seed: int = 0) -> tuple[float, float]:
    generator = torch.Generator().manual_seed(seed)
    values = torch.tensor([
        mann_whitney(a[torch.randint(len(a), (len(a),), generator=generator)],
                     b[torch.randint(len(b), (len(b),), generator=generator)])[0]
        for _ in range(draws)
    ]).sort().values
    return float(values[int(0.025 * draws)]), float(values[int(0.975 * draws)])


ROUND_LINES = {"HEL", "REH", "SK-MEL-30", "THP-1", "Rh30", "HAP1", "K-562", "NB-4", "HL-60"}


def dominant_lines(plan: Path) -> dict[str, str]:
    from collections import Counter, defaultdict
    counts: dict[str, Counter] = defaultdict(Counter)
    for line in plan.read_text().splitlines():
        gene, _, _, cell_line, _ = line.split("\t")
        counts[gene][cell_line] += 1
    return {gene: c.most_common(1)[0][0] for gene, c in counts.items()}


def composition_distance(records: dict, lines: dict[str, str]) -> float:
    from collections import Counter
    profile = lambda kind: Counter(lines.get(g, "") for g, r in records.items() if r["kind"] == kind)
    left, right = profile("positive"), profile("control")
    a, b = sum(left.values()) or 1, sum(right.values()) or 1
    return 0.5 * sum(abs(left.get(k, 0) / a - right.get(k, 0) / b) for k in set(left) | set(right))


def statistics(records: dict) -> dict[str, dict[str, torch.Tensor]]:
    out: dict[str, dict[str, list]] = {"heterogeneity": {"positive": [], "control": []},
                                       "entropy": {"positive": [], "control": []},
                                       "spread": {"positive": [], "control": []}}
    for record in records.values():
        kind = record["kind"]
        if kind not in ("positive", "control"):
            continue
        out["heterogeneity"][kind].append(record["entropy_of_mean"] - record["mean_cell_entropy"])
        out["entropy"][kind].append(record["entropy_of_mean"])
        out["spread"][kind].append(record["embedding_spread"])
    return {k: {g: torch.tensor(v) for g, v in d.items()} for k, d in out.items()}


def report(title: str, records: dict) -> None:
    counts = {k: sum(1 for r in records.values() if r["kind"] == k) for k in ("positive", "control")}
    print(f"\n=== {title} === {counts['positive']} flagged, {counts['control']} control")
    if min(counts.values()) < 3:
        print("  too few genes to test")
        return
    for name, groups in statistics(records).items():
        positive, control = groups["positive"], groups["control"]
        effect, z = mann_whitney(positive, control)
        low, high = bootstrap(positive, control)
        verdict = "separates" if low > 0.5 else ("reversed" if high < 0.5 else "not resolved")
        print(f"  {name:<15} flagged {positive.median():.4f}  control {control.median():.4f}  "
              f"effect {effect:.3f} [{low:.3f}, {high:.3f}]  z {z:+.2f}   {verdict}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery", type=Path, default=Path("outputs/probe_a/results.json"))
    parser.add_argument("--expanded", type=Path, default=Path("outputs/probe_a_large/results.json"))
    parser.add_argument("--plan", type=Path, default=Path("/home/claude/data/subset/plan.txt"))
    parser.add_argument("--plan-large", type=Path, default=Path("/home/claude/data/subset/plan2.txt"))
    args = parser.parse_args()

    discovery = json.loads(args.discovery.read_text())["genes"] if args.discovery.exists() else {}
    expanded = json.loads(args.expanded.read_text()) if args.expanded.exists() else {}
    fresh = {g: r for g, r in expanded.items() if g not in discovery}
    shared = sorted(set(discovery) & set(expanded))

    print(f"discovery {len(discovery)} genes, expanded {len(expanded)}, "
          f"shared {len(shared)}, new in expanded {len(fresh)}")

    lines = {}
    for plan in (args.plan, args.plan_large):
        if plan.exists():
            lines.update(dominant_lines(plan))

    for title, records in (("expanded set, all genes", expanded),
                           ("discovery set", discovery)):
        if records and lines:
            distance = composition_distance(records, lines)
            flag = "" if distance < 0.15 else "   <- groups were imaged in different lines"
            print(f"\ncell-line composition distance, {title}: {distance:.3f}{flag}")

    report("independent replication, genes never used before", fresh)
    report("expanded set, all genes", expanded)
    if lines:
        report("expanded set, round-morphology lines removed",
               {g: r for g, r in expanded.items() if lines.get(g) not in ROUND_LINES})
    report("discovery set", discovery)
    if lines:
        report("discovery set, round-morphology lines removed",
               {g: r for g, r in discovery.items() if lines.get(g) not in ROUND_LINES})

    if shared:
        a = torch.tensor([discovery[g]["entropy_of_mean"] - discovery[g]["mean_cell_entropy"] for g in shared])
        b = torch.tensor([expanded[g]["entropy_of_mean"] - expanded[g]["mean_cell_entropy"] for g in shared])
        centred = lambda v: v - v.mean()
        correlation = float((centred(a) * centred(b)).sum() / (centred(a).norm() * centred(b).norm()).clamp_min(1e-9))
        print(f"\n=== measurement stability on the {len(shared)} shared genes ===")
        print(f"  60 cells against 40 cells, correlation {correlation:.3f}")
        print("  This is how reproducible one gene's heterogeneity estimate is at these sample")
        print("  sizes. A low value would mean per-gene numbers are noise, whatever the group")
        print("  comparison shows.")


if __name__ == "__main__":
    main()
