"""Probe A said the HPA variable annotation does not select for measurable localization
heterogeneity. This asks the constructive question instead: does the measured quantity have
reproducible structure of its own, and if so what selects for it.

Heterogeneity is the mutual information between cell identity and predicted compartment,
H(mean prediction) - mean(per-cell entropy). Split-half reliability is computed within gene,
so it estimates how much of a gene's score is signal rather than sampling noise, without
needing a second run.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch

from pvs.eval.localization import CLASS_NAMES


def information(probability: torch.Tensor) -> float:
    entropy = lambda p: -(p.clamp_min(1e-12) * p.clamp_min(1e-12).log()).sum(-1)
    return float(entropy(probability.mean(0)) - entropy(probability).mean())


def correlation(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-12))


def mann_whitney(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    ranks = torch.cat([a, b]).argsort().argsort().float() + 1
    n, m = len(a), len(b)
    u = ranks[:n].sum().item() - n * (n + 1) / 2
    return u / (n * m), (u - n * m / 2) / max((n * m * (n + m + 1) / 12) ** 0.5, 1e-9)


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    rank = lambda x: x.argsort().argsort().float()
    return correlation(rank(a), rank(b))


def split_half(probability: torch.Tensor, draws: int, seed: int) -> tuple[float, float]:
    generator = torch.Generator().manual_seed(seed)
    n = len(probability)
    halves = []
    for _ in range(draws):
        order = torch.randperm(n, generator=generator)
        left, right = order[: n // 2], order[n // 2:]
        halves.append((information(probability[left]), information(probability[right])))
    return tuple(torch.tensor(halves).T)


def variance_explained(values: torch.Tensor, labels: list[str]) -> float:
    groups = defaultdict(list)
    for value, label in zip(values.tolist(), labels):
        groups[label].append(value)
    total = values.var(unbiased=False)
    within = sum(len(v) * torch.tensor(v).var(unbiased=False) for v in groups.values()) / len(values)
    return float(1 - within / total.clamp_min(1e-12))


def dominant_lines(plan: Path) -> dict[str, str]:
    counts: dict[str, Counter] = defaultdict(Counter)
    for line in plan.read_text().splitlines():
        gene, _, _, cell_line, _ = line.split("\t")
        counts[gene][cell_line] += 1
    return {gene: c.most_common(1)[0][0] for gene, c in counts.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("outputs/probe_a_large"))
    parser.add_argument("--plan", type=Path, default=Path("/home/claude/data/subset/plan2.txt"))
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--report", type=Path, default=Path("outputs/probe_a_large/tail.json"))
    args = parser.parse_args()

    records = json.loads((args.out / "results.json").read_text())
    lines = dominant_lines(args.plan)

    # Probe A wrote kind as positive/control from an OR of two HPA columns. The U2OS screen
    # writes a three-letter code, S/I/C for spatial variation, intensity variation and cell-cycle
    # dependency, because those columns mean different things and OR-ing them was a mistake
    # (heterogeneity-selector.md section 5b). Accept either.
    genes, scores, kinds, cell_lines, locations = [], [], [], [], []
    for gene, record in records.items():
        genes.append(gene)
        scores.append(record["entropy_of_mean"] - record["mean_cell_entropy"])
        kinds.append(record["kind"])
        cell_lines.append(lines.get(gene, "?"))
        locations.append(record["location"])
    scores = torch.tensor(scores)

    print(f"=== measured heterogeneity over {len(genes)} genes ===")
    print(f"  mean {scores.mean():.4f}  sd {scores.std():.4f}  "
          f"min {scores.min():.4f}  max {scores.max():.4f}")
    quantiles = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
    values = scores.quantile(quantiles)
    print("  " + "  ".join(f"p{int(q*100)}={v:.4f}" for q, v in zip(quantiles, values)))
    centred = scores - scores.mean()
    skew = float((centred ** 3).mean() / scores.std(unbiased=False) ** 3)
    kurtosis = float((centred ** 4).mean() / scores.std(unbiased=False) ** 4)
    print(f"  skew {skew:+.2f}  kurtosis {kurtosis:.2f}   "
          f"({'heavy tail' if kurtosis > 4 else 'no heavy tail'}, gaussian is 3.0)")

    store = args.out / "embeddings"
    available = sorted(p.stem for p in store.glob("*.pt")) if store.exists() else []
    if available:
        left, right = [], []
        for gene in available:
            probability = torch.load(store / f"{gene}.pt", map_location="cpu")["probability"].float()
            a, b = split_half(probability, args.draws, seed=abs(hash(gene)) % 10_000)
            left.append(a.mean())
            right.append(b.mean())
        left, right = torch.stack(left), torch.stack(right)
        reliability = correlation(left, right)
        corrected = 2 * reliability / (1 + reliability)
        print(f"\n=== split-half reliability, {len(available)} genes, {args.draws} splits ===")
        print(f"  half against half   {reliability:.3f}")
        print(f"  spearman-brown      {corrected:.3f}   "
              f"({'a real per-gene quantity' if corrected > 0.7 else 'mostly sampling noise'})")

    coded = all(len(k) == 3 and set(k) <= set("SIC-") for k in kinds)
    print(f"\n=== what explains the spread across genes ===")
    if coded:
        for letter, name in (("S", "spatial variation"), ("I", "intensity variation"),
                             ("C", "cell cycle dependency")):
            marked = [letter in k for k in kinds]
            flagged, control = scores[marked], scores[[not m for m in marked]]
            if min(len(flagged), len(control)) < 3:
                print(f"  {name:<24} n={len(flagged)}, too few"); continue
            effect, z = mann_whitney(flagged, control)
            print(f"  {name:<24} {variance_explained(scores, marked):.3f}"
                  f"   (marked {flagged.mean():.4f} n={len(flagged)}, "
                  f"rest {control.mean():.4f} n={len(control)}, "
                  f"AUC {effect:.3f} z {z:+.2f})")
    else:
        flagged = scores[[k == "positive" for k in kinds]]
        control = scores[[k == "control" for k in kinds]]
        print(f"  HPA variable annotation   {variance_explained(scores, kinds):.3f}"
              f"   (flagged {flagged.mean():.4f} n={len(flagged)}, "
              f"control {control.mean():.4f} n={len(control)})")
    print(f"  dominant cell line        {variance_explained(scores, cell_lines):.3f}"
          f"   ({len(set(cell_lines))} lines)")
    primary = [location.split(";")[0] for location in locations]
    print(f"  annotated compartment     {variance_explained(scores, primary):.3f}"
          f"   ({len(set(primary))} compartments)")
    argmax = [CLASS_NAMES[int(torch.tensor(records[g]['mean_probability']).argmax())] for g in genes]
    print(f"  predicted compartment     {variance_explained(scores, argmax):.3f}"
          f"   ({len(set(argmax))} predicted)")

    print(f"\n=== the top of the distribution ===")
    order = scores.argsort(descending=True)
    for rank in order[:12].tolist():
        print(f"  {scores[rank]:.4f}  {genes[rank]:<10} {kinds[rank]:<9} "
              f"{cell_lines[rank]:<12} {argmax[rank]:<22} {locations[rank][:34]}")
    top = order[: max(len(genes) // 5, 1)].tolist()
    marks = (("S", "spatial"), ("I", "intensity"), ("C", "cell cycle")) if coded \
        else (("positive", "flagged"),)
    print()
    for letter, name in marks:
        hit = (lambda k: letter in k) if coded else (lambda k: k == letter)
        share = sum(hit(kinds[i]) for i in top) / len(top)
        base = sum(hit(k) for k in kinds) / len(kinds)
        print(f"  top quintile is {share:.0%} {name} against a {base:.0%} base rate"
              f"   (enrichment {share / max(base, 1e-9):.2f}x)")
    print(f"  top quintile cell lines: {Counter(cell_lines[i] for i in top).most_common(4)}")
    print(f"  top quintile predicted:  {Counter(argmax[i] for i in top).most_common(4)}")

    args.report.write_text(json.dumps({
        "genes": genes, "heterogeneity": scores.tolist(), "kind": kinds,
        "cell_line": cell_lines, "location": locations, "predicted": argmax,
        "skew": skew, "kurtosis": kurtosis,
        "variance_explained": {
            "annotation": variance_explained(scores, kinds),
            "cell_line": variance_explained(scores, cell_lines),
            "compartment": variance_explained(scores, primary),
            "predicted": variance_explained(scores, argmax),
        },
    }, indent=2))


if __name__ == "__main__":
    main()
