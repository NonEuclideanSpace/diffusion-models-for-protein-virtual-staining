"""Selecting genes and cells from the Human Protein Atlas metadata.

The metadata is the only part of HPA that is small, and it decides everything downstream:
which genes are worth downloading, which are usable as matched controls, and how many cells
each conditioning group will actually have. Getting the selection wrong is expensive in a way
that is invisible until the statistics come out flat.
"""
from __future__ import annotations

import csv
from collections import Counter,defaultdict
from dataclasses import dataclass
from pathlib import Path

VARIATION_SPATIAL="Single-cell variation spatial"
VARIATION_INTENSITY="Single-cell variation intensity"


@dataclass(frozen=True)
class Gene:
    ensembl:str
    name:str
    reliability:str
    main_location:str
    spatial_variation:str
    intensity_variation:str

    @property
    def is_spatially_variable(self)->bool:
        return bool(self.spatial_variation.strip())

    @property
    def is_intensity_variable(self)->bool:
        return bool(self.intensity_variation.strip())

    @property
    def locations(self)->tuple[str,...]:
        return tuple(part.strip() for part in self.main_location.split(";") if part.strip())

    @property
    def multilocalizing(self)->bool:
        return len(self.locations)>=2

    @property
    def stratum(self)->tuple[str,str]:
        return self.reliability,self.main_location


def read_subcellular_location(path:str|Path)->list[Gene]:
    with Path(path).open(newline="") as handle:
        return [
            Gene(row["Gene"],row["Gene name"],row["Reliability"],row["Main location"],
                 row.get(VARIATION_SPATIAL,""),row.get(VARIATION_INTENSITY,""))
            for row in csv.DictReader(handle,delimiter="\t")
        ]


def matched_controls(
    genes:list[Gene],
    positives:list[Gene],
    per_positive:int=2,
    seed:int=0,
    cell_lines:dict[str,str]|None=None,
)->list[Gene]:
    """Draw controls sharing each positive's reliability, main location and, if supplied,
    dominant cell line.

    Reliability and main location keep a difference in measured heterogeneity from being a
    difference in annotation quality or in which compartment the protein sits.

    **Cell line has to be matched too, and omitting it invalidated a run.** Suspension and
    round-morphology lines are morphologically uniform and show markedly less apparent
    single-cell variation whatever protein is imaged: in one selection, genes dominated by
    HEL, REH or SK-MEL-30 had a median between-cell heterogeneity of 0.018 against 0.098 for
    the rest, a 5.5x difference owing nothing to the protein. Those lines fell almost entirely
    among controls, manufacturing a difference that vanished once they were removed.

    Pass `cell_lines` mapping gene name to its dominant line, derived from the crop index.
    Without it the match is on two fields and the caller must check composition afterwards.
    """
    import random

    rng=random.Random(seed)
    chosen_ids={gene.ensembl for gene in positives}
    key=lambda gene:(gene.stratum+(cell_lines.get(gene.name,""),)) if cell_lines else gene.stratum
    pools:dict[tuple,list[Gene]]=defaultdict(list)
    for gene in genes:
        if gene.ensembl not in chosen_ids:
            pools[key(gene)].append(gene)

    controls:list[Gene]=[]
    for positive in positives:
        pool=[g for g in pools[key(positive)] if g.ensembl not in chosen_ids]
        rng.shuffle(pool)
        for candidate in pool[:per_positive]:
            controls.append(candidate)
            chosen_ids.add(candidate.ensembl)
    return controls


def probe_a_selection(
    path:str|Path,
    per_positive:int=2,
    seed:int=0,
    cell_lines:dict[str,str]|None=None,
)->dict[str,list[Gene]]:
    """The gene set for probe A: does a classifier see heterogeneity known to be real.

    Pass `cell_lines` (gene name to dominant line, from `dominant_cell_lines`) — without it
    the control group can end up dominated by round suspension lines and the comparison
    measures cell shape rather than protein behaviour.
    """
    genes=read_subcellular_location(path)
    positives=[g for g in genes if g.is_spatially_variable]
    return {"positive":positives,
            "control":matched_controls(genes,positives,per_positive,seed,cell_lines)}


def dominant_cell_lines(index_rows)->dict[str,str]:
    """Most frequent cell line per gene, from crop index rows."""
    counts:dict[str,Counter]=defaultdict(Counter)
    for row in index_rows:
        counts[row["gene_names"]][row["atlas_name"]]+=1
    return {gene:line.most_common(1)[0][0] for gene,line in counts.items() if line}


def composition_distance(a:list[Gene],b:list[Gene],cell_lines:dict[str,str])->float:
    """Total variation between two gene sets' cell-line compositions. Zero is identical.

    Report this for any case-control selection. A value near 0.5 means the groups were imaged
    in largely different lines and any morphological comparison between them is confounded.
    """
    def profile(genes):
        counter=Counter(cell_lines.get(g.name,"") for g in genes)
        total=sum(counter.values()) or 1
        return {k:v/total for k,v in counter.items()}
    left,right=profile(a),profile(b)
    return 0.5*sum(abs(left.get(k,0.0)-right.get(k,0.0)) for k in set(left)|set(right))


def summarize(genes:list[Gene])->dict:
    return {
        "count":len(genes),
        "reliability":dict(Counter(g.reliability for g in genes).most_common()),
        "multilocalizing":sum(g.multilocalizing for g in genes),
        "top_locations":dict(Counter(loc for g in genes for loc in g.locations).most_common(6)),
    }


def cells_per_group(index_rows,key=("gene_names","atlas_name"))->Counter:
    """How many single cells each conditioning group has. This is the number that decides
    whether a distribution over compartments can be estimated at all."""
    counter:Counter=Counter()
    for row in index_rows:
        counter[tuple(row[k] for k in key)]+=1
    return counter
