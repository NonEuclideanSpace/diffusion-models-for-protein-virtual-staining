"""SubCell's 31 localization classes and the coarse groupings built on them.

The class index to name mapping is taken from the published inference code, and it is
independently confirmed by the positive control in `test_crops.py`: proteins annotated as
nucleoplasm, mitochondria and Golgi are assigned classes 26, 17 and 11, which this table
names Nucleoplasm, Mitochondria and Golgi apparatus.

Grouping matters as much as the names. Calibration at HPA's sample sizes is only measurable
at a small class count — 31 classes with 75 cells per gene puts every divergence below its own
noise floor. `GROUPING_3` from the SubCell annotation table collapses to seven, which
simulation at the real parameters shows is workable.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import Tensor

_CLASSES=json.loads((Path(__file__).parent/"subcell_classes.json").read_text())
CLASS_NAMES:tuple[str,...]=tuple(_CLASSES[str(i)] for i in range(len(_CLASSES)))
NUM_CLASSES=len(CLASS_NAMES)

# From annotations/location_group_mapping.tsv, column "Grouping 3". Eight rare mitotic and
# aggregate annotations are ungrouped upstream; they are collected under "Other" rather than
# dropped, so probability mass is conserved.
GROUPING_3:dict[str,str]={
    "Actin filaments":"Cytoskeleton","Centrosome":"Cytoskeleton","Focal adhesion sites":"Cytoskeleton",
    "Intermediate filaments":"Cytoskeleton","Microtubules":"Cytoskeleton",
    "Centriolar satellite":"Cytosol","Cytoplasmic bodies":"Cytosol","Cytosol":"Cytosol",
    "Endoplasmic reticulum":"Endomembrane system","Endosomes":"Endomembrane system",
    "Golgi apparatus":"Endomembrane system","Lipid droplets":"Endomembrane system",
    "Lysosomes":"Endomembrane system","Peroxisomes":"Endomembrane system","Vesicles":"Endomembrane system",
    "Mitochondria":"Mitochondria",
    "Nucleoli":"Nucleoli","Nucleoli fibrillar center":"Nucleoli","Nucleoli rim":"Nucleoli",
    "Nuclear bodies":"Nucleus","Nuclear membrane":"Nucleus","Nuclear speckles":"Nucleus",
    "Nucleoplasm":"Nucleus",
    "Cell Junctions":"Plasma membrane","Plasma membrane":"Plasma membrane",
}
OTHER="Other"
GROUP_NAMES:tuple[str,...]=tuple(sorted(set(GROUPING_3.values())))+(OTHER,)
NUM_GROUPS=len(GROUP_NAMES)


def grouping_matrix()->Tensor:
    """(NUM_CLASSES, NUM_GROUPS) indicator that sums fine probabilities into coarse ones."""
    matrix=torch.zeros(NUM_CLASSES,NUM_GROUPS)
    index={name:i for i,name in enumerate(GROUP_NAMES)}
    for class_index,name in enumerate(CLASS_NAMES):
        matrix[class_index,index.get(GROUPING_3.get(name,OTHER),index[OTHER])]=1.0
    return matrix


def to_groups(probabilities:Tensor)->Tensor:
    """Collapse 31-class probabilities onto the seven coarse groups plus Other."""
    if probabilities.shape[-1]!=NUM_CLASSES:
        raise ValueError(f"expected {NUM_CLASSES} classes, got {probabilities.shape[-1]}")
    return probabilities@grouping_matrix().to(probabilities.dtype)


def name(index:int)->str:
    return CLASS_NAMES[index]
