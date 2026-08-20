# Diffusion Models for Protein Virtual Staining

A diffusion-model project for protein virtual staining using the Human Protein Atlas and OpenCell.

## Core Implementation

Forward diffusion and closed-form noising · DDPM noise-prediction training · DDPM ancestral sampling · DDIM sampling and inversion · Conditional U-Net baseline · Diffusion Transformer with adaLN-Zero · Classifier-Free Guidance · Biological and image-quality evaluation · Interactive model comparison

## Biological Extensions

Protein-sequence conditioning with ESM-2, controlling the generated protein fluorescence channel and enabling evaluation on previously unseen proteins.

HPA-to-OpenCell domain adaptation with LoRA, demonstrating transfer from fixed-cell antibody imaging to live-cell endogenous fluorescence imaging.

## Data

**Human Protein Atlas** — primary training set. Fixed-cell immunofluorescence with DNA, microtubule, endoplasmic reticulum and protein-of-interest channels, plus gene, cell-line and subcellular-localization metadata.

**OpenCell** — live-cell confocal imaging of endogenously tagged proteins, used for LoRA fine-tuning and cross-domain evaluation rather than base training. It lacks the microtubule and ER channels, so the model must explicitly support missing conditioning channels.

**Protein sequences** — encoded with a frozen ESM-2 model, used only to produce sequence representations and not trained with the diffusion model.

## Status

In progress.
