"""The probe INTERNAL.md section 4 pre-registered, finally run.

Take public CELL-Diff weights, sample N images per protein conditioned on real landmark
channels, classify them with the same SubCell ensemble that produced pi_emp, and ask whether
D_cal is distinguishable from the null. "D_cal already small -> module is dead."

Two things about CELL-Diff shape this script. It exposes no guidance weight and never calls the
classifier-free path vendored in its DiT file, so this measures D_cal at the model's native
operating point rather than as a function of w - see `notes/celldiff-probe.md`. And its
`timestep_respacing` config field accepts a spaced schedule, so the 200-step default can be cut
to 50 without touching its code.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from urllib.request import urlopen

import torch

S3="https://czi-subcell-public.s3.us-west-2.amazonaws.com"
WEIGHTS="https://czi-celldiff-public.s3.us-west-2.amazonaws.com/v2"
UNIPROT="https://rest.uniprot.org/uniprotkb/search"

SHIM='''run=None


def init(*args,**kwargs):
    return None


def log(*args,**kwargs):
    return None
'''


def prepare_environment(repo:Path,deps:Path)->None:
    """CELL-Diff imports wandb at module scope but only uses it to log training metrics."""
    stub=deps/"wandb"
    stub.mkdir(parents=True,exist_ok=True)
    (stub/"__init__.py").write_text(SHIM)
    sys.path.insert(0,str(deps))
    sys.path.insert(0,str(repo))


def sequence_for(gene:str)->str|None:
    query=(f"{UNIPROT}?query=gene_exact:{gene}+AND+organism_id:9606+AND+reviewed:true"
           f"&format=fasta&size=1")
    try:
        with urlopen(query,timeout=60) as response:
            lines=response.read().decode().splitlines()
        return "".join(l for l in lines if not l.startswith(">")) or None
    except Exception:
        return None


def landmark_latents(vae,crop:torch.Tensor,device)->torch.Tensor:
    """CELL-Diff takes each landmark as a [0,1] grayscale plane mapped to [-1,1]."""
    from pvs.data.crops import MICROTUBULES, NUCLEUS, RETICULUM
    planes=[]
    for channel in (NUCLEUS,RETICULUM,MICROTUBULES):
        plane=(crop[channel:channel+1]*2-1)[None].to(device)
        with torch.no_grad():
            planes.append(vae.encode(plane).sample())
    return torch.cat(planes,dim=1)


def main()->None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--repo",type=Path,default=Path("third_party/CELL-Diff"))
    parser.add_argument("--deps",type=Path,default=Path("third_party/shims"))
    parser.add_argument("--weights",type=Path,default=Path("data/celldiff"))
    parser.add_argument("--ranking",type=Path,default=Path("outputs/u2os_screen/tail.json"))
    parser.add_argument("--genes",nargs="*",default=None)
    parser.add_argument("--num-genes",type=int,default=12)
    parser.add_argument("--samples",type=int,default=50)
    parser.add_argument("--steps",type=int,default=50)
    parser.add_argument("--size",type=int,default=256)
    parser.add_argument("--plan",type=Path,default=Path("data/plans/u2os_full.txt"))
    parser.add_argument("--encoder",type=Path,
                        default=Path("data/subcell/all_channels_ViT-ProtS-Pool.pth"))
    parser.add_argument("--classifiers",type=Path,default=Path("data/subcell/classifiers"))
    parser.add_argument("--out",type=Path,default=Path("outputs/celldiff_probe"))
    parser.add_argument("--device",default="auto")
    parser.add_argument("--save-images",type=int,default=0,
                        help="keep this many generated protein channels per gene, plus the real one")
    parser.add_argument("--fixed-cell",action="store_true",
                        help="hold the conditioning cell constant so the samples differ only by noise")
    parser.add_argument("--model-dtype",choices=("float32","float16","bfloat16"),
                        default="bfloat16")
    args=parser.parse_args()

    if args.device=="auto":
        args.device=("mps" if torch.backends.mps.is_available()
                     else "cuda" if torch.cuda.is_available() else "cpu")
    device=torch.device(args.device)
    args.out.mkdir(parents=True,exist_ok=True)
    prepare_environment(args.repo,args.deps)

    from copy import deepcopy
    from cell_diff.criterions.unidiff import UniDiffCriterions
    from cell_diff.data.hpa_data.vocabulary import Alphabet, convert_string_sequence_to_int_index
    from cell_diff.models.cell_diff.config import CELLDiffConfig
    from cell_diff.models.cell_diff.model import CELLDiffModel
    from cell_diff.models.vae.vae_config import VAEConfig
    from cell_diff.models.vae.vae_model import VAEModel

    from pvs.data.crops import load_crop, resize
    from pvs.eval.calibration import categorical_kl, null_distribution, total_variation
    from pvs.eval.localization import GROUPING_3, to_groups
    from pvs.eval.subcell import IMAGE_SIZE, SubCellEnsemble, subcell_input

    settings=dict(
        infer=True, ft=False,
        num_timesteps=200, timestep_respacing=str(args.steps),
        ddpm_schedule="squaredcos_cap_v2", diffusion_pred_type="noise",
        ddpm_beta_start=1e-4, ddpm_beta_end=0.02,
        num_down_blocks=3, latent_channels=4, vae_block_out_channels="128,256,512",
        block_out_channels="320,640,1280,1280", layers_per_block=2,
        mid_num_attention_heads=8, sample_size=args.size//4,
        esm_embedding="esm2", hidden_size=1280, max_protein_sequence_len=2048,
        num_hidden_layers=8, num_attention_heads=8, mlp_ratio=4, attn_drop=0.0,
        dit_patch_size=1, cell_image="nucl,er,mt", test_cell_image="nucl,er,mt",
        vae_loadcheck_path=str(args.weights/"vae_hpa_pretrained.bin"),
        loadcheck_path=str(args.weights/"cell_diff_hpa_pretrained_all.bin"),
    )
    print(json.dumps({"device":str(device),"steps":args.steps,"size":args.size,
                      "model_dtype":args.model_dtype}),flush=True)

    vae_args=deepcopy(settings); vae_args["loadcheck_path"]=settings["vae_loadcheck_path"]
    vae=VAEModel(config=VAEConfig(**vae_args)).to(device).eval()
    for parameter in vae.parameters():
        parameter.requires_grad=False
    model=CELLDiffModel(config=CELLDiffConfig(**settings),loss_fn=UniDiffCriterions)
    model_dtype={"float32":torch.float32,"float16":torch.float16,
                 "bfloat16":torch.bfloat16}[args.model_dtype]
    if device.type!="cuda" and model_dtype!=torch.float32:
        raise ValueError("reduced-precision CELL-Diff inference requires CUDA")
    model=model.to(dtype=model_dtype).to(device).eval()

    if args.genes:
        genes=args.genes
    else:
        report=json.loads(args.ranking.read_text())
        order=sorted(zip(report["genes"],report["heterogeneity"]),key=lambda p:-p[1])
        genes=[g for g,_ in order[:args.num_genes]]
    print(f"targets: {', '.join(genes)}",flush=True)

    stems={}
    for line in args.plan.read_text().splitlines():
        gene,_,_,_,stem=line.split("\t")
        stems.setdefault(gene,[]).append(stem)

    classifier=SubCellEnsemble.from_directory(args.encoder,args.classifiers).eval().to(device)
    vocab=Alphabet()
    scratch=args.out/"scratch"; scratch.mkdir(exist_ok=True)
    results={}
    results_path=args.out/"results.json"
    if results_path.exists():
        results=json.loads(results_path.read_text())

    for gene in genes:
        if gene in results:
            continue
        sequence=sequence_for(gene)
        if sequence is None:
            print(f"  {gene}: no reviewed human sequence in UniProt, skipped",flush=True); continue
        tokens=torch.LongTensor(convert_string_sequence_to_int_index(vocab,sequence))[None].to(device)

        probabilities=[]
        real_intensity=[]
        generated_intensity=[]
        started=time.time()
        keep=args.out/"images"/gene
        if args.save_images:
            keep.mkdir(parents=True,exist_ok=True)
        for index in range(args.samples):
            stem=stems[gene][0 if args.fixed_cell else index%len(stems[gene])]
            path=scratch/"cell.png"
            for attempt in range(3):
                try:
                    with urlopen(f"{S3}/{stem}_cell_image.png",timeout=90) as response:
                        path.write_bytes(response.read())
                    break
                except Exception:
                    time.sleep(2*(attempt+1))
            crop=resize(load_crop(path),args.size)
            real_intensity.append(crop[3].flatten())
            with torch.no_grad():
                landmarks=landmark_latents(vae,crop,device).to(dtype=model_dtype)
                latent=model.sequence_to_image(tokens,landmarks,
                                               progress=False,sampling_strategy="ddim")
                generated=vae.decode(latent.float()).sample
                composed=crop.clone().to(device)
                protein=((generated[0,0]+1)/2).clamp(0,1)
                generated_intensity.append(protein.cpu().flatten())
                composed[3]=protein
                batch=resize(subcell_input(composed.cpu()),IMAGE_SIZE)[None].to(device)
                _,probability=classifier(batch)
            if index<args.save_images:
                from PIL import Image as _Image
                import numpy as _np
                frame=(protein.cpu().numpy()*255).astype("uint8")
                _Image.fromarray(frame).save(keep/f"generated_{index:02d}.png")
                if index==0:
                    _Image.fromarray((crop[3].numpy()*255).astype("uint8")).save(keep/"real.png")
                    landmark=_np.stack([crop[2].numpy(),crop[1].numpy(),crop[0].numpy()],-1)
                    _Image.fromarray((landmark*255).astype("uint8")).save(keep/"landmarks.png")
                    (keep/"stem.txt").write_text(stem)
            probabilities.append(probability[0].cpu())
            if (index+1)%10==0:
                rate=(index+1)/(time.time()-started)
                print(f"  {gene}: {index+1}/{args.samples}  {rate:.2f} samples/s",flush=True)
        path.unlink(missing_ok=True)
        stacked=torch.stack(probabilities)
        real=torch.cat(real_intensity)
        generated=torch.cat(generated_intensity)
        results[gene]={"sequence_length":len(sequence),"samples":len(stacked),
                       "mean_probability":stacked.mean(0).tolist(),
                       "grouped":to_groups(stacked).mean(0).tolist(),
                       "intensity":{
                           "real_median":float(real.median()),
                           "real_p995":float(real.quantile(0.995)),
                           "generated_median":float(generated.median()),
                           "generated_p995":float(generated.quantile(0.995)),
                       }}
        results_path.write_text(json.dumps(results,indent=2))
        print(f"  {gene}: done, {len(stacked)} samples",flush=True)

    print(f"\nwrote {results_path}. Compare against pi_emp with experiments/celldiff_dcal.py")


if __name__=="__main__":
    main()
