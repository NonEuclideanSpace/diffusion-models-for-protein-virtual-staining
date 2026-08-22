"""Regenerate every image and data file the showcase site uses.

The site is static, so everything it shows has to be baked here first. Four stages, each
independently runnable, because the download stage takes half an hour and the rest take seconds:

    figures   the composited cell images and the forward-diffusion strip
    atlas     23,520 real single-cell embeddings reduced to 64 dimensions, plus a display sample
    thumbs    the sampled cells' crops, fetched and packed into sprite sheets
    toy2d     a small noise predictor trained on a 2-D mixture, with its reverse trajectory

Nothing here invents data. Every pixel comes from a Human Protein Atlas crop and every number from
a measurement already in the repository.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import torch
from PIL import Image

S3="https://czi-subcell-public.s3.us-west-2.amazonaws.com"
BG=(14,13,19)
# the Human Protein Atlas channel convention, kept because anyone who has looked at
# immunofluorescence reads it without a legend
PALETTE=[(88,132,255),(246,206,86),(244,92,82),(72,226,146)]
CHANNELS=("nucleus","endoplasmic reticulum","microtubules","protein")
SHOWCASE=[("442_C11_2_3","ACAD9","mitochondria"),("613_B10_4_7","CTSB","vesicles"),
          ("637_A2_2_5","TGOLN2","Golgi apparatus"),("1893_C15_12_20","PNISR","nuclear speckles"),
          ("663_E1_1_7","ARF6","cytosol"),("703_B10_2_5","HSP90B1","endoplasmic reticulum")]
THUMB=96
SHEET=2048
PER_ROW=SHEET//THUMB


def fetch(key:str,destination:Path)->bool:
    for attempt in range(3):
        try:
            with urlopen(f"{S3}/{key}",timeout=90) as response:
                destination.write_bytes(response.read())
            return True
        except Exception:
            time.sleep(2*(attempt+1))
    return False


def stems_by_gene(plan:Path)->dict[str,list[str]]:
    stems=defaultdict(list)
    for line in plan.read_text().splitlines():
        gene,_,_,_,stem=line.split("\t")
        stems[gene].append(stem)
    return stems


def normalise(channel:np.ndarray,percentile:float=99.6)->np.ndarray:
    """Each channel gets its own scale. Sharing one buries the protein layer under microtubules."""
    lit=channel[channel>0]
    return np.clip(channel/np.percentile(lit,percentile),0,1) if lit.size else channel


def channels_of(path:Path)->list[np.ndarray]:
    """PIL reads the crop as RGBA, which puts the channels in nucleus, ER, microtubule, protein order."""
    array=np.array(Image.open(path)).astype(np.float32)
    return [normalise(array[...,index]) for index in range(4)]


def composite(path:Path,size:int,protein_gain:float=1.45)->Image.Image:
    nucleus,reticulum,microtubules,protein=channels_of(path)
    rgb=np.stack([microtubules*0.80+reticulum*0.42+protein*0.05,
                  protein*protein_gain+reticulum*0.34,
                  nucleus+protein*0.10],-1)
    return Image.fromarray((np.clip(rgb,0,1)**0.92*255).astype(np.uint8)).resize((size,size),Image.LANCZOS)


def cosine_alpha_bar(t:float,offset:float=0.008)->float:
    f=lambda u: math.cos((u+offset)/(1+offset)*math.pi/2)**2
    return f(t)/f(0.0)


def stage_figures(args:argparse.Namespace)->None:
    out=args.showcase; out.mkdir(parents=True,exist_ok=True)
    scratch=args.scratch/"figures"; scratch.mkdir(parents=True,exist_ok=True)
    plan=stems_by_gene(args.plan)
    for stem,gene,_ in SHOWCASE:
        target=scratch/f"{stem}.png"
        if not target.exists():
            path=next(s for s in plan[gene] if Path(s).name==stem)
            if not fetch(f"{path}_cell_image.png",target):
                raise RuntimeError(f"could not fetch {stem}")
    print(f"{len(SHOWCASE)} crops in {scratch}",flush=True)

    hero=SHOWCASE[0][0]
    size,gap=420,16
    panel=Image.new("RGB",(4*size+3*gap,size),BG)
    for index,(value,colour) in enumerate(zip(channels_of(scratch/f"{hero}.png"),PALETTE)):
        tint=(value[...,None]*np.array(colour,np.float32)).astype(np.uint8)
        panel.paste(Image.fromarray(tint).resize((size,size),Image.LANCZOS),(index*(size+gap),0))
    for destination in (out,args.docs/"assets"):
        panel.save(destination/"task_four_channels.jpg",quality=91,optimize=True)

    size,gap=420,14
    grid=Image.new("RGB",(3*size+2*gap,2*size+gap),BG)
    for index,(stem,_,_) in enumerate(SHOWCASE):
        grid.paste(composite(scratch/f"{stem}.png",size),((index%3)*(size+gap),(index//3)*(size+gap)))
    for destination in (out,args.docs/"assets"):
        grid.save(destination/"compartment_grid.jpg",quality=91,optimize=True)

    size,gap=384,12
    steps=[0.0,0.14,0.28,0.42,0.58,0.74,1.0]
    base=np.array(composite(scratch/f"{hero}.png",size,1.35)).astype(np.float32)/127.5-1
    noise=np.random.default_rng(3).standard_normal(base.shape).astype(np.float32)
    strip=Image.new("RGB",(len(steps)*size+(len(steps)-1)*gap,size),BG)
    for index,t in enumerate(steps):
        bar=cosine_alpha_bar(t)
        frame=math.sqrt(bar)*base+math.sqrt(1-bar)*noise
        strip.paste(Image.fromarray(np.clip((frame+1)*127.5,0,255).astype(np.uint8)),(index*(size+gap),0))
    for destination in (out,args.docs/"assets"):
        strip.save(destination/"hero_forward_diffusion.jpg",quality=90,optimize=True)

    # the scrubber recomputes noising in the browser, so it needs the clean cell on its own
    composite(scratch/f"{hero}.png",512,1.35).save(args.docs/"assets"/"scrubber_cell.jpg",quality=92)
    wide=composite(scratch/f"{hero}.png",1400,1.5)
    width,height=wide.size
    wide.crop((0,int(height*0.18),width,int(height*0.68))).save(out/"bg_ACAD9.jpg",quality=87,optimize=True)
    # the repository header and the link-preview card are the same 1280x640 image
    banner=Image.new("RGB",(1280,640),BG)
    wide=composite(scratch/f"{hero}.png",640,1.5)
    banner.paste(wide,(0,0)); banner.paste(composite(scratch/f"{SHOWCASE[1][0]}.png",640,1.5),(640,0))
    veil=Image.new("RGB",(1280,640),BG)
    banner=Image.blend(banner,veil,0.30)
    for destination in (out,args.docs/"assets"):
        banner.save(destination/"readme_banner.jpg",quality=88,optimize=True)
    print("figures written",flush=True)


def compartment_of(location:str)->str:
    return location.split(";")[0] if location else "Unannotated"


def stage_atlas(args:argparse.Namespace)->None:
    """Reduce every screened cell to 64 dimensions and choose the subset the page will draw."""
    plan=stems_by_gene(args.plan)
    ranking={}
    header,*body=args.ranking.read_text().splitlines()
    column={name:index for index,name in enumerate(header.split("\t"))}
    for line in body:
        cell=line.split("\t")
        ranking[cell[column["gene"]]]={"location":cell[column["hpa_location"]],
                                      "heterogeneity":float(cell[column["heterogeneity"]])}

    vectors,genes,stems,skipped=[],[],[],0
    for path in sorted(args.embeddings.glob("*.pt")):
        gene=path.stem
        if gene not in plan or gene not in ranking:
            continue
        blob=torch.load(path,map_location="cpu",weights_only=False)
        embedding=blob["embedding"].float()
        # the screen wrote cells in plan order and dropped none when the counts agree, which is
        # the only condition under which a row can be matched back to its image
        if len(embedding)!=len(plan[gene]):
            skipped+=1
            continue
        vectors.append(embedding)
        genes.extend([gene]*len(embedding))
        stems.extend(plan[gene])
    matrix=torch.cat(vectors)
    print(f"{len(matrix)} cells from {len(set(genes))} genes, {skipped} genes skipped",flush=True)

    matrix=(matrix-matrix.mean(0))/matrix.std(0).clamp_min(1e-6)
    _,_,basis=torch.pca_lowrank(matrix,q=args.components,niter=4)
    reduced=(matrix@basis).numpy().astype(np.float32)

    compartments=[compartment_of(ranking[g]["location"]) for g in genes]
    counts=defaultdict(int)
    for name in compartments:
        counts[name]+=1
    keep={name for name,count in counts.items() if count>=args.min_class}
    labels=[name if name in keep else "Other" for name in compartments]

    rng=np.random.default_rng(0)
    order=rng.permutation(len(matrix))
    quota=defaultdict(int)
    per_class=max(args.sample//max(len(set(labels)),1),1)
    chosen=[]
    for index in order:
        name=labels[index]
        if quota[name]<per_class:
            quota[name]+=1; chosen.append(int(index))
    for index in order:                                  # top up to the requested size
        if len(chosen)>=args.sample:
            break
        if int(index) not in set(chosen):
            chosen.append(int(index))
    chosen=sorted(set(chosen))[:args.sample]

    np.savez_compressed(args.showcase/"atlas_pca.npz",
                        reduced=reduced,chosen=np.array(chosen,np.int32),
                        heterogeneity=np.array([ranking[g]["heterogeneity"] for g in genes],np.float32))
    (args.showcase/"atlas_meta.json").write_text(json.dumps(
        {"gene":genes,"stem":stems,"label":labels,"chosen":chosen}))
    print(f"wrote atlas_pca.npz ({reduced.shape}) and a display sample of {len(chosen)}",flush=True)


def stage_thumbs(args:argparse.Namespace)->None:
    meta=json.loads((args.showcase/"atlas_meta.json").read_text())
    chosen=meta["chosen"]; stems=meta["stem"]
    scratch=args.scratch/"thumbs"; scratch.mkdir(parents=True,exist_ok=True)
    cache=args.showcase/"thumbs"; cache.mkdir(parents=True,exist_ok=True)

    todo=[i for i in chosen if not (cache/f"{i}.jpg").exists()]
    print(f"{len(todo)} thumbnails to fetch of {len(chosen)}",flush=True)
    def one(index:int)->None:
        stem=stems[index]; temporary=scratch/f"{index}.png"
        if not fetch(f"{stem}_cell_image.png",temporary):
            return
        composite(temporary,THUMB,1.5).save(cache/f"{index}.jpg",quality=82)
        temporary.unlink(missing_ok=True)
    with ThreadPoolExecutor(args.workers) as pool:
        for done,_ in enumerate(pool.map(one,todo),1):
            if done%200==0:
                print(f"  {done}/{len(todo)}",flush=True)

    have=[i for i in chosen if (cache/f"{i}.jpg").exists()]
    per_sheet=PER_ROW*PER_ROW
    index={}
    for sheet in range((len(have)+per_sheet-1)//per_sheet):
        canvas=Image.new("RGB",(SHEET,SHEET),BG)
        for slot,cell in enumerate(have[sheet*per_sheet:(sheet+1)*per_sheet]):
            row,col=divmod(slot,PER_ROW)
            canvas.paste(Image.open(cache/f"{cell}.jpg"),(col*THUMB,row*THUMB))
            index[str(cell)]=[sheet,col,row]
        canvas.save(args.docs/"assets"/f"sprite_{sheet}.jpg",quality=84,optimize=True)
    (args.showcase/"sprite_index.json").write_text(json.dumps(index))
    print(f"packed {len(have)} thumbnails into {len(index) and max(v[0] for v in index.values())+1} sheets",flush=True)


def stage_toy2d(args:argparse.Namespace)->None:
    """Train the repository's own toy model and record where the samples are at every step."""
    from pvs.diffusion import NoiseSchedule, ddpm_loss
    from pvs.diffusion.ddim import _transport, uniform_timesteps
    from pvs.models.toy import ToyMLP

    torch.manual_seed(0)
    modes=8; radius=2.0
    angles=torch.arange(modes)*(2*math.pi/modes)
    centres=torch.stack([angles.cos(),angles.sin()],dim=1)*radius
    draw=lambda n: centres[torch.randint(0,modes,(n,))]+torch.randn(n,2)*0.12

    schedule=NoiseSchedule.make(args.timesteps,"cosine")
    model=ToyMLP()
    optimiser=torch.optim.Adam(model.parameters(),lr=2e-3)
    losses=[]
    for step in range(args.train_steps):
        loss=ddpm_loss(model,schedule,draw(512))
        optimiser.zero_grad(set_to_none=True); loss.backward(); optimiser.step()
        losses.append(round(loss.item(),4))
        if (step+1)%(args.train_steps//8)==0:
            print(f"  step {step+1:>5}  loss {sum(losses[-200:])/len(losses[-200:]):.4f}",flush=True)
    model.eval()

    grid=uniform_timesteps(len(schedule),args.frames)
    one=schedule.alpha_bars.new_ones(())
    x=torch.randn(args.points,2)
    frames=[x.numpy().round(3).tolist()]
    with torch.no_grad():
        for i in reversed(range(len(grid))):
            t=grid[i].expand(x.shape[0])
            bar=schedule.alpha_bars[grid[i]]
            bar_next=schedule.alpha_bars[grid[i-1]] if i>0 else one
            x=_transport(x,model(x,t),bar,bar_next,0.0,None)
            frames.append(x.numpy().round(3).tolist())
    target=draw(args.points).numpy().round(3).tolist()
    (args.docs/"data"/"toy2d.json").write_text(json.dumps(
        {"frames":frames,"target":target,"modes":centres.numpy().round(3).tolist(),
         "loss":losses[::max(len(losses)//200,1)],
         "parameters":sum(p.numel() for p in model.parameters()),
         "timesteps":args.timesteps,"inference_steps":args.frames}))
    print(f"wrote toy2d.json: {len(frames)} frames of {args.points} points",flush=True)


def stage_pack(args:argparse.Namespace)->None:
    """Merge the 2-D layout, the per-cell metadata and the sprite index into what the page loads."""
    meta=json.loads((args.showcase/"atlas_meta.json").read_text())
    layout=json.loads((args.showcase/"atlas_2d.json").read_text())
    sprites=json.loads((args.showcase/"sprite_index.json").read_text()) if (args.showcase/"sprite_index.json").exists() else {}
    heterogeneity=np.load(args.showcase/"atlas_pca.npz")["heterogeneity"]
    chosen=meta["chosen"]
    payload={"x":[round(v,4) for v in layout["x"]],"y":[round(v,4) for v in layout["y"]],
             "gene":[meta["gene"][i] for i in chosen],
             "label":[meta["label"][i] for i in chosen],
             "het":[round(float(heterogeneity[i]),4) for i in chosen],
             "sprite":[sprites.get(str(i)) for i in chosen]}
    if not (len(payload["x"])==len(chosen)):
        raise RuntimeError(f"layout has {len(payload['x'])} points but {len(chosen)} were chosen")
    (args.docs/"data"/"atlas.json").write_text(json.dumps(payload,separators=(",",":")))
    have=sum(1 for s in payload["sprite"] if s)
    print(f"wrote atlas.json: {len(chosen)} cells, {have} with thumbnails",flush=True)


def main()->None:
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=["figures","atlas","thumbs","toy2d","pack","all"])
    parser.add_argument("--plan",type=Path,default=Path("data/plans/u2os.txt"))
    parser.add_argument("--ranking",type=Path,default=Path("outputs/u2os_heterogeneity_ranking.tsv"))
    parser.add_argument("--embeddings",type=Path,default=Path("outputs/u2os_screen/embeddings"))
    parser.add_argument("--showcase",type=Path,default=Path("outputs/showcase"))
    parser.add_argument("--docs",type=Path,default=Path("docs"))
    parser.add_argument("--scratch",type=Path,default=Path("data/showcase"))
    parser.add_argument("--components",type=int,default=64)
    parser.add_argument("--sample",type=int,default=3000)
    parser.add_argument("--min-class",type=int,default=200)
    parser.add_argument("--workers",type=int,default=24)
    parser.add_argument("--points",type=int,default=512)
    parser.add_argument("--frames",type=int,default=60)
    parser.add_argument("--timesteps",type=int,default=200)
    parser.add_argument("--train-steps",type=int,default=4000)
    args=parser.parse_args()

    (args.docs/"assets").mkdir(parents=True,exist_ok=True)
    (args.docs/"data").mkdir(parents=True,exist_ok=True)
    args.showcase.mkdir(parents=True,exist_ok=True)
    args.scratch.mkdir(parents=True,exist_ok=True)
    stages={"figures":stage_figures,"atlas":stage_atlas,"thumbs":stage_thumbs,
            "toy2d":stage_toy2d,"pack":stage_pack}
    for name in (stages if args.stage=="all" else [args.stage]):
        print(f"\n=== {name} ===",flush=True)
        stages[name](args)


if __name__=="__main__":
    main()
