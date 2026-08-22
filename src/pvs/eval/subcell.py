"""Loader for the SubCell protein-localization encoder.

The published checkpoints predate the current transformers ViT parameter names, so the
state dict has to be remapped before it will load. Everything else is a faithful
reconstruction: a 4-channel ViT-B/16 at 448x448 followed by gated attention pooling over
the patch tokens, producing the 1536-d embedding the SubCell paper reports.

Weights: s3://czi-subcell-public/models/ (MIT). See data/fetch.sh.
"""
from __future__ import annotations

import re
from pathlib import Path

import torch
from torch import Tensor,nn

IMAGE_SIZE=448
NUM_LOCALIZATIONS=31

# SubCell consumes channels in the order its own path_list.csv names them: r, y, b, g, that is
# microtubules, ER, nucleus, protein. Crops on disk are stored nucleus, ER, microtubules,
# protein, so the first and third have to be exchanged.
CROP_TO_SUBCELL=(2,1,0,3)
PATCH_SIZE=16
HIDDEN_SIZE=768
POOL_HEADS=2
EMBEDDING_DIM=HIDDEN_SIZE*POOL_HEADS

_RENAMES=(
    (r"^encoder\.encoder\.layer\.","layers."),
    (r"^encoder\.","" ),
    (r"attention\.attention\.query","attention.q_proj"),
    (r"attention\.attention\.key","attention.k_proj"),
    (r"attention\.attention\.value","attention.v_proj"),
    (r"attention\.output\.dense","attention.o_proj"),
    (r"\bintermediate\.dense","mlp.fc1"),
    (r"^(layers\.\d+\.)output\.dense",r"\1mlp.fc2"),
)


def _remap(key:str)->str:
    for pattern,replacement in _RENAMES:
        key=re.sub(pattern,replacement,key)
    return key


def subcell_input(crop:Tensor,percentile:float=99.5)->Tensor:
    """Turn a crop in on-disk order into SubCell's expected input.

    Two things differ from what the diffusion model wants, and getting either wrong fails
    silently — every protein is assigned the same class with high confidence, which looks like
    a biological finding rather than a bug.

    **Range.** SubCell expects [0, 1]. Feeding it [-1, 1], the range a diffusion model wants,
    collapses every prediction onto one class: across four proteins with completely different
    annotations the largest total-variation distance between predictions was 0.07, against
    0.73 once the range is right.

    **Channel order.** Exchange the nucleus and microtubule channels, per CROP_TO_SUBCELL.

    Accepts (4, H, W) or (B, 4, H, W).
    """
    batched=crop.ndim==4
    stack=crop if batched else crop[None]
    if stack.shape[1]!=4:
        raise ValueError(f"expected 4 channels in crop order, got {stack.shape[1]}")
    stack=stack[:,CROP_TO_SUBCELL]
    if stack.min()<-1e-6:
        raise ValueError("crop must be in [0, 1]; pass the raw crop, not the diffusion-normalized one")
    limits=torch.quantile(stack.flatten(2),percentile/100.0,dim=2).clamp_min(1e-6)
    anchor=limits[:,2:3,None,None].clamp_min(1e-6)
    prepared=(torch.minimum(stack,limits[:,:,None,None])/anchor).clamp(0,1)
    return prepared if batched else prepared[0]


OFFICIAL_CROP=640
MASK_DILATION=7


def dilate(mask:Tensor,size:int=MASK_DILATION)->Tensor:
    """Grey dilation by a square structuring element, matching the reference preprocessing."""
    import torch.nn.functional as F

    batched=mask.ndim==4
    stack=mask if batched else mask[None]
    grown=F.max_pool2d(stack,size,stride=1,padding=size//2)
    return grown if batched else grown[0]


def centre_crop(image:Tensor,size:int=OFFICIAL_CROP)->Tensor:
    height,width=image.shape[-2:]
    top,left=(height-size)//2,(width-size)//2
    return image[...,top:top+size,left:left+size]


def subcell_official(crop:Tensor,mask:Tensor,size:int=OFFICIAL_CROP)->Tensor:
    """SubCell's published inference recipe, which differs from `subcell_input` in three ways.

    The reference implementation multiplies the crop by a dilated single-cell mask before
    anything else - "to remove the effects of surrounding cell regions", in the paper's words -
    then normalizes by a **global** min-max over all four channels jointly, then centre-crops to
    640 without resizing. `subcell_input` instead clips at a per-channel percentile, divides by
    the microtubule channel, resizes to 448, and never masks.

    That last difference is not cosmetic. In a 1024-pixel crop the centre cell covers about a
    sixth of the frame, and roughly seven in ten bright nucleus pixels belong to its neighbours,
    so an unmasked crop lets the encoder read the neighbourhood. Cells sharing an imaging field
    share neighbours, which is a mechanism for between-field structure that has nothing to do
    with either microscopy or biology.

    Masking first and normalizing second matters: zeroing the background pins the minimum at 0.
    """
    batched=crop.ndim==4
    stack=crop if batched else crop[None]
    sheet=mask if mask.ndim==4 else mask[None]
    if stack.shape[1]!=4:
        raise ValueError(f"expected 4 channels in crop order, got {stack.shape[1]}")
    masked=stack[:,CROP_TO_SUBCELL]*dilate(sheet)
    masked=centre_crop(masked,size)
    low=masked.amin(dim=(1,2,3),keepdim=True)
    high=masked.amax(dim=(1,2,3),keepdim=True)
    prepared=((masked-low)/(high-low+1e-8)).clamp(0,1)
    return prepared if batched else prepared[0]


class GatedAttentionPool(nn.Module):
    """Gated attention MIL pooling, two heads concatenated."""

    def __init__(self,hidden_size:int=HIDDEN_SIZE,attention_size:int=512,heads:int=POOL_HEADS)->None:
        super().__init__()
        self.value=nn.Linear(hidden_size,attention_size)
        self.gate=nn.Linear(hidden_size,attention_size)
        self.score=nn.Linear(attention_size,heads)

    def forward(self,tokens:Tensor)->Tensor:
        weights=self.score(torch.tanh(self.value(tokens))*torch.sigmoid(self.gate(tokens))).softmax(1)
        return torch.einsum("bnk,bnd->bkd",weights,tokens).flatten(1)


class SubCellEncoder(nn.Module):
    """ViT-B/16 over 4-channel cell crops, pooled to a 1536-d embedding."""

    def __init__(self,num_channels:int=4)->None:
        super().__init__()
        from transformers import ViTConfig,ViTModel

        self.vit=ViTModel(
            ViTConfig(
                image_size=IMAGE_SIZE,
                patch_size=PATCH_SIZE,
                num_channels=num_channels,
                hidden_size=HIDDEN_SIZE,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=4*HIDDEN_SIZE,
            ),
            add_pooling_layer=False,
        )
        self.pool=GatedAttentionPool()

    @classmethod
    def from_checkpoint(cls,path:str|Path,num_channels:int=4)->SubCellEncoder:
        state=torch.load(path,map_location="cpu",weights_only=False)
        model=cls(num_channels)
        wanted=set(model.vit.state_dict())
        encoder={}
        for key,value in state.items():
            if not key.startswith("encoder."):
                continue
            renamed=_remap(key)
            if renamed in wanted:
                encoder[renamed]=value
        if len(encoder)!=len(wanted):
            raise RuntimeError(f"remapped {len(encoder)} of {len(wanted)} encoder tensors; naming has drifted")
        model.vit.load_state_dict(encoder,strict=True)
        model.pool.load_state_dict({
            "value.weight":state["pool_model.attention_v.1.weight"],
            "value.bias":state["pool_model.attention_v.1.bias"],
            "gate.weight":state["pool_model.attention_u.1.weight"],
            "gate.bias":state["pool_model.attention_u.1.bias"],
            "score.weight":state["pool_model.attention.weight"],
            "score.bias":state["pool_model.attention.bias"],
        },strict=True)
        return model.eval()

    @torch.no_grad()
    def forward(self,images:Tensor)->Tensor:
        # SubCell's own inference runs 640x640 native rather than the 448 it trained at, so the
        # position grid has to be interpolated there. Asking for it at 448 is not free - the
        # interpolation runs on every forward pass - so it is requested only when it is needed.
        native=images.shape[-1]==IMAGE_SIZE and images.shape[-2]==IMAGE_SIZE
        hidden=(self.vit(images) if native
                else self.vit(images,interpolate_pos_encoding=True)).last_hidden_state
        return self.pool(hidden[:,1:])


class SubCellClassifier(nn.Module):
    """The published MLP head: 1536-d embedding to 31 localization logits.

    Ten seeds are released. Their disagreement is the only handle on this classifier's own
    uncertainty, and since the reference distribution for module M1 has to be predicted rather
    than looked up, that uncertainty propagates directly into the result. Use the ensemble.
    """

    def __init__(self,embedding_dim:int=EMBEDDING_DIM,num_classes:int=NUM_LOCALIZATIONS)->None:
        super().__init__()
        self.layers=nn.Sequential(
            nn.Linear(embedding_dim,512),nn.ReLU(),nn.Dropout(0.0),
            nn.Linear(512,256),nn.ReLU(),nn.Dropout(0.0),
            nn.Linear(256,num_classes),
        )

    @classmethod
    def from_checkpoint(cls,path:str|Path)->SubCellClassifier:
        state=torch.load(path,map_location="cpu",weights_only=False)
        model=cls()
        model.layers.load_state_dict(state,strict=True)
        return model.eval()

    def forward(self,embeddings:Tensor)->Tensor:
        return self.layers(embeddings)


class SubCellEnsemble(nn.Module):
    """Encoder plus every released classifier seed, averaged in probability space."""

    def __init__(self,encoder:SubCellEncoder,heads:list[SubCellClassifier])->None:
        super().__init__()
        self.encoder=encoder
        self.heads=nn.ModuleList(heads)

    @classmethod
    def from_directory(cls,encoder_path:str|Path,classifier_dir:str|Path,limit:int|None=None)->SubCellEnsemble:
        paths=sorted(Path(classifier_dir).glob("*.pth"))[:limit]
        if not paths:
            raise FileNotFoundError(f"no classifier checkpoints under {classifier_dir}")
        return cls(SubCellEncoder.from_checkpoint(encoder_path),
                   [SubCellClassifier.from_checkpoint(p) for p in paths])

    @torch.no_grad()
    def forward(self,images:Tensor)->tuple[Tensor,Tensor]:
        """Return the embedding and the seed-averaged class probabilities."""
        embeddings=self.encoder(images)
        probabilities=torch.stack([head(embeddings).softmax(-1) for head in self.heads]).mean(0)
        return embeddings,probabilities

    @torch.no_grad()
    def seed_disagreement(self,images:Tensor)->Tensor:
        """Standard deviation across seeds of the predicted probabilities, per class."""
        embeddings=self.encoder(images)
        return torch.stack([head(embeddings).softmax(-1) for head in self.heads]).std(0)
