"""Image and distribution metrics for virtual staining.

Pixel metrics and distribution metrics disagree in a way that is the whole point here. A
regression baseline trained on mean squared error wins pixel correlation and loses badly on
distribution coverage, because it predicts the conditional mean rather than a sample. Any
evaluation that reports only one family will therefore rank the models wrongly, which is why
`INTERNAL.md` section 7 requires several.

Precision and recall are the pair that matters most for the project's central claim.
Precision asks whether generated samples land inside the real distribution; recall asks how
much of the real distribution they cover. A model that produces idealized, canonical
localizations scores high precision and low recall — exactly what ProtiCelli reports, and
exactly what "learns a consensus rather than a distribution" predicts.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

EPS=1e-8


def _flat(x:Tensor)->Tensor:
    return x.flatten(1)


def pearson_correlation(a:Tensor,b:Tensor)->Tensor:
    """Per-sample linear correlation between two images."""
    x,y=_flat(a),_flat(b)
    x=x-x.mean(1,keepdim=True)
    y=y-y.mean(1,keepdim=True)
    return (x*y).sum(1)/(x.norm(dim=1)*y.norm(dim=1)).clamp_min(EPS)


def mutual_information(a:Tensor,b:Tensor,bins:int=32,value_range:tuple[float,float]=(-1.0,1.0))->Tensor:
    """Histogram estimate of I(a; b) per sample, in nats.

    Catches monotone but non-linear agreement that Pearson misses, which matters because
    fluorescence intensity relationships are rarely linear.
    """
    low,high=value_range
    quantize=lambda t:((_flat(t).clamp(low,high)-low)/(high-low)*(bins-1)).round().long()
    x,y=quantize(a),quantize(b)
    joint=torch.zeros(x.shape[0],bins*bins,device=a.device)
    joint.scatter_add_(1,x*bins+y,torch.ones_like(x,dtype=joint.dtype))
    joint=(joint/joint.sum(1,keepdim=True).clamp_min(EPS)).reshape(-1,bins,bins)
    px=joint.sum(2,keepdim=True)
    py=joint.sum(1,keepdim=True)
    ratio=(joint+EPS)/((px*py)+EPS)
    return (joint*ratio.log()).sum(dim=(1,2))


def _gaussian_kernel(size:int,sigma:float,device:torch.device)->Tensor:
    coordinates=torch.arange(size,device=device,dtype=torch.float32)-size//2
    kernel=torch.exp(-coordinates**2/(2*sigma**2))
    kernel=kernel/kernel.sum()
    return kernel[:,None]@kernel[None,:]


def ssim(a:Tensor,b:Tensor,window:int=11,sigma:float=1.5,data_range:float=2.0)->Tensor:
    channels=a.shape[1]
    kernel=_gaussian_kernel(window,sigma,a.device).expand(channels,1,window,window)
    blur=lambda t:F.conv2d(t,kernel,padding=window//2,groups=channels)
    mu_a,mu_b=blur(a),blur(b)
    var_a=blur(a*a)-mu_a**2
    var_b=blur(b*b)-mu_b**2
    covariance=blur(a*b)-mu_a*mu_b
    c1,c2=(0.01*data_range)**2,(0.03*data_range)**2
    numerator=(2*mu_a*mu_b+c1)*(2*covariance+c2)
    denominator=(mu_a**2+mu_b**2+c1)*(var_a+var_b+c2)
    return (numerator/denominator.clamp_min(EPS)).flatten(1).mean(1)


def multiscale_ssim(a:Tensor,b:Tensor,scales:int=3,**kwargs)->Tensor:
    """SSIM averaged over a resolution pyramid, so texture and layout both count."""
    values=[]
    for level in range(scales):
        if min(a.shape[-2:])<16:
            break
        values.append(ssim(a,b,**kwargs))
        if level<scales-1:
            a,b=F.avg_pool2d(a,2),F.avg_pool2d(b,2)
    return torch.stack(values).mean(0)


def glcm_features(images:Tensor,levels:int=16,distance:int=1,
                  value_range:tuple[float,float]=(-1.0,1.0))->Tensor:
    """Contrast, homogeneity, energy and correlation from a horizontal co-occurrence matrix.

    The Haralick descriptors the field uses to catch over-smoothing: a model that predicts the
    conditional mean loses high-frequency texture while keeping pixel correlation intact.
    """
    low,high=value_range
    quantized=((images.clamp(low,high)-low)/(high-low)*(levels-1)).round().long()
    left,right=quantized[...,:-distance],quantized[...,distance:]
    batch=images.shape[0]
    pairs=(left*levels+right).flatten(1)
    matrix=torch.zeros(batch,levels*levels,device=images.device)
    matrix.scatter_add_(1,pairs,torch.ones_like(pairs,dtype=matrix.dtype))
    matrix=(matrix/matrix.sum(1,keepdim=True).clamp_min(EPS)).reshape(batch,levels,levels)

    indices=torch.arange(levels,device=images.device,dtype=torch.float32)
    i=indices[None,:,None]
    j=indices[None,None,:]
    difference=(i-j).abs()
    contrast=(matrix*difference**2).sum(dim=(1,2))
    homogeneity=(matrix/(1+difference**2)).sum(dim=(1,2))
    energy=(matrix**2).sum(dim=(1,2))
    mean_i=(matrix*i).sum(dim=(1,2),keepdim=True)
    mean_j=(matrix*j).sum(dim=(1,2),keepdim=True)
    std_i=((matrix*(i-mean_i)**2).sum(dim=(1,2))).sqrt().clamp_min(EPS)
    std_j=((matrix*(j-mean_j)**2).sum(dim=(1,2))).sqrt().clamp_min(EPS)
    correlation=((matrix*(i-mean_i)*(j-mean_j)).sum(dim=(1,2)))/(std_i*std_j)
    return torch.stack([contrast,homogeneity,energy,correlation],dim=1)


def frechet_distance(real:Tensor,generated:Tensor)->float:
    """Frechet distance between two Gaussians fitted to embedding sets."""
    if real.shape[1]!=generated.shape[1]:
        raise ValueError("embedding dimensions must match")
    mu_r,mu_g=real.mean(0).double(),generated.mean(0).double()
    cov=lambda x,mu:((x.double()-mu).T@(x.double()-mu))/(x.shape[0]-1)
    sigma_r,sigma_g=cov(real,mu_r),cov(generated,mu_g)
    offset=torch.eye(sigma_r.shape[0],dtype=torch.float64,device=real.device)*1e-6
    product=(sigma_r+offset)@(sigma_g+offset)
    eigenvalues=torch.linalg.eigvals(product).real.clamp_min(0)
    return float((mu_r-mu_g).pow(2).sum()+sigma_r.trace()+sigma_g.trace()-2*eigenvalues.sqrt().sum())


def _knn_radius(x:Tensor,k:int)->Tensor:
    distances=torch.cdist(x,x)
    distances.fill_diagonal_(float("inf"))
    return distances.topk(k,largest=False).values[:,-1]


def precision_recall(real:Tensor,generated:Tensor,k:int=3)->tuple[float,float]:
    """Improved precision and recall over embedding manifolds (Kynkaanniemi et al., 2019).

    Precision is the fraction of generated samples inside the real manifold; recall is the
    fraction of real samples inside the generated one. High precision with low recall is the
    signature of a model that produces canonical outputs and misses real variability.
    """
    real_radius=_knn_radius(real,k)
    generated_radius=_knn_radius(generated,k)
    cross=torch.cdist(generated,real)
    precision=float((cross<=real_radius[None,:]).any(1).float().mean())
    recall=float((cross.T<=generated_radius[None,:]).any(1).float().mean())
    return precision,recall


def coverage_statistics(real:Tensor,generated:Tensor)->dict[str,float]:
    """Second-moment summaries that expose variance collapse independently of any manifold."""
    spread=lambda x:float((x-x.mean(0)).norm(dim=1).mean())
    return {
        "real_total_variance":float(real.var(0).sum()),
        "generated_total_variance":float(generated.var(0).sum()),
        "variance_ratio":float(generated.var(0).sum()/real.var(0).sum().clamp_min(EPS)),
        "real_distance_to_centroid":spread(real),
        "generated_distance_to_centroid":spread(generated),
    }
