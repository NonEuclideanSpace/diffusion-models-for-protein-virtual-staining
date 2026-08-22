import pytest
import torch

from pvs.diffusion.guidance import classifier_free_guidance
from pvs.models.blocks import ResidualBlock,SelfAttention
from pvs.models.conditioning import ChannelDropout,LabelEmbedding
from pvs.models.embeddings import timestep_embedding
from pvs.models.unet import ConditionalUNet


def _unet(**kwargs):
    defaults=dict(target_channels=1,condition_channels=3,base_width=16,
                  width_multipliers=(1,2),blocks_per_level=1,attention_resolutions=(16,),
                  num_classes=8,image_size=32)
    return ConditionalUNet(**{**defaults,**kwargs})


def test_timestep_embedding_is_distinct_and_smooth():
    e=timestep_embedding(torch.arange(1000),128)
    assert e.shape==(1000,128)
    assert torch.isfinite(e).all()
    near=(e[500]-e[501]).norm()
    far=(e[500]-e[900]).norm()
    assert near<far


def test_odd_embedding_dim_is_padded():
    assert timestep_embedding(torch.arange(4),65).shape==(4,65)


def test_residual_and_attention_start_as_identity():
    """Zero-initialized output projections keep a deep stack stable at step zero."""
    x,c=torch.randn(2,32,8,8),torch.randn(2,64)
    torch.testing.assert_close(ResidualBlock(32,32,64)(x,c),x)
    torch.testing.assert_close(SelfAttention(32)(x),x)


def test_attention_rejects_indivisible_head_count():
    with pytest.raises(ValueError):
        SelfAttention(30,heads=4)


def test_channel_dropout_substitutes_a_learned_plane():
    drop=ChannelDropout(3,64)
    condition=torch.randn(2,3,8,8)
    mask=torch.tensor([[1.,0.,1.],[1.,1.,1.]])
    masked,pattern=drop(condition,mask)
    torch.testing.assert_close(masked[0,0],condition[0,0])
    assert bool((masked[0,1]==drop.null_plane[1]).all())
    torch.testing.assert_close(masked[1],condition[1])
    assert pattern.shape==(2,64)
    assert not torch.equal(pattern[0],pattern[1])


def test_channel_dropout_pattern_index_is_a_bijection():
    drop=ChannelDropout(3,8)
    masks=torch.tensor([[float(b>>i&1) for i in range(3)] for b in range(8)])
    assert sorted(drop.pattern_index(masks).tolist())==list(range(8))


def test_label_embedding_reserves_a_null_index():
    labels=LabelEmbedding(8,64)
    assert labels.null_index==8
    assert labels.embedding.num_embeddings==9
    assert labels(None,batch=4,device=torch.device("cpu")).shape==(4,64)
    dropped=labels.drop(torch.zeros(4096,dtype=torch.long))
    assert 0.05<(dropped==8).float().mean().item()<0.16


def test_unet_shapes_and_zero_initialized_head():
    model=_unet()
    x,t=torch.randn(2,1,32,32),torch.randint(0,1000,(2,))
    condition,labels=torch.randn(2,3,32,32),torch.randint(0,8,(2,))
    out=model(x,t,condition,labels)
    assert out.shape==x.shape
    assert out.abs().max().item()==0.0


def test_unet_accepts_missing_channels_and_missing_labels():
    model=_unet()
    x,t=torch.randn(2,1,32,32),torch.randint(0,1000,(2,))
    condition=torch.randn(2,3,32,32)
    assert model(x,t,condition,None).shape==x.shape
    assert model(x,t,condition,torch.randint(0,8,(2,)),torch.zeros(2,3)).shape==x.shape


def test_unet_requires_the_conditioning_it_was_built_for():
    with pytest.raises(ValueError,match="conditioning channels"):
        _unet()(torch.randn(2,1,32,32),torch.randint(0,1000,(2,)))


def test_unet_gradients_reach_every_parameter():
    model=_unet()
    out=model(torch.randn(2,1,32,32),torch.randint(0,1000,(2,)),
              torch.randn(2,3,32,32),torch.randint(0,8,(2,)))
    (out.sum()+sum(p.sum() for p in model.parameters())*0).backward()
    unused=[n for n,p in model.named_parameters() if p.grad is None]
    assert not unused,unused[:5]


def test_missing_channel_changes_the_prediction():
    """If masking a channel left the output unchanged, the conditioning would be inert."""
    torch.manual_seed(0)
    model=_unet()
    for parameter in model.head[-1].parameters():
        torch.nn.init.normal_(parameter,std=0.1)
    x,t=torch.randn(2,1,32,32),torch.randint(0,1000,(2,))
    condition,labels=torch.randn(2,3,32,32),torch.randint(0,8,(2,))
    full=model(x,t,condition,labels,torch.ones(2,3))
    partial=model(x,t,condition,labels,torch.tensor([[1.,0.,0.],[1.,0.,0.]]))
    assert (full-partial).abs().max().item()>1e-6


def test_guidance_weight_one_is_the_conditional_model():
    model=_unet()
    guided=classifier_free_guidance(model,1.0)
    x,t=torch.randn(2,1,32,32),torch.randint(0,1000,(2,))
    condition,labels=torch.randn(2,3,32,32),torch.randint(0,8,(2,))
    torch.testing.assert_close(guided(x,t,condition,labels),model(x,t,condition,labels))


def test_guidance_interpolates_between_the_two_passes():
    torch.manual_seed(0)
    model=_unet()
    for parameter in model.head[-1].parameters():
        torch.nn.init.normal_(parameter,std=0.1)
    x,t=torch.randn(2,1,32,32),torch.randint(0,1000,(2,))
    condition,labels=torch.randn(2,3,32,32),torch.randint(0,8,(2,))
    conditional=model(x,t,condition,labels)
    unconditional=model(x,t,condition,None)
    for weight in (0.0,2.0,3.5):
        expected=unconditional+weight*(conditional-unconditional)
        actual=classifier_free_guidance(model,weight,batched=False)(x,t,condition,labels)
        torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-5)


def test_batched_guidance_matches_the_sequential_form():
    torch.manual_seed(0)
    model=_unet().eval()
    for parameter in model.head[-1].parameters():
        torch.nn.init.normal_(parameter,std=0.1)
    x,t=torch.randn(2,1,32,32),torch.randint(0,1000,(2,))
    condition,labels=torch.randn(2,3,32,32),torch.randint(0,8,(2,))
    a=classifier_free_guidance(model,3.0,batched=True)(x,t,condition,labels)
    b=classifier_free_guidance(model,3.0,batched=False)(x,t,condition,labels)
    torch.testing.assert_close(a,b,rtol=1e-4,atol=1e-5)


def test_guidance_rejects_an_unknown_drop_mode():
    with pytest.raises(ValueError):
        classifier_free_guidance(_unet(),2.0,drop="everything")


def test_state_embedding_starts_silent_and_matches_its_null():
    from pvs.models.conditioning import StateEmbedding
    embedding=StateEmbedding(4,32).eval()
    state=torch.tensor([[1e5,4e4,0.6,3.],[2e5,5e4,0.4,5.]])
    # Zero-initialised output layer, so a fresh model is unchanged by adding state conditioning.
    assert float(embedding(state,2,torch.device("cpu")).abs().max())==0.0
    with torch.no_grad():
        for parameter in embedding.project[-1].parameters():
            parameter.add_(0.2*torch.randn_like(parameter))
    assert float(embedding(state,2,torch.device("cpu")).abs().max())>0.0
    assert torch.allclose(embedding(state,2,torch.device("cpu"),mask=torch.zeros(2)),
                          embedding(None,2,torch.device("cpu")))


def test_state_embedding_standardises_against_running_statistics():
    from pvs.models.conditioning import StateEmbedding
    embedding=StateEmbedding(2,16)
    embedding.train()
    for _ in range(400):
        embedding.standardise(torch.randn(64,2)*torch.tensor([1e5,3.])+torch.tensor([5e5,7.]))
    embedding.eval()
    scaled=embedding.standardise(torch.tensor([[5e5,7.]]))
    assert float(scaled.abs().max())<1.0


def test_state_embedding_rejects_the_wrong_width():
    from pvs.models.conditioning import StateEmbedding
    with pytest.raises(ValueError):
        StateEmbedding(4,16)(torch.zeros(2,3),2,torch.device("cpu"))

