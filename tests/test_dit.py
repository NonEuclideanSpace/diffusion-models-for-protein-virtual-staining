import pytest
import torch

from pvs.diffusion.guidance import classifier_free_guidance
from pvs.models.dit import DiT,DiTBlock,sincos_position_embedding


def _dit(**kwargs):
    defaults=dict(image_size=32,patch_size=4,target_channels=1,condition_channels=3,
                  hidden_size=64,depth=2,heads=4,num_classes=8)
    return DiT(**{**defaults,**kwargs})


def _inputs(batch=2,size=32):
    return (torch.randn(batch,1,size,size),torch.randint(0,1000,(batch,)),
            torch.randn(batch,3,size,size),torch.randint(0,8,(batch,)))


def test_position_embedding_is_unique_per_grid_cell():
    p=sincos_position_embedding(64,8,8)
    assert p.shape==(64,64)
    assert len({tuple(row.tolist()) for row in p})==64


def test_position_embedding_encodes_the_two_axes_separately():
    """First half varies along rows only, second half along columns only."""
    p=sincos_position_embedding(64,4,4).reshape(4,4,64)
    rows,cols=p[...,:32],p[...,32:]
    assert torch.allclose(rows[1,0],rows[1,3])
    assert torch.allclose(cols[0,1],cols[3,1])
    assert not torch.allclose(rows[0,0],rows[1,0])


def test_position_embedding_requires_a_multiple_of_four():
    with pytest.raises(ValueError):
        sincos_position_embedding(66,4,4)


def test_adaln_zero_makes_every_block_start_as_the_identity():
    block=DiTBlock(64,4)
    tokens,conditioning=torch.randn(2,16,64),torch.randn(2,64)
    torch.testing.assert_close(block(tokens,conditioning),tokens)


def test_output_is_zero_at_initialization():
    x,t,condition,labels=_inputs()
    assert _dit()(x,t,condition,labels).abs().max().item()==0.0


def test_unpatchify_inverts_the_patch_layout():
    model=_dit()
    image=torch.arange(2*1*32*32,dtype=torch.float32).reshape(2,1,32,32)
    grid,patch=model.grid,model.patch_size
    patched=(image.reshape(2,1,grid,patch,grid,patch)
                  .permute(0,2,4,3,5,1)
                  .reshape(2,grid*grid,patch*patch*1))
    torch.testing.assert_close(model.unpatchify(patched),image)


def test_image_size_must_divide_by_patch_size():
    with pytest.raises(ValueError):
        _dit(image_size=30,patch_size=4)


def test_gradients_reach_every_parameter():
    model=_dit(num_modes=5)
    x,t,condition,labels=_inputs()
    model(x,t,condition,labels,None,torch.randint(0,5,(2,))).sum().backward()
    unused=[n for n,p in model.named_parameters() if p.grad is None]
    assert not unused,unused[:5]


def _activated(**kwargs):
    """adaLN-Zero output is identically zero at init, so tests of conditioning need weights."""
    torch.manual_seed(0)
    model=_dit(**kwargs)
    for block in model.blocks:
        torch.nn.init.normal_(block.modulation[-1].weight,std=0.05)
    torch.nn.init.normal_(model.final.projection.weight,std=0.05)
    torch.nn.init.normal_(model.final.modulation[-1].weight,std=0.05)
    return model.eval()


def test_every_conditioning_source_changes_the_output():
    model=_activated(num_modes=5)
    x,t,condition,labels=_inputs()
    base=model(x,t,condition,labels,None,torch.zeros(2,dtype=torch.long))
    differs=lambda other:(base-other).abs().max().item()>1e-6
    assert differs(model(x,torch.randint(0,1000,(2,)),condition,labels,None,torch.zeros(2,dtype=torch.long)))
    assert differs(model(x,t,torch.randn_like(condition),labels,None,torch.zeros(2,dtype=torch.long)))
    assert differs(model(x,t,condition,(labels+1)%8,None,torch.zeros(2,dtype=torch.long)))
    assert differs(model(x,t,condition,labels,None,torch.ones(2,dtype=torch.long)))
    assert differs(model(x,t,condition,labels,torch.zeros(2,3),torch.zeros(2,dtype=torch.long)))


def test_modes_are_ignored_when_not_supplied():
    model=_activated(num_modes=5)
    x,t,condition,labels=_inputs()
    torch.testing.assert_close(model(x,t,condition,labels),model(x,t,condition,labels,None,None))


def test_guidance_wraps_a_dit_the_same_way_as_a_unet():
    model=_activated()
    x,t,condition,labels=_inputs()
    conditional=model(x,t,condition,labels)
    unconditional=model(x,t,condition,None)
    guided=classifier_free_guidance(model,3.0,batched=False)(x,t,condition,labels)
    torch.testing.assert_close(guided,unconditional+3.0*(conditional-unconditional),rtol=1e-4,atol=1e-5)


def test_it_trains_on_a_diffusion_objective():
    """One optimizer step must move the loss, otherwise the zero initialization is a trap."""
    from pvs.diffusion import NoiseSchedule,ddpm_loss
    torch.manual_seed(0)
    model=_dit()
    schedule=NoiseSchedule.make(100,"cosine")
    x0,condition,labels=torch.randn(4,1,32,32),torch.randn(4,3,32,32),torch.randint(0,8,(4,))
    wrapped=lambda xt,t:model(xt,t,condition,labels)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    first=ddpm_loss(wrapped,schedule,x0,torch.randint(0,100,(4,)))
    for _ in range(20):
        loss=ddpm_loss(wrapped,schedule,x0,torch.randint(0,100,(4,)))
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    final=ddpm_loss(wrapped,schedule,x0,torch.randint(0,100,(4,)))
    assert final<first

def test_dit_state_conditioning_changes_the_prediction():
    torch.manual_seed(0)
    model=DiT(image_size=32,patch_size=4,target_channels=1,condition_channels=3,
              hidden_size=64,depth=2,heads=2,num_state=4).eval()
    x,t=torch.randn(3,1,32,32),torch.randint(0,100,(3,))
    condition=torch.randn(3,3,32,32)
    state=torch.tensor([[1e5,4e4,.6,3.],[2e5,5e4,.4,5.],[1.5e5,4.5e4,.5,4.]])
    # adaLN-Zero makes the whole model output zero at init, which hides any conditioning test.
    assert float(model(x,t,condition,state=state).abs().max())==0.0
    with torch.no_grad():
        for module in (model.final,model.state.project[-1]):
            for parameter in module.parameters():
                parameter.add_(0.05*torch.randn_like(parameter))
    assert not torch.allclose(model(x,t,condition,state=state),
                              model(x,t,condition,state=None),atol=1e-6)
    assert torch.allclose(model(x,t,condition,state=state,state_mask=torch.zeros(3)),
                          model(x,t,condition,state=None),atol=1e-6)


def test_dit_without_state_is_unchanged():
    model=DiT(image_size=32,patch_size=4,target_channels=1,condition_channels=3,
              hidden_size=64,depth=2,heads=2)
    assert model.state is None
    assert model(torch.randn(2,1,32,32),torch.randint(0,100,(2,)),
                 torch.randn(2,3,32,32)).shape==(2,1,32,32)
