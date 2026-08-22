from pathlib import Path

import numpy as np
import pytest
import torch

from pvs.data.cache import CachedCrops, ShardCache, ShardShuffleSampler, read_manifest


@pytest.fixture
def cache(tmp_path:Path)->Path:
    shards=tmp_path/"genes"
    shards.mkdir(parents=True)
    generator=np.random.default_rng(0)
    for gene,cells in (("AAA",5),("BBB",3),("CCC",4)):
        array=generator.integers(0,256,size=(cells,64,64,4),dtype=np.uint8)
        np.savez_compressed(shards/f"{gene}.npz",x=array,
                            meta=np.array([("positive" if gene=="AAA" else "control"),
                                           "Nucleoplasm","U2OS","64"]))
    return tmp_path


def test_manifest_reads_every_shard(cache):
    shards=read_manifest(cache)
    assert [s.gene for s in shards]==["AAA","BBB","CCC"]
    assert [s.cells for s in shards]==[5,3,4]
    assert shards[0].kind=="positive" and shards[1].kind=="control"


def test_length_is_total_cells(cache):
    assert len(CachedCrops(cache,size=32))==12


def test_locate_maps_index_to_shard_and_offset(cache):
    dataset=CachedCrops(cache,size=32)
    assert [dataset.locate(i)[0].gene for i in range(12)]== \
        ["AAA"]*5+["BBB"]*3+["CCC"]*4
    assert dataset.locate(0)[1]==0
    assert dataset.locate(4)[1]==4
    assert dataset.locate(5)[1]==0
    assert dataset.locate(11)[1]==3


def test_item_shapes_and_range(cache):
    item=CachedCrops(cache,size=32)[7]
    assert item["x0"].shape==(1,32,32)
    assert item["condition"].shape==(3,32,32)
    assert item["gene"]=="BBB"
    assert item["labels"].item()==1
    assert item["x0"].min()>=-1.0-1e-5 and item["x0"].max()<=1.0+1e-5


def test_gene_filter_restricts_and_reindexes(cache):
    dataset=CachedCrops(cache,size=32,genes=["CCC"])
    assert len(dataset)==4
    assert dataset[0]["gene"]=="CCC"
    assert dataset[0]["labels"].item()==0


def test_lru_evicts_but_stays_correct(cache):
    store=ShardCache(capacity=1)
    shards=read_manifest(cache)
    first,mask=store.get(shards[0].path)
    first=first.copy()
    assert mask is None
    store.get(shards[1].path)
    store.get(shards[2].path)
    assert len(store.entries)==1
    assert np.array_equal(store.get(shards[0].path)[0],first)


def test_sampler_is_a_permutation_and_varies_by_epoch(cache):
    dataset=CachedCrops(cache,size=32)
    sampler=ShardShuffleSampler(dataset,block=2,seed=3)
    first=list(sampler)
    assert sorted(first)==list(range(12))
    sampler.set_epoch(1)
    assert list(sampler)!=first


def test_sampler_keeps_a_block_contiguous_in_shards(cache):
    dataset=CachedCrops(cache,size=32)
    sampler=ShardShuffleSampler(dataset,block=1,seed=0)
    genes=[dataset.locate(i)[0].gene for i in sampler]
    assert len(set(genes[:1]))==1
    runs=[genes[0]]
    for gene in genes[1:]:
        if gene!=runs[-1]:
            runs.append(gene)
    assert len(runs)==3


def test_matches_direct_decode(cache):
    dataset=CachedCrops(cache,size=64)
    with np.load(cache/"genes"/"AAA.npz") as blob:
        raw=blob["x"][2]
    from pvs.data.crops import normalize, split
    crop=torch.from_numpy(raw.astype("float32")/255.0).permute(2,0,1)
    protein,landmarks=split(normalize(crop))
    assert torch.allclose(dataset[2]["x0"],protein)
    assert torch.allclose(dataset[2]["condition"],landmarks)


def test_state_covariates_come_from_the_nucleus_channel(cache):
    from pvs.data.cache import STATE_NAMES, state_covariates
    dataset=CachedCrops(cache,size=32,state=True)
    item=dataset[0]
    assert item["state"].shape==(len(STATE_NAMES),)
    assert torch.isfinite(item["state"]).all()
    # No mask in this fixture, so the cell window is the whole frame.
    assert item["state"][STATE_NAMES.index("cell_area")]==pytest.approx(64*64)


def test_state_covariates_track_dna_content():
    from pvs.data.cache import STATE_NAMES, state_covariates
    from pvs.data.crops import NUCLEUS
    dim=STATE_NAMES.index("dna")
    dim_density=STATE_NAMES.index("chromatin_density")
    dim_area=STATE_NAMES.index("nucleus_area")
    faint,bright=torch.zeros(4,16,16),torch.zeros(4,16,16)
    faint[NUCLEUS,4:12,4:12]=0.3
    bright[NUCLEUS,4:12,4:12]=0.6            # same area, twice the density
    a,b=state_covariates(faint),state_covariates(bright)
    assert float(b[dim])==pytest.approx(2*float(a[dim]))
    assert float(b[dim_density])==pytest.approx(2*float(a[dim_density]))
    assert float(b[dim_area])==pytest.approx(float(a[dim_area]))


def test_state_covariates_respect_a_mask():
    from pvs.data.cache import STATE_NAMES, state_covariates
    from pvs.data.crops import NUCLEUS
    crop=torch.zeros(4,16,16)
    crop[NUCLEUS,2:6,2:6]=0.5                # the cell of interest
    crop[NUCLEUS,10:14,10:14]=0.9            # a neighbour that must not count
    mask=torch.zeros(16,16); mask[:8,:8]=1.0
    masked=state_covariates(crop,mask)
    unmasked=state_covariates(crop)
    assert float(masked[STATE_NAMES.index("dna")])<float(unmasked[STATE_NAMES.index("dna")])
    assert float(masked[STATE_NAMES.index("cell_area")])==pytest.approx(64.0)


def test_state_is_absent_unless_asked_for(cache):
    assert "state" not in CachedCrops(cache,size=32)[0]
