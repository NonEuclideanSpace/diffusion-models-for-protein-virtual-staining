import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def cache(tmp_path:Path)->Path:
    shards=tmp_path/"cache"/"genes"
    shards.mkdir(parents=True)
    generator=np.random.default_rng(1)
    for gene in ("AAA","BBB","CCC"):
        np.savez_compressed(shards/f"{gene}.npz",
                            x=generator.integers(0,256,size=(4,48,48,4),dtype=np.uint8),
                            meta=np.array(["control","Nucleoplasm","U2OS","48"]))
    return tmp_path/"cache"


def train(cache:Path,out:Path,*extra:str)->str:
    result=subprocess.run(
        [sys.executable,"experiments/train.py","--smoke","--cache",str(cache),"--out",str(out),*extra],
        capture_output=True,text=True,timeout=600,cwd=Path(__file__).resolve().parents[1])
    assert result.returncode==0, result.stderr[-2000:]
    return result.stdout


def test_smoke_reaches_a_checkpoint(cache,tmp_path):
    out=tmp_path/"run"
    lines=train(cache,out).strip().splitlines()
    header=json.loads(lines[0])
    assert header["preset"]=="smoke" and header["cells"]==12
    assert (out/"latest.pt").exists()
    assert (out/"config.json").exists()
    assert json.loads(lines[-2])["step"]==12


def test_resume_continues_from_the_checkpoint(cache,tmp_path):
    out=tmp_path/"run"
    train(cache,out)
    output=train(cache,out,"--steps","20")
    assert "resumed at step 12" in output
    assert json.loads(output.strip().splitlines()[-2])["step"]==20


def test_explicit_flags_beat_the_smoke_preset(cache,tmp_path):
    output=train(cache,tmp_path/"run","--steps","8","--batch","2")
    assert json.loads(output.strip().splitlines()[-2])["step"]==8
