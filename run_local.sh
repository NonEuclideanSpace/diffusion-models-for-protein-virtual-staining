#!/usr/bin/env bash
# One command to run the heavy work natively on macOS, where MPS is available.
# The Cowork sandbox cannot do this: it is a Linux VM with no GPU, no network and 3 GB of RAM.
set -euo pipefail
cd "$(dirname "$0")"

DRIVE="${DRIVE:-}"
STEP="${1:-help}"

need_env() {
  [ -d .venv ] || uv sync
}

case "$STEP" in
bench)
  need_env
  uv run python -c "
import torch,time
from pvs.eval.subcell import SubCellEncoder
d='mps' if torch.backends.mps.is_available() else 'cpu'
print('device:',d)
m=SubCellEncoder().to(d).eval()
x=torch.randn(4,4,448,448,device=d)
with torch.no_grad():
    m(x)
    if d=='mps': torch.mps.synchronize()
    t=time.time()
    for _ in range(3): m(x)
    if d=='mps': torch.mps.synchronize()
    s=(time.time()-t)/12
print(f'{s*1000:.0f} ms per cell -> 2400 cells in {2400*s/60:.1f} min')
"
  ;;

weights)
  mkdir -p data/subcell/classifiers
  S3=https://czi-subcell-public.s3.us-west-2.amazonaws.com
  echo "fetching SubCell encoder (333 MB)"
  curl -# -L -C - -o data/subcell/all_channels_ViT-ProtS-Pool.pth \
    "$S3/models/all_channels_ViT-ProtS-Pool.pth"
  for s in 0 1 2 3 4 5 6 7 8 9; do
    curl -sL -C - -o "data/subcell/classifiers/all_channels_ViT_MLP_classifier_seed_${s}.pth" \
      "$S3/models/all_channels_ViT_MLP_classifier/all_channels_ViT_MLP_classifier_seed_${s}.pth"
  done
  du -sh data/subcell
  ;;

subset)
  [ -d plates ] || tar xzf hpa-index.tar.gz
  : "${DRIVE:?set DRIVE to the crop destination, e.g. DRIVE=/Volumes/TOSHIBA/hpa-subset}"
  uv run python data/fetch_subset.py --plan data/subset_plan.tsv --out "$DRIVE" --workers 14
  ;;

screen)
  # The U2OS heterogeneity screen. 24,000 cells of SubCell inference. On the two Xeon cores in
  # the cloud container this is 6.7 hours; on an M-series GPU it should be a small fraction of
  # that. Resumable: it writes one gene at a time and skips what it already has.
  need_env
  uv run python experiments/probe_a_stream.py \
    --plan data/plans/u2os.txt \
    --out outputs/u2os_screen \
    --scratch data/stream \
    --encoder data/subcell/all_channels_ViT-ProtS-Pool.pth \
    --classifiers data/subcell/classifiers \
    --threads "${THREADS:-8}" --download-workers "${WORKERS:-24}"
  ;;

celldiff-weights)
  # 6.9 GB of the 137 GB bucket. The training LMDBs are not needed.
  mkdir -p data/celldiff third_party
  B=https://czi-celldiff-public.s3.us-west-2.amazonaws.com/v2/checkpoints
  curl -# -L -C - -o data/celldiff/cell_diff_hpa_pretrained_all.bin "$B/cell_diff/hpa_pretrained_all.bin"
  curl -# -L -C - -o data/celldiff/vae_hpa_pretrained.bin "$B/vae/hpa_pretrained.bin"
  [ -d third_party/CELL-Diff ] || git clone --depth 1 https://github.com/BoHuangLab/CELL-Diff.git third_party/CELL-Diff
  # Only these are needed to import the inference path; install.sh lists far more.
  uv pip install --target third_party/shims --no-deps \
    diffusers loguru einops timm fair-esm torchvision requests urllib3 \
    charset-normalizer idna platformdirs click
  du -sh data/celldiff third_party
  ;;

probe-celldiff)
  # The probe INTERNAL.md section 4 pre-registered. Needs celldiff-weights and screen first.
  need_env
  uv run python experiments/celldiff_probe.py \
    --num-genes "${GENES:-12}" --samples "${SAMPLES:-50}" --steps "${STEPS:-50}"
  ;;

state)
  # Task 1's gate, run here instead of on rented hardware: it is SubCell inference plus mask
  # downloads, which MPS handles about as fast as a rented card and at no cost per hour.
  need_env
  uv run python experiments/state_decomposition.py \
    --genes "${GENES:-200}" --sample random --seed "${SEED:-1}" --device auto \
    --plan data/plans/u2os.txt --rank outputs/u2os_screen/results.json \
    --scratch data/state --out outputs/state_decomposition_mac.json
  ;;

probe)
  need_env
  : "${DRIVE:?set DRIVE to where the crops were written}"
  uv run python experiments/probe_a.py \
    --crops "$DRIVE" --plan data/subset_plan.tsv \
    --encoder data/subcell/all_channels_ViT-ProtS-Pool.pth \
    --classifiers data/subcell/classifiers \
    --device "$(uv run python -c 'import torch;print("mps" if torch.backends.mps.is_available() else "cpu")')" \
    --batch 16 --out outputs/probe_a
  ;;

full)
  [ -d plates ] || tar xzf hpa-index.tar.gz
  : "${DRIVE:?set DRIVE to the destination, e.g. DRIVE=/Volumes/TOSHIBA/hpa-crops}"
  python3 data/fetch_crops.py --index plates --tsv data/hpa/subcellular_location.tsv \
    --out "$DRIVE" --workers 12
  ;;

*)
  cat <<'USAGE'
usage: DRIVE=/Volumes/YOURDRIVE/... ./run_local.sh <step>

  bench     measure this machine on the heaviest kernel (ViT-B/16 at 448px). 30 seconds.
  weights   fetch the SubCell encoder and its ten classifier heads. 370 MB.
  subset    fetch the 40-gene probe-A development subset. 2400 cells, about 10 GB.
  probe     run probe A on the subset, using MPS if it is available.
  full      fetch the complete probe-A selection. 39,088 cells, about 163 GB, 90 minutes.

Nothing here needs a server. `full` is the only step that wants a large disk.
USAGE
  ;;
esac
