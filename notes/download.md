# Downloading the crops

Everything below runs on the laptop, in a normal macOS terminal. It cannot run through the
assistant: the bridge to this machine has no network access, which is why the index was built
in the cloud and the images are fetched here.

**No installation is needed.** The fetcher uses only the Python standard library — no torch,
no `uv sync`, nothing to set up. The system `python3` that ships with macOS is enough.

## 1. Find the drive

```bash
ls /Volumes
```

External drives appear there. Toshiba drives often have a space in the name, so quote the
path everywhere below. Substitute the real name for `TOSHIBA`.

## 2. Unpack the index

```bash
cd ~/Documents/ml-systems-lab/architecture-labs/dmfpvs-v0
tar xzf hpa-index.tar.gz          # creates plates/, 1219 files, 77 MB
```

The index is the map from gene to image location. Building it took about an hour of scanning
S3 manifests; unpacking it takes a second. It stays on the internal disk — it is small, and
the fetcher reads it constantly.

## 3. Look before leaping

```bash
python3 data/fetch_crops.py \
  --index plates \
  --tsv data/hpa/subcellular_location.tsv \
  --out "/Volumes/TOSHIBA/hpa-crops" \
  --dry-run
```

This downloads nothing. It prints the gene selection, how many cells each gene has, how many
clear the 50-cell threshold the calibration estimator needs, and the total size. Read those
numbers before starting — they are the ones that decide whether the study is viable.

## 4. Run it

```bash
python3 data/fetch_crops.py \
  --index plates \
  --tsv data/hpa/subcellular_location.tsv \
  --out "/Volumes/TOSHIBA/hpa-crops" \
  --workers 12
```

Roughly **51 GB, about 70 minutes** at 12 workers on a measured 14 MB/s. Progress prints every
500 cells with a running total and an estimate of the time left.

## Notes

**Interrupt it freely.** Control-C at any point, rerun the same command, and it picks up where
it stopped — each file is checked before fetching and written through a temporary name, so a
kill mid-write cannot leave a truncated image behind.

**Raising `--workers` past about 16 is unlikely to help** and is impolite to a public bucket
that is hosting this for free. 12 was measured as roughly the point where throughput stops
improving.

**`--max-cells-per-gene` defaults to 200.** That is deliberately generous against a 50-cell
requirement, leaving room to discard cells that fail segmentation checks later. Lower it to
100 to halve the download if disk or time is short.

**Layout on disk** is `<out>/<GENE>/<plate>_<position>_<sample>_<cell>_cell_image.png`, which
is what `pvs.data.crops.HpaCrops` expects.

**If the drive disconnects mid-run**, the script will start failing every fetch and report a
rising failure count rather than crashing. Stop it, remount, rerun.

## What this is for

These are the cells for probe A: does a compartment classifier detect single-cell
heterogeneity that HPA's own annotators independently flagged as real. Of the genes fetched,
150 carry HPA's `Single-cell variation spatial` flag and 267 are controls matched on both
reliability and main location. If the two groups do not separate, module M1 has no measurable
target and the project's primary claim needs to change. See `m1-feasibility.md`.
