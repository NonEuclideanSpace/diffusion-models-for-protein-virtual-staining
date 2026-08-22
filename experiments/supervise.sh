#!/usr/bin/env bash
# The container reaps long-lived background jobs. Every stage resumes from what it has already
# written, so restarting on death costs one gene instead of the run.
set -u
cd /home/claude/pvs-reference
mkdir -p outputs
log() { echo "[$(date -u +%H:%M:%S)] $*" >> outputs/supervisor.log; }

json_count() { python3 -c "import json,sys;print(len(json.load(open(sys.argv[1]))))" "$1" 2>/dev/null || echo 0; }
file_count() { ls "$1" 2>/dev/null | wc -l; }

# guard is a pgrep pattern unique to this stage, so a reaped-and-restarted supervisor never
# starts a second copy of a run that is still alive.
run_stage() {
  local name=$1 target=$2 counter=$3 path=$4 guard=$5; shift 5
  while [ "$($counter "$path")" -lt "$target" ]; do
    if pgrep -f "$guard" > /dev/null; then sleep 30; continue; fi
    log "$name at $($counter "$path")/$target, starting"
    uv run python "$@" >> "outputs/$name.log" 2>&1
    sleep 10
  done
  log "$name complete at $target"
}

run_stage expanded 116 json_count outputs/probe_a_large/results.json \
  "probe_a_stream.py .*probe_a_large" \
  experiments/probe_a_stream.py --plan /home/claude/data/subset/plan2.txt --out outputs/probe_a_large

run_stage cache 1564 file_count /home/claude/data/cache256/genes \
  "build_cache.py" \
  experiments/build_cache.py --plan /home/claude/data/subset/u2os_full.txt --out /home/claude/data/cache256

# The U2OS screen moved to the user's Mac, where MPS runs SubCell ~15x faster than these two
# cores. It is bandwidth-bound there rather than compute-bound, so duplicating it here would
# add nothing and both would write the same genes.

log "all stages complete"
