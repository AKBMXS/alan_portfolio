#!/usr/bin/env python3
# Wraps embed_fnspid_shard.py to process every shard,
# skip existing outputs, and print live progress.

import glob
import subprocess
import sys
import os
import torch

# Directory containing your .parquet shards
SHARD_DIR  = "fnspid_shards_full"
# The per-shard embed script
EMB_SCRIPT = "embed_fnspid_shard.py"
# Tweak batch size to fill your GPU memory
BATCH_SIZE = 1024

def main():
    # Find all shards
    shards = sorted(glob.glob(os.path.join(SHARD_DIR, "fnspid_*.parquet")))
    print(f"Found {len(shards)} shards in {SHARD_DIR}")

    # Report device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Embedding on device: {device}\n")

    # Loop
    for shard in shards:
        out = shard.replace(".parquet", "_embeddings.npz")
        if os.path.exists(out):
            print(f"→ Skipping {os.path.basename(shard)} (already done)")
            continue

        print(f"→ Embedding {os.path.basename(shard)} → {os.path.basename(out)}", flush=True)
        cmd = [
            sys.executable, EMB_SCRIPT,
            "-i", shard,
            "-o", out,
            "-b", str(BATCH_SIZE)
        ]
        # Run and let stdout flow; suppress stderr warnings
        subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL)

    print("\n✅ All missing shards have been embedded.")

if __name__ == "__main__":
    main()

#!python3 embed_all_shards.py