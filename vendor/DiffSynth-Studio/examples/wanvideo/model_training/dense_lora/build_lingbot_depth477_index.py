#!/usr/bin/env python3
"""Build the validated bright-40 index for preprocessed 477-frame clips."""

import argparse
import json
import os
from pathlib import Path

from dataset_477 import DepthVideoDataset


parser = argparse.ArgumentParser()
parser.add_argument("--root", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--min-rgb-mean", type=float, default=40.0)
args = parser.parse_args()

dataset = DepthVideoDataset(args.root, min_rgb_mean=args.min_rgb_mean)
payload = dataset.index_payload()
output = Path(args.output)
tmp = output.with_suffix(output.suffix + f".tmp-{os.getpid()}")
tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
os.replace(tmp, output)
print(json.dumps({"samples": len(dataset), "scan_stats": dataset.scan_stats}, indent=2))
