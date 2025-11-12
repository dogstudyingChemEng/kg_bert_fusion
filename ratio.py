#!/usr/bin/env python3
# File: ratio.py
# Usage:
#   python ratio.py
# or
#   python ratio.py --model1 /path/to/kg-bert-1 --model2 /path/to/kg-bert-2 --fused /path/to/kg-bert-fused-ot

import os
import argparse
import math
import torch


def humanize(n):
    if n >= 1_000_000_000:
        return f"{n/1_000_000_000:.3f}B"
    if n >= 1_000_000:
        return f"{n/1_000_000:.3f}M"
    if n >= 1_000:
        return f"{n/1_000:.3f}K"
    return str(n)


def count_from_state_dict(sd):
    # sd: mapping name -> tensor-like object with numel() or numpy array
    total = 0
    try:
        for v in sd.values():
            # torch tensors, numpy arrays or nested dicts
            if hasattr(v, "numel"):
                total += int(v.numel())
            elif hasattr(v, "size"):
                # numpy
                try:
                    total += int(math.prod(v.shape))
                except Exception:
                    pass
            elif isinstance(v, dict):
                total += count_from_state_dict(v)
    except Exception:
        pass
    return total


def load_and_count(path):
    """
    Try to load a model file or checkpoint and count parameters.
    Supports:
      - PyTorch state_dict files (.pt, .pth, .bin) loadable by torch.load
      - directories containing pytorch_model.bin
      - a raw state-dict-like python file (fallback not guaranteed)
    Requires torch to be installed to inspect nn.Modules.
    """
    if os.path.isdir(path):
        # common HF name
        candidates = ["pytorch_model.bin", "pytorch_model.pt", "model.bin", "model.pt"]
        for c in candidates:
            p = os.path.join(path, c)
            if os.path.exists(p):
                path = p
                break

    if not os.path.exists(path):
        raise FileNotFoundError(f"Path not found: {path}")

    data = torch.load(path, map_location="cpu")

    # If it's a full nn.Module saved, try to detect
    if isinstance(data, torch.nn.Module):
        return sum(int(p.numel()) for p in data.parameters())

    # If it's a state_dict mapping
    if isinstance(data, dict):
        # HuggingFace sometimes stores {"model": state_dict, ...}
        # or {"state_dict": state_dict}
        for k in ("state_dict", "model", "params"):
            if k in data and isinstance(data[k], dict):
                return count_from_state_dict(data[k])
        # Otherwise assume dict is the state_dict
        return count_from_state_dict(data)

    # Unknown type: attempt to call state_dict attribute
    if hasattr(data, "state_dict"):
        sd = data.state_dict()
        return count_from_state_dict(sd)

    raise RuntimeError("Unable to interpret checkpoint content for parameter counting.")


def main():
    parser = argparse.ArgumentParser(description="Count model parameters for kg-bert-1, kg-bert-2 and fused model.")
    parser.add_argument("--model1", default="./models/kg-bert-1", help="Path to kg-bert-1 checkpoint/file/dir")
    parser.add_argument("--model2", default="./models/kg-bert-2", help="Path to kg-bert-2 checkpoint/file/dir")
    parser.add_argument("--fused", default="./models/kg-bert-fused-ot", help="Path to kg-bert-fused-ot checkpoint/file/dir")
    args = parser.parse_args()

    try:
        n1 = load_and_count(args.model1)
        n2 = load_and_count(args.model2)
        nf = load_and_count(args.fused)
    except Exception as e:
        print("Error:", e)
        return

    total12 = n1 + n2

    # print(f"kg-bert-1: {n1} params ({humanize(n1)})")
    print(f"kg-bert-1: {n1} params ({humanize(n1)})")
    print(f"kg-bert-2: {n2} params ({humanize(n2)})")

    def size_on_disk(p):
        if os.path.isfile(p):
            try:
                return os.path.getsize(p)
            except Exception:
                return 0
        if os.path.isdir(p):
            total = 0
            for root, _, files in os.walk(p):
                for fn in files:
                    fp = os.path.join(root, fn)
                    try:
                        total += os.path.getsize(fp)
                    except Exception:
                        pass
            return total
        return 0

    s1 = size_on_disk(args.model1)
    s2 = size_on_disk(args.model2)
    sf = size_on_disk(args.fused)
    total_size12 = s1 + s2


    print(f"两个KG-BERT模型的总显存占用: {total_size12} ({humanize(total_size12)})")
    print(f"融合后模型的显存占用: {sf} ({humanize(sf)})")

    if total_size12 > 0:
        disk_reduction_pct = (total_size12 - sf) / total_size12 * 100
    else:
        disk_reduction_pct = 0.0
    print(f"显存占用减少的百分比为: {disk_reduction_pct:.2f}%")

    print(f"两个KG-BERT模型的总参数量: {total12} ({humanize(total12)})")
    print(f"融合后模型的参数量: {nf} ({humanize(nf)})")

    diff = total12 - nf
    print(f"参数复杂度减少的百分比为: {(total12-diff)/total12 * 100:.2f}%")

if __name__ == "__main__":
    main()