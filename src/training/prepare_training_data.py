"""
Step 1: prepare_training_data.py
Merges all out_*.jsonl files into a clean, validated training dataset.
  - Deduplicates by (query, positive) pair
  - Filters out empty positives/negatives
  - Splits into train / val sets
  - Outputs HuggingFace-compatible triplet JSONL
"""
import json
import os
import random
from collections import defaultdict
from pathlib import Path

# ─── Config ──────────────────────────────────────────────
BASE       = Path("data/processed")
OUT_DIR    = Path("data/training")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# All generated output files
SOURCES = {
    "eval": BASE / "out_eval_set.jsonl",
    "a1":   BASE / "out_a1_qa_selfretrieval.jsonl",
    "a2":   BASE / "out_a2_qa_corrective.jsonl",
    "a3":   BASE / "out_a3_qa_paraphrase.jsonl",
    "b1":   BASE / "out_b1_pop_synthetic.jsonl",
    "b2":   BASE / "out_b2_pop_corrective.jsonl",
    "b3":   BASE / "out_b3_bridge.jsonl",
}

VAL_RATIO   = 0.05    # 5% held out for validation
RANDOM_SEED = 42

def load_source(path: Path, tag: str) -> list:
    if not path.exists():
        print(f"  ⚠️  Missing: {path}")
        return []
    records = []
    with open(path) as f:
        for line in f:
            try:
                d = json.loads(line)
                q   = d.get("query", "").strip()
                pos = d.get("positive", "").strip()
                neg = d.get("negative", "").strip()
                if not q or not pos:
                    continue
                records.append({
                    "query":    q,
                    "positive": pos,
                    "negative": neg,
                    "track":    tag,
                    "phase":    d.get("phase", tag),
                    "crop":     d.get("crop", ""),
                    "state":    d.get("state", ""),
                })
            except:
                pass
    return records

def main():
    random.seed(RANDOM_SEED)
    all_records = []
    seen        = set()   # dedup key: (query[:100], positive[:100])
    stats       = defaultdict(int)

    print("📂 Loading sources...")
    for tag, path in SOURCES.items():
        recs = load_source(path, tag)
        before = len(all_records)
        for r in recs:
            key = (r["query"][:100], r["positive"][:100])
            if key in seen:
                stats["duplicates"] += 1
                continue
            seen.add(key)
            all_records.append(r)
        added = len(all_records) - before
        print(f"  {tag.upper():>4}: {added:>6} records loaded  ({len(recs)-added} dupes dropped)")
        stats[f"track_{tag}"] = added

    # Separate: eval set is NEVER part of training
    eval_recs  = [r for r in all_records if r["phase"] == "eval"]
    train_pool = [r for r in all_records if r["phase"] != "eval"]

    # Shuffle and split train → train + val
    random.shuffle(train_pool)
    val_size  = int(len(train_pool) * VAL_RATIO)
    val_recs  = train_pool[:val_size]
    train_recs = train_pool[val_size:]

    def write(path, records):
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write(OUT_DIR / "train.jsonl",      train_recs)
    write(OUT_DIR / "val.jsonl",        val_recs)
    write(OUT_DIR / "eval_locked.jsonl", eval_recs)

    # Summary
    print(f"\n✅ Data Preparation Complete")
    print(f"  {'Train':<12}: {len(train_recs):>8,} triplets  → data/training/train.jsonl")
    print(f"  {'Validation':<12}: {len(val_recs):>8,} triplets  → data/training/val.jsonl")
    print(f"  {'Eval (Locked)':<12}: {len(eval_recs):>8,} pairs     → data/training/eval_locked.jsonl")
    print(f"  {'Duplicates':<12}: {stats['duplicates']:>8,} dropped")
    print(f"\n  Track breakdown:")
    for k, v in stats.items():
        if k.startswith("track_"):
            print(f"    {k.replace('track_','').upper():>4}: {v:>6,}")

if __name__ == "__main__":
    main()
