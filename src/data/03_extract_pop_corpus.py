"""
extract_pop_corpus.py
══════════════════════
Dump the full POP corpus from staging MongoDB.

Source: golden_db.pop  (31k chunks)

Output: pop_corpus.jsonl
Each record:
{
  "chunk_id":   str,
  "content":    str,       # the text of the chunk
  "state":      str,
  "crop":       str,
  "metadata":   dict,      # all original metadata fields
}
"""

import json
import os
import sys
from pathlib import Path

from pymongo import MongoClient
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
import config


def extract_pop_corpus():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print("Connecting to staging MongoDB ...")
    client = MongoClient(config.STAGING_URI, serverSelectionTimeoutMS=10_000)
    db     = client[config.STAGING_GOLDEN_DB]
    col    = db[config.STAGING_POP]

    total = col.count_documents({})
    print(f"Total POP documents in collection: {total:,}")

    records = []
    for doc in tqdm(col.find({}), total=total, desc="POP chunks"):
        meta = doc.get("metadata", {})

        # The text content field name may vary — check all common names
        content = (
            doc.get("text")
            or doc.get("content")
            or doc.get("chunk_text")
            or doc.get("page_content")
            or ""
        )
        content = content.strip() if isinstance(content, str) else ""

        if not content:
            continue   # skip empty chunks

        records.append({
            "chunk_id": str(doc.get("_id", "")),
            "content":  content,
            "state":    meta.get("state", meta.get("State", "")),
            "crop":     meta.get("crop",  meta.get("Crop",  "")),
            "metadata": meta,
        })

    print(f"\nValid POP chunks : {len(records):,}  (skipped {total - len(records):,} empty)")

    with open(config.POP_DUMP_JSONL, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    print(f"Written → {config.POP_DUMP_JSONL}")

    # ── Quick stats ───────────────────────────────────────────────────────────
    states = {}
    crops  = {}
    for r in records:
        s = r["state"] or "unknown"
        c = r["crop"]  or "unknown"
        states[s] = states.get(s, 0) + 1
        crops[c]  = crops.get(c, 0)  + 1

    print(f"\nTop 10 states  : {sorted(states.items(), key=lambda x: -x[1])[:10]}")
    print(f"Top 10 crops   : {sorted(crops.items(),  key=lambda x: -x[1])[:10]}")

    client.close()


if __name__ == "__main__":
    extract_pop_corpus()
