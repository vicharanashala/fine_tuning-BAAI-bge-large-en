import json
import os
import sys
from pathlib import Path

# ── path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
import config

def clean_logs(filepath):
    if not os.path.exists(filepath): return
    
    seen_signals = set()
    cleaned = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            
            tool_q = data.get("tool_query", "").strip().lower()
            if not tool_q: continue
            
            # Identify result set for deduplication
            retrieved = data.get("retrieved", [])
            valid_items = []
            for r in retrieved:
                if "raw_text_error" in r: continue
                # We want: id, text, rank
                item = {
                    "id": r.get("question_id") or r.get("chunk_id") or r.get("id") or "",
                    "text": r.get("question_text") or r.get("answer_text") or r.get("content") or r.get("text") or "",
                    "rank": r.get("rank")
                }
                if item["id"] or item["text"]:
                    valid_items.append(item)
            
            if not valid_items: continue
            
            # Deduplication key
            res_ids = tuple(sorted([i["id"] for i in valid_items]))
            signal_key = (tool_q, res_ids)
            if signal_key in seen_signals: continue
            seen_signals.add(signal_key)
            
            # Pruned record
            clean_rec = {
                "farmer_query": data.get("original_farmer_query"),
                "tool_query": data.get("tool_query"),
                "state": data.get("meta", {}).get("state") or data.get("meta", {}).get("state_code", ""),
                "crop": data.get("meta", {}).get("crop", ""),
                "split": data.get("split"),
                "retrieved": valid_items
            }
            cleaned.append(clean_rec)
            
    with open(path, 'w', encoding='utf-8') as f:
        for rec in cleaned:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(cleaned)

def clean_qa_corpus():
    path = config.QA_CORPUS
    if not os.path.exists(path): return
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            records.append({
                "id": data.get("question_id"),
                "question": data.get("question"),
                "answer": data.get("answer"),
                "state": data.get("state"),
                "crop": data.get("crop")
            })
    with open(path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(records)

def clean_pop_corpus():
    path = config.POP_CORPUS
    if not os.path.exists(path): return
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            records.append({
                "id": data.get("chunk_id"),
                "text": data.get("content"),
                "state": data.get("state"),
                "crop": data.get("crop")
            })
    with open(path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(records)

if __name__ == "__main__":
    print(f"Cleaned QA Logs: {clean_logs(config.LOG_QA_JSONL)}")
    print(f"Cleaned POP Logs: {clean_logs(config.LOG_POP_JSONL)}")
    print(f"Cleaned QA Corpus: {clean_qa_corpus()}")
    print(f"Cleaned POP Corpus: {clean_pop_corpus()}")
