"""
Full Blueprint Orchestrator — run_full_pipeline.py
Generates ALL data as per the final blueprint:
  Phase 0: Eval Set (from expanded_candidates.jsonl)
  Phase 1: A1 — QA Self-Retrieval
  Phase 2: A3 — QA Paraphrase (Hinglish / Broken English)
  Phase 3: A2 — QA Log Correctives  (from expanded_candidates, type=qa)
  Phase 4: B1 — Synthetic POP Queries
  Phase 5: B2 — POP Log Correctives (from expanded_candidates, type=pop)
  Phase 6: B3 — QA→POP Bridge

Constraints:
  - GPU 0 only (Port 9100)
  - CONCURRENCY = 60 (optimal from benchmark)
  - Min confidence threshold = 0.80
  - Checkpointing: resumes from where it left off
  - Hard negative must be same crop as query (A1)
"""

import asyncio
import aiohttp
import json
import os
import time
import random
import sys
from collections import defaultdict
from typing import List, Dict, Optional

sys.path.insert(0, os.path.dirname(__file__))
from data_factory import DataFactory

# ─── Config ───────────────────────────────────────────────
CONCURRENCY    = 60
CONFIDENCE_MIN = 0.80
WORKER_URL     = "http://localhost:9100"
BASE_DIR       = "data/processed"

# Source files
QA_CORPUS      = f"{BASE_DIR}/qa_corpus.jsonl"
POP_CORPUS     = f"{BASE_DIR}/pop_corpus.jsonl"
EXPANDED       = f"{BASE_DIR}/expanded_candidates.jsonl"

# Output files (append mode — safe to resume)
OUT = {
    "eval": f"{BASE_DIR}/out_eval_set.jsonl",
    "a1":   f"{BASE_DIR}/out_a1_qa_selfretrieval.jsonl",
    "a2":   f"{BASE_DIR}/out_a2_qa_corrective.jsonl",
    "a3":   f"{BASE_DIR}/out_a3_qa_paraphrase.jsonl",
    "b1":   f"{BASE_DIR}/out_b1_pop_synthetic.jsonl",
    "b2":   f"{BASE_DIR}/out_b2_pop_corrective.jsonl",
    "b3":   f"{BASE_DIR}/out_b3_bridge.jsonl",
}

LOG_FILE = f"{BASE_DIR}/pipeline_run.log"

# ─── Helpers ──────────────────────────────────────────────

def load_jsonl(path: str) -> List[Dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

def already_done(out_path: str) -> set:
    """Return set of query IDs already written (for resuming)."""
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    if "query" in d:
                        done.add(d["query"][:80])
                except:
                    pass
    return done

def write_triplet(out_path: str, record: Dict):
    with open(out_path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

# ─── Phase 0: Eval Set ────────────────────────────────────

async def run_eval_set(factory: DataFactory, semaphore: asyncio.Semaphore):
    """Tournament over expanded_candidates for locked gold eval."""
    items = load_jsonl(EXPANDED)
    done  = already_done(OUT["eval"])
    todo  = [it for it in items if it["query"][:80] not in done]
    log(f"[Eval] {len(todo)} to process ({len(done)} already done)")

    async def process(session, item):
        async with semaphore:
            query    = item["query"]
            cands    = item.get("expanded_retrieval", [])[:10]
            if not cands:
                return
            winner = await factory.run_tournament(session, query, cands)
            if not winner:
                return
            # Find hard negative = top candidate tournament LOSER (crop-matched)
            losers = [c for c in cands if c.get("id") != winner.get("id")]
            neg    = losers[0] if losers else None
            record = {
                "phase": "eval",
                "query": query,
                "positive": winner.get("text", ""),
                "negative": neg.get("text", "") if neg else "",
                "positive_id": winner.get("id"),
                "negative_id": neg.get("id") if neg else None,
                "crop":  item.get("crop", ""),
                "state": item.get("state", ""),
                "type":  item.get("type", ""),
            }
            write_triplet(OUT["eval"], record)

    async with aiohttp.ClientSession() as session:
        tasks = [process(session, it) for it in todo]
        for i in range(0, len(tasks), CONCURRENCY * 2):
            await asyncio.gather(*tasks[i:i + CONCURRENCY * 2])
            log(f"  [Eval] {min(i + CONCURRENCY*2, len(tasks))}/{len(tasks)}")

# ─── Phase 1: A1 — QA Self-Retrieval ─────────────────────

async def run_a1(factory: DataFactory, semaphore: asyncio.Semaphore):
    """For each QA pair, use Q as query, same-pair answer as positive.
    Hard negative = same-crop question with DIFFERENT intent (Type 1)."""
    qa = load_jsonl(QA_CORPUS)
    # Build crop index for fast same-crop lookup
    by_crop = defaultdict(list)
    for item in qa:
        crop = item.get("metadata", {}).get("Crop", item.get("crop", "unknown"))
        by_crop[crop].append(item)

    done = already_done(OUT["a1"])
    todo = [it for it in qa if it.get("question", "")[:80] not in done]
    log(f"[A1] {len(todo)} to process ({len(done)} already done)")

    async def process(session, item):
        async with semaphore:
            q    = item.get("question", "").strip()
            ans  = item.get("answer", "").strip()
            crop = item.get("metadata", {}).get("Crop", item.get("crop", "unknown"))
            if not q or not ans:
                return

            # Pick hard negative candidates from the same crop pool
            pool = [x for x in by_crop.get(crop, []) if x.get("question") != q]
            if not pool:
                return

            # Sample up to 10 candidates for intent checking
            candidates = random.sample(pool, min(10, len(pool)))
            hard_neg = None
            for cand in candidates:
                res = await factory.judge_intent(session, q, cand.get("question", ""))
                if res and not res.get("same_intent") and res.get("confidence", 0) >= CONFIDENCE_MIN:
                    hard_neg = cand
                    break

            if not hard_neg:
                return  # No high-confidence negative found — skip

            record = {
                "phase": "a1",
                "query":    q,
                "positive": f"Question: {q}\n\nAnswer: {ans}",
                "negative": f"Question: {hard_neg.get('question')}\n\nAnswer: {hard_neg.get('answer', '')}",
                "crop":  crop,
            }
            write_triplet(OUT["a1"], record)

    async with aiohttp.ClientSession() as session:
        tasks = [process(session, it) for it in todo]
        for i in range(0, len(tasks), CONCURRENCY * 3):
            await asyncio.gather(*tasks[i:i + CONCURRENCY * 3])
            if i % (CONCURRENCY * 15) == 0:
                log(f"  [A1] {min(i + CONCURRENCY*3, len(tasks))}/{len(tasks)}")

# ─── Phase 2: A3 — Paraphrase ─────────────────────────────

async def run_a3(factory: DataFactory, semaphore: asyncio.Semaphore):
    """Paraphrase QA questions into Hinglish / Broken English."""
    qa   = load_jsonl(QA_CORPUS)
    done = already_done(OUT["a3"])
    todo = [it for it in qa if it.get("question", "")[:80] not in done]
    log(f"[A3] {len(todo)} to process ({len(done)} already done)")

    async def process(session, item):
        async with semaphore:
            q   = item.get("question", "").strip()
            ans = item.get("answer", "").strip()
            if not q:
                return
            result = await factory.generate_paraphrases(session, q)
            if not result:
                return
            for style, paraphrase in [("hinglish", result.get("hinglish")), ("broken_english", result.get("broken_english"))]:
                if not paraphrase:
                    continue
                record = {
                    "phase":     "a3",
                    "style":     style,
                    "query":     paraphrase,
                    "positive":  f"Question: {q}\n\nAnswer: {ans}",
                    "negative":  "",  # filled separately via intent check
                    "source_q":  q,
                }
                write_triplet(OUT["a3"], record)

    async with aiohttp.ClientSession() as session:
        tasks = [process(session, it) for it in todo]
        for i in range(0, len(tasks), CONCURRENCY * 4):
            await asyncio.gather(*tasks[i:i + CONCURRENCY*4])
            if i % (CONCURRENCY * 20) == 0:
                log(f"  [A3] {min(i + CONCURRENCY*4, len(tasks))}/{len(tasks)}")

# ─── Phase 3: A2 — QA Log Correctives ────────────────────

async def run_a2(factory: DataFactory, semaphore: asyncio.Semaphore):
    """For log queries that retrieved wrong QA docs, find the best QA match via tournament."""
    items = [it for it in load_jsonl(EXPANDED) if it.get("type") == "qa"]
    done  = already_done(OUT["a2"])
    todo  = [it for it in items if it["query"][:80] not in done]
    log(f"[A2] {len(todo)} to process ({len(done)} already done)")

    async def process(session, item):
        async with semaphore:
            query = item["query"]
            cands = item.get("expanded_retrieval", [])[:20]
            if not cands:
                return
            winner = await factory.run_tournament(session, query, cands)
            if not winner:
                return
            losers = [c for c in cands if c.get("id") != winner.get("id")]
            neg    = losers[0] if losers else None
            record = {
                "phase":    "a2",
                "query":    query,
                "positive": winner.get("text", ""),
                "negative": neg.get("text", "") if neg else "",
                "crop":     item.get("crop", ""),
                "state":    item.get("state", ""),
            }
            write_triplet(OUT["a2"], record)

    async with aiohttp.ClientSession() as session:
        tasks = [process(session, it) for it in todo]
        for i in range(0, len(tasks), CONCURRENCY * 2):
            await asyncio.gather(*tasks[i:i + CONCURRENCY * 2])
            log(f"  [A2] {min(i + CONCURRENCY*2, len(tasks))}/{len(tasks)}")

# ─── Phase 4: B1 — Synthetic POP Queries ─────────────────

async def run_b1(factory: DataFactory, semaphore: asyncio.Semaphore):
    """For each POP chunk, generate 4 farmer-style queries."""
    pop  = load_jsonl(POP_CORPUS)
    done = already_done(OUT["b1"])
    # Can't dedup by query since we're the ones creating them; dedup by chunk[:80]
    done_chunks = set()
    if os.path.exists(OUT["b1"]):
        with open(OUT["b1"]) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    done_chunks.add(d.get("source_chunk", "")[:80])
                except:
                    pass

    todo = [it for it in pop if it.get("text", "")[:80] not in done_chunks]
    log(f"[B1] {len(todo)} to process ({len(done_chunks)} already done)")

    async def process(session, item):
        async with semaphore:
            chunk = item.get("text", "").strip()
            if not chunk or len(chunk) < 50:  # skip header-only chunks
                return
            result = await factory.generate_pop_queries(session, chunk)
            if not result:
                return
            queries = result.get("queries", [])
            for q in queries:
                if not q or len(q) < 10:
                    continue
                record = {
                    "phase":        "b1",
                    "query":        q,
                    "positive":     chunk,
                    "negative":     "",
                    "source_chunk": chunk[:80],
                    "state":        item.get("state", ""),
                }
                write_triplet(OUT["b1"], record)

    async with aiohttp.ClientSession() as session:
        tasks = [process(session, it) for it in todo]
        for i in range(0, len(tasks), CONCURRENCY * 4):
            await asyncio.gather(*tasks[i:i + CONCURRENCY*4])
            if i % (CONCURRENCY * 20) == 0:
                log(f"  [B1] {min(i + CONCURRENCY*4, len(tasks))}/{len(tasks)}")

# ─── Phase 5: B2 — POP Log Correctives ───────────────────

async def run_b2(factory: DataFactory, semaphore: asyncio.Semaphore):
    """For log queries that retrieved wrong POP chunks, find the best via tournament."""
    items = [it for it in load_jsonl(EXPANDED) if it.get("type") == "pop"]
    done  = already_done(OUT["b2"])
    todo  = [it for it in items if it["query"][:80] not in done]
    log(f"[B2] {len(todo)} to process ({len(done)} already done)")

    async def process(session, item):
        async with semaphore:
            query = item["query"]
            cands = item.get("expanded_retrieval", [])[:20]
            if not cands:
                return
            winner = await factory.run_tournament(session, query, cands)
            if not winner:
                return
            losers = [c for c in cands if c.get("id") != winner.get("id")]
            neg    = losers[0] if losers else None
            record = {
                "phase":    "b2",
                "query":    query,
                "positive": winner.get("text", ""),
                "negative": neg.get("text", "") if neg else "",
                "state":    item.get("state", ""),
            }
            write_triplet(OUT["b2"], record)

    async with aiohttp.ClientSession() as session:
        tasks = [process(session, it) for it in todo]
        for i in range(0, len(tasks), CONCURRENCY * 2):
            await asyncio.gather(*tasks[i:i + CONCURRENCY * 2])
            log(f"  [B2] {min(i + CONCURRENCY*2, len(tasks))}/{len(tasks)}")

# ─── Phase 6: B3 — Bridge ─────────────────────────────────

async def run_b3(factory: DataFactory, semaphore: asyncio.Semaphore):
    """Cross-corpus: for each QA question, find the most relevant POP chunk."""
    qa   = load_jsonl(QA_CORPUS)
    pop  = load_jsonl(POP_CORPUS)
    done = already_done(OUT["b3"])
    todo = [it for it in qa if it.get("question", "")[:80] not in done]
    log(f"[B3] {len(todo)} to process ({len(done)} already done)")

    # Pre-sample a random POP pool per call (we don't have pre-retrieved POP for each QA)
    # We use a random 5-chunk reservoir and let the LLM pick the most relevant
    POP_SAMPLE_SIZE = 5

    async def process(session, item):
        async with semaphore:
            q   = item.get("question", "").strip()
            ans = item.get("answer", "").strip()
            if not q:
                return

            pop_candidates = random.sample(pop, min(POP_SAMPLE_SIZE, len(pop)))
            best_chunk = None
            best_conf  = 0.0
            for pc in pop_candidates:
                chunk = pc.get("text", "")
                if not chunk:
                    continue
                res = await factory.judge_relevance(session, q, chunk)
                if res and res.get("relevant") and res.get("confidence", 0) >= CONFIDENCE_MIN:
                    conf = res.get("confidence", 0)
                    if conf > best_conf:
                        best_conf  = conf
                        best_chunk = chunk

            if not best_chunk:
                return

            # Negative = a POP chunk about same crop but different topic
            neg_chunk = None
            for pc in random.sample(pop, min(10, len(pop))):
                chunk = pc.get("text", "")
                res   = await factory.judge_relevance(session, q, chunk)
                if res and not res.get("relevant") and res.get("confidence", 0) >= CONFIDENCE_MIN:
                    neg_chunk = chunk
                    break

            record = {
                "phase":    "b3",
                "query":    q,
                "positive": best_chunk,
                "negative": neg_chunk or "",
                "source_qa": f"Question: {q}\n\nAnswer: {ans}",
            }
            write_triplet(OUT["b3"], record)

    async with aiohttp.ClientSession() as session:
        tasks = [process(session, it) for it in todo]
        for i in range(0, len(tasks), CONCURRENCY * 2):
            await asyncio.gather(*tasks[i:i + CONCURRENCY * 2])
            if i % (CONCURRENCY * 10) == 0:
                log(f"  [B3] {min(i + CONCURRENCY*2, len(tasks))}/{len(tasks)}")

# ─── Main Orchestrator ────────────────────────────────────

async def main():
    factory    = DataFactory(WORKER_URL)
    semaphore  = asyncio.Semaphore(CONCURRENCY)
    start_time = time.time()

    log("=" * 60)
    log("FULL BLUEPRINT PIPELINE STARTED — GPU 0 | Batch=60 | Threshold=0.80")
    log("=" * 60)

    phases = [
        ("Phase 0: Eval Set",           run_eval_set),
        ("Phase 1: A1 QA Self-Retrieval", run_a1),
        ("Phase 2: A3 QA Paraphrase",   run_a3),
        ("Phase 3: A2 QA Corrective",   run_a2),
        ("Phase 4: B1 POP Synthetic",   run_b1),
        ("Phase 5: B2 POP Corrective",  run_b2),
        ("Phase 6: B3 Bridge",          run_b3),
    ]

    for name, fn in phases:
        phase_start = time.time()
        log(f"\n{'─'*50}\n▶ {name}\n{'─'*50}")
        await fn(factory, semaphore)
        elapsed = (time.time() - phase_start) / 60
        log(f"✅ {name} done in {elapsed:.1f} min")

    total = (time.time() - start_time) / 60
    log("\n" + "=" * 60)
    log(f"🏁 ALL PHASES COMPLETE — Total time: {total:.1f} min")
    log("=" * 60)

    # Final count summary
    log("\n📊 Output Summary:")
    for key, path in OUT.items():
        if os.path.exists(path):
            with open(path) as f:
                count = sum(1 for _ in f)
            log(f"  {key.upper()}: {count} triplets → {path}")

if __name__ == "__main__":
    asyncio.run(main())
