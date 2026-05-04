import asyncio
import aiohttp
import json
import os
from typing import List, Dict, Optional, Any
from tqdm.asyncio import tqdm
from data_factory import DataFactory

# --- Configuration ---
INPUT_CANDIDATES = "data/processed/expanded_candidates.jsonl"
OUTPUT_EVAL_GOLD = "data/processed/eval_gold_triplets.jsonl"
BATCH_SIZE = 10 # Concurrent queries (each with top-20)
STRICT_MODE = True # Dual-run consistency for Eval

async def process_query(factory: DataFactory, session: aiohttp.ClientSession, data: Dict):
    query = data["query"]
    q_type = data["type"]
    candidates = data["expanded_retrieval"]

    # 1. Broad Filter (Parallel)
    judgments = []
    if q_type == "qa":
        tasks = [factory.judge_intent(session, query, cand.get("text", ""), strict=STRICT_MODE) for cand in candidates]
    else:
        tasks = [factory.judge_relevance(session, query, cand.get("text", ""), strict=STRICT_MODE) for cand in candidates]
    
    outcomes = await asyncio.gather(*tasks)
    
    valid_candidates = []
    for cand, res in zip(candidates, outcomes):
        if res and (res.get("same_intent") or res.get("relevant")) and res.get("confidence", 0) >= 0.9:
            cand["confidence"] = res["confidence"]
            cand["reason"] = res["reason"]
            valid_candidates.append(cand)

    if not valid_candidates:
        return None

    # 2. Pick the Gold standard (Tournament)
    final_gold = await factory.run_tournament(session, query, valid_candidates)
    
    return {
        "query": query,
        "type": q_type,
        "gold_doc": final_gold.get("text"),
        "doc_id": final_gold.get("id"),
        "confidence": final_gold.get("confidence"),
        "reason": final_gold.get("reason"),
        "crop": data.get("crop"),
        "state": data.get("state")
    }

async def main():
    factory = DataFactory()
    
    if not os.path.exists(INPUT_CANDIDATES):
        print(f"Error: {INPUT_CANDIDATES} not found.")
        return

    with open(INPUT_CANDIDATES, "r") as f:
        all_lines = f.readlines()
    
    print(f"Starting Eval Generation for {len(all_lines)} queries...")
    
    results = []
    async with aiohttp.ClientSession() as session:
        for i in range(0, len(all_lines), BATCH_SIZE):
            batch = [json.loads(line) for line in all_lines[i:i+BATCH_SIZE]]
            tasks = [process_query(factory, session, item) for item in batch]
            
            outcomes = await tqdm.gather(*tasks, desc=f"Batch {i//BATCH_SIZE + 1}")
            
            # Save progress incrementally
            with open(OUTPUT_EVAL_GOLD, "a") as out_f:
                for res in outcomes:
                    if res:
                        out_f.write(json.dumps(res) + "\n")
                        results.append(res)
    
    print(f"\nEval Generation Complete! Saved {len(results)} high-precision pairs to {OUTPUT_EVAL_GOLD}")

if __name__ == "__main__":
    asyncio.run(main())
