import asyncio
import aiohttp
import json
import os
import time
from typing import List, Dict, Optional

# --- Configuration ---
LOG_QA_PATH = "data/processed/logs_qa_retrieval.jsonl"
OUTPUT_PATH = "data/processed/eval_gold_set_v1.jsonl"
ENDPOINTS = ["http://localhost:11434", "http://localhost:11435"]
MODEL = "gemma4:e4b"
CONCURRENCY_PER_GPU = 8
MAX_RETRIES = 3

# --- Prompts ---
JUDGE_PROMPT_TEMPLATE = """You are an expert agricultural judge. Verify if the retrieved document correctly and completely answers the farmer's query.

Farmer's Query: {query}
Retrieved Document: {doc}

Follow these rules:
1. 'relevant' must be true only if the doc precisely answers the query for the specific crop/pest.
2. 'confidence' must be 0.0 to 1.0 (0.90+ for eval set).
3. 'reason' brief explanation.

Return ONLY valid JSON:
{{"relevant": boolean, "confidence": float, "reason": "string"}}"""

class DualGPUJudge:
    def __init__(self, endpoints: List[str]):
        self.endpoints = endpoints
        self.semaphore = asyncio.Semaphore(len(endpoints) * CONCURRENCY_PER_GPU)
        self.endpoint_idx = 0

    async def call_ollama(self, endpoint: str, prompt: str, temp: float) -> Optional[Dict]:
        payload = {
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temp},
            "format": "json"
        }
        for attempt in range(MAX_RETRIES):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(f"{endpoint}/api/generate", json=payload, timeout=60) as resp:
                        if resp.status == 200:
                            result = await resp.json()
                            return json.loads(result['response'])
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    print(f"Error calling {endpoint}: {e}")
            await asyncio.sleep(1)
        return None

    async def judge_sample(self, sample: Dict) -> Optional[Dict]:
        async with self.semaphore:
            # Round-robin endpoint selection
            endpoint = self.endpoints[self.endpoint_idx % len(self.endpoints)]
            self.endpoint_idx += 1
            
            query = sample.get('tool_query', sample.get('query', ''))
            candidates = sample.get('retrieved', [])[:5] # Check top 5
            
            for candidate in candidates:
                doc_text = candidate.get('text', '')
                prompt = JUDGE_PROMPT_TEMPLATE.format(query=query, doc=doc_text)
                
                # Blueprint Step 0: Consistency Check (Two Runs)
                r1_task = self.call_ollama(endpoint, prompt, 0.0)
                r2_task = self.call_ollama(endpoint, prompt, 0.1)
                
                r1, r2 = await asyncio.gather(r1_task, r2_task)
                
                if r1 and r2:
                    # Consistency: Relevant Yes/No must match
                    if r1.get('relevant') == r2.get('relevant') and r1.get('relevant') == True:
                        # Confidence: Average or lowest? Blueprint says strict.
                        conf = min(r1.get('confidence', 0), r2.get('confidence', 0))
                        if conf >= 0.90:
                            return {
                                "query_id": sample.get("query_id", ""),
                                "query": query,
                                "gold_doc": doc_text,
                                "doc_id": candidate.get("id"),
                                "confidence": conf,
                                "reason": r1.get("reason")
                            }
            return None

    async def run(self):
        # Load samples
        with open(LOG_QA_PATH, 'r') as f:
            all_samples = [json.loads(line) for line in f]
        
        # SMOKE RUN: Take first 20 items
        target_samples = all_samples[:20]
        
        print(f"Starting judging for {len(target_samples)} samples on {len(self.endpoints)} GPUs...")
        start_time = time.time()
        
        tasks = [self.judge_sample(s) for s in target_samples]
        results = await asyncio.gather(*tasks)
        
        # Filter and Save
        gold_samples = [r for r in results if r]
        
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        with open(OUTPUT_PATH, 'a') as f:
            for item in gold_samples:
                f.write(json.dumps(item) + '\n')
                
        end_time = time.time()
        print(f"Finished. Found {len(gold_samples)} Gold pairs.")
        print(f"Time: {end_time - start_time:.2f}s | Speed: {len(target_samples)/(end_time - start_time):.2f} samples/sec")

if __name__ == "__main__":
    asyncio.run(DualGPUJudge(ENDPOINTS).run())
