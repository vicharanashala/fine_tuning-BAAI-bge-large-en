"""
Batch Size Benchmark — GPU 0 only (Port 9100)
Tests concurrency levels with UNIQUE queries to bypass vLLM KV cache.
Monitors VRAM at each tier.
"""
import asyncio
import aiohttp
import time
import json
import random
import subprocess

VLLM_URL = "http://localhost:9100/v1/completions"
MODEL = "/home/models/gemma4-e4b"
QA_CORPUS = "data/processed/qa_corpus.jsonl"

TEST_BATCH_SIZES = [20, 40, 60, 80, 100]
REQUESTS_PER_TEST = 200  # 200 unique calls per tier

TYPE1_TEMPLATE = """You are an agricultural expert. Determine if these two questions have the SAME intent.

Q1: {q1}
Q2: {q2}

Output ONLY this JSON:
{{
  "same_intent": true/false,
  "confidence": 0.0-1.0,
  "reason": "one sentence"
}}"""

def get_vram_usage():
    """Get GPU 0 VRAM usage in MB."""
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "--id=0"],
            text=True
        )
        return int(result.strip())
    except:
        return -1

def load_unique_questions(n: int) -> list:
    """Load n unique questions from qa_corpus."""
    questions = []
    with open(QA_CORPUS, "r") as f:
        for line in f:
            item = json.loads(line)
            q = item.get("question", "").strip()
            if q and len(q) > 20:
                questions.append(q)
            if len(questions) >= n * 2:
                break
    return questions

async def benchmark_tier(questions: list, tier_size: int):
    """Run REQUESTS_PER_TEST unique calls with given concurrency."""
    semaphore = asyncio.Semaphore(tier_size)
    errors = 0

    # Build unique prompts — each call uses a different pair of questions
    pairs = [(questions[i], questions[i+1]) for i in range(0, REQUESTS_PER_TEST * 2, 2)]

    async def call_one(q1, q2):
        nonlocal errors
        async with semaphore:
            prompt = TYPE1_TEMPLATE.format(q1=q1, q2=q2)
            payload = {
                "model": MODEL,
                "prompt": prompt,
                "max_tokens": 80,
                "temperature": 0.0
            }
            try:
                async with session.post(VLLM_URL, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status != 200:
                        errors += 1
                    await resp.read()
                    return True
            except Exception:
                errors += 1
                return False

    vram_before = get_vram_usage()
    async with aiohttp.ClientSession() as session:
        start = time.time()
        tasks = [call_one(p[0], p[1]) for p in pairs[:REQUESTS_PER_TEST]]
        await asyncio.gather(*tasks)
        duration = time.time() - start

    vram_after = get_vram_usage()
    throughput = (REQUESTS_PER_TEST - errors) / duration
    return throughput, duration, errors, vram_before, vram_after

async def main():
    print("Loading unique questions from qa_corpus.jsonl...")
    questions = load_unique_questions(REQUESTS_PER_TEST * 2 + 10)
    if len(questions) < REQUESTS_PER_TEST * 2:
        print(f"Warning: only {len(questions)} questions found, adjusting...")

    print(f"\n{'Batch':>6} | {'req/s':>8} | {'Duration':>9} | {'Errors':>7} | {'VRAM Before':>11} | {'VRAM After':>10} | Status")
    print("-" * 85)

    optimal = None
    peak_throughput = 0

    for size in TEST_BATCH_SIZES:
        throughput, duration, errors, vram_b, vram_a = await benchmark_tier(questions, size)
        
        vram_pct = (vram_a / 97887) * 100  # 97887 MB = total VRAM on RTX PRO 6000
        status = "✅ SAFE" if vram_pct < 88 and errors == 0 else ("⚠️  WARN" if vram_pct < 93 else "❌ RISKY")
        
        if throughput > peak_throughput and errors == 0 and vram_pct < 88:
            peak_throughput = throughput
            optimal = size

        print(f"{size:>6} | {throughput:>8.2f} | {duration:>8.1f}s | {errors:>7} | {vram_b:>9}MB | {vram_a:>8}MB | {status}")
        
        # Cooldown between tiers to let VRAM settle
        await asyncio.sleep(3)

    print("-" * 85)
    if optimal:
        eta_hours = 132000 / (optimal * 60 * peak_throughput / optimal) if peak_throughput > 0 else 0
        print(f"\n🎯 OPTIMAL BATCH SIZE: {optimal}")
        print(f"   Peak Throughput   : {peak_throughput:.2f} req/s")
        print(f"   Est. Full Run ETA : ~{132000 / peak_throughput / 3600:.1f} hours for 132,000 calls")
    else:
        print("\n⚠️  No safe optimal found — investigate VRAM or errors above.")

if __name__ == "__main__":
    asyncio.run(main())
