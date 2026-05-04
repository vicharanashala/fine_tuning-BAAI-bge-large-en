import asyncio
import aiohttp
import json
import os
import random
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

# --- Configuration ---
WORKER_URL = "http://localhost:9100"
MODEL_NAME = "/home/models/gemma4-e4b"

# --- Blueprint Prompt Templates ---

TYPE_1_INTENT = """You are evaluating whether two agricultural questions have the same intent.

Question 1: {q1}
Question 2: {q2}

Step 1 — Same crop check (paddy != wheat even if problem similar)
Step 2 — Same specific problem check (aphid != stem borer)
Step 3 — Compatible info type (treatment vs identification)

Output ONLY this JSON:
{{
  "crop_match": true/false,
  "problem_match": true/false,
  "info_type_match": true/false,
  "same_intent": true/false,
  "confidence": 0.0-1.0,
  "reason": "one sentence"
}}"""

TYPE_2_RELEVANCE = """You are evaluating whether an agricultural knowledge chunk answers a farmer's question.

Farmer's Question: {query}
Knowledge Chunk: {chunk_text}

Step 1 — Crop match?
Step 2 — Specific problem (pest/disease) match?
Step 3 — Correct information type?

Output ONLY this JSON:
{{
  "relevant": true/false,
  "confidence": 0.0-1.0,
  "reason": "one sentence citing specific crop and pest"
}}"""

TYPE_3_TOURNAMENT = """You are selecting the BETTER agricultural information for a farmer's question.

Farmer's Question: {query}

Option A: {a}
Option B: {b}

Choose the option that BETTER answers the farmer's exact question on crop, problem, and info-need.

Output ONLY this JSON:
{{
  "better": "A" or "B" or "neither",
  "confidence": 0.0-1.0,
  "reason": "one sentence"
}}"""

TYPE_5_PARAPHRASE = """Rewrite this agricultural question in 2 different styles.
Styles: Hinglish (informal), Broken English (simple).
Focus on crop-specific domain terms.

Question: {question}

Output ONLY this JSON:
{{
  "hinglish": "...",
  "broken_english": "..."
}}"""

TYPE_6_POP_QUERY = """Generate 4 diverse farmer-style queries for this agricultural knowledge chunk.
Ensure the queries cover different aspects (identification, treatment, prevention).

Chunk Text: {chunk_text}

Output ONLY this JSON:
{{
  "queries": ["q1", "q2", "q3", "q4"]
}}"""

# ---------------------------------------------------

class DataFactory:
    def __init__(self, worker_url: str = WORKER_URL):
        self.worker_url = worker_url

    async def call_llm(self, session: aiohttp.ClientSession, prompt: str, temperature: float = 0.0) -> Optional[Dict]:
        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": 300,
        }
        try:
            async with session.post(f"{self.worker_url}/v1/chat/completions", json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    text = data["choices"][0]["message"]["content"].strip()
                    # Clean markdown
                    text = text.replace("```json", "").replace("```", "").strip()
                    return json.loads(text)
        except Exception as e:
            # print(f"LLM Error: {e}")
            pass
        return None

    async def judge_intent(self, session: aiohttp.ClientSession, q1: str, q2: str, strict: bool = False) -> Optional[Dict]:
        """Type 1 Judgment"""
        prompt = TYPE_1_INTENT.format(q1=q1, q2=q2)
        if strict:
            # Dual-run consistency
            r1, r2 = await asyncio.gather(self.call_llm(session, prompt, 0.0), self.call_llm(session, prompt, 0.1))
            if r1 and r2 and r1.get("same_intent") == r2.get("same_intent"):
                r1["confidence"] = (r1.get("confidence", 0) + r2.get("confidence", 0)) / 2
                return r1
            return None
        return await self.call_llm(session, prompt, 0.0)

    async def judge_relevance(self, session: aiohttp.ClientSession, query: str, chunk: str, strict: bool = False) -> Optional[Dict]:
        """Type 2 Judgment"""
        prompt = TYPE_2_RELEVANCE.format(query=query, chunk_text=chunk)
        if strict:
            r1, r2 = await asyncio.gather(self.call_llm(session, prompt, 0.0), self.call_llm(session, prompt, 0.1))
            if r1 and r2 and r1.get("relevant") == r2.get("relevant"):
                r1["confidence"] = (r1.get("confidence", 0) + r2.get("confidence", 0)) / 2
                return r1
            return None
        return await self.call_llm(session, prompt, 0.0)

    async def run_tournament(self, session: aiohttp.ClientSession, query: str, candidates: List[Dict]) -> Optional[Dict]:
        """Type 3 Bracket Tournament for Top-N candidates"""
        if not candidates: return None
        if len(candidates) == 1: return candidates[0]

        current_winner = candidates[0]
        for next_cand in candidates[1:]:
            prompt = TYPE_3_TOURNAMENT.format(query=query, a=current_winner.get("text", ""), b=next_cand.get("text", ""))
            res = await self.call_llm(session, prompt, 0.0)
            if res and res.get("better") == "B":
                current_winner = next_cand
        return current_winner

    async def generate_paraphrases(self, session: aiohttp.ClientSession, question: str) -> Optional[Dict]:
        """Type 5 Generation"""
        prompt = TYPE_5_PARAPHRASE.format(question=question)
        return await self.call_llm(session, prompt, 1.0) # High temp for variety

    async def generate_pop_queries(self, session: aiohttp.ClientSession, chunk: str) -> Optional[Dict]:
        """Type 6 Generation"""
        prompt = TYPE_6_POP_QUERY.format(chunk_text=chunk)
        return await self.call_llm(session, prompt, 1.0)

async def test_factory():
    factory = DataFactory()
    async with aiohttp.ClientSession() as session:
        # Simple test
        res = await factory.judge_intent(session, "paddy leaf blast treatment", "how to cure blast in rice crop")
        print(f"Test Intent: {res}")

if __name__ == "__main__":
    asyncio.run(test_factory())
