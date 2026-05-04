# Ajrasakha Embedding Fine-Tuning — Final Blueprint
> **Single source of truth.** LLM-only verification. No cross-encoder. No commercial API cost.

---

## System Architecture

```
Farmer Query
     │
     ▼
┌──────────────────────────────────────┐
│  BAAI/bge-large-en  (one model)      │
│  encodes query → vector              │
└───────────────┬──────────────────────┘
                │
       ┌────────┴────────┐
       ▼                 ▼
  Q&A Index          POP Index
  12k Q&A            31k POP
  questions          chunks
       │                 │
  Similarity         Similarity
  Search             Search
       │                 │
  Match found?           │
  YES → Answer     NO → Answer
  from Q&A              from POP
```

**One model, two separate indices, two separate retrieval tasks.**

The model must learn to embed farmer queries close to:
1. Semantically matching **Q&A questions** (not answers) — Q&A retrieval task
2. Relevant **POP chunks** (when no Q&A match exists) — POP retrieval task

---

## Guiding Principles

1. **Real farmer queries > synthetic.** The 12k Q&A questions are in domain language. Protect them.
2. **False negative > no negative.** LLM judges relevance — any candidate judged relevant is excluded from negatives, not reclassified.
3. **Evaluation set first.** Build it before touching training data.
4. **Both tasks must be evaluated separately.** Improvement in one cannot hide regression in the other.
5. **Local LLM is the single judge.** No cross-encoder. No BGE cosine pre-filter. No commercial API.

---

## Verification System: Local LLM Only

All verification tasks that previously required an agri expert are now handled exclusively by the **local open-source LLM** running on your VM.

### How It Works

```
BGE retrieves top-K candidates per query   (K = 5 to 10 depending on source)
                    ↓
ALL candidates → Local LLM (structured prompt)
                    ↓
      confidence >= 0.80  →  ACCEPT or REJECT
      confidence <  0.80  →  DISCARD (ambiguous — do not use)
```

No pre-filter. No cross-encoder. One model makes all decisions.

### False Negative Prevention

Without a cross-encoder threshold, false negatives are handled directly:
- Any candidate the LLM labels as **relevant** (any confidence) → excluded from negatives entirely
- Only LLM-confirmed **irrelevant** candidates (confidence ≥ 0.80) become hard negatives
- LLM is the gatekeeper — if it's uncertain, the candidate is discarded rather than risked

### Why Reduce K Instead of Pre-Filtering

Rather than retrieving top-50 and pre-filtering to reduce LLM calls, we retrieve **fewer candidates from the start**:

| Source | BGE retrieves | Why |
|---|---|---|
| A1 (Q&A pairs) | top-5 | Same-intent questions cluster tightly in embedding space |
| B3 (Q&A→POP) | top-5 | Dense domain, correct chunk is usually in top-5 |
| A2, B2 (logs) | top-10 | More uncertainty, need wider candidate pool |
| Eval set | top-10 | Need maximum coverage for gold labels |

This brings total LLM calls to **~70,000–80,000**, completable in ~4–7 days of overnight batch processing on a local 8B model.

### Consistency Check (Eval Set + A2 + B2)

For high-stakes labels, run the same prompt twice:
```python
r1 = llm(prompt, temperature=0.0)
r2 = llm(prompt, temperature=0.1)

if r1["label"] == r2["label"]:
    if min(r1["confidence"], r2["confidence"]) >= threshold:
        ACCEPT  # use average confidence
else:
    DISCARD  # inconsistent = genuinely ambiguous, do not use
```

---

## Four LLM Judgment Types

### Type 1 — Q&A Intent Matching
*Used in: A1 (finding same-intent Q&A pairs), A2 (confirming a Q&A retrieval failure)*

```
You are evaluating whether two agricultural questions have the same intent.

Question 1: {q1}
Question 2: {q2}

Step 1 — Extract from Question 1:
  - Crop: ___
  - Problem (disease/pest/practice): ___
  - Information needed (identification/treatment/dosage/prevention): ___

Step 2 — Extract from Question 2:
  - Crop: ___
  - Problem: ___
  - Information needed: ___

Step 3 — Compare:
  SAME INTENT requires ALL THREE to match:
  - Same crop (paddy ≠ wheat, even if problem is similar)
  - Same specific problem (aphid ≠ stem borer, blast ≠ blight)
  - Compatible info type (treatment ≈ dosage = compatible; identification ≠ treatment = not)

Output ONLY this JSON:
{
  "crop_match": true/false,
  "problem_match": true/false,
  "info_type_match": true/false,
  "same_intent": true/false,
  "confidence": 0.0–1.0,
  "reason": "one sentence"
}
```

**Accept if:** `same_intent=true`, `confidence >= 0.85`, all three component booleans true.
**Consistency check:** Run twice — keep only if both agree on `same_intent`.

---

### Type 2 — Query-to-POP Relevance
*Used in: B2 (log POP failures), B3 (Q&A→POP bridge), Eval set (POP)*

```
You are evaluating whether an agricultural knowledge chunk answers a farmer's question.

Farmer's Question: {query}
Knowledge Chunk: {chunk_text}

Step 1 — What does the farmer need?
  - Crop the farmer is asking about: ___
  - Specific problem (name the pest/disease/condition exactly): ___
  - Type of information needed (symptom identification / chemical treatment /
    dosage / timing / prevention): ___

Step 2 — What does the chunk provide?
  - Crop the chunk is about: ___
  - Specific problem the chunk addresses: ___
  - Type of information in the chunk: ___

Step 3 — Relevance check:
  RELEVANT only if:
  ✓ Same crop
  ✓ Same specific problem (not just same crop category)
  ✓ Chunk provides the information type the farmer asked for

  IRRELEVANT if:
  ✗ Different crop
  ✗ Same crop, different pest or disease
  ✗ Correct crop and problem, but wrong information type

Output ONLY this JSON:
{
  "farmer_crop": "...",
  "farmer_problem": "...",
  "farmer_info_need": "...",
  "chunk_crop": "...",
  "chunk_problem": "...",
  "chunk_info_type": "...",
  "relevant": true/false,
  "confidence": 0.0–1.0,
  "reason": "cite specific crop and pest from both query and chunk"
}
```

**Accept if:** `relevant=true`, `confidence >= 0.80`, `farmer_crop` matches `chunk_crop` (string check), `reason` contains a crop or pest name (rejects generic/hallucinated reasoning).

---

### Type 3 — Pairwise Ranking
*Used in: Selecting the single best positive from multiple confirmed-relevant candidates*

```
You are selecting the better agricultural information for a farmer's question.

Farmer's Question: {query}

Option A:
{chunk_a}

Option B:
{chunk_b}

Compare on:
1. Does it address the same crop as the question?
2. Does it address the same specific pest/disease/problem?
3. Is the information directly actionable for the farmer's need?

Choose the option that BETTER answers the farmer's exact question.
If neither is right, say "neither".

Output ONLY this JSON:
{
  "better": "A" or "B" or "neither",
  "confidence": 0.0–1.0,
  "reason": "one sentence citing crop and problem from chosen option"
}
```

**Tournament bracket for 5 candidates (3 LLM calls, not 5):**
```
Round 1: A vs B → winner_1  |  C vs D → winner_2
Round 2: winner_1 vs E → winner_3
Round 3: winner_3 vs winner_2 → BEST POSITIVE
```
Accept final winner only if last-round `confidence >= 0.80`.

---

### Type 4 — Q&A Retrieval Failure Annotation
*Used in: A2 (confirming log Q&A failures and finding correct positive)*

**Sub-step 4A — Confirm the retrieval was wrong:**
```
You are evaluating whether a Q&A match was correct for a farmer's question.

Farmer's Question: {farmer_query}
Q&A Question Retrieved: {retrieved_qa_question}

Is the Q&A question a correct match? Apply intent check:
same crop + same specific problem + compatible information type.

Output ONLY this JSON:
{
  "correct_match": true/false,
  "confidence": 0.0–1.0,
  "reason": "one sentence"
}
```

**Sub-step 4B — Find the correct Q&A question:**
If confirmed wrong → BGE retrieves top-5 Q&A questions for farmer_query → run Type 3 pairwise tournament → winner is the positive, original wrong question is the hard negative.

---

## Step 0: Build Evaluation Sets (Before Any Training)

Strictest thresholds here: **`confidence >= 0.90`**, consistency check mandatory. Built once, locked permanently.

### 0A — Q&A Retrieval Eval Set

**Source:** Ajrasakha logs — sessions where a Q&A match WAS found

```
Extract (farmer_query, retrieved_qa_question) from logs
                    ↓
BGE retrieves top-10 Q&A candidates for farmer_query
                    ↓
ALL 10 → LLM Type 1 judgment (conf >= 0.90, run twice)
                    ↓
Best confirmed match = eval positive
Discard entire query if no candidate reaches confidence >= 0.90
```

Also: hold out **10% of Q&A (1,200 pairs)** randomly — training is forbidden from using these.

**Target:** 400 verified `{farmer_query → correct_qa_question_id}` pairs

### 0B — POP Retrieval Eval Set

**Source:** Ajrasakha logs — sessions where retrieval went to POP (Q&A path failed)

```
Extract (farmer_query, top retrieved_pop_chunk_ids) from logs
                    ↓
BGE retrieves top-10 POP candidates for farmer_query
                    ↓
ALL 10 → LLM Type 2 judgment (conf >= 0.90, run twice)
                    ↓
Among relevant candidates: Type 3 pairwise → single best = eval positive
If no candidate reaches threshold: discard entire query
```

**Target:** 400 verified `{farmer_query → correct_pop_chunk_id}` pairs

### Eval Set Summary

| Eval Set | Size | Purpose |
|---|---|---|
| Q&A retrieval | ~400 pairs | Can model find the right Q&A question? |
| POP retrieval | ~400 pairs | Can model find the right POP chunk? |
| Held-out Q&A | 1,200 pairs | General domain coverage |
| **Total locked** | **~2,000** | Never touched in training |

---

## Training Data — Two Parallel Tracks

---

### TRACK A: Q&A Retrieval Training

**Goal:** Teach the model that farmer queries must embed close to same-intent Q&A questions.

---

#### A1 — Q&A Self-Retrieval Pairs (~9,000 triplets)

The 12k Q&A questions are simultaneously the queries and the documents.

```
For each Q&A question q_i (training set only):
           ↓
BGE retrieves top-5 most similar Q&A questions
           ↓
ALL 5 → LLM Type 1 judgment
           ↓
same_intent=true,  conf >= 0.85  →  candidate positive
same_intent=false, conf >= 0.80  →  confirmed hard negative
conf < 0.80                      →  DISCARD

If multiple positives: LLM Type 3 pairwise → pick single best positive
All same_intent=false candidates with conf >= 0.80 → hard negatives
Any same_intent=true candidates (even low confidence) → excluded from negatives
```

**Format:**
```
Query    = q_i
Positive = q_j (same-intent, LLM confirmed)
Negative = q_k (different intent, LLM confirmed, confusingly similar)
```

---

#### A2 — Log Q&A Retrieval Failures (~3,000–4,000 triplets) [Phase 2 only]

**Source:** Logs where Q&A retrieval path was used. LLM classifies whether the retrieved Q&A question is correct. Three distinct cases arise:

**Case 1 — Correct retrieval (rank 1 = correct):**
```
Log:  farmer_query  +  correct_qa_question
LLM:  correct_match = true
                   ↓
Positive = correct_qa_question  (directly from log — no search needed)
Negative = MISSING
  → BGE retrieves top-15 from Q&A index
  → Exclude the confirmed positive
  → LLM Type 1 finds a different-intent candidate  (conf >= 0.80)

Triplet: (farmer_query, log_correct_question, new_hard_negative)
```

**Case 2 — Wrong retrieval, but correct answer exists in rank 6–15:**
```
Log:  farmer_query  +  wrong_qa_question  (rank 1)
LLM:  correct_match = false
                   ↓
Negative = wrong_qa_question  (confirmed from log)
Positive = MISSING
  → BGE extends retrieval to top-15
  → ONLY new candidates (rank 6–15) sent to LLM Type 1
    (rank 1–5 already confirmed wrong — skip them)
  → If same_intent=true found (conf >= 0.85) → Positive (was ranked too low)

Triplet: (farmer_query, rank8_correct, rank1_wrong)
Training signal: "Rank-8 item should rank above rank-1 item" — precise ranking correction
```

**Case 3 — All top-15 irrelevant:**
```
DISCARD — BGE cannot find the answer in top-15.
Do not force a positive.
```

**Format:**
```
Query    = farmer_query (real colloquial language from logs)
Positive = correct Q&A question (from log directly OR found in rank 6–15)
Negative = wrongly retrieved Q&A question (model's actual mistake)
```

---

#### A3 — Q&A Paraphrase Augmentation (~17,000 triplets)

**Goal:** Bridge the gap between formal Q&A question language and how farmers actually type.
**No LLM verification needed** — positive is always the source question (deterministic).

**Generation prompt:**
```
Rewrite this agricultural question in 3 different styles.
Match the style of these real farmer queries:
  - "paddy patte peele ho rahe hain" (Hinglish, informal)
  - "my paddy leaves turning yellow what to do" (broken English)
  - "mere gehun mein kala daag aa raha hai" (Hinglish)

Question: {qa_question}

Output JSON:
{"hinglish": "...", "broken_english": "...", "formal_english": "..."}
```

**Format:**
```
Query    = LLM-generated paraphrase (one of 3 styles)
Positive = original Q&A question (deterministic — no LLM verification)
Negative = different Q&A question, same crop, different problem
           (LLM Type 1 confirms different intent, conf >= 0.80)
```

---

**Track A Total: ~30,000 triplets**

| Source | Count | Phase |
|---|---|---|
| A1: Q&A self-retrieval | ~9,000 | Phase 1 |
| A3: Paraphrase augmentation | ~17,000 | Phase 1 |
| A2: Log Q&A failures | ~4,000 | Phase 2 only |

---

### TRACK B: POP Retrieval Training

**Goal:** Teach the model to embed farmer queries close to relevant POP chunks when Q&A fails.

---

#### B1 — Clustered Synthetic Queries from POP (~9,600 triplets)

Implements CGPT approach (Paper 2601): cluster first, then generate queries per cluster.

**Clustering:**
```
1. Embed all 31k POP chunks with current BGE (~1–2 hours, one-time)
2. K-means → k=600 clusters
3. From each cluster: randomly sample 4 chunks
   (random intra-cluster sample, NOT centroid-only — preserves semantic diversity)
4. Output: 600 × 4 = 2,400 representative chunks
```

**Synthetic Query Generation (farmer-language grounded):**
```
You generate agricultural queries in the style of Indian farmers.

Real farmer query examples (match this style exactly):
- "paddy patte peele ho rahe hain kya spray karein"
- "meri makka ki fasal sukh rahi hai"
- "cotton mein kala keeda aa gaya"
- "wheat mein rust ka kya ilaj hai"
- "tomato pe safed daag aa rahe hain"

Generate 4 diverse farmer-style queries for this content:
{chunk_text}

Output JSON: {"queries": ["q1", "q2", "q3", "q4"]}
```

**Positive:** source chunk (deterministic — no LLM verification needed)

**Negative finding:**
```
For each (synthetic_query, source_chunk) pair:
           ↓
BGE retrieves top-5 chunks from DIFFERENT clusters
           ↓
ALL 5 → LLM Type 2 judgment
           ↓
relevant=true  (any confidence) → exclude from negatives (false negative risk)
relevant=false, conf >= 0.80   → confirmed hard negative
conf < 0.80                    → DISCARD
```

---

#### B2 — Log POP Retrieval Failures (~3,000–4,000 triplets) [Phase 2 only]

**Source:** Logs where farmer query bypassed Q&A and went to POP. The log contains the top-k chunks shown by the system. LLM classifies each chunk. Three distinct cases arise:

**Case 1 — Partial correct (≥1 chunk relevant in top-5):**
```
Log:  farmer_query  +  [chunk_A✓, chunk_B✗, chunk_C✗, chunk_D✗, chunk_E✗]
LLM:  chunk_A relevant, rest irrelevant
                   ↓
Positive  = chunk_A  (directly from log — no search needed)
Negatives = chunk_B, C, D, E  (directly from log)
No extra search needed — log gives everything.

Triplet: (farmer_query, chunk_A, chunk_B/C/D/E)
```

**Case 2 — All chunks wrong, but correct chunk exists in rank 6–15:**
```
Log:  farmer_query  +  [chunk_A✗, chunk_B✗, chunk_C✗, chunk_D✗, chunk_E✗]
LLM:  all irrelevant
                   ↓
Negatives = top-5 wrong chunks  (confirmed from log)
Positive = MISSING
  → BGE extends retrieval to top-15
  → ONLY new candidates (rank 6–15) sent to LLM Type 2
    (rank 1–5 already confirmed wrong — skip them)
  → If relevant chunk found (conf >= 0.80) → Positive (was ranked too low)
  → If multiple relevant: LLM Type 3 pairwise → single best

Triplet: (farmer_query, rank9_correct_chunk, rank1_wrong_chunk)
Training signal: "Rank-9 chunk should rank above rank-1 chunk" — precise ranking correction
```

**Case 3 — All top-15 irrelevant:**
```
DISCARD — BGE cannot find the correct chunk in top-15.
Do not force a positive.
```

**Format:**
```
Query    = farmer_query (real, bypassed Q&A)
Positive = correct POP chunk (from log directly OR found in rank 6–15)
Negative = wrongly retrieved POP chunk (model's actual mistake)
```

---

#### B3 — Q&A Questions as POP Proxy Queries (~7,000 triplets)

Q&A questions are written in farmer language AND cover agricultural topics. Use them as natural-language queries to train POP retrieval. This is NOT linking answers to POP — it uses Q&A *questions* as training queries for the POP index.

```
For each Q&A question (training set only):
           ↓
BGE retrieves top-5 POP chunks
ALL 5 → LLM Type 2 judgment
           ↓
relevant=true,  conf >= 0.80 → candidate positive
relevant=false, conf >= 0.80 → confirmed hard negative
conf < 0.80                  → DISCARD

If multiple positives: LLM Type 3 pairwise → single best positive
Yield triplet only if a confirmed positive exists
```

**Expected yield:** ~7,000 triplets (not all Q&A questions map to a qualifying POP chunk)

---

**Track B Total: ~20,600 triplets**

| Source | Count | Phase |
|---|---|---|
| B1: Clustered synthetic | ~9,600 | Phase 1 |
| B3: Q&A→POP bridge | ~7,000 | Phase 1 |
| B2: Log POP failures | ~4,000 | Phase 2 only |

---

## Final Training Dataset

| ID | Source | Count | Track | Phase | Quality |
|---|---|---|---|---|---|
| A1 | Q&A self-retrieval pairs | ~9,000 | A | 1 | ⭐⭐⭐⭐⭐ Real queries, LLM verified |
| A3 | Q&A paraphrase augmentation | ~17,000 | A | 1 | ⭐⭐⭐⭐ Farmer-language bridging |
| B1 | Clustered synthetic POP | ~9,600 | B | 1 | ⭐⭐⭐ Coverage expansion |
| B3 | Q&A questions → POP chunks | ~7,000 | B | 1 | ⭐⭐⭐⭐ Real language × POP knowledge |
| A2 | Log Q&A failures | ~3,000–4,000 | A | 2 | ⭐⭐⭐⭐⭐ Real errors, precise ranking correction |
| B2 | Log POP failures | ~3,000–4,000 | B | 2 | ⭐⭐⭐⭐⭐ Real errors, precise ranking correction |
| **Total** | | **~48,600–50,600** | | | |

**Phase 1 data:** ~42,600 triplets (A1 + A3 + B1 + B3)
**Phase 2 data:** ~6,000–8,000 triplets (A2 + B2)
**Track ratio:** Track A ≈ 60% | Track B ≈ 40%

---

## Detailed Dataset Reference

### Training Sources

| ID | Raw Asset Used | Task Trained | Query Source | Positive Source | Negative Source | How Positive Found | How Negative Found | LLM Judgment | Count | Phase |
|---|---|---|---|---|---|---|---|---|---|---|
| **A1** | 12k Q&A questions | Q&A retrieval | Each Q&A question (is its own query) | Another Q&A question with same crop + same problem + same info type | Q&A question that looks similar but is different intent (same crop, different disease) | BGE top-5 → LLM Type 1 (conf ≥ 0.85) → Type 3 pairwise if multiple | BGE top-5 → LLM Type 1 confirms different intent (conf ≥ 0.80); LLM-relevant excluded | Type 1 intent match | ~9,000 | 1 |
| **A3** | 12k Q&A questions | Q&A retrieval | LLM paraphrase of Q&A question in Hinglish / broken English / formal English | Original Q&A question (deterministic — always the source question) | Different Q&A question, same crop, different disease | Deterministic — no search needed | LLM Type 1 confirms different intent (conf ≥ 0.80) | Type 1 (negatives only) | ~17,000 | 1 |
| **A2** (Case 1) | Ajrasakha logs (Q&A path, correct retrieval) | Q&A retrieval | Real farmer query from log (colloquial, Hinglish capable) | Retrieved Q&A question (confirmed correct by LLM — directly from log) | Does not exist in log → BGE top-15, exclude positive, LLM Type 1 finds different-intent question | Directly from log (LLM confirms correct_match=true) | BGE top-15 → LLM Type 1 (conf ≥ 0.80) | Type 4 (sub-step 4A) | ~1,500–2,000 | 2 |
| **A2** (Case 2) | Ajrasakha logs (Q&A path, wrong retrieval) | Q&A retrieval | Real farmer query from log | Does not exist in log → BGE extends to rank 6–15, LLM Type 1 finds correct one | Retrieved Q&A question from log (confirmed wrong — model's actual mistake) | BGE rank 6–15 → LLM Type 1 (conf ≥ 0.85) — answer was there, just ranked too low | Directly from log (LLM confirms correct_match=false) | Type 4 (sub-steps 4A + 4B) | ~1,500–2,000 | 2 |
| **B1** | 31k POP chunks | POP retrieval | LLM-generated farmer-style query (few-shot grounded in real log queries) | Source POP chunk the query was generated from (deterministic) | POP chunks from different K-means clusters, confirmed irrelevant by LLM | Deterministic — always the source chunk | BGE top-5 from different clusters → LLM Type 2 (conf ≥ 0.80); LLM-relevant excluded | Type 2 (negatives only) | ~9,600 | 1 |
| **B3** | 12k Q&A questions + 31k POP chunks | POP retrieval | Q&A question text (real farmer domain language — not synthetic) | Most relevant POP chunk for that question's topic, LLM pairwise winner | POP chunk on same crop but different disease/problem | BGE top-5 → LLM Type 2 (conf ≥ 0.80) → Type 3 pairwise selects single best | LLM Type 2 confirms irrelevant (conf ≥ 0.80); LLM-relevant excluded | Type 2 + Type 3 | ~7,000 | 1 |
| **B2** (Case 1) | Ajrasakha logs (POP path, ≥1 chunk correct) | POP retrieval | Real farmer query from log that bypassed Q&A | Relevant chunk confirmed by LLM — directly from log | Other irrelevant chunks — directly from log (model's actual mistakes) | Directly from log (LLM confirms relevant=true) | Directly from log (LLM confirms relevant=false) | Type 2 | ~1,500–2,000 | 2 |
| **B2** (Case 2) | Ajrasakha logs (POP path, all wrong) | POP retrieval | Real farmer query from log that bypassed Q&A | Does not exist in log → BGE extends to rank 6–15, LLM Type 2 finds correct chunk | Top-5 wrong chunks from log (confirmed irrelevant — model's actual mistakes) | BGE rank 6–15 → LLM Type 2 (conf ≥ 0.80) — chunk was there, just ranked too low | Directly from log (LLM confirms relevant=false) | Type 2 + Type 3 | ~1,500–2,000 | 2 |

---

### Evaluation Sources

| ID | Raw Asset Used | Tests | Query Source | Correct Label Source | How Label Found | LLM Threshold | Count |
|---|---|---|---|---|---|---|---|
| **QA Eval** | Ajrasakha logs (Q&A sessions, last 2 weeks) | Q&A retrieval quality: can model find the right Q&A question for a real farmer query? | Real farmer query from log (Q&A retrieval path was triggered) | The Q&A question that correctly matches the farmer's intent | BGE top-10 Q&A candidates → LLM Type 1 judges intent → Type 3 pairwise selects best | conf ≥ 0.90, run twice, both must agree | ~400 pairs |
| **POP Eval** | Ajrasakha logs (POP sessions, last 2 weeks) | POP retrieval quality: can model find the right POP chunk for a query that bypassed Q&A? | Real farmer query from log (Q&A path failed, went to POP) | The POP chunk that correctly and specifically answers the farmer's query | BGE top-10 POP candidates → LLM Type 2 judges relevance → Type 3 pairwise selects best | conf ≥ 0.90, run twice, both must agree | ~400 pairs |
| **Held-out Q&A** | 12k Q&A questions | Domain coverage: model hasn't forgotten general Q&A domain knowledge | Q&A question from held-out 10% | The same question exists in the full 12k Q&A index (trivial self-match) | Random 10% hold-out before any training data is touched | Not needed — self-match | 1,200 pairs |

---

### Log Case Taxonomy (A2 & B2)

| Log Case | What Log Gives You | Missing Piece | How to Find It | Discard? |
|---|---|---|---|---|
| A2: Correct retrieval | Positive (confirmed correct) ✓ | Negative | BGE top-15 → LLM Type 1 finds different-intent Q&A | No |
| A2: Wrong retrieval, answer in rank 6–15 | Negative (confirmed wrong) ✓ | Positive | BGE extends to rank 6–15 → LLM Type 1 finds correct Q&A | No |
| A2: All top-15 wrong | Negative ✓ | Positive — cannot find reliably | — | **YES — discard** |
| B2: ≥1 chunk relevant | Positive ✓ + Negatives ✓ | Nothing | Log has everything | No |
| B2: All wrong, correct in rank 6–15 | Negatives ✓ | Positive | BGE extends to rank 6–15 → LLM Type 2 finds correct chunk | No |
| B2: All top-15 wrong | Negatives ✓ | Positive — cannot find reliably | — | **YES — discard** |

---

## LLM Verification: Call Volume & Time Estimate

| Task | BGE top-K | LLM Calls | Est. Time (8B model) |
|---|---|---|---|
| Eval set (× 2 runs) | top-10 | ~4,000 | ~8 hrs |
| A1 Q&A self-retrieval | top-5 | ~60,000 | ~120 hrs |
| A2 log Q&A failures | top-5 | ~6,000 | ~12 hrs |
| B2 log POP failures | top-10 | ~8,000 | ~16 hrs |
| B3 Q&A→POP | top-5 | ~54,000 | ~108 hrs |
| **Total** | | **~132,000** | **~264 hrs ≈ ~7 days batch** |

All costs: **$0** — local LLM on VM.

> B1 and A3 require minimal LLM verification calls (synthetic and paraphrase positives are deterministic). LLM is only needed for negative confirmation in B1 (~9,600 calls) and no verification in A3.

---

## LLM Quality Check Before Scaling

Spot-check 50 randomly sampled LLM judgments before running at full scale:

| Metric | Acceptable Threshold |
|---|---|
| LLM label agreement with manual check | > 85% |
| Average confidence on accepted labels | > 0.88 |
| % inputs discarded (conf < threshold) | 10–30% (expected and healthy) |
| % false positives in accepted positives | < 8% |

If agreement < 85% → revise prompts before scaling. Add more agricultural few-shot examples.

---

## Batching Strategy (Critical for Multi-Task Training)

**Never mix Track A and Track B in the same batch.**

A Track A in-batch negative is a Q&A question about a different topic. A Track B positive for the same query is a POP chunk on the same topic. If mixed, the POP chunk becomes a false in-batch negative for the Q&A retrieval sample.

```
Batch 1  →  Track A samples only  (crop-balanced)
Batch 2  →  Track B samples only  (cluster-balanced)
Batch 3  →  Track A
Batch 4  →  Track B
...
```

Within Track A batches: each batch of 64 spans ≥ 16 different crops.
Within Track B batches: each batch of 64 spans ≥ 16 different K-means clusters.

---

## Step 5: Training

### Two Non-Negotiable Rules

**Rule 1 — Instruction Formatting Is Asymmetric:**
```python
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

def format_query(q: str) -> str:
    return QUERY_INSTRUCTION + q   # Always prefixed for queries

def format_passage(p: str) -> str:
    return p                        # NEVER prefixed for documents
```
If the instruction prefix is accidentally added to any passage/chunk, the embedding space becomes misaligned with production inference.

**Rule 2 — REFINE Fusion After Every Training Phase (Paper 2410):**
```python
LAMBDA = 0.35  # Frozen model contributes 35%, trained contributes 65%

def apply_refine_fusion(trained_model, frozen_model):
    for (_, p_train), (_, p_frozen) in zip(
        trained_model.named_parameters(),
        frozen_model.named_parameters()
    ):
        p_train.data = LAMBDA * p_frozen.data + (1 - LAMBDA) * p_train.data
```
Without this, Phase 2 erases Phase 1 gains.

---

### Phase 1 — Broad Domain Adaptation

**Data:** A1 + A3 + B1 + B3 (~42,600 triplets)
**Loss:** MultipleNegativesRankingLoss
**Frozen anchor:** Original `BAAI/bge-large-en`

```python
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader

model = SentenceTransformer("BAAI/bge-large-en")

frozen_anchor = SentenceTransformer("BAAI/bge-large-en")
for p in frozen_anchor.parameters():
    p.requires_grad = False

track_a_examples = load_triplets(["A1", "A3"])
track_b_examples = load_triplets(["B1", "B3"])

train_dataloader = AlternatingDataLoader(
    track_a=DataLoader(track_a_examples,
                       sampler=CropBalancedSampler(track_a_examples),
                       batch_size=64),
    track_b=DataLoader(track_b_examples,
                       sampler=ClusterBalancedSampler(track_b_examples),
                       batch_size=64),
)

train_loss = losses.MultipleNegativesRankingLoss(model)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=2,
    warmup_steps=200,
    optimizer_params={"lr": 1e-5},
)

apply_refine_fusion(model, frozen_anchor)
model.save("./phase1_fused")
```

---

### Phase 2 — Hard Negative Refinement

**Data:** A2 + B2 (~8,000 triplets — log failures only)
**Frozen anchor:** Phase 1 fused model (not original BGE)

```python
frozen_anchor_p2 = SentenceTransformer("./phase1_fused")
for p in frozen_anchor_p2.parameters():
    p.requires_grad = False

model_p2 = SentenceTransformer("./phase1_fused")

hard_examples = load_triplets(["A2", "B2"])

train_loss_p2 = losses.MultipleNegativesRankingLoss(model_p2)

model_p2.fit(
    train_objectives=[(
        AlternatingDataLoader(
            track_a=DataLoader([x for x in hard_examples if x.track == "A"], batch_size=32),
            track_b=DataLoader([x for x in hard_examples if x.track == "B"], batch_size=32),
        ),
        train_loss_p2
    )],
    epochs=1,
    warmup_steps=100,
    optimizer_params={"lr": 5e-6},
)

apply_refine_fusion(model_p2, frozen_anchor_p2)
model_p2.save("./final_ajrasakha_model")
```

---

### Hyperparameter Summary

| Parameter | Phase 1 | Phase 2 |
|---|---|---|
| Data | A1 + A3 + B1 + B3 | A2 + B2 |
| Batch size | 64 | 32 |
| Learning rate | 1e-5 | 5e-6 |
| Epochs | 2 | 1 |
| Warmup steps | 200 | 100 |
| REFINE lambda | 0.35 | 0.35 |
| Frozen anchor | Original BGE | Phase 1 fused |
| Batching | Alternating A/B | Alternating A/B |

---

## Step 6: Evaluation

### Q&A Retrieval Evaluation

```python
from sentence_transformers.evaluation import InformationRetrievalEvaluator

qa_evaluator = InformationRetrievalEvaluator(
    queries={q_id: format_query(farmer_q) for q_id, farmer_q in qa_eval_queries},
    corpus={qa_id: qa_question_text for qa_id, qa_question_text in full_qa_index},
    relevant_docs={q_id: {correct_qa_id} for q_id, correct_qa_id in qa_eval_labels},
    k_values=[1, 3, 5],
    name="qa_retrieval"
)
```

### POP Retrieval Evaluation

```python
pop_evaluator = InformationRetrievalEvaluator(
    queries={q_id: format_query(farmer_q) for q_id, farmer_q in pop_eval_queries},
    corpus={chunk_id: chunk_text for chunk_id, chunk_text in full_pop_index},
    relevant_docs={q_id: {correct_chunk_id} for q_id, correct_chunk_id in pop_eval_labels},
    k_values=[1, 3, 5],
    name="pop_retrieval"
)
```

### Results Table

| Checkpoint | QA R@1 | QA R@3 | QA R@5 | POP R@1 | POP R@3 | POP R@5 | MRR (avg) |
|---|---|---|---|---|---|---|---|
| Vanilla BGE (baseline) | | | | | | | |
| Phase 1 only | | | | | | | |
| Phase 1 + Phase 2 (final) | | | | | | | |

> **Primary metric: Recall@1 for both tasks.** First retrieved result is what the LLM uses. Both tasks must improve — regression in either = do not deploy.

### Ablation Tests

| Ablation | Validates |
|---|---|
| No REFINE fusion after Phase 2 | Does fusion prevent catastrophic forgetting? |
| No CGPT clustering (flat synthetic gen) | Does K-means clustering help POP coverage? |
| No LLM false-negative exclusion | Does excluding LLM-relevant candidates from negatives matter? |
| Random batching (no track separation) | Does alternating batch strategy matter? |
| Track A only training | Confirms QA improves, POP baseline or degrades |
| Track B only training | Confirms POP improves, QA baseline or degrades |

---

## Complete Pipeline at a Glance

```
WEEK 1: EVALUATION SET + DATA PREPARATION
══════════════════════════════════════════════════════════

[0] Build Eval Sets (LOCKED — never used in training)
    QA eval:  ~400 pairs  │  BGE top-10 → LLM Type 1+3  │  conf >= 0.90, run twice
    POP eval: ~400 pairs  │  BGE top-10 → LLM Type 2+3  │  conf >= 0.90, run twice
    Held-out: 1,200 Q&A pairs (random 10% hold-out, never trained on)

[A1] Q&A Self-Retrieval Pairs
    BGE top-5 per Q&A question → ALL to LLM Type 1 (conf >= 0.85)
    LLM-relevant excluded from negatives
    → ~9,000 triplets

[A3] Q&A Paraphrase Augmentation
    LLM generates Hinglish + broken English + formal English variants
    Positive is deterministic (source question), no verification
    Negatives: LLM Type 1 confirms different intent (conf >= 0.80)
    → ~17,000 triplets

[B1] Clustered Synthetic POP Queries
    K-means 600 clusters → sample 4/cluster → 2,400 chunks
    LLM generates 4 farmer-style queries per chunk (few-shot grounded)
    Negatives: BGE top-5 from different clusters → LLM Type 2 (conf >= 0.80)
    → ~9,600 triplets

[B3] Q&A Questions → POP Bridge
    BGE top-5 POP chunks per Q&A question → ALL to LLM Type 2 (conf >= 0.80)
    LLM Type 3 pairwise selects single best positive
    LLM-relevant excluded from negatives
    → ~7,000 triplets

WEEK 2: LOG ANNOTATION (PHASE 2 DATA)
══════════════════════════════════════════════════════════

[A2] Log Q&A Failures
    Time-split logs (older than 2 weeks for training)
    BGE top-5 → LLM Type 4 (Sub-steps 4A + 4B) + consistency check
    → ~4,000 triplets

[B2] Log POP Failures
    Same time-split logs (POP retrieval path)
    BGE top-10 → LLM Type 2 + Type 3 pairwise + consistency check
    → ~4,000 triplets

WEEK 3: TRAINING
══════════════════════════════════════════════════════════

[Phase 1]
    Data: A1 + A3 + B1 + B3 (~42,600 triplets)
    Loss: MNRL | Alternating A/B batches | Crop & cluster balanced
    2 epochs, LR 1e-5, batch 64, warmup 200
    → REFINE fusion λ=0.35 (anchor: original BGE)
    → Save: phase1_fused

[Phase 2]
    Data: A2 + B2 (~8,000 triplets)
    Loss: MNRL | Alternating A/B batches
    1 epoch, LR 5e-6, batch 32, warmup 100
    → REFINE fusion λ=0.35 (anchor: phase1_fused)
    → Save: final_ajrasakha_model

WEEK 4: EVALUATION + DEPLOYMENT
══════════════════════════════════════════════════════════

[Offline Eval]
    QA Recall@1/3/5 + POP Recall@1/3/5 + MRR
    Run ablations (REFINE, clustering, false-neg exclusion, batching)

[Shadow Test]
    New model runs in parallel, not shown to users
    LLM judges divergent results (Type 1 for QA path, Type 2 for POP path)
    Deploy if win rate > 60% on divergent cases

[Re-embed]
    All 31k POP chunks + all 12k Q&A questions
    Using final_ajrasakha_model
    Both indices must use same model — never mix old embeddings with new model
```

---

## Research Paper Contributions

| Paper | Technique | Where Applied |
|---|---|---|
| `2505.19274` | False-negative prevention principle | LLM-relevant candidates excluded from negatives across all sources |
| `2410.12890` | REFINE model fusion (λ=0.35 frozen anchor) | After Phase 1, after Phase 2 |
| `2601.15849` | CGPT: K-means cluster before synthetic query gen | B1 |
| `2408.11868` | Soft/filtered labels via confidence thresholding | All LLM judgment acceptance thresholds |

---

## Failure Mode Reference

| Symptom | Cause | Fix |
|---|---|---|
| QA R@1 improves, POP R@1 degrades | Track A dominates training | Rebalance toward 50/50 |
| POP R@1 improves, QA R@1 degrades | Track B dominates | Rebalance toward 50/50 |
| Both degrade after Phase 2 | REFINE fusion not applied | Apply λ interpolation strictly |
| Paraphrase data hurts QA R@1 | LLM paraphrases too far from real queries | Add more real log queries as few-shots |
| Training loss hits 0 too fast | Negatives too easy | Reduce BGE top-K to get harder candidates |
| Track A/B interference | Tracks mixed in same batch | Enforce strict alternating batch strategy |
| LLM annotation agreement < 85% | Prompts too vague | Add agri-specific examples; enforce structured output |
| High LLM discard rate (> 40%) | Threshold too strict | Verify LLM is following structured format; check local model quality |
| Shadow test win rate < 60% | Model trained on wrong distribution | Review A3 paraphrase style; check log time-split |
