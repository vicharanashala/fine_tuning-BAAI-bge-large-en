# Future Roadmap: The Path to +20% Improvement

Achieving a **15%–20% improvement** over the world's best model (BGE-Large) is an extremely ambitious goal—it would essentially make your model the #1 English embedding model in existence by a wide margin.

To get from our current **+2.6%** to **+20%**, we must transition from "tweaking" to **"Architectural Warfare."** This document outlines the four pillars of that strategy.

---

## 1. Hard Negative Mining (The +5-7% Boost)
Currently, the model often faces negative answers that are too easy to distinguish. 

*   **The Strategy**: Use our current Champion (**Fused-15**) to find the Top 5 documents it is *confused* about for every query in the training set. Then, train the model specifically to tell the difference between the "Right Answer" and those "Top 5 Confusing Answers."
*   **Why it works**: This forces the model to learn the fine-grained differences between specific fertilizers, pests, and crops that "look" similar but are factually distinct.

## 2. Domain-Specific Vocabulary (The +3-5% Boost)
BGE was trained on general internet text. It lacks specialized "tokens" for complex agricultural chemicals or local Indian crop names.

*   **The Strategy**: We "expand" the model's vocabulary by adding 500–1000 specific agricultural terms (pesticide names, soil types, vernacular crop names) and pre-train the embedding layer on your raw text for a few hours before resuming triplet training.
*   **Why it works**: It allows the model to "understand" a chemical name as a single, coherent concept rather than a string of random, disconnected characters.

## 3. The "Giant Slayer": Cross-Encoder Re-ranking (The +15-20% Boost)
If you want a massive jump in retrieval quality, embeddings (Bi-Encoders) alone might hit a mathematical ceiling.

*   **The Strategy**: Use our Fused-15 model to find the Top 50 results (Bi-Encoder). Then, train a **Cross-Encoder** (a model that looks at the Query and the Document *together* in the same attention window) to re-rank those 50.
*   **The Reality**: Cross-Encoders are computationally slower but are historically **10%–30% more accurate** than Bi-Encoders. This is the "industry secret" for hitting elite retrieval numbers.

## 4. Synthetic Data Generation (LLM-as-a-Teacher)
A model is only as good as its training data. 

*   **The Strategy**: Use a state-of-the-art model (like GPT-4o or Claude 3.5) to review our 140,000 existing triplets and "rewrite" or "refine" the answers to ensure they are 100% factually dense and perfectly aligned with the query intent.
*   **Why it works**: High-quality synthetic data from a superior teacher provides a much cleaner gradient for the smaller embedding model to follow.

---

## Phase 3: Immediate Next Steps
To move toward the 20% mark, we recommend focusing on **Hard Negative Mining** first. It provides the highest "return on investment" using the infrastructure we have already built.

**Goal**: Break the **0.40 NDCG@10** barrier.
