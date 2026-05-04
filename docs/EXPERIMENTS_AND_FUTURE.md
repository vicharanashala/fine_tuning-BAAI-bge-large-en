# Experiments, Results, and Future Work

This document summarizes the performance of the various model iterations and outlines the roadmap for future improvements.

## 1. Experiment Details
All experiments were conducted on NVIDIA RTX 6000 Ada GPUs using the `SentenceTransformerTrainer` with Multi-Negative Ranking Loss (MNRL).

| Phase | LR | Batch | Optimizer | Epochs |
|---|---|---|---|---|
| Phase 1 | 1e-5 | 128 | AdamW | 2 |
| Phase 2 | 5e-6 | 32 | AdamW | 1 |

---

## 2. Tournament Results (NDCG@10)

The "Grand Audit" compared our internal checkpoints, the souped model, and various fused hybrids against industry leaders.

| Rank | Model | NDCG@10 | Status |
| :--- | :--- | :--- | :--- |
| **1** | **Fused-15 (15% FT / 85% Base)** | **0.3894** | **CHAMPION (+2.6%)** |
| 2 | Original BGE-Large-en-v1.5 | 0.3794 | Baseline |
| 3 | Multilingual E5 Large | 0.3063 | External Leader |
| 4 | Raw Specialist (CP 6400) | 0.3273 | Pure FT |
| 5 | Nomic Embed Text v1.5 | 0.2761 | External |

**Key Takeaway**: Our specialized model outperforms Multilingual E5 by **~26%** and the original BGE-Large by **~2.6%**.

---

## 3. Future Roadmap

To achieve the goal of **+15-20% improvement**, the following strategies are proposed:

### A. LoRA (Low-Rank Adaptation)
Transition from full-parameter fine-tuning to LoRA. By freezing 99% of the base weights and only training a tiny adapter, we can avoid catastrophic forgetting and likely achieve a cleaner domain injection.

### B. Hard Negative Mining (Round 2)
Use the current champion (Fused-15) to mine even harder negatives from the corpus. Training on the "confusion set" of our best model will force it to learn the most difficult semantic boundaries.

### C. Domain Vocabulary Expansion
Add agricultural-specific tokens (pesticide names, Indian crop varieties) to the tokenizer and pre-train the embedding layer before triplet training.

### D. Cross-Encoder Re-ranking
Develop a domain-specific Cross-Encoder to re-rank the Top 50 results from our Fused-15 Bi-Encoder. This is the most likely path to reaching a 15-20% jump in retrieval accuracy.
