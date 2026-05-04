# Research and Fine-Tuning Methodology

This project is built upon state-of-the-art research in embedding optimization and model merging. This document outlines the papers referenced and the specific components utilized.

## 1. Research References

| Paper ID | Contribution | Application in Project |
|---|---|---|
| **`2505.19274`** | Cross-Encoder Filtering | Used a 60% threshold to filter and verify hard negatives for Track A and B. |
| **`2410.12890`** | **REFINE** Model Fusion | Applied linear weight interpolation ($\lambda=0.15$) to blend fine-tuned weights with the base model. |
| **`2601.15849`** | CGPT Clustering | Implemented K-means clustering before synthetic query generation to maximize diversity. |
| **`2408.11868`** | LLM-as-a-Judge | Used for soft-labeling and filtering the final triplet dataset for quality. |

---

## 2. Fine-Tuning Methodology

The project followed a two-phase "Exploration-Refinement" strategy to achieve a stable global minimum.

### Phase 1: Domain Exploration
- **Goal**: Large-scale adaptation to the agricultural vocabulary.
- **Config**: 2 Epochs, LR 1e-5, Batch Size 64/128.
- **Outcome**: Successfully learned domain terms but suffered from "Catastrophic Forgetting" (16% drop in general logic).

### Phase 2: Domain Polishing
- **Goal**: Refine the boundaries using corrective log data.
- **Config**: 1 Epoch, LR 5e-6, Batch Size 32.
- **Outcome**: Stabilized the weights and improved the Recall@1 significantly.

---

## 3. Post-Training Optimization

### Weight Souping (Intra-Model Averaging)
We averaged the top 3 checkpoints from both phases to create a "Stable Specialist." This process smooths out training oscillations and provides a more consistent representation across various crop types.

### REFINE Fusion (Inter-Model Blending)
To recover the general reasoning power of the base model while keeping the agricultural knowledge, we used **Model Fusion**.
- **Formula**: $Model_{final} = (1 - \lambda) \cdot Model_{base} + \lambda \cdot Model_{specialist}$
- **Optimal Ratio**: $\lambda = 0.15$ (15% Specialist / 85% Base).
- **Result**: This specific ratio allowed us to beat the original BGE-Large baseline.
