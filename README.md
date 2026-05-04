# Agricultural Embedding Fine-Tuning (Ajrasakha)

This repository contains the full pipeline for fine-tuning the BGE-Large-en-v1.5 embedding model specifically for the Indian agricultural domain. 

The project achieved a **+2.6% improvement in NDCG@10** and a **+3.2% improvement in Recall@1** over the original baseline on specialized agricultural retrieval tasks.

## 🚀 Key Results
- **Champion Model**: Fused-15 (85% Base / 15% Fine-tuned)
- **NDCG@10**: 0.3894 (vs 0.3794 Baseline)
- **Domain Lift**: Outperforms Multilingual E5 by ~26% in retrieval accuracy.
- **Dependency Map**: Full list of libraries available in `requirements.txt`.

## 🛠️ Installation
```bash
# Clone the repository
git clone https://github.com/your-username/embedding-model-fine-tuning.git
cd embedding-model-fine-tuning

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 📁 Project Structure
- `src/data/`: Data extraction from logs/POP and triplet generation.
- `src/training/`: Phase 1 and Phase 2 fine-tuning scripts (Multi-GPU support).
- `src/evaluation/`: Model Tournament, Weight Souping, and Model Fusion (REFINE).
- `docs/`: Technical documentation:
    - [Data Pipeline](docs/DATA_PIPELINE.md)
    - [Final Fine-Tuning Blueprint](docs/FINAL_FINE_TUNING_BLUEPRINT.md)
    - [Research & Methodology](docs/RESEARCH_AND_METHODOLOGY.md)
    - [Experiments & Results](docs/EXPERIMENTS_AND_FUTURE.md)
    - [Future Roadmap (+20% Goal)](docs/FUTURE_ROADMAP.md)
- `research/`: Reference papers and technical notes.
- `scripts/`: Automation and infrastructure management.

## 🛠️ Methodology
We used a two-phase training approach followed by model merging:
1. **Phase 1**: Large-scale domain exploration.
2. **Phase 2**: High-precision polishing on log failures.
3. **Weight Souping**: Averaging top checkpoints for stability.
4. **Model Fusion**: Blending specialist weights with the original BGE "anchor" to prevent catastrophic forgetting.

## 📖 References
Based on research including **REFINE** (Model Fusion), **CGPT** (Clustered Synthetic Generation), and **Hard Negative Mining** techniques. See [RESEARCH_AND_METHODOLOGY.md](docs/RESEARCH_AND_METHODOLOGY.md) for details.

---
*Built for the Ajrasakha RAG Ecosystem.*
