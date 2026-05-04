# Agricultural Embedding Data Pipeline

This document details the data generation and processing pipeline used to fine-tune the BGE-Large embedding model for the Ajrasakha RAG system.

## 1. Data Architecture: Dual-Track Strategy
To ensure the model performs well on both Q&A retrieval and Package of Practice (POP) document retrieval, we employed a two-track data generation strategy.

### Track A: Q&A Retrieval (Semantic Similarity)
Focuses on mapping colloquial farmer queries to existing formal Q&A pairs.
- **A1: Identity/Self-Retrieval (~9,000 triplets)**: Mapping formal questions to themselves or close semantic matches.
- **A3: Paraphrase Augmentation (~17,000 triplets)**: Using LLMs to generate "farmer-style" colloquial variants of formal questions.
- **A2: QA Corrective (Phase 2)**: Hard examples derived from real log failures where the model initially retrieved the wrong answer.

### Track B: POP Retrieval (Document Retrieval)
Focuses on mapping queries to specific agricultural manual chunks.
- **B1: Clustered Synthetic Queries (~9,600 triplets)**: 
    - **Method**: K-means clustering (600 clusters) on the POP corpus.
    - **Generation**: Sampling 4 records per cluster and using an LLM to generate diverse queries.
- **B3: Q&A-to-POP Bridge (~7,000 triplets)**: Mapping formal Q&A questions to the relevant technical manual (POP) sections.
- **B2: POP Corrective (Phase 2)**: Real-world failure cases where technical retrieval failed.

---

## 2. Advanced Generation Techniques

### CGPT Clustering (`2601.15849`)
Instead of random sampling, we used K-means clustering to group similar agricultural concepts. This ensures that the synthetic queries cover the entire breadth of the knowledge base (from pest control to soil health) without redundant over-sampling of common topics.

### LLM-as-a-Judge (`2408.11868`)
To ensure high-quality training triplets, every generated positive-negative pair was subjected to an LLM audit.
- **Strict Mode**: Dual-run consistency checks.
- **Filtering**: Removing triplets where the LLM could not clearly distinguish the positive from the negative or where the reasoning was weak.

### Hard Negative Mining
We implemented a 60% Cross-Encoder threshold (inspired by `2505.19274`) to select "Hard Negatives"—sentences that are semantically close but factually incorrect—to force the model to learn fine-grained boundaries.
