# Sparse Token Selection: Transformer vs MLP

Empirically demonstrates the core architectural separation from Wang, Wei, Hsu & Lee (ICML 2024): transformers can generalise to longer sequences out-of-distribution while fully-connected networks fail by architecture.

## Task
Sequences of N tokens (dim=8). Label determined by sum at 3 fixed sparse positions. Model must identify relevant positions and ignore the rest.

## Key result
- Transformer trained on N=30 generalises to sequences of length 60 (OOD)
- MLP cannot even process OOD sequences — fixed input size is an architectural constraint
- This demonstrates the core claim: transformers have structural advantages over FCNs for sparse token selection

## Run
```bash
pip install torch
python3 hsu_sparse_tokens.py
```

## Reference
Wang et al., "Transformers Provably Learn Sparse Token Selection While Fully-Connected Nets Cannot", ICML 2024, arXiv:2406.06893