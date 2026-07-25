# Experiment: VSA-Conditioned Decoder Feasibility

## Setup
- **Model:** 6-layer TransformerEncoder (d_model=256, 4 heads, ~15M params, randomly initialised)
- **Conditioning:** 10000-d bipolar VSA vector projected to 256-d, added to all token embeddings
- **Data:** 2000 template sentences from 20-word vocabulary
- **Training:** 50 epochs, AdamW, lr=1e-4, masked LM loss (15% masked)
- **Eval:** Iterative unmasking (50 steps, confidence-based re-mask)

## Results
- Final loss: ~0.8 (vs ~3.0 random — model learned language structure)
- Exact match accuracy on held-out sentences: **0.0%**
- Order discrimination ("dog bites man" vs "man bites dog"): **0.0%**

## Conclusion
**Embedding-level VSA conditioning is insufficient.** The transformer learns plausible token distributions but cannot reliably decode word order from a permuted-bundle vector added uniformly to all positional embeddings.

The conditioning signal is too weak to penetrate 12 transformer layers uniformly. Each layer's self-attention and FFN operations mix the VSA signal across all positions, diluting its positional information below recoverability.

## Next Step
Try **adaLN per-layer modulation** (adaptive LayerNorm shift/scale from the VSA vector at each layer), which is the standard approach used by MDLM and SEDD for time-step conditioning. This gives the VSA signal a direct gradient path into every layer rather than only the embedding layer.
