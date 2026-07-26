# Decoder Training — Post-Mortem

## What was attempted
Train a 9.9M-parameter DiT (adaLN-conditioned diffusion transformer) to map VSA sentence vectors → text.

**Architecture:**
- `VSAConditionedDiT`: 6-layer, 256-dim, 4-head transformer with adaLN modulation
- VSA vector (10000-d) projected to cond_dim via SiLU MLP, then modulates every layer
- Discrete diffusion with absorbing-state [MASK] corruption
- Training: masked LM loss on corrupted positions only

**Data:** 8000 template sentences (3–6 words) from 110-word vocabulary, encoded to 10000-d bipolar vectors via permute-by-position → bind → bundle → sign.

## Results
- **Training loss**: ~0.0002 (effectively zero — model memorised training pairs)
- **Test exact-match accuracy**: ~6% (stuck, never improved beyond this)
- **Valid token rate**: 100% (model always produces plausible tokens)

## Root cause
A diffusion model with adaLN conditioning learns a *generative distribution* P(text | VSA_vector). For a fixed VSA vector, it generates *plausible* continuations, not *deterministic* reconstructions. The exact-match evaluation (does the output exactly match the training sentence?) is the wrong metric — it tests retrieval, not generation.

The 6% accuracy represents the fraction of test sentences where the stochastic generative process happened to land on the exact training output.

**Why loss is near-zero but accuracy is low:**
The cross-entropy loss on masked tokens only measures how well the model predicts each token *independently*. Individual token predictions are accurate, but the *joint* sequence rarely matches the training example exactly because the model treats decoding as sampling from a distribution, not as retrieving a stored string.

## Implications for the architecture
The decoder cannot replace the Hopfield net. The correct division of labour is:

```
Query → Hopfield recall → nearest stored VSA vector → Decoder → fluent text
```

The Hopfield net does exact retrieval (perfect recall within capacity). The decoder renders that retrieved pattern as readable text. The decoder should never see novel VSA vectors — only vectors that are exactly stored patterns (cleaned by the Hopfield net).

## What would work
**Option A: Non-generative decoder (autoregressive, not diffusion)**
- Replace the diffusion head with an autoregressive LM head conditioned on the VSA vector
- Train on (Hopfield-cleaned VSA, text) pairs
- At inference: VSA vector conditions every step of left-to-right generation
- This turns it into conditional generation with a deterministic prefix, which the diffusion approach lacks

**Option B: Use Hopfield energy as decoder**
- Skip learned decoding entirely
- Store text strings alongside VSA vectors in the associative store
- At retrieval: Hopfield resolves query → nearest stored VSA → return associated text string
- This is simpler, deterministic, and works within the existing architecture (the store already pairs keys and values)

**Option C: Diffusion with semantic matching**
- Keep the diffusion decoder but evaluate with BLEU/ROUGE/Semantic similarity instead of exact match
- The decoder generates text that says the *same thing* as the training sentence, not necessarily the *same string*
- This matches what diffusion models are actually good at: generating semantically coherent outputs from a conditioning signal

## Recommendation
**Pursue Option B first** — it's already implemented (AssociativeStore pairs VSA vectors with values). The decoder becomes unnecessary for the core retrieval loop. The LLM harness already demonstrates the full pipeline working without the learned decoder.

**Defer Options A and C** until retrieval semantics are fully validated, then revisit if fluent generation is needed for downstream tasks.

## Files
- `experiments/train_decoder.py` — the training pipeline (8000 sentences, 200 epochs)
- `src/decoder/` — VSAConditionedDiT, diffusion, sampler
- `checkpoints/decoder_best.pt` — best checkpoint (6% exact match)
- `checkpoints/decoder_ep*.pt` — epoch checkpoints
