# A Biologically Grounded Architecture for Reasoning AI Beyond the Transformer

## 1. Motivation

Current AI systems (transformers trained by backpropagation) have specific, well-characterised failure modes:

1. Fine-tuning risks damaging unrelated capability, because learning is driven by a global gradient rather than a local update. There is no clean way to add knowledge without risking the old.
2. Models cannot reliably tell when they are operating outside their trained territory. Hallucination is largely a symptom of this: no internal signal for "I don't know this."
3. Response to novelty is close to binary, either the full model runs or a fixed guardrail blocks something, with no graduated, proportionate response.
4. There is no runtime self-check: no way for a system to notice, at inference, that its own internal state has drifted from "behaving as intended."
5. Relational reasoning is tied specifically to self-attention (parallel, all-to-all token comparison), which is structurally incompatible with local, event-driven, biologically plausible learning rules.
6. Extended, coherent generation is currently only demonstrated well via autoregressive decoding, which itself depends on the attention mechanism above.

Rather than patching these onto the existing transformer/backprop substrate, this paper works through what nature already does to solve equivalent problems, and asks whether those mechanisms can be combined into a coherent alternative architecture.

The approach taken throughout: identify the functional requirement first, then find the biological system that already solves it, rather than starting from a biological metaphor and looking for a use.

## 2. Requirements for a working AI system

| # | Requirement | Description |
|---|---|---|
| 1 | Encoding | Transform raw input into usable internal representations |
| 2 | Storage/retrieval | Hold information short and long term, retrieve on demand |
| 3 | Relational reasoning | Relate pieces of knowledge, generalise, compose new inferences |
| 4 | Continual learning | Learn without destroying prior capability |
| 5 | Self-knowledge | Know what it doesn't know, detect internal drift |
| 6 | Relevance selection | Select what matters, moment to moment, from everything stored |
| 7 | Extended generation | Produce long, coherent, structured output |
| 8 | Goal-directed action | Pursue objectives across multiple steps |
| 9 | Learning from consequences | Adjust behaviour based on outcomes |
| 10 | Broad generalisation | Perform across many kinds of task |

## 3. The five subsystems

The architecture below assembles five biological mechanisms, each targeting specific requirements from the table above, and all built on the same underlying computational primitive: **local, content-addressable, non-destructive memory**, rather than a global loss function computed over the whole system.

### 3.1 Substrate: spiking, locally-trained network

**Targets:** Requirement 1 (encoding), and provides the foundation everything else runs on.

**Biological basis:** Spiking neural networks communicate via discrete, timed events. Only neurons that spike consume computation, so the network is sparse and event-driven by construction, not by a tuning choice. Learning happens through local rules, spike-timing-dependent plasticity (STDP), equilibrium propagation, or feedback alignment, rather than a global backward pass through the whole network. This removes the "locking" problem of backpropagation, where nothing can update until the full forward and backward pass completes.

**Current state:** Proven for efficiency and small-to-medium scale tasks (vision, sensor data, some language classification). Not proven at large-language-model scale. Existing "spiking LLMs" are mostly conversions or distillations of already-trained transformers, not systems trained natively from spikes, because sequential spiking dynamics are structurally at odds with parallel self-attention.

**Programmatic implementation:**
- Frameworks: `snnTorch`, `Brian2`, `Nengo`, Intel's `Lava` (targets Loihi neuromorphic hardware), `BindsNET`
- Local learning rule: implement STDP directly (weight update as a function of relative spike timing between pre- and post-synaptic neurons), or use surrogate gradients only where local backprop-free training is not yet practical
- Encoding scheme: rate coding (spike frequency represents intensity) or temporal coding (spike timing carries information) depending on modality
- Hardware note: can be simulated on GPU for prototyping (`snnTorch` runs on PyTorch tensors), true efficiency gains require neuromorphic hardware (Loihi, SpiNNaker) or event-based sensors

### 3.2 Relational memory: hippocampal-style binding

**Targets:** Requirement 2 (storage/retrieval) and requirement 3 (relational reasoning), as an alternative to self-attention.

**Biological basis:** The hippocampus and entorhinal cortex are not solely for spatial navigation. The same circuitry encodes general relational memory, spatial and conceptual alike. Three components do the work: pattern separation (dentate gyrus) keeps similar memories distinct; pattern completion (CA3) retrieves a full memory from a partial cue, via recurrent attractor dynamics; grid cells provide a structured, compositional coordinate system underlying both. Vector Symbolic Architectures (VSA), also called hyperdimensional computing, formalise this: symbols are high-dimensional vectors, combined through multiplicative binding (role-filler pairs) and additive bundling (sets), giving compositional structure without needing parallel token comparison.

**Current state:** VSA binding/bundling has been demonstrated on compositional and analogical reasoning benchmarks (Raven's-matrices-style tasks, abductive reasoning). Formal models (Vector-HaSH, Tolman-Eichenbaum Machine) explicitly connect this to the actual entorhinal-hippocampal circuitry. Not yet demonstrated at open-ended, language-scale reasoning. This is the single largest open uncertainty in the whole architecture.

**Programmatic implementation:**
- Libraries: `torchhd` (PyTorch-based HDC/VSA library, supports binding, bundling, permutation operations), OpenHD
- Memory structure: an associative store of high-dimensional vectors (typically 1,000 to 10,000 dimensions), with cosine similarity or Hamming distance for retrieval
- Pattern completion: implement as a Hopfield network or modern continuous Hopfield network (dense associative memory), which is mathematically an attractor network trained by local Hebbian-style updates, directly analogous to CA3
- Grid-cell-style structure: encode compositional/positional structure using fractional power encoding or structured VSA (see "Grid Cell-Inspired Structured Vector Algebra") for representing scales and relations explicitly rather than as opaque embeddings

### 3.3 Continual learning and self-monitoring: immune-style layer

**Targets:** Requirement 4 (continual learning) and requirement 5 (self-knowledge).

**Biological basis:** Two mechanisms from adaptive and innate immunity. Clonal selection: when a receptor matches something, it clones and locally mutates (affinity maturation), and some resulting cells persist as memory, without erasing existing memory cells, addressing catastrophic forgetting structurally rather than by regularisation. Negative selection: cells that react against "self" are filtered out during development, leaving a population that only reacts to what fails to match self. Applied to internal activations rather than external antigens, this gives a live self-consistency check: does the current internal state match the learned baseline of normal, intended operation.

**Current state:** Artificial Immune Systems (negative selection algorithms, clonal selection algorithms, immune networks, danger theory/dendritic cell algorithms) are an established field, but mostly applied to anomaly detection and optimisation, not integrated with deep learning or LLM-scale systems. The specific application (self-consistency monitoring of internal activations) is a genuine gap in current research, not an established technique.

**Programmatic implementation:**
- Clonal memory: maintain a growing, pruneable pool of small adapter modules (LoRA-sized), each associated with a "receptor" vector. A novel input that a module partially matches triggers cloning of that module plus local fine-tuning (mutation); modules unused for a set period decay/get pruned
- Negative selection: during a clean baseline period, train a set of detectors (could be as simple as one-class SVMs, autoencoders trained to reconstruct only "self" activation patterns, or density estimators like normalising flows) on internal activation statistics. At inference, flag activations with high reconstruction error or low likelihood under the baseline model
- Danger-theory gating: rather than gating purely on novelty, gate additionally on contextual harm signals (loss spikes, contradiction between outputs, downstream error signals), weighting the immune-style response by context, not just pattern mismatch

### 3.4 Action selection: basal-ganglia-style relevance and planning

**Targets:** Requirement 6 (relevance selection), requirement 8 (goal-directed action), requirement 9 (learning from consequences).

**Biological basis:** The basal ganglia, via the direct and indirect pathways, resolve competition between candidate actions or memories for access to limited resources, which is functionally what attention does for relevance in a transformer, but via competitive selection rather than parallel comparison. Dopaminergic neurons encode reward prediction error (the difference between predicted and actual reward), driving a three-factor Hebbian learning rule (pre-synaptic activity, post-synaptic activity, and the reward signal), which is local and compatible with the rest of the architecture's learning constraints.

**Current state:** Well-established computational neuroscience models exist (actor-critic architectures with direct/indirect pathway structure, Bayesian-Hebbian learning rules), validated against biological data on action selection and reward learning tasks. This is the most mature and directly reusable subsystem of the five, closest to existing, working reinforcement learning methods.

**Programmatic implementation:**
- Framework: actor-critic reinforcement learning (e.g., using `stable-baselines3` as a starting point for the RL scaffolding, replacing the standard backprop-trained networks with the three-factor Hebbian rule below where local learning is required)
- Three-factor learning rule: weight update proportional to (pre-synaptic activity) x (post-synaptic activity) x (reward prediction error), computed locally at each synapse rather than backpropagated
- Dual pathway structure: implement separate "Go" (direct pathway, promotes an action) and "NoGo" (indirect pathway, suppresses an action) populations, with the reward-prediction-error signal modulating both, matching the biological Go/NoGo architecture
- Relevance selection: use this same competitive selection mechanism over candidate memories retrieved from the VSA store (section 3.2), not just over motor actions, treating "which memory to attend to" as an action-selection problem

### 3.5 Generation: developmental, coarse-to-fine unfolding

**Targets:** Requirement 7 (extended generation), the least resolved requirement.

**Biological basis:** A genome does not store the organism, it stores a compact set of local rules. Cells follow local interaction rules (chemical gradients, neighbour signalling), and the organism's coherent overall structure emerges from repeated local application of those rules, not from a central plan being read out linearly. Global coarse structure forms first (a rough body plan), then local detail refines progressively within that structure. Separately, DNA replication and transcription maintain fidelity over very long sequences through proofreading built into the synthesis process itself (exonuclease proofreading, mismatch repair), catching errors as the sequence is built rather than checking only the finished product.

**Current state:** Neural Cellular Automata (NCA) are the working computational analogue: small, local update rules, applied repeatedly, that grow into coherent global structures, with demonstrated coarse-to-fine refinement and self-repair. This is an active but still small-scale research area (image and pattern generation, not yet language or open-ended sequential output). This is the only one of the five generation-relevant mechanisms that is structurally compatible with the rest of the architecture (local, non-token-parallel), which is why it is preferred here over trying to force sequential decoding onto a spiking substrate.

**Programmatic implementation:**
- Base method: Neural Cellular Automata, implemented as a small convolutional (or graph) update rule applied repeatedly to a state grid, trained end-to-end initially with standard gradient descent (existing NCA work uses this for tractability), with a longer-term goal of retraining the update rule using the local rules from section 3.1
- Coarse-to-fine control: condition the NCA's early iterations on a coarse target representation (drawn from the VSA/hippocampal memory store in section 3.2), then allow local iteration to fill in detail, giving a hierarchical, top-down-then-local-refinement generation process rather than left-to-right decoding
- In-process fidelity checks: add a lightweight local consistency check at each iteration step (comparable to exonuclease proofreading), rejecting or correcting local updates that contradict already-settled structure, rather than relying solely on a final evaluation

## 4. How the subsystems connect

All five subsystems share one computational primitive: local, content-addressable, non-destructive memory operating without a global backward pass. This is what makes them a coherent system rather than five bolted-together modules.

- Pattern completion (3.2) and clonal memory (3.3) both operate as queries and insertions into an associative store, the same underlying data structure
- Negative selection (3.3) computes its "does this match self" check as a pattern-completion query against a stored baseline, using the same machinery as ordinary retrieval, not a separate module
- Action selection (3.4) treats "which memory is relevant right now" as the same kind of competitive selection problem as "which action to take right now"
- Generation (3.5) draws its coarse target representation directly from the VSA memory store (3.2), rather than generating from nothing

## 5. Honest state of the whole

**What's proven, separately:** local learning rules work at meaningful scale for specific tasks. Spiking substrates are demonstrably more efficient. VSA binding demonstrates compositional reasoning on benchmark tasks. Basal-ganglia-style actor-critic models are validated against real neural data. Immune-inspired anomaly detection works for its established use cases. Neural Cellular Automata demonstrably grow coherent structure from local rules.

**What's not proven:** nobody has assembled these five into one system. Each exists as an active but separate research thread. There is no published architecture combining hippocampal-style binding, spiking substrates, immune-style continual memory, basal-ganglia action selection, and developmental generation.

**The load-bearing uncertainty:** whether pattern completion (3.2) can substitute for attention at the scale and flexibility needed for open-ended reasoning over long, varied context, not just small benchmark tasks. Everything else in the architecture is either well-established individually or a reasonable, testable extension of established work. This one piece is the genuine unknown the whole design rests on.

## 6. Suggested path to a first prototype

Rather than attempting the full system, a first prototype should validate the riskiest assumption first and defer generation entirely.

1. **Toy relational reasoning task** (not language): test whether VSA binding plus Hopfield-style pattern completion can solve simple relational/analogical tasks (e.g., simplified Raven's-matrices-style problems) using only local learning rules, no backprop, no attention.
2. **Add continual learning**: introduce new relations over time and test whether clonal-style expansion avoids degrading performance on earlier-learned relations, compared to a baseline that overwrites.
3. **Add self-monitoring**: test whether negative-selection-style detectors trained on the network's own "normal" activation patterns can flag genuinely novel or out-of-distribution relations before they degrade output quality.
4. **Add action selection**: wrap the above in a simple reinforcement-learning task where relevance (which stored relation to retrieve) is chosen by a basal-ganglia-style actor-critic loop rather than a fixed rule.
5. **Only then, generation**: attempt a small Neural Cellular Automata generation task conditioned on the VSA store, on a simple structured output domain (e.g., small grid patterns or short symbolic sequences), before considering anything language-scale.

This ordering deliberately keeps language generation, the least resolved and most substrate-incompatible requirement, out of the critical path until the other four subsystems are shown to work together on simpler output.

## 7. Sources consulted

This paper draws on current research surveyed during its development, including work on artificial immune systems, spiking neural networks and local learning rules, vector symbolic architectures and grid-cell-inspired encoding, basal ganglia reinforcement learning models, and neural cellular automata for morphogenesis. Specific papers and reviews are available on request; this document synthesises rather than reproduces them.
