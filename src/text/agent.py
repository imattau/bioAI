import torch
import re
from pathlib import Path

from src.vsa import VSA, VSAHashStore, HopfieldNet
from src.clonal import ClonalPool
from src.immune import SelfMonitor
from src.basal import GoNoGoActorCritic
from src.text.encoder import VSAEncoder
from src.text.decoder import VSADecoder
from src.text.token_library import TokenLibrary
from src.text.semantic_vsa import OllamaEmbedder, SemanticVSAEncoder
from src.text.response_generator import OllamaResponseGenerator
from src.text.consolidation import ConsolidationMemory
from src.text.chunk_composer import LearnedChunkComposer
from src.text.response_candidates import (
    FixedSpliceCandidateGenerator,
    SequenceCandidateScorer,
)
from src.text.vsa_sequence_ranker import (
    SemanticChunkVSASequenceEncoder,
    VSASequenceRanker,
)


class BioAIDialogueAgent:
    def __init__(self, vsa_dim: int = 1000):
        self.vsa = VSA(dim=vsa_dim, device="cpu")
        self.encoder = VSAEncoder(vsa=self.vsa)
        self.decoder = VSADecoder(vsa=self.vsa, store_capacity=200,
                                  encoder=self.encoder)
        self.hash_store = VSAHashStore(self.vsa)
        self.hopfield = HopfieldNet(dim=vsa_dim)
        self.monitor = SelfMonitor(self.hopfield, energy_threshold=3.0)
        self.clonal = ClonalPool(input_dim=vsa_dim, max_modules=50)
        self.gonogo = GoNoGoActorCritic(input_dim=vsa_dim, n_actions=4)
        self.history: list[dict] = []
        self._monitor_calibrated = False
        self.turn_count = 0
        self._text_index: dict[bytes, str] = {}
        self._last_query_vec: torch.Tensor | None = None
        self._cached_context: str = ""
        self._ctx_cache_threshold: float = 0.7
        self._retrieval_threshold: float = 0.38
        self._retrieval_margin: float = 0.1
        self.library = TokenLibrary(vector_cache_size=1024)
        self.consolidation = ConsolidationMemory()
        self._hot_memory_capacity = 200
        self._history_capacity = 1000
        self.semantic_encoder: SemanticVSAEncoder | None = None
        self.semantic_model: str | None = None
        self.response_generator = None
        self.response_model: str | None = None
        self.chunk_composer: LearnedChunkComposer | None = None
        self.candidate_generator = FixedSpliceCandidateGenerator()
        self.candidate_scorer = SequenceCandidateScorer()
        self.sequence_ranker: VSASequenceRanker | None = None

    def enable_semantic_retrieval(
        self,
        model: str = "qwen3-embedding:0.6b",
        vsa_dim: int = 1024,
    ):
        self.semantic_model = model
        self.semantic_encoder = SemanticVSAEncoder(
            OllamaEmbedder(model), vsa_dim=vsa_dim
        )

    def enable_response_generation(
        self,
        model: str = "qwen2.5-coder:1.5b",
    ):
        self.response_model = model
        self.response_generator = OllamaResponseGenerator(model)

    def enable_chunk_composition(self):
        self.chunk_composer = LearnedChunkComposer()

    def learn_conversation(self, prompt: str, response: str) -> None:
        if self.chunk_composer is None:
            self.enable_chunk_composition()
        self.chunk_composer.learn(prompt, response)

    def enable_vsa_sequence_ranking(self, dimension: int = 512):
        self.sequence_ranker = VSASequenceRanker(dimension=dimension)

    def enable_semantic_sequence_ranking(
        self,
        model: str = "qwen3-embedding:0.6b",
        dimension: int = 1024,
    ):
        if self.semantic_encoder is None:
            self.enable_semantic_retrieval(model=model, vsa_dim=dimension)
        self.sequence_ranker = VSASequenceRanker(
            encoder=SemanticChunkVSASequenceEncoder(self.semantic_encoder)
        )

    def _calibrate_monitor(self):
        if self._monitor_calibrated:
            return
        vecs = [self.encoder.encode(f"baseline {i}") for i in range(3)]
        for v in vecs:
            self.hopfield.store(v)
        recalls = [self.hopfield.recall(v + 0.1 * self.vsa.make_vector(), steps=5)
                   for v in vecs]
        self.monitor.calibrate(recalls)
        self._monitor_calibrated = True

    def _store_turn(self, user_text: str, user_vec: torch.Tensor,
                    long_term: bool = True):
        self.history.append({
            "turn": self.turn_count,
            "user": user_text,
            "user_vec": user_vec,
            "long_term": long_term,
        })
        if len(self.history) > self._history_capacity:
            self.history.pop(0)
        if not long_term:
            return
        semantic_vector = (
            self.semantic_encoder.encode(user_text)
            if self.semantic_encoder is not None else None
        )
        sentence_id = self.library.add(
            user_text, semantic_vector=semantic_vector
        )
        self.consolidation.observe_relation(user_text, sentence_id)
        if semantic_vector is not None:
            for term in self.library.semantic_terms(user_text):
                posting = self.library.semantic_postings[term]
                if term in self.consolidation.concepts:
                    self.consolidation.observe_concept(
                        term, semantic_vector, sentence_id
                    )
                elif len(posting) >= self.consolidation.promotion_threshold:
                    evidence = [
                        (source, self.library.semantic_vector(source))
                        for source in posting
                    ]
                    evidence = [
                        (source, vector)
                        for source, vector in evidence if vector is not None
                    ]
                    self.consolidation.observe_concept(
                        term,
                        semantic_vector,
                        sentence_id,
                        initial_vectors=[vector for _, vector in evidence],
                        initial_source_ids=[source for source, _ in evidence],
                    )
        if len(self.hash_store) < self._hot_memory_capacity:
            self.hash_store.insert(user_vec, user_vec)
        if len(self.hopfield) < self._hot_memory_capacity:
            self.hopfield.store(user_vec)
        if len(self.decoder) < self._hot_memory_capacity:
            self.decoder.ingest(user_text)
        key = user_vec.cpu().numpy().tobytes()
        self._text_index[key] = user_text

    def _detect_novelty(self, user_vec: torch.Tensor) -> dict:
        recall = self.hopfield.recall(user_vec, steps=5)
        drift = self.monitor.score(recall)
        is_novel = drift["is_anomaly"]
        if is_novel:
            _, created = self.clonal.process(user_vec)
            return {"novel": True, "energy_z": drift["energy_z"],
                    "clonal_created": created}
        return {"novel": False, "energy_z": drift["energy_z"],
                "clonal_created": False}

    @staticmethod
    def _infer_intent(user_input: str) -> str:
        text = user_input.strip().lower()
        question_starters = (
            "what ", "who ", "where ", "when ", "why ", "how ",
            "which ", "is ", "are ", "was ", "were ", "do ", "does ",
            "did ", "can ", "could ", "would ", "should ",
        )
        return "question" if (
            text.endswith("?") or text.startswith(question_starters)
        ) else "statement"

    @staticmethod
    def _content_words(text: str) -> set[str]:
        stop_words = {
            "a", "an", "the", "is", "are", "was", "were", "of", "to",
            "in", "on", "at", "for", "and", "or", "what", "who", "where",
            "when", "why", "how", "which", "do", "does", "did", "kind",
        }
        return {
            word for word in re.findall(r"[a-z0-9]+", text.lower())
            if word not in stop_words
        }

    def _retrieve_context(self, user_input: str, user_vec: torch.Tensor) -> dict:
        indexed_candidates = self.library.candidate_ids(user_input, limit=100)
        relation_scores = dict(
            self.consolidation.relation_candidates(user_input)
        )
        learned_semantic_scores = {}
        if self.semantic_encoder is not None:
            semantic_query = self.semantic_encoder.encode(user_input)
            semantic_candidates = self.library.semantic_candidate_ids(
                semantic_query, limit=100
            )
            learned_semantic_scores = dict(semantic_candidates)
            prototype_matches = self.consolidation.query_concepts(
                semantic_query, limit=8
            )
            for match in prototype_matches:
                if match["score"] <= 0:
                    continue
                for sentence_id in match["source_ids"]:
                    learned_semantic_scores[sentence_id] = max(
                        learned_semantic_scores.get(sentence_id, -1.0),
                        match["score"],
                    )
            candidate_ids = {
                sentence_id for sentence_id, _ in indexed_candidates
            } | {
                sentence_id for sentence_id, _ in semantic_candidates
            } | set(learned_semantic_scores) | set(relation_scores)
        else:
            candidate_ids = {
                sentence_id for sentence_id, _ in indexed_candidates
            } | set(relation_scores)
        query_words = self._content_words(user_input)
        query_concepts = self.library.semantic_terms(user_input)
        reranked = []
        for sentence_id in candidate_ids:
            text = self.library.texts[sentence_id]
            memory_words = self._content_words(text)
            lexical_score = (
                len(query_words & memory_words) / len(query_words)
                if query_words else 0.0
            )
            memory_concepts = self.library.semantic_terms(text)
            semantic_score = (
                len(query_concepts & memory_concepts)
                / min(len(query_concepts), len(memory_concepts))
                if query_concepts and memory_concepts else 0.0
            )
            memory_vector = self.library.vector(sentence_id, self.encoder)
            vector_score = max(
                0.0, self.vsa.similarity(user_vec, memory_vector).item()
            )
            relation_score = relation_scores.get(sentence_id, 0.0)
            if self.semantic_encoder is not None:
                learned_score = max(
                    0.0, learned_semantic_scores.get(sentence_id, 0.0)
                )
                combined_score = (
                    0.55 * learned_score
                    + 0.15 * lexical_score
                    + 0.2 * semantic_score
                    + 0.1 * vector_score
                    + 0.25 * relation_score
                )
            else:
                combined_score = (
                    0.3 * lexical_score
                    + 0.5 * semantic_score
                    + 0.2 * vector_score
                    + 0.25 * relation_score
                )
            reranked.append((sentence_id, text, combined_score))
        candidates = sorted(
            reranked, key=lambda item: item[2], reverse=True
        )[:3]
        if not candidates:
            return {
                "accepted": False,
                "text": "",
                "score": 0.0,
                "margin": 0.0,
                "candidates": [],
            }
        best_id, best_text, best_score = candidates[0]
        second_score = candidates[1][2] if len(candidates) > 1 else -1.0
        margin = best_score - second_score
        accepted = (
            best_score >= self._retrieval_threshold
            and margin >= self._retrieval_margin
        )
        return {
            "accepted": accepted,
            "memory_id": best_id if accepted else None,
            "text": best_text if accepted else "",
            "score": best_score,
            "margin": margin,
            "candidates": [
                {"id": sentence_id, "text": text, "score": score}
                for sentence_id, text, score in candidates
            ],
        }

    def process_turn(self, user_input: str, ctx_cache: bool = True) -> dict:
        self.turn_count += 1
        user_vec = self.encoder.encode(user_input)
        intent = self._infer_intent(user_input)
        self._calibrate_monitor()

        novelty = self._detect_novelty(user_vec)
        if intent == "question":
            reasoning = self.consolidation.reason_path(user_input)
            if reasoning is not None:
                source_ids = reasoning["source_ids"]
                retrieval = {
                    "accepted": True,
                    "memory_id": source_ids[0],
                    "source_ids": source_ids,
                    "text": reasoning["text"],
                    "score": 1.0,
                    "margin": 1.0,
                    "reasoning": reasoning,
                    "candidates": [
                        {
                            "id": source_id,
                            "text": self.library.texts[source_id],
                            "score": 1.0,
                        }
                        for source_id in source_ids
                    ],
                }
            else:
                retrieval = self._retrieve_context(user_input, user_vec)
        else:
            retrieval = {
                "accepted": False,
                "memory_id": None,
                "text": "",
                "score": 0.0,
                "margin": 0.0,
                "candidates": [],
            }
        context = retrieval["text"]
        self._cached_context = context
        self._last_query_vec = user_vec

        self._store_turn(
            user_input, user_vec, long_term=(intent == "statement")
        )

        if intent == "statement":
            response = "I'll remember that."
        elif not retrieval["accepted"]:
            response = "I don't have a sufficiently relevant memory for that."
        elif self.chunk_composer is not None:
            evidence = [
                candidate["text"] for candidate in retrieval["candidates"]
            ]
            candidates = self.candidate_generator.generate(
                user_input, evidence, self.chunk_composer
            )
            ranker = (
                self.sequence_ranker
                if self.sequence_ranker is not None
                and self.sequence_ranker.updates > 0
                else self.candidate_scorer
            )
            ranked_candidates = ranker.rank(user_input, candidates, evidence)
            response = (
                ranked_candidates[0]["text"] if ranked_candidates else context
            )
        elif self.response_generator is not None:
            response = self.response_generator(
                user_input,
                [{
                    "id": retrieval["memory_id"],
                    "text": retrieval["text"],
                    "score": retrieval["score"],
                }],
            )
        else:
            response = context

        if intent == "statement":
            response_mode = "acknowledgement"
        elif not retrieval["accepted"]:
            response_mode = "abstention"
        elif self.response_generator is None:
            response_mode = (
                "learned_chunk_composition"
                if self.chunk_composer is not None
                else "retrieved_memory"
            )
        else:
            response_mode = getattr(
                self.response_generator, "last_mode", "grounded_generation"
            )
        if retrieval.get("reasoning") is not None:
            response_mode = "consolidated_reasoning"

        source_ids = retrieval.get(
            "source_ids",
            [retrieval["memory_id"]] if retrieval["accepted"] else [],
        )
        sources = [
            {
                "id": source_id,
                "text": self.library.texts[source_id],
                "score": retrieval["score"],
            }
            for source_id in source_ids
        ]

        return {
            "response": response,
            "response_generated": (
                retrieval["accepted"] and self.response_generator is not None
            ),
            "response_mode": response_mode,
            "sources": sources,
            "intent": intent,
            "retrieval_accepted": retrieval["accepted"],
            "retrieval_score": retrieval["score"],
            "retrieval_margin": retrieval["margin"],
            "retrieval_candidates": retrieval["candidates"],
            "reasoning": retrieval.get("reasoning"),
            "drift_detected": novelty["novel"],
            "energy_z": novelty["energy_z"],
            "clonal_created": novelty["clonal_created"],
            "turn": self.turn_count,
        }

    def recall(self, query: str) -> str:
        exact = self.library.exact_lookup(query)
        if exact is not None:
            return exact
        qv = self.encoder.encode(query)
        key = qv.cpu().numpy().tobytes()
        if key in self._text_index:
            return self._text_index[key]
        stored = self.hash_store.lookup(qv)
        if stored is None:
            return ""
        key = stored.cpu().numpy().tobytes()
        if key in self._text_index:
            return self._text_index[key]
        results = self.decoder.decode(stored, k=1)
        return results[0][0] if results else ""

    def save(self, path: str | Path, max_history: int = 1000):
        import copy
        history = self.history[-max_history:] if len(self.history) > max_history else self.history
        word_cache = {w: v.to(torch.float16) for w, v in self.encoder._word_cache.items()}
        pos_vectors = self.encoder._pos_vectors.to(torch.float16)
        state = {
            "vsa_dim": self.vsa.dim,
            "turn_count": self.turn_count,
            "energy_threshold": self.monitor.energy_threshold,
            "_monitor_calibrated": self._monitor_calibrated,
            "monitor_energy_mean": self.monitor.energy_mean,
            "monitor_energy_std": self.monitor.energy_std,
            "word_cache": word_cache,
            "pos_vectors": pos_vectors,
            "hash_store": self.hash_store.get_state(),
            "agent_hopfield": self.hopfield.get_state(),
            "clonal": self.clonal.get_state(),
            "decoder": self.decoder.get_state(),
            "history": history,
            "text_index_keys": list(self._text_index.keys()),
            "text_index_values": list(self._text_index.values()),
            "token_library": self.library.get_state(),
            "consolidation": self.consolidation.get_state(),
            "hot_memory_capacity": self._hot_memory_capacity,
            "history_capacity": self._history_capacity,
            "semantic_model": self.semantic_model,
            "semantic_encoder": (
                self.semantic_encoder.get_state()
                if self.semantic_encoder is not None else None
            ),
            "response_model": self.response_model,
            "chunk_composer": (
                self.chunk_composer.get_state()
                if self.chunk_composer is not None else None
            ),
            "candidate_scorer": self.candidate_scorer.get_state(),
            "sequence_ranker": (
                self.sequence_ranker.get_state()
                if self.sequence_ranker is not None else None
            ),
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path: str | Path) -> "BioAIDialogueAgent":
        import torchhd
        from array import array
        torch.serialization.add_safe_globals([
            torchhd.tensors.map.MAPTensor,
            array,
        ])
        state = torch.load(path)
        vsa = VSA(dim=state["vsa_dim"], device="cpu")
        encoder = VSAEncoder(vsa=vsa)
        encoder._word_cache = {w: v.to(torch.float32) for w, v in state["word_cache"].items()}
        encoder._pos_vectors = state["pos_vectors"].to(torch.float32)
        decoder = VSADecoder(vsa=vsa, encoder=encoder)
        decoder.set_state(state["decoder"])
        agent = cls.__new__(cls)
        agent.vsa = vsa
        agent.encoder = encoder
        agent.decoder = decoder
        agent.hash_store = VSAHashStore(vsa)
        agent.hash_store.set_state(state["hash_store"])
        agent.hopfield = HopfieldNet(dim=vsa.dim)
        agent.hopfield.set_state(state["agent_hopfield"]["patterns"])
        agent.monitor = SelfMonitor(agent.hopfield,
                                     energy_threshold=state["energy_threshold"])
        agent.clonal = ClonalPool(input_dim=vsa.dim)
        agent.clonal.set_state(state["clonal"])
        agent.gonogo = GoNoGoActorCritic(input_dim=vsa.dim, n_actions=4)
        agent.history = state["history"]
        agent._text_index = dict(zip(state["text_index_keys"],
                                      state["text_index_values"]))
        agent._monitor_calibrated = state["_monitor_calibrated"]
        agent.turn_count = state["turn_count"]
        agent._last_query_vec = None
        agent._cached_context = ""
        agent._ctx_cache_threshold = 0.7
        agent._retrieval_threshold = 0.38
        agent._retrieval_margin = 0.1
        if "token_library" in state:
            agent.library = TokenLibrary.from_state(state["token_library"])
        else:
            agent.library = TokenLibrary(vector_cache_size=1024)
            for text in decoder._texts:
                agent.library.add(text)
        agent.consolidation = ConsolidationMemory.from_state(
            state.get("consolidation", {})
        )
        agent._hot_memory_capacity = state.get("hot_memory_capacity", 200)
        agent._history_capacity = state.get("history_capacity", 1000)
        agent.semantic_model = state.get("semantic_model")
        semantic_state = state.get("semantic_encoder")
        if semantic_state and agent.semantic_model:
            agent.semantic_encoder = SemanticVSAEncoder(
                OllamaEmbedder(agent.semantic_model),
                vsa_dim=semantic_state["vsa_dim"],
                seed=semantic_state["seed"],
                projection=semantic_state["projection"],
            )
        else:
            agent.semantic_encoder = None
        agent.response_model = state.get("response_model")
        agent.response_generator = (
            OllamaResponseGenerator(agent.response_model)
            if agent.response_model else None
        )
        composer_state = state.get("chunk_composer")
        agent.chunk_composer = (
            LearnedChunkComposer.from_state(composer_state)
            if composer_state else None
        )
        agent.candidate_generator = FixedSpliceCandidateGenerator()
        agent.candidate_scorer = SequenceCandidateScorer.from_state(
            state.get("candidate_scorer", {})
        )
        sequence_state = state.get("sequence_ranker")
        sequence_embedder = (
            OllamaEmbedder(agent.semantic_model)
            if agent.semantic_model else None
        )
        agent.sequence_ranker = (
            VSASequenceRanker.from_state(
                sequence_state, embedder=sequence_embedder
            )
            if sequence_state else None
        )
        if agent._monitor_calibrated:
            agent.monitor.calibrated = True
            agent.monitor.energy_mean = state["monitor_energy_mean"]
            agent.monitor.energy_std = state["monitor_energy_std"]
        return agent
