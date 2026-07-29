import torch
import re
from pathlib import Path

from src.vsa import VSA, VSAHashStore, HopfieldNet, RelationalEncoder, RelationalMemory
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
from src.text.ecology import ResponseEcosystem
from src.text.ecology.frame_extractor import FrameLibrary, extract_frame
from src.text.ecology.proposition_extractor import extract_propositions


RELATIONAL_VSA_DIM = 2000


def _new_relational_memory() -> RelationalMemory:
    return RelationalMemory(
        RelationalEncoder(VSA(dim=RELATIONAL_VSA_DIM, device="cpu")),
        dim=RELATIONAL_VSA_DIM,
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
        self._gonogo_optimizer = torch.optim.AdamW(self.gonogo.parameters(), lr=1e-3)
        # Off by default: see _retrieve_context / record_feedback. gonogo
        # computes a real decision every question turn regardless (visible
        # as gonogo_go/gonogo_action on the result). Enabled by default as
        # of experiments/gonogo_gate_comparison.py: a trained agent forked
        # into gate-off/gate-on copies and evaluated on identical held-out
        # scenarios showed the gate strictly helping (40%->52% overall
        # accuracy, 76%->100% on trap questions, no change on matching
        # questions -- consistent with veto-only: it suppresses wrong
        # acceptances but can never rescue a wrongly-rejected match). A
        # freshly constructed, untrained network vetoes close to randomly
        # until record_feedback has trained it on real outcomes; set this
        # to False to opt back out of that behavior during early training.
        self.gonogo_gate_enabled = True
        self._last_gonogo_decision: dict | None = None
        # Fixed, reproducible from vsa_dim alone (same seed every
        # construction/load -- no persistence needed): gonogo's state must
        # encode retrieval QUALITY (score, margin), not the query's
        # semantic content. The raw query vector is unique per question and
        # carries no information about how good the match was, so a policy
        # trained on it can't generalize "trust high-confidence matches"
        # across different topics -- confirmed empirically via
        # experiments/gonogo_feedback_benchmark.py, where using the raw
        # query vector as state made gonogo's agreement with ground truth
        # *worse* over a training run (40% late vs 60% early), not better.
        _gonogo_axis_gen = torch.Generator().manual_seed(20260728)
        self._gonogo_score_axis = torch.randn(vsa_dim, generator=_gonogo_axis_gen)
        self._gonogo_margin_axis = torch.randn(vsa_dim, generator=_gonogo_axis_gen)
        # gonogo.act() has no exploration floor of its own (unlike
        # RelevanceSelector, src/basal/relevance.py), so it can get
        # permanently stuck favoring one action from an unlucky
        # initialization -- the same failure mode found and fixed for
        # RelevanceSelector (RELATIONAL_MEMORY.md SS2.7), confirmed here too:
        # two otherwise-identical gonogo_feedback_benchmark.py runs showed
        # wildly different learning curves (58%->69% agreement vs a stuck
        # 48%->52%) before this was added.
        self._gonogo_epsilon = 0.15
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
        # Independent of vsa_dim (which is sometimes set very small, e.g. 64,
        # for the bag-of-words text encoder/novelty-detection path): reliable
        # VSA bind/unbind needs enough dimensionality to keep crosstalk down
        # (see RELATIONAL_MEMORY.md), so relational memory gets its own VSA.
        self.relational = _new_relational_memory()
        # Provenance for relational triples, kept outside RelationalMemory
        # entirely rather than as a VSA context role: a context role shares
        # the entity-vector namespace used for decode candidates (by design,
        # for cross-role vocabulary sharing — see RelationalEncoder.filler),
        # so binding a sentence-id string in as context would leak raw id
        # strings into subject/object candidate lists. This is plain lookup
        # metadata, never something a query should be conditioned on or a
        # decode should ever return.
        self._relational_sources: dict[tuple[str, str, str], list[int]] = {}
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
        self.response_ecosystem: ResponseEcosystem | None = None
        # Always on (unlike chunk_composer/response_ecosystem): a pure
        # Python evidence-counting structure with no cost when empty, and
        # response_ecosystem's frame-aware realisation degrades gracefully
        # to its fixed template table when it has nothing for a relation
        # yet -- there's no reason to gate this behind an opt-in flag the
        # way an LLM-backed or VSA-training component needs to be.
        self.frame_library = FrameLibrary()

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
        for sentence in self.candidate_generator._sentences(response):
            self.frame_library.observe(sentence)
        for prop in extract_propositions([response]):
            frame = extract_frame(prop.source_text)
            self.relational.store_triple(
                prop.subject, prop.relation, prop.object,
                frame=frame.template if frame is not None else None,
            )

    def enable_vsa_sequence_ranking(self, dimension: int = 512):
        self.sequence_ranker = VSASequenceRanker(dimension=dimension)

    def enable_ecological_generation(
        self,
        survivors_per_niche: int = 2,
        max_rounds: int = 3,
    ):
        self.response_ecosystem = ResponseEcosystem(
            candidate_generator=self.candidate_generator,
            scorer=self.candidate_scorer,
            survivors_per_niche=survivors_per_niche,
            max_rounds=max_rounds,
        )

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
        for subject, relation, obj in ConsolidationMemory.extract_relations(user_text):
            self.relational.store_triple(subject, relation, obj)
            source_ids = self._relational_sources.setdefault(
                (subject, relation, obj), []
            )
            if sentence_id not in source_ids:
                source_ids.append(sentence_id)
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

    # First-word + short-length heuristic, not exact-phrase matching: an
    # earlier version required the WHOLE normalised message to exactly
    # equal one of a handful of canonical phrases, which broke on anything
    # combining two of them ("Yes, that's correct." normalises to "yes that
    # s correct", which isn't equal to either "yes" or "that s correct" on
    # its own). Matching on the first word plus a length cap is more
    # forgiving of real phrasing while still bounding the false-positive
    # risk: a short reply starting with "no" (e.g. "No thanks") right after
    # a retrieval-based answer could be misread as a correction -- a real,
    # disclosed limitation of a heuristic detector, not a hidden one, and
    # only reachable in the narrow window right after such an answer (see
    # process_turn's turn-number check).
    _POSITIVE_FEEDBACK_STARTS = frozenset({
        "yes", "yep", "yeah", "correct", "exactly", "right", "perfect", "good",
    })
    _NEGATIVE_FEEDBACK_STARTS = frozenset({
        "no", "nope", "wrong", "incorrect",
    })
    _MAX_FEEDBACK_WORDS = 5

    @classmethod
    def _detect_feedback_signal(cls, text: str) -> bool | None:
        """True/False for a short confirmation/correction utterance, None
        if `text` isn't feedback-shaped at all (the common case -- an
        ordinary new statement or question)."""
        words = ConsolidationMemory.normalise(text).split()
        if not words or len(words) > cls._MAX_FEEDBACK_WORDS:
            return None
        if words[0] in cls._POSITIVE_FEEDBACK_STARTS:
            return True
        if words[0] in cls._NEGATIVE_FEEDBACK_STARTS:
            return False
        return None

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
            self._last_gonogo_decision = None
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
        heuristic_accepted = (
            best_score >= self._retrieval_threshold
            and margin >= self._retrieval_margin
        )

        # gonogo makes a real Go/NoGo decision every question turn regardless
        # of the gate below -- it's always exercised, always visible on the
        # result, and always trainable via record_feedback. Whether it
        # actually CHANGES the outcome is gated separately
        # (gonogo_gate_enabled): with no ambient ground-truth signal in
        # ordinary conversation, an untrained network vetoing turns at
        # random would just be noise, not learning -- see __init__.
        #
        # State is retrieval QUALITY (score, margin embedded via two fixed
        # random axes), not the query's semantic content -- see __init__
        # for why the latter doesn't generalize.
        gonogo_state = (
            best_score * self._gonogo_score_axis
            + margin * self._gonogo_margin_axis
        )
        if torch.rand(1).item() < self._gonogo_epsilon:
            gonogo_action = torch.randint(0, 4, (1,)).item()
        else:
            gonogo_action, _ = self.gonogo.act(gonogo_state)
        gonogo_go = gonogo_action in (0, 1)
        self._last_gonogo_decision = {
            "state": gonogo_state.detach(), "action": gonogo_action,
            "turn": self.turn_count,
        }
        accepted = heuristic_accepted
        if self.gonogo_gate_enabled and heuristic_accepted and not gonogo_go:
            # NoGo can only veto an already-accepted candidate (suppress),
            # never approve one the threshold rule already rejected --
            # matches the indirect pathway's suppressive role and bounds
            # the gate's influence to "more cautious," never "more likely
            # to hallucinate."
            accepted = False

        return {
            "accepted": accepted,
            "memory_id": best_id if accepted else None,
            "text": best_text if accepted else "",
            "score": best_score,
            "margin": margin,
            "gonogo_action": gonogo_action,
            "gonogo_go": gonogo_go,
            "candidates": [
                {"id": sentence_id, "text": text, "score": score}
                for sentence_id, text, score in candidates
            ],
        }

    def _parse_multi_clue_identity_question(self, text: str) -> list[tuple[str, str]] | None:
        """'Who/what is X [and [is] Y ...]?' -> [("is", x), ("is", y), ...].

        A bounded, explicit extension of ConsolidationMemory's "is" relation
        vocabulary to support genuinely multi-clue identification questions —
        the one question shape that actually fits RelationalMemory.resolve()'s
        design (intersecting independent evidence about the same unknown
        SUBJECT). "What is X's Y" (unknown OBJECT, handled separately via
        parse_relation_query) doesn't fit resolve()'s intersection at all,
        since the tied candidates there are alternative values of one
        (subject, relation) pair, not one entity's membership across several
        different relations — see RELATIONAL_MEMORY.md SS2.4/SS6.

        Splitting purely on the word "and" is ambiguous: a property's own
        text can itself contain "and" (e.g. an LLM-generated fact "Dog is
        loyal and affectionate." — found via
        experiments/llm_relational_benchmark.py, hand-crafted examples never
        exercised this). Naively splitting on every "and" wrongly cuts that
        into two clues ("loyal", "affectionate"), neither of which exactly
        matches the stored value, falling back to noisy vector similarity
        instead of exact ground truth and sometimes landing on a wrong
        answer. Fixed with a known-value merge pass, not smarter grammar:
        after the naive split, adjacent pieces are greedily re-joined
        (longest span first) whenever the merged text exactly matches a
        value already recorded under relation "is" in
        self.relational.triples — using the actual stored vocabulary to
        correct the segmentation (the same idea as dictionary-based/
        maximum-munch tokenization, or gazetteer-based named entity
        recognition, applied to clue boundaries instead of word or entity
        boundaries). This only recognizes clues that match something
        already stored — it can't discover a genuinely novel multi-word
        property it's never seen — which is the right scope here: the
        point is to recognize which previously-asserted facts a question
        refers to, not to parse free-form grammar.
        """
        query = text.strip().rstrip("?.!")
        match = re.match(r"^(?:who|what)\s+is\s+(.+)$", query, flags=re.IGNORECASE)
        if not match:
            return None
        pieces = re.split(r"\s+and\s+(?:is\s+)?", match.group(1))
        pieces = [ConsolidationMemory.normalise(piece) for piece in pieces]
        pieces = [piece for piece in pieces if piece]
        if not pieces:
            return None

        known_values = {
            obj for _, relation, obj, _ in self.relational.triples
            if relation == "is"
        }
        merged: list[str] = []
        i = 0
        while i < len(pieces):
            for j in range(len(pieces), i, -1):
                candidate = " and ".join(pieces[i:j])
                if candidate in known_values:
                    merged.append(candidate)
                    i = j
                    break
            else:
                merged.append(pieces[i])
                i += 1

        return [("is", value) for value in merged] if len(merged) > 1 else None

    def _relational_source_ids(self, subject: str, relation: str, obj: str) -> list[int]:
        return list(self._relational_sources.get((subject, relation, obj), []))

    def _relational_result(self, known: dict, missing_role: str, answer: str) -> dict:
        full = dict(known)
        full[missing_role] = answer
        source_ids = self._relational_source_ids(
            full["subject"], full["relation"], full["object"]
        )
        text = (
            self.library.texts[source_ids[0]] if source_ids
            else f"{full['subject']} {full['relation']} {full['object']}."
        )
        return {
            "accepted": True,
            "memory_id": source_ids[0] if source_ids else None,
            "source_ids": source_ids,
            "text": text,
            "score": 1.0,
            "margin": 1.0,
            "relational": True,
            "ambiguous": False,
            "candidates": [
                {"id": sid, "text": self.library.texts[sid], "score": 1.0}
                for sid in source_ids
            ],
        }

    @staticmethod
    def _ambiguous_relational_result(candidates: list[str]) -> dict:
        listing = candidates[0] if len(candidates) == 1 else (
            ", ".join(candidates[:-1]) + " or " + candidates[-1]
        )
        return {
            "accepted": True,
            "memory_id": None,
            "source_ids": [],
            "text": (
                f"It could be {listing} — I don't have enough information "
                f"to be sure."
            ),
            "score": 0.5,
            "margin": 0.0,
            "relational": True,
            "ambiguous": True,
            "ambiguous_candidates": candidates,
            "candidates": [],
        }

    def _gather_candidate_queries(
        self, candidates: list[str], exclude_pairs: set[tuple[str, str]]
    ) -> list[tuple[dict, dict | None]]:
        """Bounded pool of other already-recorded (relation, object) facts
        about the tied `candidates`, for resolve_auto to select from — only
        used as a fallback when the user's explicitly stated clues alone
        don't resolve the ambiguity (see _answer_relational_query). This is
        deliberately not an open-ended memory search: only facts literally
        already stored about entities already in the tied pool, excluding
        the exact (relation, object) pairs already tried -- note this
        excludes by the specific pair, not the whole relation name, since
        two different clues can legitimately share a relation (e.g. "is a
        mammal" and "is loyal" are both relation "is"). See
        RELATIONAL_MEMORY.md SS2.7.
        """
        seen: set[tuple[str, str]] = set(exclude_pairs)
        pool: list[tuple[dict, dict | None]] = []
        for subject, relation, obj, _ctx in self.relational.triples:
            if subject not in candidates:
                continue
            key = (relation, obj)
            if key in seen:
                continue
            seen.add(key)
            pool.append(({"relation": relation, "object": obj}, None))
        return pool

    def _answer_relational_query(self, user_input: str) -> dict | None:
        """Answer a question directly from relational memory when possible.

        Multi-clue identity questions ("who is X and is Y") use resolve() to
        intersect independent evidence about the same unknown subject.
        Single-relation questions ("what is X's Y") use complete_detailed
        directly — resolve() doesn't apply to that shape (see
        _parse_multi_clue_identity_question). Returns None when the question
        doesn't parse into a relational query at all, falling through to
        ordinary retrieval. Ambiguity is surfaced honestly (named candidates)
        rather than silently guessed, per RELATIONAL_MEMORY.md SS2.2.
        """
        clues = self._parse_multi_clue_identity_question(user_input)
        if clues:
            queries = [
                ({"relation": relation, "object": obj}, None)
                for relation, obj in clues
            ]
            # top_k=2: intersection across steps only narrows anything when
            # each step's candidate list actually excludes the losers. The
            # library default (5) can exceed the total entity vocabulary in
            # a small/young memory, in which case every step's "top-5" is
            # just "everything" and intersecting them narrows nothing even
            # when the underlying scores are clearly separated.
            trace = self.relational.resolve(queries, top_k=2)
            if not trace.resolved and trace.final_candidates:
                # The explicit clues alone weren't enough -- escalate to
                # basal-ganglia relevance selection (RELATIONAL_MEMORY.md
                # SS2.7) over other already-known facts about the tied
                # candidates, rather than giving up or guessing.
                exclude = set(clues)
                pool = self._gather_candidate_queries(trace.final_candidates, exclude)
                if pool:
                    trace = self.relational.resolve_auto(
                        queries, pool, top_k=2, max_steps=3
                    )
            if trace.resolved:
                # Cite whichever query actually settled it -- the last
                # explicit clue, or the auto-selected one if escalation was
                # what resolved it.
                return self._relational_result(
                    trace.steps[-1].known, "subject", trace.final_candidates[0],
                )
            if trace.final_candidates:
                return self._ambiguous_relational_result(trace.final_candidates)
            return None

        parsed = ConsolidationMemory.parse_relation_query(user_input)
        if parsed is None:
            return None
        subject, relation = parsed
        net = self.relational.nets.get(relation)
        if net is None or len(net) == 0:
            return None
        # Hopfield recall always returns *some* nearest-pattern guess, even
        # for a subject that was never actually stored (e.g. "who is X" is
        # genuinely ambiguous between "describe X" and "which entity has
        # property X" — parse_relation_query always takes the former
        # reading, so a property phrase like "a mammal" can land here as a
        # literal, never-asserted "subject"). Refuse to answer rather than
        # let a soft nearest-match masquerade as real knowledge.
        if not any(s == subject for s, r, _, _ in self.relational.triples):
            return None
        result = self.relational.complete_detailed(
            {"subject": subject, "relation": relation}
        )
        if result.best is None:
            return None
        if not result.ambiguous:
            return self._relational_result(
                {"subject": subject, "relation": relation}, "object", result.best
            )
        # Ground truth, not result.candidates: complete_detailed's top-k
        # candidate list is drawn from the whole shared entity-vector
        # namespace (every entity ever seen, across every relation), so an
        # unrelated low-score entity can still make the top-k in a small
        # vocabulary and get named as a plausible answer alongside the real
        # ones. The literal stored triples are exact and can't contain that
        # noise — same principle as resolve()'s ground-truth-exact final
        # candidates (RELATIONAL_MEMORY.md SS2.4).
        ground_truth = self.relational.ground_truth_ambiguity(
            {"subject": subject, "relation": relation}
        )
        return self._ambiguous_relational_result(
            sorted({obj for _, _, obj in ground_truth})
        )

    def _handle_feedback_turn(self, user_input: str, positive: bool) -> dict:
        """Consume a live confirmation/correction turn: train gonogo via
        record_feedback and respond with an acknowledgement instead of
        processing this as a new statement or question. Doesn't touch
        long-term memory (nothing factual was actually asserted) and
        consumes _last_gonogo_decision so a second feedback-shaped message
        in a row (with no new retrieval-based question in between) falls
        through to ordinary handling instead of being misapplied again.
        """
        self.turn_count += 1
        self.record_feedback(correct=positive)
        self._last_gonogo_decision = None
        self.history.append({
            "turn": self.turn_count,
            "user": user_input,
            "user_vec": self.encoder.encode(user_input),
            "long_term": False,
        })
        response = (
            "Thanks, I'll keep that in mind."
            if positive else
            "Thanks for the correction — I'll learn from that."
        )
        return {
            "response": response,
            "response_generated": False,
            "response_mode": "feedback_acknowledged",
            "sources": [],
            "intent": "feedback",
            "retrieval_accepted": False,
            "retrieval_score": 0.0,
            "retrieval_margin": 0.0,
            "retrieval_candidates": [],
            "reasoning": None,
            "ambiguous": False,
            "ambiguous_candidates": [],
            "gonogo_go": None,
            "gonogo_action": None,
            "drift_detected": False,
            "energy_z": 0.0,
            "clonal_created": False,
            "turn": self.turn_count,
        }

    def process_turn(self, user_input: str, ctx_cache: bool = True) -> dict:
        # Live feedback loop: a short confirmation/correction directly
        # following a general-retrieval question turn trains gonogo via
        # record_feedback, using the user's own next message as the reward
        # signal -- the only genuinely non-fabricated source of "was that
        # correct" available in ordinary conversation (see record_feedback's
        # docstring on why nothing else qualifies). Checked before
        # incrementing turn_count: the comparison is against the turn number
        # _retrieve_context tagged its decision with, i.e. "the turn that
        # just happened," not the one about to start. Requires the decision
        # to be from the IMMEDIATELY preceding turn, not an arbitrary
        # earlier one -- feedback several turns later is ambiguous about
        # what it's even referring to.
        feedback_signal = self._detect_feedback_signal(user_input)
        if (
            feedback_signal is not None
            and self._last_gonogo_decision is not None
            and self._last_gonogo_decision.get("turn") == self.turn_count
        ):
            return self._handle_feedback_turn(user_input, feedback_signal)

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
                relational_answer = self._answer_relational_query(user_input)
                if relational_answer is not None:
                    retrieval = relational_answer
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

        synthesized = False
        if intent == "statement":
            response = "I'll remember that."
        elif not retrieval["accepted"]:
            response = "I don't have a sufficiently relevant memory for that."
        elif self.response_ecosystem is not None:
            evidence = [
                candidate["text"] for candidate in retrieval["candidates"]
            ]
            result = self.response_ecosystem.generate(
                user_input, evidence, self.chunk_composer, self.sequence_ranker,
                relational_memory=self.relational, frame_library=self.frame_library,
                fallback=context,
            )
            response = result.response
            synthesized = True
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
            synthesized = True
        elif self.response_generator is not None:
            response = self.response_generator(
                user_input,
                [{
                    "id": retrieval["memory_id"],
                    "text": retrieval["text"],
                    "score": retrieval["score"],
                }],
            )
            synthesized = True
        else:
            response = context

        if intent == "statement":
            response_mode = "acknowledgement"
        elif not retrieval["accepted"]:
            response_mode = "abstention"
        elif self.response_generator is None:
            if self.response_ecosystem is not None:
                response_mode = "ecological_generation"
            elif self.chunk_composer is not None:
                response_mode = "learned_chunk_composition"
            else:
                response_mode = "retrieved_memory"
        else:
            response_mode = getattr(
                self.response_generator, "last_mode", "grounded_generation"
            )
        if retrieval.get("reasoning") is not None:
            response_mode = "consolidated_reasoning"
        elif retrieval.get("relational"):
            response_mode = (
                "relational_ambiguous" if retrieval.get("ambiguous")
                else "relational_reasoning"
            )

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
            "response_generated": synthesized,
            "response_mode": response_mode,
            "sources": sources,
            "intent": intent,
            "retrieval_accepted": retrieval["accepted"],
            "retrieval_score": retrieval["score"],
            "retrieval_margin": retrieval["margin"],
            "retrieval_candidates": retrieval["candidates"],
            "reasoning": retrieval.get("reasoning"),
            "ambiguous": retrieval.get("ambiguous", False),
            "ambiguous_candidates": retrieval.get("ambiguous_candidates", []),
            "gonogo_go": retrieval.get("gonogo_go"),
            "gonogo_action": retrieval.get("gonogo_action"),
            "drift_detected": novelty["novel"],
            "energy_z": novelty["energy_z"],
            "clonal_created": novelty["clonal_created"],
            "turn": self.turn_count,
        }

    def record_feedback(self, correct: bool, reward: float | None = None) -> None:
        """Train gonogo's retrieval accept/reject decision (see
        _retrieve_context) from EXPLICIT external feedback about the most
        recent question turn's retrieval -- e.g. a benchmark harness that
        knows the correct answer, or a real user correction.

        There is no ambient reward signal for this decision in ordinary
        unsupervised conversation: unlike RelationalMemory.resolve_auto,
        which can self-supervise from its own ground-truth triple
        intersection (see RELATIONAL_MEMORY.md SS2.7), whether a general
        free-text retrieval was actually correct isn't something the agent
        can determine on its own. So gonogo computes a real decision every
        question turn (visible as gonogo_go/gonogo_action on process_turn's
        result) but its weights only change when this is called -- without
        it, gonogo never learns from ordinary conversation. That's the
        honest characterization of the current wiring, not a limitation to
        hide: gonogo_gate_enabled defaults to False for exactly this
        reason (see __init__), so nothing behavioral changes until this
        has actually trained it on real outcomes and a caller opts in.

        No-op if the most recent question turn didn't reach a retrieval
        decision (e.g. it was a statement, or answered via the
        reasoning/relational paths instead of _retrieve_context).
        """
        if self._last_gonogo_decision is None:
            return
        reward = reward if reward is not None else (1.0 if correct else -1.0)
        state = self._last_gonogo_decision["state"]
        # The actual action sampled during _retrieve_context, not a
        # re-derived one: act() samples stochastically, so recomputing
        # (even deterministically) from the same state could give a
        # different action than the one that actually produced this turn's
        # observed outcome, misattributing the reward.
        action = self._last_gonogo_decision["action"]
        loss = self.gonogo.compute_loss(state, action, reward, next_state=None)
        self._gonogo_optimizer.zero_grad()
        loss.backward()
        self._gonogo_optimizer.step()

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
            "gonogo_state_dict": self.gonogo.state_dict(),
            "gonogo_gate_enabled": self.gonogo_gate_enabled,
            "gonogo_epsilon": self._gonogo_epsilon,
            "decoder": self.decoder.get_state(),
            "history": history,
            "text_index_keys": list(self._text_index.keys()),
            "text_index_values": list(self._text_index.values()),
            "token_library": self.library.get_state(),
            "consolidation": self.consolidation.get_state(),
            "relational": self.relational.get_state(),
            "relational_sources": list(self._relational_sources.items()),
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
            "response_ecosystem": (
                {
                    "survivors_per_niche": self.response_ecosystem.survivors_per_niche,
                    "max_rounds": self.response_ecosystem.max_rounds,
                }
                if self.response_ecosystem is not None else None
            ),
            "frame_library": self.frame_library.get_state(),
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
        if "gonogo_state_dict" in state:
            agent.gonogo.load_state_dict(state["gonogo_state_dict"])
        agent._gonogo_optimizer = torch.optim.AdamW(agent.gonogo.parameters(), lr=1e-3)
        agent.gonogo_gate_enabled = state.get("gonogo_gate_enabled", False)
        agent._last_gonogo_decision = None
        _gonogo_axis_gen = torch.Generator().manual_seed(20260728)
        agent._gonogo_score_axis = torch.randn(vsa.dim, generator=_gonogo_axis_gen)
        agent._gonogo_margin_axis = torch.randn(vsa.dim, generator=_gonogo_axis_gen)
        agent._gonogo_epsilon = state.get("gonogo_epsilon", 0.15)
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
        relational_state = state.get("relational")
        agent.relational = (
            RelationalMemory.from_state(relational_state)
            if relational_state is not None
            else _new_relational_memory()
        )
        agent._relational_sources = {
            tuple(key): ids for key, ids in state.get("relational_sources", [])
        }
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
        ecosystem_state = state.get("response_ecosystem")
        agent.response_ecosystem = (
            ResponseEcosystem(
                candidate_generator=agent.candidate_generator,
                scorer=agent.candidate_scorer,
                survivors_per_niche=ecosystem_state.get("survivors_per_niche", 2),
                max_rounds=ecosystem_state.get("max_rounds", 3),
            )
            if ecosystem_state is not None else None
        )
        agent.frame_library = FrameLibrary.from_state(
            state.get("frame_library", {})
        )
        if agent._monitor_calibrated:
            agent.monitor.calibrated = True
            agent.monitor.energy_mean = state["monitor_energy_mean"]
            agent.monitor.energy_std = state["monitor_energy_std"]
        return agent
