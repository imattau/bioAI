import torch
from pathlib import Path

from src.vsa import VSA, VSAHashStore, HopfieldNet
from src.clonal import ClonalPool
from src.immune import SelfMonitor
from src.basal import GoNoGoActorCritic
from src.text.encoder import VSAEncoder
from src.text.decoder import VSADecoder


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

    def _store_turn(self, user_text: str, user_vec: torch.Tensor):
        self.history.append({
            "turn": self.turn_count,
            "user": user_text,
            "user_vec": user_vec,
        })
        self.hash_store.insert(user_vec, user_vec)
        self.decoder.ingest(user_text)
        self.hopfield.store(user_vec)
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

    def _retrieve_context(self, user_vec: torch.Tensor) -> str:
        results = self.decoder.decode(user_vec, k=1)
        if results:
            return results[0][0]
        return ""

    def process_turn(self, user_input: str, ctx_cache: bool = True) -> dict:
        self.turn_count += 1
        user_vec = self.encoder.encode(user_input)
        self._calibrate_monitor()

        novelty = self._detect_novelty(user_vec)

        if ctx_cache and self._last_query_vec is not None:
            sim = self.vsa.similarity(user_vec, self._last_query_vec).item()
            if sim > self._ctx_cache_threshold:
                context = self._cached_context
            else:
                context = self._retrieve_context(user_vec)
                self._cached_context = context
                self._last_query_vec = user_vec
        else:
            context = self._retrieve_context(user_vec)
            self._cached_context = context
            self._last_query_vec = user_vec

        self._store_turn(user_input, user_vec)

        if not context:
            response = "I need to learn more about that."
        elif novelty["novel"]:
            response = f"This seems new. Closest I know: {context}"
        else:
            response = context

        return {
            "response": response,
            "drift_detected": novelty["novel"],
            "energy_z": novelty["energy_z"],
            "clonal_created": novelty["clonal_created"],
            "turn": self.turn_count,
        }

    def recall(self, query: str) -> str:
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
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path: str | Path) -> "BioAIDialogueAgent":
        import torchhd
        torch.serialization.add_safe_globals([torchhd.tensors.map.MAPTensor])
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
        if agent._monitor_calibrated:
            agent.monitor.calibrated = True
            agent.monitor.energy_mean = state["monitor_energy_mean"]
            agent.monitor.energy_std = state["monitor_energy_std"]
        return agent
