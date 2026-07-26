import torch

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

    def process_turn(self, user_input: str) -> dict:
        self.turn_count += 1
        user_vec = self.encoder.encode(user_input)
        self._calibrate_monitor()

        novelty = self._detect_novelty(user_vec)
        context = self._retrieve_context(user_vec)
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
