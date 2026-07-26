import torch
import torchhd

from src.vsa import VSA


class VSAEncoder:
    def __init__(self, vsa: VSA | None = None, max_seq_len: int = 64,
                 mode: str = 'vsa', sem_model_name: str = 'all-MiniLM-L6-v2'):
        self.vsa = vsa or VSA()
        self.mode = mode
        self.max_seq_len = max_seq_len
        self._word_cache: dict[str, torch.Tensor] = {}
        self._pos_vectors = torchhd.random(max_seq_len, self.vsa.dim,
                                           device=self.vsa.device)
        self._sem_model = None
        self._sem_proj = None
        if mode == 'semantic':
            self._init_semantic(sem_model_name)

    def _init_semantic(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
            self._sem_model = SentenceTransformer(model_name)
            sem_dim = self._sem_model.get_sentence_embedding_dimension()
            self._sem_proj = torch.nn.Linear(sem_dim, self.vsa.dim)
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )

    def _ensure_word(self, word: str) -> torch.Tensor:
        if word not in self._word_cache:
            self._word_cache[word] = self.vsa.make_vector()
        return self._word_cache[word]

    def encode_vsa(self, text: str) -> torch.Tensor:
        words = text.lower().strip(".,!?;:'\"").split()
        words = words[:self.max_seq_len]
        if not words:
            return torch.zeros(self.vsa.dim, device=self.vsa.device)
        vecs = torch.stack([self._ensure_word(w) for w in words])
        pos = self._pos_vectors[:len(vecs)]
        bound = torchhd.bind(vecs, pos)
        bundle = torchhd.multiset(bound)
        return bundle.sign()

    def encode_semantic(self, text: str) -> torch.Tensor:
        emb = self._sem_model.encode(text, convert_to_tensor=True)
        projected = self._sem_proj(emb.to(self.vsa.device))
        return projected.sign()

    def encode(self, text: str) -> torch.Tensor:
        if self.mode == 'semantic':
            return self.encode_semantic(text)
        return self.encode_vsa(text)

    def encode_batch(self, texts: list[str]) -> torch.Tensor:
        return torch.stack([self.encode(t) for t in texts])

    def similarity(self, a: str, b: str) -> float:
        va = self.encode(a)
        vb = self.encode(b)
        return self.vsa.similarity(va, vb).item()

    def similarity_batch(self, texts_a: list[str],
                         texts_b: list[str]) -> torch.Tensor:
        va = self.encode_batch(texts_a)
        vb = self.encode_batch(texts_b)
        return torchhd.cosine_similarity(va, vb)
