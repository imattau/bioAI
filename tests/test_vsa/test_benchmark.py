from src.vsa import VSA
from src.vsa.benchmark import RavenLikeTask, accuracy


class TestBenchmark:
    def setup_method(self):
        self.vsa = VSA(dim=1000)
        self.task = RavenLikeTask(self.vsa)

    def test_analogy_encoding(self):
        ab, cd = self.task.make_analogy("a", "b", "c", "d")
        assert ab.dim() == 1 and ab.shape[0] == 1000
        assert cd.dim() == 1 and cd.shape[0] == 1000

    def test_accuracy_function(self):
        assert accuracy([0, 1, 2], [0, 1, 2]) == 1.0
        assert accuracy([0, 1], [1, 0]) == 0.0
