"""Tests for save/load persistence of BioAIDialogueAgent."""

import tempfile
from pathlib import Path

from src.text import BioAIDialogueAgent
from src.vsa import VSA, HopfieldNet, AssociativeStore, VSAHashStore


class TestSaveLoad:
    def setup_method(self):
        self.agent = BioAIDialogueAgent(vsa_dim=1000)
        self.tmp = Path(tempfile.mktemp(suffix=".pt"))

    def teardown_method(self):
        if self.tmp.exists():
            self.tmp.unlink()

    def test_save_load_empty(self):
        self.agent.save(self.tmp)
        loaded = BioAIDialogueAgent.load(self.tmp)
        assert loaded.turn_count == 0
        assert len(loaded.decoder) == 0
        assert len(loaded._text_index) == 0
        assert len(loaded.history) == 0

    def test_save_load_after_turns(self):
        facts = [
            "The capital of France is Paris",
            "The sky is blue",
            "Water freezes at 0 degrees",
            "Mars is the red planet",
            "Dogs are mammals",
        ]
        for f in facts:
            self.agent.process_turn(f)

        self.agent.save(self.tmp)
        loaded = BioAIDialogueAgent.load(self.tmp)

        assert loaded.turn_count == len(facts)
        assert len(loaded.decoder) == len(facts)
        assert len(loaded._text_index) == len(facts)
        for f in facts:
            recalled = loaded.recall(f)
            assert recalled == f, f"'{f}' not recallable after load, got '{recalled}'"

    def test_continue_after_load(self):
        for i in range(5):
            self.agent.process_turn(f"initial fact {i}")
        self.agent.save(self.tmp)

        loaded = BioAIDialogueAgent.load(self.tmp)
        assert loaded.turn_count == 5

        for i in range(5, 10):
            loaded.process_turn(f"new fact {i}")
        assert loaded.turn_count == 10
        for i in range(10):
            recalled = loaded.recall(f"initial fact {i}" if i < 5 else f"new fact {i}")
            assert recalled == (f"initial fact {i}" if i < 5 else f"new fact {i}"), \
                f"Lost fact {i}: '{recalled}'"

    def test_monitor_calibration_preserved(self):
        for i in range(5):
            self.agent.process_turn(f"calibration fact {i}")
        assert self.agent._monitor_calibrated

        self.agent.save(self.tmp)
        loaded = BioAIDialogueAgent.load(self.tmp)

        assert loaded._monitor_calibrated
        assert loaded.monitor.energy_mean == self.agent.monitor.energy_mean
        assert loaded.monitor.energy_std == self.agent.monitor.energy_std

    def test_clonal_pool_preserved(self):
        for i in range(10):
            self.agent.process_turn(f"distinct fact number {i}")
        self.agent.save(self.tmp)
        loaded = BioAIDialogueAgent.load(self.tmp)

        assert len(loaded.clonal) == len(self.agent.clonal)
        for orig_mod, load_mod in zip(self.agent.clonal.modules, loaded.clonal.modules):
            sim = self.agent.vsa.similarity(orig_mod.receptor, load_mod.receptor).item()
            assert sim > 0.99, f"Clonal receptor mismatch: sim={sim}"

    def test_hash_store_preserved(self):
        for i in range(10):
            self.agent.process_turn(f"hash fact {i}")
        self.agent.save(self.tmp)
        loaded = BioAIDialogueAgent.load(self.tmp)

        assert len(loaded.hash_store) == len(self.agent.hash_store)
        for i in range(10):
            hv = loaded.encoder.encode(f"hash fact {i}")
            result = loaded.hash_store.lookup(hv)
            assert result is not None, f"Hash store lost fact {i}"

    def test_decoder_store_preserved(self):
        for i in range(10):
            self.agent.process_turn(f"decoder fact {i}")
        self.agent.save(self.tmp)
        loaded = BioAIDialogueAgent.load(self.tmp)

        assert len(loaded.decoder) == len(self.agent.decoder)
        for i in range(10):
            recalled = loaded.decoder.decode_from_text(f"decoder fact {i}", k=1)
            assert recalled[0][0] == f"decoder fact {i}", \
                f"Decoder lost fact {i}: '{recalled[0][0]}'"

    def test_hopfield_patterns_preserved(self):
        for i in range(10):
            self.agent.process_turn(f"hopfield fact {i}")
        self.agent.save(self.tmp)
        loaded = BioAIDialogueAgent.load(self.tmp)

        assert len(loaded.hopfield) == len(self.agent.hopfield)
        hv_src = self.agent.encoder.encode("hopfield fact 0")
        hv_dst = loaded.encoder.encode("hopfield fact 0")
        rec_src = self.agent.hopfield.recall(hv_src, steps=5)
        rec_dst = loaded.hopfield.recall(hv_dst, steps=5)
        sim = self.agent.vsa.similarity(rec_src, rec_dst).item()
        assert sim > 0.99, f"Hopfield patterns diverged: sim={sim}"

    def test_save_size_is_reasonable(self):
        import os
        n_facts = 500
        agent = BioAIDialogueAgent(vsa_dim=1000)
        common_prefix = "the value of something is uniquely determined by"
        for i in range(n_facts):
            agent.process_turn(f"{common_prefix} its specific fact number {i}")
        path = Path(tempfile.mktemp(suffix=".pt"))
        agent.save(path)
        mb = path.stat().st_size / 1_000_000
        print(f"\n  {n_facts} facts (shared words): {mb:.1f}MB")
        assert mb < 20, f"Save file too large: {mb:.1f}MB for {n_facts} facts (expected < 20MB)"
        path.unlink()


