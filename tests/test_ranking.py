import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/Tabidachi"))
from ranking_metrics import metrics_for_group, ranking_metrics
from evaluate_modernbert_4input_v16 import ranking_metrics as model_metrics


def row(label, prediction, source="a", data_id=0):
    return {"source_file": source, "data_id": data_id, "score": label, "predicted_score": prediction}


class RankingTests(unittest.TestCase):
    def test_multiple_positives_distinguish_hit_and_recall(self):
        metrics = metrics_for_group([1, 0, 1], [1, 3])
        self.assertEqual(metrics["HR@1"], 1)
        self.assertEqual(metrics["Recall@1"], 0.5)
        self.assertAlmostEqual(metrics["NDCG@3"], 1.5 / (1 + 1 / math.log2(3)))

    def test_perfect_multiple_positives_are_normalized(self):
        metrics = model_metrics([row(1, 3), row(1, 2), row(0, 1)], [3])[0]
        self.assertEqual(metrics["NDCG@3"], 1)

    def test_mrr_uses_first_positive_with_cutoff(self):
        metrics = metrics_for_group([0, 0, 1, 1], [1, 3, 10])
        self.assertEqual(metrics["MRR@1"], 0)
        self.assertEqual(metrics["MRR@3"], 1 / 3)
        self.assertEqual(metrics["MRR@10"], 1 / 3)

    def test_no_positive_is_included_as_zero(self):
        self.assertTrue(all(v == 0 for v in metrics_for_group([0, 0], [1, 5]).values()))

    def test_source_file_prevents_group_collision_and_macro_averages(self):
        metrics, count = ranking_metrics([row(1, 0.1), row(0, 0.9, "b"), row(1, 0.2, "b")], [1, 3])
        self.assertEqual(count, 2)
        self.assertEqual(metrics["HR@1"], 0.5)
        self.assertEqual(metrics["MRR@3"], 0.75)

    def test_ties_keep_input_order(self):
        self.assertEqual(ranking_metrics([row(0, 1), row(1, 1)], [1])[0]["HR@1"], 0)
        self.assertEqual(ranking_metrics([row(1, 1), row(0, 1)], [1])[0]["HR@1"], 1)

    def test_custom_score_key(self):
        data = [{**row(1, 0), "alternative": 1}]
        self.assertEqual(ranking_metrics(data, [1], "alternative")[0]["HR@1"], 1)

    def test_invalid_ks_and_empty_inputs(self):
        for ks in ([], [0], [-1], [1.5]):
            with self.subTest(ks=ks), self.assertRaises(ValueError):
                metrics_for_group([1], ks)
        with self.assertRaises(ValueError):
            ranking_metrics([], [1])
        with self.assertRaises(ValueError):
            metrics_for_group([], [1])

    def test_invalid_rows_fail_instead_of_silently_truncating(self):
        for invalid in ({}, None, row(0.5, 1), row(-1, 1), row(1, float("nan")), row(1, float("inf")), row(1, 1, data_id="1")):
            with self.subTest(row=invalid), self.assertRaises(ValueError):
                ranking_metrics([invalid], [1])


class CliTests(unittest.TestCase):
    def run_cli(self, *args, cwd=None):
        return subprocess.run(
            [sys.executable, str(ROOT / "src/Tabidachi/evaluate_precomputed_ranking.py"), *args],
            cwd=cwd or ROOT, capture_output=True, text=True, encoding="utf-8",
            env={**os.environ, "PYTHONUTF8": "1"},
        )

    def test_example_and_plain_filename_outputs_from_other_working_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            result = self.run_cli("--input-dir", str(ROOT / "examples/ranking"), "--ks", "1", "3", "5", "--save-json", "metrics.json", "--save-scored", "rows.json", cwd=temp)
            self.assertEqual(result.returncode, 0, result.stderr)
            actual = json.loads((Path(temp) / "metrics.json").read_text())
            expected = json.loads((ROOT / "results/example-metrics.json").read_text())
            self.assertEqual(actual, expected)
            self.assertEqual(len(json.loads((Path(temp) / "rows.json").read_text())), 6)

    def test_missing_empty_and_malformed_input_errors(self):
        with tempfile.TemporaryDirectory() as temp:
            for directory in (str(Path(temp) / "missing"), temp):
                result = self.run_cli("--input-dir", directory)
                self.assertEqual(result.returncode, 2)
                self.assertNotIn("Traceback", result.stderr)
            (Path(temp) / "bad.json").write_text('{}')
            self.assertEqual(self.run_cli("--input-dir", temp).returncode, 2)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PreprocessingTests(unittest.TestCase):
    def test_tabidachi_merges_consecutive_speakers_and_aligns_labels(self):
        module = load_module("src/Tabidachi/create_dataset_1.py", "tabidachi_transform")
        result = module.transform_data({
            "context": [{"speaker": "operator", "utterance": "hello"}, {"speaker": "customer", "utterance": "quiet"}, {"speaker": "customer", "utterance": "indoors"}],
            "candidates": [{"id": "x", "detail": {"Summary": "museum"}}, {"id": "y", "detail": {"Summary": "park"}}],
            "mentioned": {"id": "y"},
        })
        self.assertEqual(result["dialogue"], "A: hello\nB: quiet indoors")
        self.assertEqual(result["score"], [0, 1])

    def test_chatrec_threshold_and_speaker_mapping(self):
        module = load_module("src/ChatRec/data_preprocessing.py", "chatrec_preprocess")
        result = module.process_data({
            "dialogue": [{"speaker": "alice", "utterance": "hello"}, {"speaker": "bob", "utterance": "hi"}],
            "place": [{"id": "x", "name": "synthetic", "description": "example"}],
            "questionnaire": {"alice": {"evaluation": [{"id": "x", "score": 3}]}, "bob": {"evaluation": [{"id": "x", "score": 2}]}},
        })
        self.assertEqual(result["dialogue"], "A:hello\nB:hi\n")
        self.assertEqual(result["place"][0]["A_Score"], 1)
        self.assertEqual(result["place"][0]["B_Score"], 0)


if __name__ == "__main__":
    unittest.main()
