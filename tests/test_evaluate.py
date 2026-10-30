"""tests/test_evaluate.py – Unit tests for the evaluation module."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from src.evaluate import compute_metrics, run_evaluation, save_eval
from src.data_loader import generate_synthetic_reviews


@pytest.fixture(scope="module")
def synthetic_df():
    return generate_synthetic_reviews()


@pytest.fixture()
def eval_output_dir(tmp_path):
    """Provide a temp directory for eval.json output."""
    return str(tmp_path / "eval_test")


class TestComputeMetrics:

    def test_returns_required_keys(self, synthetic_df):
        metrics = compute_metrics(synthetic_df, threshold=0.50)
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "false_positive_rate" in metrics
        assert "confusion_matrix" in metrics
        assert "per_detector_recall" in metrics
        assert "time_to_detect" in metrics

    def test_scores_in_range(self, synthetic_df):
        metrics = compute_metrics(synthetic_df, threshold=0.50)
        assert 0.0 <= metrics["precision"] <= 1.0
        assert 0.0 <= metrics["recall"] <= 1.0
        assert 0.0 <= metrics["f1"] <= 1.0
        assert 0.0 <= metrics["false_positive_rate"] <= 1.0

    def test_confusion_matrix_sums_to_total(self, synthetic_df):
        metrics = compute_metrics(synthetic_df, threshold=0.50)
        cm = metrics["confusion_matrix"]
        assert cm["tp"] + cm["fp"] + cm["fn"] + cm["tn"] == metrics["total_reviews"]

    def test_threshold_changes_results(self, synthetic_df):
        low = compute_metrics(synthetic_df, threshold=0.20)
        high = compute_metrics(synthetic_df, threshold=0.80)
        # Lower threshold → higher recall (or equal), lower precision (or equal)
        assert low["recall"] >= high["recall"]

    def test_per_detector_recall_has_all_detectors(self, synthetic_df):
        metrics = compute_metrics(synthetic_df, threshold=0.50)
        pdr = metrics["per_detector_recall"]
        assert set(pdr.keys()) == {"velocity", "similarity", "account", "rating_drop"}

    def test_time_to_detect_is_list(self, synthetic_df):
        metrics = compute_metrics(synthetic_df, threshold=0.50)
        assert isinstance(metrics["time_to_detect"], list)
        if metrics["time_to_detect"]:
            assert "campaign" in metrics["time_to_detect"][0]
            assert "hours_to_detect" in metrics["time_to_detect"][0]


class TestSaveEval:

    def test_eval_json_is_produced(self, synthetic_df, eval_output_dir):
        metrics = compute_metrics(synthetic_df, threshold=0.50)
        path = save_eval(metrics, output_dir=eval_output_dir)
        assert path.exists()
        assert path.name == "eval.json"

        with open(path) as f:
            data = json.load(f)
        assert "precision" in data
        assert "generated_at" in data
        assert "note" in data

    def test_eval_json_roundtrip(self, synthetic_df, eval_output_dir):
        metrics = compute_metrics(synthetic_df, threshold=0.50)
        path = save_eval(metrics, output_dir=eval_output_dir)

        with open(path) as f:
            loaded = json.load(f)

        assert loaded["precision"] == metrics["precision"]
        assert loaded["recall"] == metrics["recall"]
        assert loaded["f1"] == metrics["f1"]


class TestRunEvaluation:

    def test_produces_eval_json(self, eval_output_dir):
        """Integration test: run_evaluation should produce data/eval.json."""
        metrics = run_evaluation(threshold=0.50, output_dir=eval_output_dir)
        expected_path = Path(eval_output_dir) / "eval.json"
        assert expected_path.exists(), f"eval.json not found at {expected_path}"

        with open(expected_path) as f:
            data = json.load(f)
        assert data["threshold"] == 0.50
        assert isinstance(data["precision"], float)
