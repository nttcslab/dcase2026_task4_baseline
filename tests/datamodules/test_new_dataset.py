import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datamodules.new_dataset import DatasetS3, collate_fn
from src.utils import LABELS
from tests.helpers import (
    N_SAMPLES,
    SR,
    make_mock_synthesize_output,
    make_waveform_dataset,
    write_mono_wav,
)


class TestCollateFn(unittest.TestCase):
    def test_stacks_tensor_fields(self):
        batch = [
            {
                "mixture": torch.zeros(4, 100),
                "labels": torch.tensor([0, 18, 18]),
                "active": torch.tensor([True, False, False]),
            },
            {
                "mixture": torch.ones(4, 100),
                "labels": torch.tensor([1, 2, 18]),
                "active": torch.tensor([True, True, False]),
            },
        ]
        out = collate_fn(batch)

        self.assertEqual(out["mixture"].shape, (2, 4, 100))
        self.assertEqual(out["labels"].shape, (2, 3))
        self.assertEqual(out["active"].shape, (2, 3))
        self.assertTrue(torch.equal(out["mixture"][0], torch.zeros(4, 100)))
        self.assertTrue(torch.equal(out["mixture"][1], torch.ones(4, 100)))

    def test_leaves_non_tensor_fields_as_lists(self):
        batch = [
            {"label": ["Speech", "silence"], "mixture": torch.zeros(4, 8)},
            {"label": ["Clapping", "silence"], "mixture": torch.ones(4, 8)},
        ]
        out = collate_fn(batch)
        self.assertEqual(out["label"], [batch[0]["label"], batch[1]["label"]])
        self.assertEqual(out["mixture"].shape, (2, 4, 8))


class TestLabelUtilities(unittest.TestCase):
    def _make_dataset(self, label_vector_mode="multihot", silence_label_mode="zeros"):
        config = {
            "mode": "generate",
            "dupse_rate": 0.0,
            "dupse_min_angle": 60.0,
            "max_n_dupse": 1,
            "dupse_exclusion_folder_depth": 0,
            "spatial_sound_scene": {"sr": SR},
            "snr_range": [5, 20],
            "nevent_range": [1, 2],
            "dataset_length": 1,
            "shuffle_label": False,
            "fg_return": {"dry": True},
        }
        return DatasetS3(
            config=config,
            n_sources=3,
            label_set="dcase2026t4",
            label_vector_mode=label_vector_mode,
            silence_label_mode=silence_label_mode,
            return_source=False,
        )

    def test_silence_idx_equals_num_labels(self):
        ds = self._make_dataset()
        self.assertEqual(ds.silence_idx, len(LABELS["dcase2026t4"]))

    def test_label_vector_multihot(self):
        ds = self._make_dataset(label_vector_mode="multihot")
        labels = ["Speech", "Clapping", "silence"]
        vec = ds._get_label_vector(labels)
        self.assertEqual(vec.shape, (len(LABELS["dcase2026t4"]),))
        speech_idx = LABELS["dcase2026t4"].index("Speech")
        clap_idx = LABELS["dcase2026t4"].index("Clapping")
        self.assertEqual(vec[speech_idx].item(), 1.0)
        self.assertEqual(vec[clap_idx].item(), 1.0)

    def test_label_vector_concat(self):
        ds = self._make_dataset(label_vector_mode="concat")
        labels = ["Speech", "silence", "silence"]
        vec = ds._get_label_vector(labels)
        self.assertEqual(vec.shape, (3 * len(LABELS["dcase2026t4"]),))

    def test_label_vector_stack(self):
        ds = self._make_dataset(label_vector_mode="stack")
        labels = ["Speech", "Clapping", "silence"]
        vec = ds._get_label_vector(labels)
        self.assertEqual(vec.shape, (3, len(LABELS["dcase2026t4"])))

    def test_silence_onehot_mode(self):
        ds = self._make_dataset(silence_label_mode="onehot")
        silence_vec = ds.get_onehot("silence")
        self.assertEqual(silence_vec.shape, (len(LABELS["dcase2026t4"]) + 1,))
        self.assertEqual(silence_vec[-1].item(), 1.0)

    def test_unsupported_label_vector_mode_raises(self):
        ds = self._make_dataset()
        ds.label_vector_mode = "invalid_mode"
        with self.assertRaises(NotImplementedError):
            ds._get_label_vector(["Speech"])


class TestWaveformMode(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _make_dataset(self, n_sources=3, include_oracle=True, return_source=True):
        paths = make_waveform_dataset(self.root, include_oracle=include_oracle)
        config = {
            "mode": "waveform",
            "soundscape_dir": paths["soundscape_dir"],
            "oracle_target_dir": paths["oracle_target_dir"],
            "sr": SR,
        }
        return DatasetS3(
            config=config,
            n_sources=n_sources,
            label_set="dcase2026t4",
            return_source=return_source,
            label_vector_mode="stack",
            silence_label_mode="onehot",
        )

    def test_waveform_init_and_len(self):
        ds = self._make_dataset()
        self.assertEqual(len(ds), 1)

    def test_waveform_getitem_shapes(self):
        ds = self._make_dataset(n_sources=3)
        item = ds[0]

        self.assertEqual(item["mixture"].shape, (4, N_SAMPLES))
        self.assertEqual(item["soundscape"], "scene01")
        self.assertEqual(len(item["label"]), 3)
        self.assertEqual(item["label"][-1], "silence")
        self.assertEqual(item["dry_sources"].shape, (3, 1, N_SAMPLES))
        self.assertEqual(item["label_vector"].shape, (3, len(LABELS["dcase2026t4"]) + 1))

    def test_waveform_without_sources(self):
        ds = self._make_dataset(return_source=False)
        item = ds[0]
        self.assertIn("mixture", item)
        self.assertIn("label", item)
        self.assertNotIn("dry_sources", item)

    def test_get_data_waveform_missing_dir_raises(self):
        ds = self._make_dataset()
        with self.assertRaises(FileNotFoundError):
            ds._get_data_waveform(ds.data, "ref", str(self.root / "missing"))

    def test_get_data_waveform_parses_optional_index(self):
        oracle_dir = self.root / "oracle_target_alt"
        oracle_dir.mkdir()
        write_mono_wav(oracle_dir / "scene01_Speech.wav")
        data = [{"soundscape": "scene01"}]
        ds = self._make_dataset()
        ds._get_data_waveform(data, "ref", str(oracle_dir))
        self.assertEqual(data[0]["ref_label"], ["Speech"])


class TestGenerateOutput(unittest.TestCase):
    def _make_dataset(self, n_sources=3, return_source=True, shuffle_label=False):
        config = {
            "mode": "generate",
            "dupse_rate": 0.0,
            "dupse_min_angle": 60.0,
            "max_n_dupse": 1,
            "dupse_exclusion_folder_depth": 0,
            "spatial_sound_scene": {"sr": SR},
            "snr_range": [5, 20],
            "nevent_range": [1, 2],
            "dataset_length": 1,
            "shuffle_label": shuffle_label,
            "fg_return": {"dry": True, "dry_channel": 0, "metadata": True},
        }
        return DatasetS3(
            config=config,
            n_sources=n_sources,
            label_set="dcase2026t4",
            return_source=return_source,
            label_vector_mode="stack",
            silence_label_mode="onehot",
        )

    def test_generate_returns_joint_loss_fields(self):
        ds = self._make_dataset(n_sources=3)
        labels = ["Speech", "Clapping"]
        mock_s3 = MagicMock()
        mock_s3.synthesize.return_value = make_mock_synthesize_output(labels)

        item = ds._generate(mock_s3)

        self.assertEqual(item["labels"].tolist(), [0, 1, ds.silence_idx])
        self.assertEqual(item["doas"].shape, (3, 3))
        self.assertEqual(item["active"].tolist(), [True, True, False])
        self.assertEqual(item["label"], ["Speech", "Clapping", "silence"])
        self.assertEqual(item["waveforms"].shape, (3, N_SAMPLES))
        self.assertEqual(item["dry_sources"].shape, (3, 1, N_SAMPLES))
        self.assertEqual(item["mixture"].shape, (4, N_SAMPLES))

    def test_generate_pads_single_event(self):
        ds = self._make_dataset(n_sources=3)
        mock_s3 = MagicMock()
        mock_s3.synthesize.return_value = make_mock_synthesize_output(["Pour"])

        item = ds._generate(mock_s3)

        self.assertEqual(item["labels"].tolist(), [LABELS["dcase2026t4"].index("Pour"), ds.silence_idx, ds.silence_idx])
        self.assertFalse(item["active"][1].item())
        self.assertTrue(torch.all(item["doas"][1:] == 0))

    def test_generate_without_sources(self):
        ds = self._make_dataset(return_source=False)
        mock_s3 = MagicMock()
        mock_s3.synthesize.return_value = make_mock_synthesize_output(["Speech"])

        item = ds._generate(mock_s3)
        self.assertNotIn("waveforms", item)
        self.assertNotIn("dry_sources", item)

    def test_generate_includes_metadata_when_requested(self):
        ds = self._make_dataset(return_source=False)
        ds.return_meta = True
        output = make_mock_synthesize_output(["Speech"])
        mock_s3 = MagicMock()
        mock_s3.synthesize.return_value = output

        item = ds._generate(mock_s3)
        self.assertEqual(item["metadata"], output)


class TestMetadataMode(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    @patch("src.datamodules.new_dataset.SpAudSyn")
    def test_metadata_getitem_uses_from_metadata(self, mock_sp_aud_syn_cls):
        metadata_dir = self.root / "metadata"
        metadata_dir.mkdir()
        metadata_file = metadata_dir / "scene0001.json"
        metadata_file.write_text("{}", encoding="utf-8")

        metadata_list = self.root / "valid.json"
        metadata_list.write_text(
            '[{"metadata_path": "scene0001.json"}]',
            encoding="utf-8",
        )

        mock_s3 = MagicMock()
        mock_s3.synthesize.return_value = make_mock_synthesize_output(["Speech"])
        mock_sp_aud_syn_cls.from_metadata.return_value = mock_s3

        config = {
            "mode": "metadata",
            "metadata_list": str(metadata_list),
            "fg_return": {"dry": True},
            "sr": SR,
        }
        ds = DatasetS3(
            config=config,
            n_sources=3,
            label_set="dcase2026t4",
            return_source=True,
        )

        item = ds[0]
        mock_sp_aud_syn_cls.from_metadata.assert_called_once_with(str(metadata_file))
        self.assertEqual(item["soundscape"], "soundscape_0000")
        self.assertIn("labels", item)


class TestGetPosition(unittest.TestCase):
    def _make_dataset(self):
        config = {
            "mode": "generate",
            "dupse_rate": 0.5,
            "dupse_min_angle": 60.0,
            "max_n_dupse": 1,
            "dupse_exclusion_folder_depth": 0,
            "spatial_sound_scene": {"sr": SR},
            "snr_range": [5, 20],
            "nevent_range": [2, 2],
            "dataset_length": 1,
            "shuffle_label": False,
            "fg_return": {"dry": True},
        }
        return DatasetS3(
            config=config,
            n_sources=3,
            label_set="dcase2026t4",
            return_source=False,
        )

    def test_get_position_respects_min_angle(self):
        ds = self._make_dataset()
        ref_pos = np.array([[1.0, 0.0, 0.0]])
        all_pos = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
            ]
        )
        chosen = ds._get_position(ref_pos, all_pos)
        self.assertTrue(np.allclose(chosen, [0.0, 1.0, 0.0]) or np.allclose(chosen, [-1.0, 0.0, 0.0]))


if __name__ == "__main__":
    unittest.main()
