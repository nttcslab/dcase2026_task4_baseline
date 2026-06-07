import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from torch.utils.data import RandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datamodules.new_datamodule import DataModule, initialize_config
from tests.helpers import SR, make_waveform_dataset


class TestInitializeConfig(unittest.TestCase):
    def test_initialize_config_builds_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = make_waveform_dataset(root, include_oracle=False)
            module_cfg = {
                "module": "src.datamodules.new_dataset",
                "main": "DatasetS3",
                "args": {
                    "config": {
                        "mode": "waveform",
                        "soundscape_dir": paths["soundscape_dir"],
                        "sr": SR,
                    },
                    "n_sources": 3,
                    "label_set": "dcase2026t4",
                    "return_source": False,
                },
            }
            dataset = initialize_config(module_cfg)
            self.assertEqual(len(dataset), 1)
            self.assertTrue(hasattr(dataset, "collate_fn"))


class TestDataModule(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.paths = make_waveform_dataset(self.root, include_oracle=True)

        self.train_config = {
            "batch_size": 2,
            "num_workers": 0,
            "persistent_workers": False,
            "dataset": {
                "module": "src.datamodules.new_dataset",
                "main": "DatasetS3",
                "args": {
                    "config": {
                        "mode": "waveform",
                        "soundscape_dir": self.paths["soundscape_dir"],
                        "oracle_target_dir": self.paths["oracle_target_dir"],
                        "sr": SR,
                    },
                    "n_sources": 3,
                    "label_set": "dcase2026t4",
                    "return_source": True,
                    "label_vector_mode": "stack",
                    "silence_label_mode": "onehot",
                },
            },
        }

        self.val_config = {
            "batch_size": 1,
            "num_workers": 0,
            "persistent_workers": False,
            "dataset": {
                "module": "src.datamodules.new_dataset",
                "main": "DatasetS3",
                "args": {
                    "config": {
                        "mode": "waveform",
                        "soundscape_dir": self.paths["soundscape_dir"],
                        "oracle_target_dir": self.paths["oracle_target_dir"],
                        "sr": SR,
                    },
                    "n_sources": 3,
                    "label_set": "dcase2026t4",
                    "return_source": True,
                    "label_vector_mode": "stack",
                    "silence_label_mode": "onehot",
                },
            },
        }

    def tearDown(self):
        self.tmp.cleanup()

    def test_datamodule_without_validation(self):
        dm = DataModule(train_dataloader=self.train_config)
        self.assertIsNotNone(dm.train_dataset)
        self.assertIsNone(dm.val_dataset)

    def test_datamodule_with_validation(self):
        dm = DataModule(
            train_dataloader=self.train_config,
            val_dataloader=self.val_config,
        )
        self.assertIsNotNone(dm.val_dataset)

    def test_train_dataloader_yields_batch(self):
        dm = DataModule(train_dataloader=self.train_config)
        loader = dm.train_dataloader()
        batch = next(iter(loader))

        self.assertEqual(batch["mixture"].shape[0], 1)
        self.assertEqual(batch["mixture"].shape[1], 4)
        self.assertEqual(batch["dry_sources"].shape[0], 1)
        self.assertEqual(batch["dry_sources"].shape[1], 3)

    def test_val_dataloader_no_shuffle(self):
        dm = DataModule(
            train_dataloader=self.train_config,
            val_dataloader=self.val_config,
        )
        val_loader = dm.val_dataloader()
        # PyTorch 2.x DataLoader has no .shuffle; check sampler type instead
        self.assertNotIsInstance(val_loader.sampler, RandomSampler)

    def test_train_dataloader_uses_dataset_collate_fn(self):
        dm = DataModule(train_dataloader=self.train_config)
        loader = dm.train_dataloader()
        self.assertIs(loader.collate_fn, dm.train_dataset.collate_fn)


class TestDataModuleMetadataMode(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

        # mirrors data/dev_set/metadata/valid.json + valid/*.json layout
        metadata_list = self.root / "metadata" / "valid.json"
        metadata_list.parent.mkdir(parents=True)
        (metadata_list.parent / "valid").mkdir(exist_ok=True)
        (metadata_list.parent / "valid" / "scene0001.json").write_text("{}", encoding="utf-8")
        metadata_list.write_text('[{"metadata_path": "valid/scene0001.json"}]', encoding="utf-8")

        self.train_config = {
            "batch_size": 1,
            "num_workers": 0,
            "persistent_workers": False,
            "dataset": {
                "module": "src.datamodules.new_dataset",
                "main": "DatasetS3",
                "args": {
                    "config": {
                        "mode": "metadata",
                        "metadata_list": str(metadata_list),
                        "fg_return": {"dry": True},
                        "sr": SR,
                    },
                    "n_sources": 3,
                    "label_set": "dcase2026t4",
                    "return_source": True,
                },
            },
        }

    def tearDown(self):
        self.tmp.cleanup()

    @patch("src.datamodules.new_dataset.SpAudSyn")
    def test_metadata_train_dataloader(self, mock_sp_aud_syn_cls):
        from tests.helpers import make_mock_synthesize_output

        mock_s3 = MagicMock()
        mock_s3.synthesize.return_value = make_mock_synthesize_output(["Speech"])
        mock_sp_aud_syn_cls.from_metadata.return_value = mock_s3

        dm = DataModule(train_dataloader=self.train_config)
        batch = next(iter(dm.train_dataloader()))

        self.assertIn("labels", batch)
        self.assertEqual(batch["labels"].shape, (1, 3))
        self.assertEqual(batch["mixture"].shape[0], 1)


if __name__ == "__main__":
    unittest.main()
