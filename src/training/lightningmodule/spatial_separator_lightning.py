"""
SpatialSeparatorLightning
=========================
BaseLightningModule를 상속하여 SpatialSeparatorModel (new_models.py) 학습.

배치 계약 (DatasetS3 generate 모드 + fg_return에 metadata 포함):
    batch['mixture']      : [B, 4, T]
    batch['dry_sources']  : [B, K, 1, T]  ← dataset이 1-ch dry source를 줌
    batch['label_vector'] : [B, K, 18]    (silence_label_mode='zeros', label_vector_mode='stack')
    batch['metadata']     : list of dicts, metadata['fg_events'][k]['event_position'] = [[x,y,z]]
                            ※ return_meta=True 일 때만 존재

DoA ground-truth는 fg_events[k]['event_position'][0] 에서 꺼냄.
silence 슬롯(label all-zero)은 DoA target = [0, 0, 0].
"""
import torch
from .base_lightningmodule import BaseLightningModule


def _extract_doa_from_metadata(metadata_list, K: int, device) -> torch.Tensor:
    """
    metadata_list : list of B 개 dict (SpAudSyn synthesize 출력)
                    metadata['fg_events'][k]['metadata']['event_position'] = [[x,y,z]]
    Returns: doas [B, K, 3]
    """
    B = len(metadata_list)
    doas = torch.zeros(B, K, 3, device=device)
    for b, meta in enumerate(metadata_list):
        events = meta.get('fg_events', [])
        for k, ev in enumerate(events):
            if k >= K:
                break
            pos = ev.get('event_position', [[0, 0, 0]])
            xyz = pos[0] if isinstance(pos[0], (list, tuple)) else pos
            doas[b, k, 0] = xyz[0]
            doas[b, k, 1] = xyz[1]
            doas[b, k, 2] = xyz[2]
    return doas


class SpatialSeparatorLightning(BaseLightningModule):
    """
    YAML lightning_module 섹션:
        module: src.training.lightningmodule.spatial_separator_lightning
        main: SpatialSeparatorLightning
        args:
            model:
                module: src.models.new_models
                main: SpatialSeparatorModel
                args: {}
            loss:
                module: src.training.loss.final_joint_pit_loss
                main: get_loss_func
                args:
                    w_wav: 1.0
                    w_doa: 0.5
                    w_cls: 0.5
            optimizer:
                module: torch.optim
                main: AdamW
                args:
                    params: null
                    lr: 0.0001
                    weight_decay: 0.01
            is_validation: false
    """

    # ──────────────────────────────────────────────────────────
    # 내부 헬퍼
    # ──────────────────────────────────────────────────────────

    def _build_targets(self, batch: dict) -> dict:
        """배치에서 target dict 구성."""
        device = batch['mixture'].device
        B = batch['mixture'].shape[0]
        K = batch['dry_sources'].shape[1]   # K = n_sources

        # ── waveform target: [B, K, T] (1-ch dry source squeeze)
        tgt_wav = batch['dry_sources'].squeeze(2)      # [B, K, 1, T] → [B, K, T]

        # ── label_vector: [B, K, 18]
        tgt_lv = batch['label_vector']                 # [B, K, 18]  (stack 모드)
        if tgt_lv.dim() == 2:
            # concat 모드 fallback: [B, K*18] → [B, K, 18]
            n_cls = tgt_lv.shape[1] // K
            tgt_lv = tgt_lv.view(B, K, n_cls)

        # ── DoA target: [B, K, 3]
        if 'metadata' in batch:
            tgt_doa = _extract_doa_from_metadata(batch['metadata'], K, device)
        else:
            # metadata 없으면 zero (smoke test 대응)
            tgt_doa = torch.zeros(B, K, 3, device=device)

        return {
            'waveforms':    tgt_wav,
            'doas':         tgt_doa,
            'label_vector': tgt_lv,
        }

    # ──────────────────────────────────────────────────────────
    # training_step_processing (BaseLightningModule 추상 메서드)
    # ──────────────────────────────────────────────────────────

    def training_step_processing(self, batch: dict, batch_idx: int):
        batchsize = batch['mixture'].shape[0]

        # forward
        output = self.model(batch['mixture'])   # SpatialSeparatorModel.forward()

        # target
        target = self._build_targets(batch)

        # loss
        loss_dict = self.loss_func(output, target)

        return batchsize, loss_dict

    # ──────────────────────────────────────────────────────────
    # validation_step_processing (is_validation=True 일 때 호출)
    # ──────────────────────────────────────────────────────────

    def validation_step_processing(self, batch: dict, batch_idx: int):
        batchsize = batch['mixture'].shape[0]

        output = self.model(batch['mixture'])
        target = self._build_targets(batch)

        loss_dict = self.loss_func(output, target)
        loss_dict = {k: v.item() for k, v in loss_dict.items()}

        return batchsize, loss_dict
