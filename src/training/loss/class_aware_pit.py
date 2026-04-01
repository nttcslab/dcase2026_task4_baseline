import torch
import numpy as np
import torch
from torch import Tensor
from itertools import permutations
from typing import Callable, Literal
from torchmetrics.functional.audio import signal_noise_ratio as snr


def snr_loss_return_batch(preds, target):
    metric = -snr(preds, target)
    if metric.dim() == 1: return metric
    return  metric.flatten(start_dim=1).mean(dim=1)

def class_aware_permutation_invariant_training(
        waveform_pred: Tensor,                       # shape [B, S, ...]
        waveform_target: Tensor,                     # shape [B, S, ...]
        label: Tensor,                 # shape [B, S, L]
        metric_func: Callable[[Tensor, Tensor], Tensor],  # returns [B]
        eval_func: Literal["max", "min"] = "max",
    ) -> tuple[Tensor, Tensor]:

        B, S = waveform_pred.shape[0], waveform_pred.shape[1]
        device = waveform_pred.device

        if eval_func not in ["max", "min"]: raise ValueError(f"eval_func must be 'max' or 'min', got {eval_func}")
        eval_op = torch.max if eval_func == "max" else torch.min
        penalty_value = -1e9 if eval_func == "max" else 1e9

        # Prepare metric matrix [B, S_target, S_pred]
        metric_mtx = torch.full((B, S, S), penalty_value, device=device, dtype=torch.float64) # [B, S, S]
        is_silence = (label == 0).all(dim=2) # [B, S] # TODO: silence label assignment
        silence_mask = is_silence.unsqueeze(2) | is_silence.unsqueeze(1) # B, S, S

        # silence will not be permutated, and silence score will be 0
        metric_mtx.diagonal(dim1=1, dim2=2)[silence_mask.diagonal(dim1=1, dim2=2)] = 0

        same_label = (label.unsqueeze(2) == label.unsqueeze(1)).all(dim=3) # B, S, S
        valid_mask = same_label & ~silence_mask # [B, S, S] only valid with same-label permutation, except silence

        for i in range(S):
            for j in range(S):
                valid = valid_mask[:, i, j]  # [B]
                if valid.any():
                    m = metric_func(waveform_pred[:, j, :], waveform_target[:, i, :])  # [B]
                    metric_mtx[valid, i, j] = m[valid].to(dtype=metric_mtx.dtype)

        # Exhaustive search for best permutation
        perms = torch.tensor(list(permutations(range(S))), device=device)  # [P, S]
        P = perms.shape[0]

        # [B, S, P]
        perm_mtx = perms.T[None, :, :].expand(B, S, P)
        gathered = torch.gather(metric_mtx, 2, perm_mtx)  # [B, S, P]
        scores = gathered.mean(dim=1)  # [B, P]

        best_metric, best_idx = eval_op(scores, dim=1)  # [B]
        best_perm = perms[best_idx]  # [B, S]

        return best_metric, best_perm

def get_loss_func():
    def loss_func(output, target):
        loss_val_all_sources, _ = class_aware_permutation_invariant_training(
            waveform_pred = output['waveform'],
            waveform_target = target['waveform'],
            label = target['label_vector'],
            metric_func = snr_loss_return_batch,
            eval_func = 'min',
        )
        loss_val = loss_val_all_sources.mean()
        loss_dict = {
            'loss': loss_val, # main loss, for back propagation
        }
        return loss_dict
    return loss_func
def get_metric_func():
    def loss_func(output, target):
        loss_val_all_sources, _ = class_aware_permutation_invariant_training(
            waveform_pred = output['waveform'],
            waveform_target = target['waveform'],
            label = target['label_vector'],
            metric_func = snr_loss_return_batch,
            eval_func = 'min',
        )
        loss_dict = {
            'loss': loss_val_all_sources, # main loss, for back propagation
        }
        return loss_dict
    return loss_func

