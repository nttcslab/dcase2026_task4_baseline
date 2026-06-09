"""
FinalJointPITLoss
=================
배치 계약:
  output  : waveforms [B,K,T], doa_pred [B,K,3], class_logits [B,K,19], active [B,K]
  target  : waveforms [B,K,T], doas [B,K,3],     label_vector [B,K,18], active [B,K]

모든 출력(파형·DoA·클래스)을 하나의 순열로 정렬하는 Joint-PIT.
CAPI-SDRi 평가 기준에서 permutation이 섞이면 점수가 망가지므로
단일 순열로 세 Loss를 함께 묶는 것이 필수.

순열 탐색 기준: α*SI-SDR_score − β*DoA_cos_dist
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import permutations


# ─────────────────────────────────────────────
# 기본 metric 함수들
# ─────────────────────────────────────────────

def si_sdr_batch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    SI-SDR (Scale-Invariant SDR), higher = better.
    pred, target : [B, T]
    returns      : [B]
    """
    eps = 1e-8
    target = target - target.mean(dim=-1, keepdim=True)
    pred   = pred   - pred.mean(dim=-1, keepdim=True)

    dot   = (pred * target).sum(dim=-1, keepdim=True)
    t_pow = (target * target).sum(dim=-1, keepdim=True) + eps
    s_target = dot / t_pow * target                   # [B, T]
    e_noise  = pred - s_target

    si_sdr = 10 * torch.log10(
        (s_target ** 2).sum(-1) / ((e_noise ** 2).sum(-1) + eps) + eps
    )                                                  # [B]
    return si_sdr


def doa_cos_sim_batch(pred_xyz: torch.Tensor, target_xyz: torch.Tensor) -> torch.Tensor:
    """
    DoA 코사인 유사도, higher = better.
    pred_xyz, target_xyz : [B, 3]
    returns              : [B]
    """
    eps = 1e-8
    p = F.normalize(pred_xyz,   dim=-1, eps=eps)
    t = F.normalize(target_xyz, dim=-1, eps=eps)
    return (p * t).sum(dim=-1)   # [B], range [-1, 1]


# ─────────────────────────────────────────────
# Joint PIT 순열 탐색
# ─────────────────────────────────────────────

def find_best_perm(
    pred_wav:  torch.Tensor,    # [B, K, T]
    tgt_wav:   torch.Tensor,    # [B, K, T]
    pred_doa:  torch.Tensor,    # [B, K, 3]
    tgt_doa:   torch.Tensor,    # [B, K, 3]
    tgt_label: torch.Tensor,    # [B, K, 18]  (silence = all-zero)
    alpha: float = 1.0,
    beta:  float = 0.5,
) -> torch.Tensor:
    """
    Returns best_perm : [B, K]   (indices mapping pred → target)

    순열 탐색 기준 (maximise):
        score(π) = mean_k [ α * SI-SDR(pred_π(k), tgt_k)
                           + β * cos_sim(doa_pred_π(k), doa_tgt_k) ]
    단, silence 슬롯(label all-zero)끼리만 매칭 허용 (class-aware).
    """
    B, K, T  = pred_wav.shape
    device   = pred_wav.device
    is_sil   = (tgt_label == 0).all(dim=-1)  # [B, K] True = silence

    perms = list(permutations(range(K)))      # K! 가지 (K=3 → 6)
    n_perms = len(perms)
    perm_tensor = torch.tensor(perms, device=device)  # [P, K]

    # score matrix [B, K_target, K_pred]
    wav_score = torch.zeros(B, K, K, device=device)
    doa_score = torch.zeros(B, K, K, device=device)

    for i in range(K):
        for j in range(K):
            wav_score[:, i, j] = si_sdr_batch(pred_wav[:, j], tgt_wav[:, i])
            doa_score[:, i, j] = doa_cos_sim_batch(pred_doa[:, j], tgt_doa[:, i])

    # class-aware masking: silence target ↔ silence pred만 허용
    #   - silence tgt (i) 는 silence pred (j) 와만 매칭
    #   - non-silence tgt는 non-silence pred와만 매칭
    for i in range(K):
        for j in range(K):
            mismatch = is_sil[:, i] ^ is_sil[:, j]   # [B]
            if mismatch.any():
                wav_score[mismatch, i, j] = -1e9
                doa_score[mismatch, i, j] = -2.0      # cos sim 최솟값 아래

    joint_score = alpha * wav_score + beta * doa_score  # [B, K, K]

    # 모든 순열 점수 계산
    perm_scores = torch.zeros(B, n_perms, device=device)
    for p_idx, perm in enumerate(perms):
        perm_t = torch.tensor(perm, device=device)         # [K]
        # joint_score[b, i, perm[i]] → [B, K] → mean over K
        gathered = joint_score[:, torch.arange(K), perm_t]  # [B, K]
        perm_scores[:, p_idx] = gathered.mean(dim=1)

    best_idx  = perm_scores.argmax(dim=1)    # [B]
    best_perm = perm_tensor[best_idx]        # [B, K]
    return best_perm


# ─────────────────────────────────────────────
# 메인 Loss 클래스
# ─────────────────────────────────────────────

class FinalJointPITLoss(nn.Module):
    """
    Joint-PIT Loss (waveform + DoA + class).

    Args:
        w_wav  : SI-SDR loss 가중치
        w_doa  : DoA cosine loss 가중치
        w_cls  : classification CE loss 가중치
        alpha  : 순열 탐색 시 SI-SDR 기여 비율
        beta   : 순열 탐색 시 DoA cosine 기여 비율
        n_classes : silence 제외 클래스 수 (default 18, silence index = 18)
    """
    def __init__(
        self,
        w_wav:     float = 1.0,
        w_doa:     float = 0.5,
        w_cls:     float = 0.5,
        alpha:     float = 1.0,
        beta:      float = 0.5,
        n_classes: int   = 18,
    ):
        super().__init__()
        self.w_wav     = w_wav
        self.w_doa     = w_doa
        self.w_cls     = w_cls
        self.alpha     = alpha
        self.beta      = beta
        self.n_classes = n_classes
        self.silence_idx = n_classes   # class_logits shape = [B, K, n_classes+1]

    def forward(self, output: dict, target: dict) -> dict:
        """
        output keys : waveforms [B,K,T], doa_pred [B,K,3], class_logits [B,K,n_classes+1]
        target keys : waveforms [B,K,T], doas     [B,K,3], label_vector [B,K,18]
                      (optional) active [B,K] bool

        반환: {'loss': scalar, 'loss_wav': scalar, 'loss_doa': scalar, 'loss_cls': scalar}
        """
        pred_wav  = output['waveforms']          # [B, K, T]
        pred_doa  = output['doa_pred']           # [B, K, 3]
        pred_cls  = output['class_logits']       # [B, K, n_classes+1]

        tgt_wav   = target['waveforms']          # [B, K, T]
        tgt_doa   = target['doas']               # [B, K, 3]
        tgt_label = target['label_vector']       # [B, K, 18]

        B, K, T = pred_wav.shape
        device  = pred_wav.device

        # ── 1. 최적 순열 탐색 ──────────────────────────────────
        with torch.no_grad():
            best_perm = find_best_perm(
                pred_wav, tgt_wav, pred_doa, tgt_doa, tgt_label,
                alpha=self.alpha, beta=self.beta,
            )  # [B, K]

        # best_perm[b, i] = j 의미: 타깃 슬롯 i ← 예측 슬롯 j
        # 예측을 타깃 순서에 맞게 재정렬
        expand_wav = best_perm.unsqueeze(-1).expand(B, K, T)
        pred_wav_r = torch.gather(pred_wav, 1, expand_wav)          # [B, K, T]

        expand_doa = best_perm.unsqueeze(-1).expand(B, K, 3)
        pred_doa_r = torch.gather(pred_doa, 1, expand_doa)          # [B, K, 3]

        expand_cls = best_perm.unsqueeze(-1).expand(B, K, self.n_classes + 1)
        pred_cls_r = torch.gather(pred_cls, 1, expand_cls)          # [B, K, n_classes+1]

        # ── 2. Waveform loss (neg SI-SDR) ─────────────────────
        is_sil  = (tgt_label == 0).all(dim=-1)  # [B, K]
        active  = ~is_sil                        # [B, K]

        wav_losses = []
        for k in range(K):
            sdr = si_sdr_batch(pred_wav_r[:, k], tgt_wav[:, k])  # [B]
            # silence 슬롯은 0 loss (이미 올-zero target에 대해 학습 압력 없음)
            sdr = sdr * active[:, k].float()
            wav_losses.append(-sdr)
        loss_wav = torch.stack(wav_losses, dim=1).mean()           # scalar

        # ── 3. DoA loss (1 - cosine similarity) ───────────────
        doa_losses = []
        for k in range(K):
            cos = doa_cos_sim_batch(pred_doa_r[:, k], tgt_doa[:, k])  # [B]
            cos = cos * active[:, k].float()
            doa_losses.append(1.0 - cos)
        loss_doa = torch.stack(doa_losses, dim=1).mean()           # scalar

        # ── 4. Class loss (cross-entropy) ──────────────────────
        # silence 타깃 → silence_idx (= n_classes) 로 매핑
        tgt_class_idx = tgt_label.argmax(dim=-1)                   # [B, K] 0..17
        tgt_class_idx = tgt_class_idx * active.long() + \
                        self.silence_idx * (~active).long()        # silence → 18

        # CE: [B*K, n_classes+1] vs [B*K]
        loss_cls = F.cross_entropy(
            pred_cls_r.reshape(B * K, -1),
            tgt_class_idx.reshape(B * K),
        )

        # ── 5. 총 loss ─────────────────────────────────────────
        loss = self.w_wav * loss_wav + self.w_doa * loss_doa + self.w_cls * loss_cls

        return {
            'loss':     loss,
            'loss_wav': loss_wav.detach(),
            'loss_doa': loss_doa.detach(),
            'loss_cls': loss_cls.detach(),
        }


def get_loss_func(w_wav=1.0, w_doa=0.5, w_cls=0.5):
    """YAML initialize_config 호환 팩토리 함수."""
    return FinalJointPITLoss(w_wav=w_wav, w_doa=w_doa, w_cls=w_cls)
