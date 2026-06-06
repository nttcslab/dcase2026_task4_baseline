import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import permutations

class FinalJointPITLoss(nn.Module):
    def __init__(self, n_classes=18, weights=None):
        super().__init__()
        # 원래 클래스 0~17에 더해, 빈 슬롯을 위한 '침묵(Silence)' 클래스를 18번으로 명시
        self.silence_class_idx = n_classes 
        
        # Loss 스케일 보정을 위한 가중치 (매우 중요)
        # SNR은 절대값이 크므로 낮추고, DoA MSE는 값이 매우 작으므로 크게 뻥튀기합니다.
        self.weights = weights or {
            'snr': 0.1,      # 보통 -15 ~ 15 단위
            'energy': 0.05,  # 빈 슬롯의 꼼수(Trivial zero) 방지를 위해 아주 작게 설정
            'ce': 1.0,       # 보통 0 ~ 5 단위
            'doa': 50.0      # 보통 0.01 ~ 1.0 단위 (강하게 학습시켜 K를 맞추게 유도)
        }

    def compute_snr(self, pred_wav, target_wav, eps=1e-8):
        """ 순수 SNR 연산 모듈 """
        noise_energy = torch.sum((pred_wav - target_wav) ** 2, dim=-1) + eps
        target_energy = torch.sum(target_wav ** 2, dim=-1) + eps
        return 10 * torch.log10(target_energy / noise_energy)

    def forward(self, preds, targets):
        """
        preds: {
            'waveforms': [B, K, T], 
            'class_logits': [B, K, 19],  # 헤드 출력 차원이 19여야 함 (18 + 1)
            'doa_pred': [B, K, 3]
        }
        targets: {
            'waveforms': [B, K, T], 
            'labels': [B, K],            # 빈 슬롯은 18(침묵)로 전처리되어 들어와야 함
            'doas': [B, K, 3],           # 빈 슬롯은 [0,0,0]이어야 함
            'active': [B, K] bool        # 실제 소리 존재 여부
        }
        """
        device = preds['waveforms'].device
        B, K, T = preds['waveforms'].shape
        
        # 통합 Cost Matrix 초기화 [B, Target_K, Pred_K]
        cost_mtx = torch.zeros((B, K, K), device=device)

        # 1. 조합별 Cost Matrix 생성
        for t_idx in range(K):
            t_active = targets['active'][:, t_idx] # [B]
            
            for p_idx in range(K):
                # -- (1) 파형 Cost (SNR or Energy) --
                pred_w = preds['waveforms'][:, p_idx, :]
                targ_w = targets['waveforms'][:, t_idx, :]
                
                snr_val = self.compute_snr(pred_w, targ_w)
                pred_energy = torch.sum(pred_w ** 2, dim=-1)
                
                # Active면 -SNR(최소화), Inactive면 예측 파형의 에너지 패널티
                c_snr = torch.where(t_active, -snr_val * self.weights['snr'], 
                                              pred_energy * self.weights['energy'])
                
                # -- (2) CE Cost --
                # t_active가 False이면 targets['labels']가 18(silence)을 향하도록 학습됨
                c_ce = F.cross_entropy(preds['class_logits'][:, p_idx, :], 
                                       targets['labels'][:, t_idx], 
                                       reduction='none') * self.weights['ce']
                
                # -- (3) DoA Cost --
                # 타겟이 [0,0,0]이면 예측도 자연스럽게 벡터 크기가 0이 되도록 유도됨
                c_doa = F.mse_loss(preds['doa_pred'][:, p_idx, :], 
                                   targets['doas'][:, t_idx, :], 
                                   reduction='none').mean(dim=-1) * self.weights['doa']
                
                # 가중합산
                cost_mtx[:, t_idx, p_idx] = c_snr + c_ce + c_doa

        # 2. 최적 순서(Permutation) 탐색
        perms = torch.tensor(list(permutations(range(K))), device=device) # [6, K]
        P = perms.shape[0]
        
        perm_costs = torch.zeros((B, P), device=device)
        for i, p in enumerate(perms):
            # 해당 순서쌍의 Cost 합산
            cost_sum = torch.stack([cost_mtx[:, t, p[t]] for t in range(K)], dim=1).sum(dim=1)
            perm_costs[:, i] = cost_sum

        # 가장 Cost가 낮은 순서의 Index 추출
        best_idx = torch.argmin(perm_costs, dim=1) # [B]
        best_perms = perms[best_idx] # [B, K]

        # 3. 최적 순서에 따른 최종 Loss 역전파(Backprop) 계산
        final_loss = 0.0
        loss_components = {'snr': 0.0, 'ce': 0.0, 'doa': 0.0}
        
        for b in range(B):
            p = best_perms[b]
            for t_idx in range(K):
                p_idx = p[t_idx]
                t_act = targets['active'][b, t_idx]
                
                pred_w = preds['waveforms'][b:b+1, p_idx, :]
                targ_w = targets['waveforms'][b:b+1, t_idx, :]
                
                # SNR / Energy Loss
                if t_act:
                    l_snr = -self.compute_snr(pred_w, targ_w)[0] * self.weights['snr']
                else:
                    l_snr = torch.sum(pred_w ** 2) * self.weights['energy']
                
                # CE Loss
                l_ce = F.cross_entropy(preds['class_logits'][b:b+1, p_idx, :], 
                                       targets['labels'][b:b+1, t_idx]) * self.weights['ce']
                
                # DoA Loss
                l_doa = F.mse_loss(preds['doa_pred'][b, p_idx, :], 
                                   targets['doas'][b, t_idx, :]) * self.weights['doa']
                
                # 배치 내 슬롯 단위로 합산
                final_loss += (l_snr + l_ce + l_doa)
                
                # 로깅을 위한 분리 저장
                loss_components['snr'] += l_snr.item()
                loss_components['ce'] += l_ce.item()
                loss_components['doa'] += l_doa.item()

        # 전체(B * K) 평균으로 보정
        total_items = B * K
        final_loss = final_loss / total_items
        
        return {
            'loss': final_loss, # Optimizer가 step()을 밟을 최종 스칼라 텐서
            'loss_snr_scaled': loss_components['snr'] / total_items,
            'loss_ce_scaled': loss_components['ce'] / total_items,
            'loss_doa_scaled': loss_components['doa'] / total_items
        }