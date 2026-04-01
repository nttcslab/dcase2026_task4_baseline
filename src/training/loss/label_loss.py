import torch.nn.functional as F
from torchmetrics.functional.audio import permutation_invariant_training

def celoss(output, target): # [bs, nclass]
    loss_val = F.cross_entropy(output, target, reduction='none')
    return loss_val # [bs]
def get_loss_func():
    def loss_func(output, target):
        target_prob = target['probabilities'] # [bs, n_out, nclass]
        output_prob = output['probabilities'] # [bs, n_out, nclass]
        loss_val_all_sources, perms = permutation_invariant_training(output_prob,
                                                          target_prob,
                                                          celoss,
                                                          mode='speaker-wise',
                                                          eval_func='min')
        loss_val = loss_val_all_sources.mean()
        loss_dict = {
            'loss': loss_val, # main loss, for back propagation
            # 'perm': perms,
        }
        return loss_dict
    return loss_func
