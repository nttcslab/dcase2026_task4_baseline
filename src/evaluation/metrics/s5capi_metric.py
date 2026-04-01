from torchmetrics.functional import signal_noise_ratio as snr
import torch
import numpy as np
from itertools import combinations, permutations

class S5ClassAwareMetric():
    def __init__(self, metricfunc='sdr'):
        if metricfunc == 'sdr':
            self.metric_func = snr
            self.metric_name = 'CAPI-SDRi'
            self.min_max = 'max'
        else: raise ValueError(f"metricfunc of '{metricfunc}' is not implemented!!")
        self.metric_values = []
        self.reset()
    def update(self, batch_est_labels, batch_est_waveforms, batch_ref_labels, batch_ref_waveforms, batch_mixture):
        mvalues = self.compute_batch(batch_est_labels, batch_est_waveforms, batch_ref_labels, batch_ref_waveforms, batch_mixture)
        self.metric_values.extend(mvalues)
        return mvalues
        
    def compute(self, is_print=False):
        if not self.metric_values: return None
        non_None_metric_values = [v for v in self.metric_values if v is not None]
        reobj = {
            'mean': sum(non_None_metric_values)/len(non_None_metric_values)
        }
        if is_print:
            print('%s: %.3f'%(self.metric_name, reobj['mean']))
        return 
        
    def reset(self): self.metric_values = []
    
    def compute_batch(self, batch_est_labels, batch_est_waveforms, batch_ref_labels, batch_ref_waveforms, batch_mixture):
        return [self.compute_sample(est_lb, est_wf, ref_lb, ref_wf, mixture)
                for est_lb, est_wf, ref_lb, ref_wf, mixture in
                zip(batch_est_labels,  batch_est_waveforms, batch_ref_labels, batch_ref_waveforms, batch_mixture)]


    def _pi_metric(self,
                   est_wf, # nevent, wlen
                    ref_wf, # nevent, wlen
                    mixture, # 1, wlen
                  ):
        assert est_wf.shape[0] != 0 and ref_wf.shape[0] != 0
        TP = min(est_wf.shape[0], ref_wf.shape[0])
    
        # all possible permutation
        perms = []
        perm_est_wfs = []
        perm_ref_wfs = []
        for rp in combinations(range(ref_wf.shape[0]), TP):
            for ep in permutations(range(est_wf.shape[0]), TP):
                rp = list(rp)
                ep = list(ep)
                perms.append((rp, ep))
                perm_ref_wfs.append(ref_wf[rp, :])
                perm_est_wfs.append(est_wf[ep, :])
    
        perm_est_wfs_stack = torch.stack(perm_est_wfs, dim=0) # nperm, n_tp, wlen
        perm_ref_wfs_stack = torch.stack(perm_ref_wfs, dim=0) # nperm, n_tp, wlen
    
        mixture_repeat = mixture.view(1, 1, -1).expand(perm_est_wfs_stack.shape[0], perm_est_wfs_stack.shape[1], -1)
    
        # calculate metric
        metrics = self.metric_func(perm_est_wfs_stack, perm_ref_wfs_stack) # nperm, n_tp
        metrics_mixture = self.metric_func(mixture_repeat, perm_ref_wfs_stack) # nperm, n_tp
        metrics_i = metrics - metrics_mixture # metric improvement
    
        # find the best permutation
        metrics_mean = metrics.mean(dim=tuple(range(1, metrics.dim()))) # n_perm
        if self.min_max == 'max':   best_i = torch.argmax(metrics_mean).item()
        elif self.min_max == 'min': best_i = torch.argmin(metrics_mean).item()
        else: raise NotImplementedError(f"min_max '{self.min_max}' has not been implemented.")
    
        # extract the best permutation results
        best_metric = metrics[best_i] # n_tp
        best_metric_i = metrics_i[best_i] # n_tp
        best_ref_perm, best_est_perm = perms[best_i]
    
        # append TP or FP penalties of any
        if est_wf.shape[0] != ref_wf.shape[0]:
            fnfp = abs(est_wf.shape[0] - ref_wf.shape[0])
            best_metric = torch.cat((best_metric, torch.zeros(fnfp)))
            best_metric_i = torch.cat((best_metric_i, torch.zeros(fnfp)))
    
        return {
            'metric': best_metric,
            'metric_i': best_metric_i,
            'est_perm': best_est_perm, # local indices
            'ref_perm': best_ref_perm,
        }


    def compute_sample(self,
                  est_lb, # list [lb1, lb2,...]
                  est_wf, # [nevent, wlen]
                  ref_lb, # list [lb1, lb2, ...]
                  ref_wf, # [nevent, wlen]
                  mixture, # [wlen, ]
                  ):
    
        all_labels = (set(est_lb) | set(ref_lb)) - {'silence'}
        if not all_labels: return None # true silence prediction
    
        # collect waveform of the same class
        est_lists = {lb: [] for lb in all_labels}
        ref_lists = {lb: [] for lb in all_labels}
    
        for i, (lb, wf) in enumerate(zip(est_lb, est_wf)):
            if lb != 'silence':
                est_lists[lb].append(wf)
    
        for i, (lb, wf) in enumerate(zip(ref_lb, ref_wf)):
            if lb != 'silence':
                ref_lists[lb].append(wf)
    
        est_dict = {
            lb: torch.stack(est_lists[lb], dim=0) if est_lists[lb] else torch.empty(
                (0, *est_wf.shape[1:]), dtype=est_wf.dtype, device=est_wf.device
            )
            for lb in all_labels
        }
        ref_dict = {
            lb: torch.stack(ref_lists[lb], dim=0) if ref_lists[lb] else torch.empty(
                (0, *ref_wf.shape[1:]), dtype=ref_wf.dtype, device=ref_wf.device
            )
            for lb in all_labels
        }
    
        metric_i = []
        for lb in all_labels:
            est_wf_1c = est_dict[lb]
            ref_wf_1c = ref_dict[lb]
            assert est_wf_1c.shape[0] != 0 or ref_wf_1c.shape[0] != 0
            if est_wf_1c.shape[0] == 0: # all False Negative
                metric_i.append(torch.zeros(ref_wf_1c.shape[0]))
            elif ref_wf_1c.shape[0] == 0: # all False Positive
                metric_i.append(torch.zeros(est_wf_1c.shape[0]))
            else:
                output = self._pi_metric(est_wf = est_wf_1c,
                                   ref_wf = ref_wf_1c,
                                   mixture = mixture)
                metric_i.append(output['metric_i'])
        metric_i = torch.cat(metric_i)

        return  metric_i.mean().item()
