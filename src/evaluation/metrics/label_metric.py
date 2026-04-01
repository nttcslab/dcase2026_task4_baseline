from torchmetrics.functional import signal_noise_ratio as snr
import torch
from collections import Counter

class LabelMetric():
    def __init__(self):
        self.metric_values = []
        self.metric_name = 'label_metrics'
        self.reset()

    def update(self, batch_est_labels, batch_ref_labels, **args):
        mvalues = self.compute_batch(batch_est_labels, batch_ref_labels)
        self.metric_values.extend(mvalues)
        return mvalues

    def compute(self, is_print=False):
        metric_values = [m for m in self.metric_values]
        acc = [v['acc'] for v in metric_values]
        TP = sum([v['tp'] for v in metric_values])
        FP = sum([v['fp'] for v in metric_values])
        FN = sum([v['fn'] for v in metric_values])
        reobj = {
            'accuracy_mix': sum(acc)*100/len(acc),
            'accuracy_src': TP*100/(TP+FP+FN)
        }
        if is_print:
            print('Accuracy (mixture): %.3f'%(reobj['accuracy_mix']))
            print('Accuracy (source) : %.3f'%(reobj['accuracy_src']))
        return reobj

    def reset(self): self.metric_values = []

    def compute_batch(self, batch_est_labels, batch_ref_labels):
        return [self.compute_sample(est_lb, ref_lb)  for est_lb, ref_lb in
                                zip(batch_est_labels, batch_ref_labels)]

    def compute_sample(self, est_lb, ref_lb):
        """
        Args:
            est_lb (list of str): Labels for estimated sources, length n_est_sources.
            ref_lb (list of str): Labels for reference sources, length n_ref_sources.
        """
        est_label = list([r for r in est_lb if r != 'silence'])
        ref_label = list([r for r in ref_lb if r != 'silence'])

        est_count = Counter(est_label)
        ref_count = Counter(ref_label)
        common = est_count & ref_count

        acc = (est_count == ref_count)
        num_ref = len(ref_label)
        tp = sum(list(common.values()))
        fp = len(est_label) - tp
        fn = len(ref_label) - tp
        assert ((fn + fp) == 0) == acc # 2 ways to calculate accuracy
        return {
            'ref_label': ref_label,
            'acc': acc,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'nref': num_ref,
        }
