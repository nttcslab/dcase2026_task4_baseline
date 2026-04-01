import torch
import importlib

from src.utils import LABELS, initialize_config

class S5(torch.nn.Module):
    def __init__(
        self,
        tagger_config,
        separator_config,
        label_set,
        tagger_ckpt=None,
        separator_ckpt=None,
    ):
        super().__init__()

        tagger = initialize_config(tagger_config) # checkpoint loaded
        separator = initialize_config(separator_config)

        if separator_ckpt is not None:
            self._load_ckpt(separator_ckpt, separator)
        if tagger_ckpt is not None:
            self._load_ckpt(tagger_ckpt, tagger)
        
        separator.eval();
        tagger.eval();

        self.tagger = tagger
        self.separator = separator

        self.label_set = label_set
        self.labels = LABELS[self.label_set].copy()
        self.onehots = torch.eye(len(self.labels), requires_grad=False).to(torch.float32)
        self.label_onehots = {label: self.onehots[idx] for idx, label in enumerate(self.labels)}
        # append silence, silence is all-zero vector
        self.labels.append('silence')
        self.label_onehots['silence'] = torch.zeros(self.onehots.size(1), requires_grad=False, dtype=torch.float32)
    
    def _load_ckpt(self, path, model):
        model_ckpt = torch.load(path, weights_only=False, map_location='cpu')['state_dict']
        if set(model.state_dict().keys()) != set(model_ckpt.keys()): # remove prefix, incase the ckpt is of lightning module
            one_model_key = next(iter(model.state_dict().keys()))
            ckpt_corresponding_key = [k for k in model_ckpt.keys() if k.endswith(one_model_key)]
            prefix = ckpt_corresponding_key[0][:-len(one_model_key)]
            model_ckpt = {k[len(prefix):]: v for k, v in model_ckpt.items() if k.startswith(prefix)}# remove prefix
        model.load_state_dict(model_ckpt)  
    
    def _get_label(self, batch_label_vector): # [bs, nclass]
        labels = []
        for label_vectors in batch_label_vector: # nout, onehot
            labels.append([self.labels[i] for i in torch.argmax(label_vectors, dim = 1)])
        return labels # [[], [], ...]
    def predict_label(self, batch_mixture): # TODO: change to kwards
        output = self.tagger.predict({'waveform': batch_mixture}) # output['label_vector'], bs, nout, onehot
        labels = self._get_label(output['label_vector'])
        return {'label': labels, # [[lb1, lb2, ...], [lb1, lb2, lb3..], ...]
                'probabilities': output['probabilities'], # [bs, nout]
                'label_vector': output['label_vector'][..., :-1] #, [bs, nout, onehot] silence becomes all zeros
                }

    def _get_label_vector(self, batch_labels):
        return torch.stack([torch.stack([self.label_onehots[label] for label in labels]).flatten() for labels in batch_labels])
    def separate(self, batch_mixture, batch_label):
        # mixture [bs, 4, wlen]
        # labels [[lb1, lb2,...], ...   ] or [bs, nevent*onehotlen]

        if isinstance(batch_label, list): # support text labels
            batch_label = self._get_label_vector(batch_label).to(batch_mixture.device)

        separator_input = {
            'mixture': batch_mixture, 
            'label_vector': batch_label,
        }
        separator_output = self.separator(separator_input)
        return {'waveform': separator_output['waveform']} # {'waveform': []}
    
    def predict_label_separate(self, mixture):
        # mixtures[bs, 4, wlen]
         # [bs, wlen]
        with torch.no_grad(): # in-place indexing is not good for training, modify it if using this Class for tranining
            label_out = self.predict_label(mixture)
            batch_label = label_out['label']
            batch_probs = label_out['probabilities'] # [bs, nout]
            batch_label_vector = label_out['label_vector']#, [bs, nout, onehot]
    
            for ib in range(len(batch_label)):
                sorted_indices = sorted(range(len(batch_label[ib])),
                                        key=lambda i: batch_label[ib][i] == 'silence') # move silence to the end
                batch_label[ib] = [batch_label[ib][i] for i in sorted_indices]
                batch_label_vector[ib] = batch_label_vector[ib][sorted_indices]
                batch_probs[ib] = batch_probs[ib][sorted_indices]
    
            batch_label_vector = batch_label_vector.flatten(start_dim = 1) # [bs, nevent*onehotlen]
            predict_waveforms = self.separate(mixture, batch_label_vector)
            reobj = {
                'label': batch_label,
                'probabilities': batch_probs,
                'waveform': predict_waveforms['waveform']
            }
            return reobj











