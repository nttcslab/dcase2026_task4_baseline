from . portable_m2d import PortableM2D
from timm.models.layers import trunc_normal_
import torch
import torch.nn.functional as F
import torch.nn as nn
import itertools
import math

import logging
logger = logging.getLogger(__name__)

class HeadAvgLinear(nn.Module):
    def __init__(self, m2d_feature_d, num_outputs, num_classes,
                 track_input_dim=1024,
                 track_hidden_dim=512):
        super().__init__()
        self.num_outputs = num_outputs
        self.track_input_dim = track_input_dim

        self.m2d_out_norm = nn.BatchNorm1d(m2d_feature_d, eps=1e-05, momentum=0.1, affine=False)

        # divide output into tracks
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(m2d_feature_d, track_input_dim*self.num_outputs),
            nn.LayerNorm(track_input_dim*self.num_outputs),
            nn.ReLU()
        )
        if track_hidden_dim is None: # MLP layers for each tracks?
            self.tracks = nn.ModuleList([
                nn.Linear(track_input_dim, num_classes)
                for _ in range(num_outputs)
            ])
        else:
            self.tracks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(track_input_dim, track_hidden_dim),
                    nn.LayerNorm(track_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(track_hidden_dim, num_classes)
                ) for _ in range(num_outputs)
            ])

    def forward(self, x): # [bs, seq, D] output of m2d.encode
        x = x.mean(1) # [bs, D]
        x = self.m2d_out_norm(x) # [bs, D]
        x = self.linear(x) # [bs, num_outputs*track_input_dim]
        x = x.view(x.shape[0], self.num_outputs, self.track_input_dim) # [bs, num_output, track_input_dim]
        outputs = []
        for i, track in enumerate(self.tracks):
            out = track(x[:, i, :])  # [bs, nclasses]
            outputs.append(out.unsqueeze(1))  # # [bs, 1, nclasses]
        x = torch.cat(outputs, dim=1)# [bs, num_output, nclasses]
        return x

class M2dAt(PortableM2D):
    def __init__(self,
                 weight_file,
                 num_classes,
                 num_outputs,
                 finetuning_layers='head', # head, backbone_out, 1_blocks, 2_blocks, ..., 15_blocks, all
                 head_layer='average_linear',
                 head_args= None,
                 ref_channel=None,
                 ):
        super().__init__(weight_file, num_classes=527, freeze_embed=False, flat_features=None)
        self.finetuning_layers = finetuning_layers
        self.num_classes = num_classes
        self.ref_channel = ref_channel
        self.num_outputs = num_outputs

        if head_layer == 'average_linear':
            if head_args is None: head_args = {'track_input_dim': 1024, 'track_hidden_dim': 512}
            logger.info(f'Use average_linear head: {head_args}')
            self.head = HeadAvgLinear(self.cfg.feature_d, num_outputs, num_classes, **head_args)
        elif head_layer == 'identity':
            logger.info(f'No head is used')
            self.head = nn.Identity()
        else:
            raise NotImplementedError(f"head_layer mode '{head_layer}' has not been implemented")

        modules = [self.backbone.cls_token, self.backbone.pos_embed, self.backbone.patch_embed, self.backbone.pos_drop, self.backbone.patch_drop, self.backbone.norm_pre] # 0-5
        for block in self.backbone.blocks: modules.append(block) # 6-17
        modules.extend([self.backbone.norm, self.backbone.fc_norm, self.backbone.head_drop]) # 18-20
        modules.extend([self.head_norm, self.head]) # 21-22
        self.md = modules

        finetuning_modules_idx = {
            'head': 21, # modules ~20 is frozen
            'backbone_out': 18, # modules ~18 is frozen
            'all': 0,
        }
        for i in range(1, len(self.backbone.blocks) + 1):
            finetuning_modules_idx[f'{i}_blocks'] = 17 - i + 1
        self.finetuning_modules_idx = finetuning_modules_idx
        if self.finetuning_layers in finetuning_modules_idx.keys():
            logger.info(f'finetuning: {self.finetuning_layers}')
            modules_idx = finetuning_modules_idx[self.finetuning_layers]
            for i, module in enumerate(modules):
                self._set_requires_grad(module, i >= modules_idx) # from modules_idx, set requires grad == True
        else:
            raise NotImplementedError(f"finetuning_layers mode '{self.finetuning_layers}' has not been implemented")
    
    
    def _set_requires_grad(self, model, requiregrad):
        if isinstance(model, torch.nn.parameter.Parameter): model.requires_grad = requiregrad
        else:
            for param in model.parameters(): param.requires_grad = requiregrad

    
    # copy from PortableM2D, change input output to dict
    # def forward(self, batch_audio, average_per_time_frame=False):
    def forward(self, input_dict):
        batch_audio = input_dict['waveform'] # [bs, wlen] or [bs, nch, wlen]
        if batch_audio.dim() == 3:
            assert self.ref_channel is not None
            batch_audio = batch_audio[:, self.ref_channel, :] # [bs, wlen]
        x = self.encode(batch_audio, average_per_time_frame=False) # [bs, time, freq]
        # x = x.mean(1)  # B, D
        # x = self.head_norm(x.unsqueeze(-1)).squeeze(-1) # BatchNorm1d(768, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
        x = self.head(x)
        
        output_dict = {'probabilities': x} # [bs, n_output, n_classes]
        return output_dict

    def predict(self, input_dict):
        output_dict = self.forward(input_dict) # [bs, num_outputs, num_classes]
        probs = torch.softmax(output_dict['probabilities'], dim=-1)
        values, indices = torch.max(probs, dim=-1)
        one_hot = torch.nn.functional.one_hot(indices, num_classes=self.num_classes).to(torch.float32)  
        
        return {'label_vector': one_hot, # [bs, num_outputs, num_classes]
                'probabilities': values} # [bs, num_outputs] one probability for one one_hot vector (one predicted label)








