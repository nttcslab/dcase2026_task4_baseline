import torch
import torch.nn.functional as F
import torch.nn as nn
import itertools
from .m2dat import M2dAt, HeadAvgLinear

import logging
logger = logging.getLogger(__name__)

class M2dAtSpatial(nn.Module):
    def __init__(self,
                 weight_file,
                 num_classes,
                 num_outputs,
                 nchan,
                 finetuning_layers='head', # head, backbone_out, 1_blocks, 2_blocks, ..., 15_blocks, all
                 head_layer='average_linear',
                 head_args= None,
                 ):
        super().__init__()

        self.num_classes = num_classes
        self.num_outputs = num_outputs
        self.head_layer = head_layer

        self.m2d = M2dAt(weight_file=weight_file,
            num_classes=self.num_classes,
            num_outputs=self.num_outputs,
            finetuning_layers=finetuning_layers,
            head_layer='identity', # no head
            ref_channel=None,    # each branch process one channel
        )
        m2d_feature_d = self.m2d.cfg.feature_d

        if head_layer == 'catchan_average_linear':
            if head_args is None: head_args = {'track_input_dim': 1024,
                                               'track_hidden_dim': 512
                                               }
            logger.info(f'Use average_linear head: {head_args}')
            self.head = HeadAvgLinear(nchan*m2d_feature_d, num_outputs, num_classes, **head_args)
        else: raise NotImplementedError(f"head_layer mode '{head_layer}' has not been implemented")

    # copy from PortableM2D, change input output to dict
    # def forward(self, batch_audio, average_per_time_frame=False):
    def forward(self, input_dict):
        x = input_dict['waveform'] # [bs, nchan, wlen]
        bs, nchan, wlen = x.shape
        x = x.view(bs*nchan, wlen) # [bs*nchan, wlen]
        x = self.m2d.encode(x, average_per_time_frame=False) # [bs*nchan, seq, D]
        _, seq, D = x.shape
        x = x.view(bs, nchan, seq, D) # bs, nchan, seq, D
        if self.head_layer.startswith('catchan'):
            x = x.permute(0, 2, 1, 3).contiguous()  # bs, seq, nchan, D
            x = x.view(bs, seq, D * nchan) # bs, seq, D*nchan
            if hasattr(self, 'before_head'):
                x = self.before_head(x)
            x = self.head(x)
        elif self.head_layer == 'flatchan_transformer':
            x = self.head(x)

        output_dict = {'probabilities': x}
        return output_dict

    def predict(self, input_dict):
        output_dict = self.forward(input_dict) # [bs, num_outputs, num_classes]
        probs = torch.softmax(output_dict['probabilities'], dim=-1)
        values, indices = torch.max(probs, dim=-1)
        one_hot = torch.nn.functional.one_hot(indices, num_classes=self.num_classes).to(torch.float32)  
        
        return {'label_vector': one_hot, # [bs, num_outputs, num_classes]
                'probabilities': values} # [bs, num_outputs] one probability for one one_hot vector (one predicted label)


