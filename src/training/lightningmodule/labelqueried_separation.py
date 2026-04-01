from .base_lightningmodule import BaseLightningModule

class LabelQueriedSeparationLightning(BaseLightningModule):
    def training_step_processing(self, batch_data_dict, batch_idx):
        batchsize = batch_data_dict['mixture'].shape[0]

        input_dict = {
            'mixture': batch_data_dict['mixture'], # [bs, nch, wlen]
            'label_vector': batch_data_dict['label_vector'] # [bs, label_len]
        }
        output_dict = self.model(input_dict) # {'waveform': [bs, nch, wlen]}

        copylb = batch_data_dict['label_vector'].clone()
        if copylb.dim() == 3:
            assert copylb.shape[:2] == output_dict['waveform'].shape[:2]
        elif copylb.dim() == 2:
            assert copylb.shape[1] % output_dict['waveform'].shape[1] == 0
            copylb = copylb.view(
                copylb.shape[0],
                output_dict['waveform'].shape[1],
                copylb.shape[1] // output_dict['waveform'].shape[1])
        target_dict = {'waveform': batch_data_dict['dry_sources'],
                       'label_vector': copylb
                       }
        loss_dict = self.loss_func(output_dict, target_dict)

        return batchsize, loss_dict

    def validation_step_processing(self, batch_data_dict, batch_idx):
        batchsize = batch_data_dict['mixture'].shape[0]

        input_dict = {
            'mixture': batch_data_dict['mixture'], # [bs, nch, wlen]
            'label_vector': batch_data_dict['label_vector'] # [bs, label_len]
        }
        output_dict = self.model(input_dict) # {'waveform': [bs, nch, wlen]}

        copylb = batch_data_dict['label_vector'].clone()
        if copylb.dim() == 3:
            assert copylb.shape[:2] == output_dict['waveform'].shape[:2]
        elif copylb.dim() == 2:
            assert copylb.shape[1] % output_dict['waveform'].shape[1] == 0
            copylb = copylb.view(
                copylb.shape[0],
                output_dict['waveform'].shape[1],
                copylb.shape[1] // output_dict['waveform'].shape[1])
        target_dict = {'waveform': batch_data_dict['dry_sources'],
                       'label_vector': copylb
                       }
        loss_dict = self.loss_func(output_dict, target_dict)

        loss_dict = {k: v.item() for k,v in loss_dict.items()}
        if self.metric_func: # add metrics
            metric = self.metric_func(output_dict, target_dict)
            for k,v in metric.items():
                loss_dict[k] = v.mean().item() # torch tensor size [bs]

        return batchsize, loss_dict
