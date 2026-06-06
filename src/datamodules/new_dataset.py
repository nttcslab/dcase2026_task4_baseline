import os
import torch
import re
import numpy as np
import librosa
import random
import warnings
import json

from src.modules.spatial_audio_synthesizer.spatial_audio_synthesizer import SpAudSyn
from src.utils import LABELS

def collate_fn(list_data_dict):
    """
    배치(Batch)로 묶어주는 함수. 
    우리가 새롭게 추가한 labels, doas, active, waveforms 텐서들도 
    여기서 [B, K, ...] 형태로 자동으로 예쁘게 쌓입니다(stacked).
    """
    data = {k: [] for k in list_data_dict[0].keys()}
    for ddict in list_data_dict:
        for k in data:
            data[k].append(ddict[k])
    for k in data.keys():
        if type(data[k][0]) is torch.Tensor:
            data[k] = torch.stack(data[k], 0)
    return data

class DatasetS3(torch.utils.data.Dataset):
    def __init__(self,
                 config,
                 n_sources,
                 label_set, 
                 label_vector_mode='multihot',
                 silence_label_mode='zeros', 
                 return_meta=None, 
                 return_source=True, 
                ):
        super().__init__()
        self.return_meta = return_meta
        self.label_set = label_set
        self.config = config
        self.n_sources = n_sources
        self.return_source = return_source
        self.label_vector_mode = label_vector_mode
        self.silence_label_mode = silence_label_mode
        self.labels = LABELS[self.label_set].copy()

        # [추가됨] 침묵(Silence) 라벨의 정수 인덱스(보통 18)를 저장해 둠 (Final Loss용)
        self.silence_idx = len(self.labels)

        if self.config['mode']== 'waveform':
            self.soundscape_dir = self.config['soundscape_dir']
            self.oracle_target_dir = self.config.get('oracle_target_dir', None)
            self.estimate_target_dir = self.config.get('estimate_target_dir', None)
            self.data = [{'soundscape': f[:-4],
                      'mixture_path': os.path.join(self.soundscape_dir, f)
                      } for f in os.listdir(self.soundscape_dir) if f.endswith(".wav")]
            self.data = sorted(self.data, key=lambda x: x['soundscape'])
            if self.oracle_target_dir is not None:
                self._get_data_waveform(self.data, 'ref', self.oracle_target_dir)
            if self.estimate_target_dir is not None:
                self._get_data_waveform(self.data, 'est', self.estimate_target_dir)

            self.sr = self.config['sr']
            self.dataset_length = len(self.data)
            
        elif self.config['mode']== 'metadata':
            self.sr = self.config['sr']
            self.fg_return = self.config['fg_return']
            self.metadata_list = self.config['metadata_list']

            self.metadata_dir = os.path.dirname(self.metadata_list)
            with open(self.metadata_list) as f:
                self.data = json.load(f)
            self.dataset_length = len(self.data)
            self.shuffle_label = False

        elif self.config['mode']== 'generate':
            self.dupse_rate = self.config['dupse_rate'] 
            self.dupse_min_angle = np.deg2rad(self.config['dupse_min_angle']) 
            self.max_n_dupse = self.config['max_n_dupse'] 
            self.dupse_exclusion_folder_depth = self.config['dupse_exclusion_folder_depth'] 

            self.spatial_sound_scene = self.config['spatial_sound_scene']
            self.sr = self.config['spatial_sound_scene']['sr']
            self.snr_range = self.config['snr_range']
            self.nevent_range = self.config['nevent_range']
            self.dataset_length = self.config['dataset_length']
            self.shuffle_label = self.config['shuffle_label']
            self.fg_return = self.config['fg_return']

        print(self.labels, flush=True)
        print(len(self.labels), flush=True)
        
        if self.silence_label_mode == 'zeros':
            self.onehots = torch.eye(len(self.labels), requires_grad=False).to(torch.float32)
            self.label_onehots = {label: self.onehots[idx] for idx, label in enumerate(self.labels)}
            self.label_onehots['silence'] = torch.zeros(self.onehots.size(1), requires_grad=False,  dtype=torch.float32)
        elif self.silence_label_mode == 'onehot':
            self.onehots = torch.eye(len(self.labels) + 1, requires_grad=False).to(torch.float32)
            self.label_onehots = {label: self.onehots[idx] for idx, label in enumerate(self.labels)}
            self.label_onehots['silence'] = self.onehots[-1]

        self.collate_fn = collate_fn

    def get_onehot(self, label):
        return self.label_onehots[label]

    def __len__(self):
        return self.dataset_length

    def _get_label_vector(self, labels):
        label_vector_all = torch.stack([self.get_onehot(label) for label in labels]) 
        if self.label_vector_mode == 'multihot': label_vector_all = torch.any(label_vector_all.bool(), dim=0).float()
        elif self.label_vector_mode == 'concat': label_vector_all = label_vector_all.flatten() 
        elif self.label_vector_mode == 'stack': pass  
        else: raise NotImplementedError(f'label_vector_mode "{self.label_vector_mode}" not implemented')
        return label_vector_all

    def __getitem__(self, idx):
        if self.config['mode']== 'waveform':
            soundscene = self._get_item_waveform(idx)
            soundscene['soundscape'] = self.data[idx]['soundscape']
        elif self.config['mode']== 'generate':
            soundscene = self._get_item_generate(idx)
            soundscene['soundscape'] = 'soundscape_%08d'%(idx)
        elif self.config['mode']== 'metadata':
            soundscene = self._get_item_metadata(idx)
            soundscene['soundscape'] = 'soundscape_%04d'%(idx)

        return soundscene

    #=====================================================
    # Utilizations for waveform mode
    #=====================================================
    def _get_data_waveform(self, data, est_ref, source_dir):
        if not os.path.exists(source_dir):
            raise FileNotFoundError(f"Source directory '{source_dir}' does not exist.")
        all_wav = [f for f in os.listdir(source_dir) if f.endswith(".wav")]
        for d in data:
            sources = sorted([w for w in all_wav if w.startswith(d['soundscape'])])
            d[est_ref + '_label'] = []
            d[est_ref + '_source_paths'] = []
            for source in sources:
                pattern = rf"^{re.escape(d['soundscape'])}(?:_(\d+))?_(.+)\.wav$" 
                match = re.match(pattern, source)
                if match:
                    label = match.group(2)
                else:
                    print(f"Error: filename '{source}' does not match expected pattern")

                assert label in self.labels, f'"{source}" is not a valid filename'
                d[est_ref + '_label'].append(label)
                d[est_ref + '_source_paths'].append(os.path.join(source_dir, source))

    def _get_label_waveform(self, info, est_ref):
        labels = list(info[est_ref + '_label'])
        if len(labels) < self.n_sources:
            labels.extend(['silence'] * (self.n_sources - len(labels)))
        return labels

    def _get_source_waveform(self, info, est_ref, wlen):
        dry_sources = []
        for source_path in info[est_ref + '_source_paths']:
            dry_source, sr = librosa.load(source_path, sr = None)
            assert sr == self.sr
            dry_sources.append(dry_source)
        assert len(dry_sources) == len(info[est_ref + '_label'])

        if len(dry_sources) < self.n_sources:
            for _ in range(self.n_sources - len(dry_sources)):
                dry_sources.append(np.zeros(wlen, dtype=np.float32))
        return torch.from_numpy(np.stack(dry_sources))[:, None, :].to(torch.float32) 

    def _get_item_waveform(self, idx):
        # 파형/테스트 모드용. 여기서는 모델 평가만 하므로 DoA나 정답 마스크가 완벽하지 않아도 됨
        info = self.data[idx]
        mixture, sr = librosa.load(info['mixture_path'], sr = None, mono=False)
        assert sr == self.sr
        item = {
            'mixture': torch.from_numpy(mixture).to(torch.float32), 
        }

        if self.oracle_target_dir is not None:
            item['label'] = self._get_label_waveform(info, 'ref')
            item['label_vector'] = self._get_label_vector(item['label'])
            if self.return_source: item['dry_sources'] = self._get_source_waveform(info, 'ref', mixture.shape[-1]) 

        if self.estimate_target_dir is not None:
            item['est_label'] = self._get_label_waveform(info, 'est')
            item['est_label_vector'] = self._get_label_vector(item['est_label'])
            if self.return_source: item['est_dry_sources'] = self._get_source_waveform(info, 'est', mixture.shape[-1]) 
            
        if self.return_meta: item['metadata'] = info
        return item

    #=====================================================
    # Utilizations for generate mode
    #=====================================================
    def _get_position(self, ref_pos, all_pos):
        ref_pos = np.atleast_2d(ref_pos)
        ref_unit = ref_pos / np.linalg.norm(ref_pos, axis=1, keepdims=True)
        all_unit = all_pos / np.linalg.norm(all_pos, axis=1, keepdims=True)
        cos_theta = all_unit @ ref_unit.T

        angles = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        valid_mask = np.all(angles > self.dupse_min_angle, axis=1)
        valid_indices = np.where(valid_mask)[0]

        return all_pos[random.choice(valid_indices)]
        
    def _generate(self, s3):
        """
        [핵심 수정 영역] FinalJointPITLoss에서 요구하는 포맷으로 반환값을 재구성합니다.
        """
        output = s3.synthesize(fg_return=self.fg_return,
                               int_return=self.config.get('int_return', {}),
                               bg_return=self.config.get('bg_return', {}),)
        mixture = output['mixture'] # [4, wlen]

        if self.shuffle_label:
            random.shuffle(output['fg_events'])

        num_actual_events = len(output['fg_events'])
        npad = self.n_sources - num_actual_events
        
        # 1. 라벨 추출 (정수 인덱싱, 빈 슬롯은 silence_idx로 패딩)
        labels = [self.labels.index(fge['metadata']['label']) for fge in output['fg_events']]
        if npad > 0:
            labels.extend([self.silence_idx] * npad)
            
        # 2. DoA 좌표 추출 (빈 슬롯은 [0.0, 0.0, 0.0] 패딩)
        doas = [fge['event_position'][0] for fge in output['fg_events']]
        if npad > 0:
            doas.extend([[0.0, 0.0, 0.0] for _ in range(npad)])
            
        # 3. Active 마스크 생성 (실제 소리가 있는 슬롯만 True)
        active_mask = [True] * num_actual_events + [False] * npad

        # 베이스라인 기존 코드 호환성용 라벨명 리스트
        label_names = [fge['metadata']['label'] for fge in output['fg_events']]
        if npad > 0: label_names.extend(['silence'] * npad)

        return_obj = {
            'mixture': torch.from_numpy(mixture).to(torch.float32), 
            'labels': torch.tensor(labels, dtype=torch.long),       # [K_max]
            'doas': torch.tensor(doas, dtype=torch.float32),        # [K_max, 3]
            'active': torch.tensor(active_mask, dtype=torch.bool),  # [K_max]
            # 하위 호환성 유지
            'label': label_names, 
            'label_vector': self._get_label_vector(label_names),
        }

        # 4. 건조 음원(Dry source) 파형 추출 및 패딩
        if self.return_source:
            sources = [fge['waveform_dry'] for fge in output['fg_events']]
            if npad > 0:
                sources.extend([np.zeros((1, mixture.shape[-1]), dtype=mixture.dtype) for _ in range(npad)])
                
            dry_sources_tensor = torch.from_numpy(np.stack(sources)) # [K_max, 1, wlen]
            
            # Loss 계산을 위한 2D 텐서 추가 [K_max, wlen]
            return_obj['waveforms'] = dry_sources_tensor.squeeze(1)  
            # 기존 베이스라인 호환용 [K_max, 1, wlen]
            return_obj['dry_sources'] = dry_sources_tensor           

        if self.return_meta: 
            return_obj['metadata'] = output

        return return_obj

    def _get_item_generate(self, idx):
        s3 = SpAudSyn(**self.spatial_sound_scene)
        nevents = random.randint(self.nevent_range[0], self.nevent_range[1])
        if nevents < 2 or random.random() > self.dupse_rate:
            for i in range(nevents):
                s3.add_event(
                    label={'method': 'choose_wo_replacement'},
                    source_file={'method': 'choose'},
                    source_time={'method': 'choose'},
                    event_time={'method': 'choose'},
                    event_position={'method': 'choose', 'get_position_args': {'mode': 'point'}},
                    snr={'method': 'uniform', 'range': self.config['snr_range']},
                )
        else:
            n_dupse = random.randint(1, min(nevents-1, self.max_n_dupse))
            for _ in range(nevents-n_dupse):
                s3.add_event(
                    label={'method': 'choose_wo_replacement'},
                    source_file={'method': 'choose'},
                    source_time={'method': 'choose'},
                    event_time={'method': 'choose'},
                    event_position={'method': 'choose', 'get_position_args': {'mode': 'point'}},
                    snr={'method': 'uniform', 'range': self.config['snr_range']},
                )
            selected_labels = [e['label'] for e in s3.fg_events]
            for _ in range(n_dupse):
                ref_label = random.choice(selected_labels)
                ref_positions = [e['event_position'][0] for e in s3.fg_events if e['label'] == ref_label]
                selected_position = self._get_position(ref_pos = ref_positions,
                                                all_pos = s3.room.get_all_positions())
                selected_position = [selected_position.tolist()]
                s3.add_event(
                    label={'method': 'const', 'value': ref_label},
                    source_file={'method': 'choose_wo_replacement', 'exclusion_folder_depth': self.dupse_exclusion_folder_depth},
                    source_time={'method': 'choose'},
                    event_time={'method': 'choose'},
                    event_position={'method': 'const', 'value': selected_position},
                    snr={'method': 'uniform', 'range': self.config['snr_range']},
                )
        assert self.nevent_range[0] <= len(s3.fg_events) <=self.nevent_range[1]
        assert len(s3.fg_events) == nevents

        if 'interference_dir' in self.config['spatial_sound_scene']:
            ninteferences = random.randint(self.config['ninterference_range'][0], self.config['ninterference_range'][1])
            for _ in range(ninteferences):
                s3.add_interference(
                    label={'method': 'choose'},
                    source_file={'method': 'choose'},
                    source_time={'method': 'choose'},
                    event_time={'method': 'choose'},
                    event_position={'method': 'choose', 'get_position_args': {'mode': 'point'}},
                    snr={'method': 'uniform', 'range': self.config['inteference_snr_range']},
                )
        if self.spatial_sound_scene['background_dir']: 
            s3.add_background(source_file={'method': 'choose'},)
            
        return self._generate(s3)

    #=====================================================
    # Utilizations for metadata mode
    #=====================================================
    def _get_item_metadata(self, idx):
        metadata_path = os.path.join(self.metadata_dir, self.data[idx]['metadata_path'])
        s3 = SpAudSyn.from_metadata(metadata_path)
        return self._generate(s3)