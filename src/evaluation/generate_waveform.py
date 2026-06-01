import numpy as np
import yaml
import json
from tqdm import tqdm
import torch
import os, sys
import argparse
import soundfile as sf
import copy
import re

from torch.utils.data import DataLoader
from src.utils import LABELS, initialize_config, ignore_warnings
ignore_warnings()

all_labels = set(LABELS['dcase2026t4'])

from .evaluate import Evaluator

def verify_output_filename(soundscape_dir, output_dir):
    soundscapes = [f for f in os.listdir(soundscape_dir) if f.endswith(".wav")]
    soundscape_names = [f[:-4] for f in soundscapes]
    assert(len(set(soundscape_names)) == len(soundscapes)), 'duplicated soundscapes name'
    est_waveforms = [f for f in os.listdir(output_dir) if f.endswith(".wav")]
    
    for est_waveform_name in est_waveforms:
        snames = [s for s in soundscape_names if est_waveform_name.startswith(s)]
        assert len(snames) == 1
        pattern = rf"^{re.escape(snames[0])}(?:_(\d+))?_(.+)\.wav$" # soundscapes[:-4]_number_label.wav     or    
        
        match = re.match(pattern, est_waveform_name)
        if match: label = match.group(2)
        else: print(f"Error: filename '{est_waveform_name}' does not match expected pattern "
                  f"'{snames[0][:-4]}_<number>_<label>.wav'")
        assert label in all_labels, f'"{label}" is not a valid label'
    print('Output filenames are verified successfully!')

def verify_json(eval_results_json, output_dir):
    with open(eval_results_json) as f: eval_results = json.load(f)
    all_wav_out = [est['filename'] for e in eval_results['probabilities'] for est in e['estimate']]
    assert len(set(all_wav_out)) == len(all_wav_out), 'duplicated wav filename in eval_results.json'

    # get all wav files in output_dir
    wav_files = [f for f in os.listdir(output_dir) if f.endswith(".wav")]
    assert set(all_wav_out) == set(wav_files), 'wav files in eval_results.json do not match wav files in output_dir'
    print('eval_results.json is verified successfully!')

def verify_dev_set_test_json(test_json, soundscape_dir):
    with open(test_json) as f: test_results = json.load(f)
    all_soundscapes = [e['soundscape'] + '.wav' for e in test_results['details']]

    assert len(set(all_soundscapes)) == len(all_soundscapes), 'duplicated soundscape filename in test_json'

    # get all wav files in soundscape_dir
    wav_files = [f for f in os.listdir(soundscape_dir) if f.endswith(".wav")]
    assert set(all_soundscapes) == set(wav_files), 'wav files in test_json do not match wav files in soundscape_dir'
    print('dev_set_test_json is verified successfully!')


def main(args):
    sr = 32000
    # Create output dir
    outputdir = os.path.join(args.output_dir, args.output_name, 'eval_out')
    probs_json_dir = os.path.join(args.output_dir, args.output_name, 'eval_results.json')
    
    with open(args.config) as f: config = yaml.safe_load(f)
    print(json.dumps(config, indent=4), flush=True)

    if os.path.isdir(outputdir) and os.listdir(outputdir):
        raise ValueError(f'{outputdir} exists and is not empty!! Please remove it or choose another submission_number')

    os.makedirs(outputdir, exist_ok=True)

    # dev_set/test
    test_json_name = 'dev_set_test_results.json'
    test_json_path = os.path.join(args.output_dir, args.output_name)

    print('Evaluating dev/test set...')
    evalobj = Evaluator(
                config_path = args.config,
                waveform_output_dir = '',
                result_dir = test_json_path,
                result_filename = test_json_name,
                batch_size = args.batchsize
                )
    evalobj.evaluate()
    print('Verifying dev_set/test_json')
    verify_dev_set_test_json(os.path.join(test_json_path, test_json_name),
                             config['dataset']['args']['config']['soundscape_dir'])

    # eval_set
    print('Generating waveforms for eval set...')
    
    
    dataset_config = copy.deepcopy(config['eval_dataset'])
    dataset_config['oracle_target_dir'] = None
    dataset_config['estimate_target_dir'] = None
    dataset = initialize_config(dataset_config, reload=True)

    dataloader = DataLoader(dataset,
                            batch_size=args.batchsize,
                            shuffle=False,
                            collate_fn=dataset.collate_fn,
                            num_workers=args.batchsize)

    # Load models
    
    model = initialize_config(config['model'], reload=True)
    model.eval(); model = model.to('cuda')

    probs_list = []
    for batch in tqdm(dataloader):
        with torch.no_grad():
            output = model.predict_label_separate(batch['mixture'].to('cuda'))

        for labels, waveforms, probs, soundscape_name in zip(output['label'], 
                                                      output['waveform'].cpu()[:, :, 0, :],
                                                      output['probabilities'],
                                                      batch['soundscape']):
            est_obj = {'soundscape': soundscape_name, 'estimate': []}
            for i, (label, waveform, prob) in enumerate(zip(labels, waveforms, probs)):
                if label != 'silence':
                    filename = soundscape_name + '_' + str(i) + '_' + label + '.wav'
                    wavpath = os.path.join(outputdir, filename)
                    sf.write(wavpath, waveform.numpy(), sr)
                    est_obj['estimate'].append({
                        'filename': filename,
                        'label': label,
                        'probability': prob.item(),
                    })
            probs_list.append(est_obj)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    result_summary = {'nparams': total_params,
                      'probabilities': probs_list,}
    if 'config' in locals(): result_summary['config'] = config
    with open(probs_json_dir, "w") as outfile:
        json.dump(result_summary, outfile, indent=4)
    print('Verifying output filenames')
    verify_output_filename(dataset_config['args']['config']['soundscape_dir'], outputdir)
    print('Verifying eval_results.json')
    verify_json(probs_json_dir, outputdir)
    
    print('Finish')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batchsize","-b", type=int, required=False, default=2)
    parser.add_argument("--config", "-c", type=str, required=True,)

    parser.add_argument("--output_dir", type=str, required=True,)
    parser.add_argument("--output_name", type=str, required=False, default='submission',)

    args = parser.parse_args()
    print('START')
    main(args)

# python -m src.evaluation.generate_waveform -c src/evaluation/eval_configs/m2dat_4c_resunetk_eval.yaml --output_dir workspace/submission --output_name Nguyen_NTT_task4_1
# python -m src.evaluation.generate_waveform -c src/evaluation/eval_configs/m2dat_1c_resunetk_eval.yaml --output_dir workspace/submission --output_name Nguyen_NTT_task4_2
