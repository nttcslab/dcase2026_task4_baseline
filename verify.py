import os
import argparse

def verify_source_structure(source_dir):
    print(f'\nVerify source directory: {os.path.abspath(source_dir)}', flush=True)

    assert os.path.isdir(source_dir), f'Not a directory: {source_dir}'
    assert os.path.isdir(os.path.join(source_dir, 'src')), f'Missing folder: src'
    print('src: OK', flush=True)

    assert os.path.isfile(os.path.join(source_dir, 'checkpoint/m2d_as_vit_base-80x1001p16x16p32k-240413_AS-FT_enconly/weights_ep69it3124-0.47998.pth')), f'Missing checkpoint: checkpoint/m2d_as_vit_base-80x1001p16x16p32k-240413_AS-FT_enconly/weights_ep69it3124-0.47998.pth'
    print('M2D checkpoint: OK', flush=True)

    spAudSyn_files = ['src/modules/spatial_audio_synthesizer/spatial_audio_synthesizer.py',
                      'src/modules/spatial_audio_synthesizer/room.py',
                      'src/modules/spatial_audio_synthesizer/utils.py',
             ]
    for f in spAudSyn_files:
        assert os.path.isfile(f), f'Missing: {f}'
    assert os.path.isfile(os.path.join(source_dir, 'src/modules/spatial_audio_synthesizer/spatial_audio_synthesizer.py'))
    print('SpAudSyn modules: OK', flush=True)

    assert os.path.isfile(os.path.join(source_dir, 'data/dev_set/metadata/valid.json')), "missing valid.json"
    datasubdirs = [
        'dev_set',
        'dev_set/config',
        'dev_set/interference',
        'dev_set/interference/train',
        'dev_set/interference/valid',
        'dev_set/noise',
        'dev_set/noise/train',
        'dev_set/noise/valid',
        'dev_set/room_ir',
        'dev_set/room_ir/train',
        'dev_set/room_ir/valid',
        'dev_set/sound_event',
        'dev_set/sound_event/train',
        'dev_set/sound_event/valid',
        'dev_set/synthesized/test/soundscape',
        'dev_set/synthesized/test/oracle_target',
        'dev_set/metadata/valid',
    ]
    for subdir in datasubdirs:
        folder = os.path.join(source_dir, 'data', subdir)
        assert os.path.isdir(folder), f'Missing folder: {folder}'
    print('data folders: OK', flush=True)
    print("Source directory verified successfully.", flush=True)
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", type=str, required=False, default='')
    args = parser.parse_args()

    if args.source_dir: # verify structure of the source directory
        verify_source_structure(args.source_dir)
