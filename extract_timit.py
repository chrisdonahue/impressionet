from collections import defaultdict
import glob
import os
import random
import sys

import essentia
from essentia.standard import MonoLoader, FrameGenerator, Resample, Windowing, Spectrum, MelBands
import numpy as np
from tqdm import tqdm

def mel40_analyzer():
    window = Windowing(size=256, type='blackmanharris62')
    spectrum = Spectrum(size=256)
    mel = MelBands(
            inputSize=129,
            numberBands=40,
            lowFrequencyBound=27.5,
            highFrequencyBound=8000.0,
            sampleRate=16000.0)
    def analyzer(samples):
        feats = []
        for frame in FrameGenerator(samples, 256, 160):
            frame_feats = mel(spectrum(window(frame)))
            frame_feats = np.log(frame_feats + 1e-16)
            feats.append(frame_feats)
        return np.array(feats)
    return analyzer

def raw_analyzer():
    def analyzer(samples):
        return samples
    return analyzer

_TEST_SPEAKERS = set([
    'MDAB0', 'MWBT0', 'FELC0',
    'MTAS1', 'MWEW0', 'FPAS0',
    'MJMP0', 'MLNT0', 'FPKT0',
    'MLLL0', 'MTLS0', 'FJLM0',
    'MBPM0', 'MKLT0', 'FNLP0',
    'MCMJ0', 'MJDH0', 'FMGD0',
    'MGRT0', 'MNJM0', 'FDHC0',
    'MJLN0', 'MPAM0', 'FMLD0'
])

def load_timit(
        timit_dir,
        analyzer,
        limit=None):
    glob_fp = os.path.join(timit_dir, '**/**/**/*.PHN')
    phn_fps = glob.glob(glob_fp)
    test_speakers = set()
    valid_speakers = set()
    train_speakers = set()
    shapes = []
    num_sa = 0
    train_set = []
    valid_set = []
    test_set = []
    if limit:
        phn_fps = phn_fps[:limit]
    for phn_fp in tqdm(phn_fps):
        path_data = phn_fp.split(os.sep)[-4:]
        dataset = path_data[0].lower()
        dialect = path_data[1]
        sex = path_data[2][:1]
        speaker_id = path_data[2]
        utterance_name = os.path.splitext(os.path.split(phn_fp)[1])[0]

        if 'SA' in utterance_name:
            num_sa += 1
            continue

        # build test speakers
        if dataset == 'test' and speaker_id not in _TEST_SPEAKERS:
            dataset = 'valid'

        # extract PHN
        phoneme_list = []
        with open(phn_fp, 'r') as f:
            for l in f.read().strip().splitlines():
                start, stop, phoneme = l.split()
                phoneme_list.append((start, stop, phoneme))

        # extract WAV
        wav_fp = '{}.WAV'.format(os.path.splitext(phn_fp)[0])
        loader = MonoLoader(filename=wav_fp, sampleRate=16000)
        utterance_feats = analyzer(loader())
        assert utterance_feats.dtype == np.float32
        shapes.append(utterance_feats.shape)

        # add to list
        entry = ((dialect, sex, speaker_id, utterance_name), phoneme_list, utterance_feats)
        if dataset == 'train':
            train_set.append(entry)
            train_speakers.add(speaker_id)
        elif dataset == 'valid':
            valid_set.append(entry)
            valid_speakers.add(speaker_id)
        elif dataset == 'test':
            test_set.append(entry)
            test_speakers.add(speaker_id)

    print np.mean(shapes, axis=0)

    if limit is None:
        assert num_sa == 1260
        assert len(train_speakers) == 462
        assert len(valid_speakers) == 144
        assert len(test_speakers) == 24

    return train_set, valid_set, test_set

if __name__ == '__main__':
    import cPickle as pickle

    timit_dir = '/home/cdonahue/timit/TIMIT'
    limit = 10

    tag = 'raw'
    analyzer = raw_analyzer()
    #tag = 'mel40'
    #analyzer = mel40_analyzer()

    train, valid, test = load_timit(timit_dir, analyzer, limit=limit)

    if limit is None:
        with open('timit_{}_train.pkl'.format(tag), 'wb') as f:
            pickle.dump(train, f)
        with open('timit_{}_valid.pkl'.format(tag), 'wb') as f:
            pickle.dump(valid, f)
        with open('timit_{}_test.pkl'.format(tag), 'wb') as f:
            pickle.dump(test, f)
