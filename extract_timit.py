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
    wrd_fps = glob.glob(os.path.join(timit_dir, '**/**/**/*.WRD'))
    dataset_to_speakers = defaultdict(set)
    for wrd_fp in wrd_fps:
        path_data = wrd_fp.split(os.sep)[-3:]
        dialect = path_data[0]
        sex = path_data[1][:1]
        speaker_id = path_data[1]
        sentence_id = os.path.splitext(path_data[3])[0]
        dataset_to_speakers[dataset].add(speaker_id)
    print {k:len(v) for k, v in dataset_to_speakers.items()}


    limit = limit if limit else len(wrd_fps)
    results = []
    for wrd_fp in tqdm(wrd_fps[:limit]):
        wav_fp = '{}.WAV'.format(os.path.splitext(wrd_fp)[0])
        path_data = wrd_fp.split(os.sep)[-3:]
        dialect = path_data[0]
        sex = path_data[1][:1]
        speaker_id = path_data[1][1:]
        sentence_id = os.path.splitext(path_data[2])[0]
        metadata = (dialect, sex, speaker_id, sentence_id)

        loader = MonoLoader(filename=wav_fp, sampleRate=16000)
        utterance = loader()

        feats = analyzer(utterance)
        print metadata
        print feats.shape
        results.append((metadata, feats))

    return results

if __name__ == '__main__':
    import cPickle as pickle

    timit_dir = '/media/cdonahue/bulk1/datasets/timit/TIMIT'

    analyzer = mel40_analyzer()
    timit_feats = load_timit(timit_dir, analyzer, limit=10)
