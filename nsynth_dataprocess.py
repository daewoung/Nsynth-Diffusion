import numpy as np
import torch
import torchaudio as T
import torchaudio.transforms as TT
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from tqdm import tqdm


class AttrDict(dict):
  def __init__(self, *args, **kwargs):
      super(AttrDict, self).__init__(*args, **kwargs)
      self.__dict__ = self

  def override(self, attrs):
    if isinstance(attrs, dict):
      self.__dict__.update(**attrs)
    elif isinstance(attrs, (list, tuple, set)):
      for attr in attrs:
        self.override(attr)
    elif attrs is not None:
      raise NotImplementedError
    return self


params = AttrDict(
    sample_rate=16000,
    n_mels=80,
    n_fft=1024,
    hop_samples=256,
    audio_len = 16000*5
)


def transform(filename):

  print(f'Processing {filename}')
  audio, sr = T.load(filename)
  audio = torch.clamp(audio[0], -1.0, 1.0)

  if params.sample_rate != sr:
    raise ValueError(f'Invalid sample rate {sr}.')
  mel_args = {
      'sample_rate': sr,
      'win_length': params.hop_samples * 4,
      'hop_length': params.hop_samples,
      'n_fft': params.n_fft,
      'f_min': 20.0,
      'f_max': sr / 2.0,
      'n_mels': params.n_mels,
      'power': 1.0,
      'normalized': True,
  }
  mel_spec_transform = TT.MelSpectrogram(**mel_args)

  with torch.no_grad():
    spectrogram = mel_spec_transform(audio)
    spectrogram = 20 * torch.log10(torch.clamp(spectrogram, min=1e-5)) - 20
    spectrogram = torch.clamp((spectrogram + 100) / 100, 0.0, 1.0)
    np.save(f'{filename}.spec.npy', spectrogram.cpu().numpy())


def main(args):
  filenames = glob(f'{args.dir}/**/*.wav', recursive=True)
  with ProcessPoolExecutor() as executor:
    list(tqdm(executor.map(transform, filenames), desc='Preprocessing', total=len(filenames)))


if __name__ == '__main__':
  parser = ArgumentParser(description='prepares a dataset to train DiffWave')
  parser.add_argument('dir',
      help='directory containing .wav files for training')
  main(parser.parse_args())
