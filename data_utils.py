import random
import numpy as np
import torch
import torch.utils.data
from utils import load_filepaths_and_text, load_wav_to_torch
import librosa

class TextAudioSpeakerLoader(torch.utils.data.Dataset):
    def __init__(self, audiopaths, hparams, all_in_mem: bool = False, vol_aug: bool = True, pitch_aug: bool = True, is_slice = True, rank = 0, world_size = 1):
        self.audiopaths = load_filepaths_and_text(audiopaths)
        self.hparams = hparams
        self.max_wav_value = hparams.data.max_wav_value
        self.sampling_rate = hparams.data.sampling_rate
        self.filter_length = hparams.data.filter_length
        self.hop_length = hparams.data.hop_length
        self.win_length = hparams.data.win_length
        self.sampling_rate = hparams.data.sampling_rate
        self.segment_size = hparams.train.segment_size
        self.vol_aug = hparams.train.vol_aug and vol_aug
        self.is_slice = is_slice
        self.pitch_aug = pitch_aug
        random.seed(1234)
        random.shuffle(self.audiopaths)
        
        if world_size > 1:
            self.audiopaths = self.audiopaths[rank::world_size]
            
        self.all_in_mem = all_in_mem
        if self.all_in_mem:
            self.cache = [self.get_audio(p[0]) for p in self.audiopaths]

    def get_audio(self, filename):
        filename = filename.replace("\\", "/")
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError("Sample Rate not match. Expect {} but got {} from {}".format(self.sampling_rate, sampling_rate, filename))
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = audio_norm[:, :audio.shape[-1] //self.hop_length * self.hop_length]
        return audio_norm

    def random_slice(self, audio_norm):
        if random.choice([True, False, False]) and self.pitch_aug:
            audio_norm = librosa.resample(y = audio_norm.numpy(),  orig_sr=self.sampling_rate, target_sr=int(self.sampling_rate * random.uniform(1, 1/(2**(15/12)))))
            audio_norm = torch.from_numpy(audio_norm)
            if audio_norm.abs().max() >= 1.0:
                audio_norm /= audio_norm.abs().max()
                audio_norm *= 0.95

        if random.choice([True, False]) and self.vol_aug:
            max_amp = float(torch.max(torch.abs(audio_norm))) + 1e-5
            max_shift = min(1, np.log10(1/max_amp))
            log10_vol_shift = random.uniform(-1, max_shift)
            audio_norm = audio_norm * (10 ** log10_vol_shift)

        
        if self.is_slice:
            if (audio_norm.shape[-1] // self.hop_length) > self.segment_size:
                start = random.randint(0, int(audio_norm.shape[-1] // self.hop_length) - self.segment_size)
                end = start + self.segment_size - 10
                audio_norm = audio_norm[start * self.hop_length : end * self.hop_length]

        return audio_norm

    def __getitem__(self, index):
        if self.all_in_mem:
            return self.random_slice(*self.cache[index])
        else:
            return self.random_slice(*self.get_audio(self.audiopaths[index][0]))

    def __len__(self):
        return len(self.audiopaths)


class TextAudioCollate:
    def __call__(self, batch):
        batch = [b for b in batch if b is not None]
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x.shape[-1] for x in batch]),
            dim=0, descending=True)

        max_wav_len = max([x.size(-1) for x in batch])

        lengths = torch.LongTensor(len(batch))

        wav_padded = torch.zeros(len(batch), max_wav_len)

        for i in range(len(ids_sorted_decreasing)):
            wav = batch[ids_sorted_decreasing[i]]
            
            lengths[i] = wav.size(-1)
            wav_padded[i, :wav.size(-1)] = wav

        return wav_padded, lengths
