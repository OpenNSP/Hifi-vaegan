import argparse
import concurrent.futures
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

import librosa
import numpy as np
from rich.progress import track
from scipy.io import wavfile

def load_wav(wav_path):
    return librosa.load(wav_path, sr=None)

def trim_wav(wav, top_db=40):
    return librosa.effects.trim(wav, top_db=top_db)

def normalize_peak(wav, threshold=1.0):
    peak = np.abs(wav).max()
    if peak > threshold:
        wav = 0.98 * wav / peak
    return wav

def resample_wav(wav, sr, target_sr):
    return librosa.resample(wav, orig_sr=sr, target_sr=target_sr)

def save_wav_to_path(wav, save_path, sr):
    wavfile.write(save_path, sr, (wav * np.iinfo(np.int16).max).astype(np.int16))

def process(item):
    wav_name, args = item

    wav_path = os.path.join(args.in_dir, wav_name)
    if os.path.exists(wav_path) and '.wav' in wav_path:
        os.makedirs(args.out_dir, exist_ok=True)

        wav, sr = load_wav(wav_path)
        wav, _ = trim_wav(wav)
        wav = normalize_peak(wav)
        resampled_wav = resample_wav(wav, sr, args.sr)

        resampled_wav /= np.max(np.abs(resampled_wav))

        save_path = os.path.join(args.out_dir, wav_name)
        save_wav_to_path(resampled_wav, save_path, args.sr)

def process_all_speakers():
    process_count = 30 if os.cpu_count() > 60 else (os.cpu_count() - 2 if os.cpu_count() > 4 else 1)
    with ProcessPoolExecutor(max_workers=process_count) as executor:
        futures = [executor.submit(process, (i, args)) for i in os.listdir(args.in_dir) if i.endswith("wav")]
        for _ in track(concurrent.futures.as_completed(futures), total=len(futures), description="Resampling:"):
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr", type=int, default=44100, help="sampling rate")
    parser.add_argument("--in_dir", type=str, default="./dataset_raw", help="path to source dir")
    parser.add_argument("--out_dir", type=str, default="./dataset/44k", help="path to target dir")
    args = parser.parse_args()

    print(f"CPU count: {cpu_count()}")
    process_all_speakers()
