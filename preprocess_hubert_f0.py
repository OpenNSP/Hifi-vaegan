import argparse
import logging
import multiprocessing
import os
import random
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from random import shuffle

import librosa
import numpy as np
import torch
from tqdm import tqdm

import utils

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

hps = utils.get_hparams_from_file("configs/config.json")
sampling_rate = hps.data.sampling_rate
hop_length = hps.data.hop_length
speech_encoder = hps["model"]["speech_encoder"]


def process_one(filename, f0p):
    # print(filename)
    wav, sr = librosa.load(filename, sr=sampling_rate)
    audio_norm = torch.FloatTensor(wav)
    audio_norm = audio_norm.unsqueeze(0)

    f0_path = filename + ".f0.npy"
    if not os.path.exists(f0_path):
        f0_predictor = utils.get_f0_predictor(f0p,sampling_rate=sampling_rate, hop_length=hop_length,device=None,threshold=0.05)
        f0,uv = f0_predictor.compute_f0_uv(
            wav
        )
        np.save(f0_path, np.asanyarray((f0,uv),dtype=object))


def process_batch(file_chunk, f0p):
    for filename in tqdm(file_chunk):
        process_one(filename,  f0p)

def parallel_process(filenames, num_processes, f0p):
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        tasks = []
        for i in range(num_processes):
            start = int(i * len(filenames) / num_processes)
            end = int((i + 1) * len(filenames) / num_processes)
            file_chunk = filenames[start:end]
            tasks.append(executor.submit(process_batch, file_chunk, f0p))

        for task in tqdm(tasks):
            task.result()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_dir", type=str, default="dataset/44k", help="path to input dir"
    )
    parser.add_argument(
        '--f0_predictor', type=str, default="dio", help='Select F0 predictor, can select crepe,pm,dio,harvest,rmvpe, default pm(note: crepe is original F0 using mean filter)'
    )
    parser.add_argument(
        '--num_processes', type=int, default=1, help='You are advised to set the number of processes to the same as the number of CPU cores'
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    f0p = args.f0_predictor
    print(f0p)

    filenames = glob(f"{args.in_dir}/*.wav", recursive=True)  # [:10]
    shuffle(filenames)
    multiprocessing.set_start_method("spawn", force=True)

    num_processes = args.num_processes
    if num_processes == 0:
        num_processes = os.cpu_count()
    
    parallel_process(filenames, num_processes, f0p)