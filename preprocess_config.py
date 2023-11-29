import argparse
import json
import os
import re
import wave
from random import shuffle

pattern = re.compile(r'^[\.a-zA-Z0-9_\/]+$')

def get_wav_duration(file_path):
    try:
        with wave.open(file_path, 'rb') as wav_file:
            n_frames = wav_file.getnframes()
            framerate = wav_file.getframerate()
            return n_frames / float(framerate)
    except Exception as e:
        print(f"Reading {file_path}")
        raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_num", type=int, default=5, help="The number of audios used as the validation set")
    parser.add_argument("--source_dir", type=str, default="./dataset/44k", help="path to source dir")
    parser.add_argument("--vol_aug", action="store_true", help="Whether to use volume embedding and volume augmentation")
    args = parser.parse_args()
    
    config_template = json.load(open("configs_template/config_template.json"))
    train = []
    val = []
    idx = 0
    wavs = []

    os.makedirs(args.source_dir, exist_ok=True)

    for file_name in os.listdir(os.path.join(args.source_dir)):
        if not file_name.endswith("wav"):
            continue
        if file_name.startswith("."):
            continue

        file_path = "/".join([args.source_dir, file_name])

        if not pattern.match(file_name):
            print("Detected non-ASCII file name: " + file_path)

        if get_wav_duration(file_path) < 0.3:
            print("Skip too short audio: " + file_path)
            continue

        wavs.append(file_path)

    shuffle(wavs)
    train += wavs[args.val_num:]
    val += wavs[:args.val_num]

    shuffle(train)
    shuffle(val)

    print("Writing filelists/train.txt")
    with open("filelists/train.txt", "w") as f:
        for fname in train:
            wavpath = fname
            f.write(wavpath + "\n")

    print("Writing filelists/val.txt")
    with open("filelists/val.txt", "w") as f:
        for fname in val:
            wavpath = fname
            f.write(wavpath + "\n")

    print("Writing configs/config.json")
    with open("configs/config.json", "w") as f:
        json.dump(config_template, f, indent=2)
