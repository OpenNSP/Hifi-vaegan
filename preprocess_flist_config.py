import argparse
import json
import os
import re
import wave
from random import shuffle

from loguru import logger
from tqdm import tqdm

pattern = re.compile(r'^[\.a-zA-Z0-9_\/]+$')

def get_wav_duration(file_path):
    try:
        with wave.open(file_path, 'rb') as wav_file:
            # 获取音频帧数
            n_frames = wav_file.getnframes()
            # 获取采样率
            framerate = wav_file.getframerate()
            # 计算时长（秒）
            return n_frames / float(framerate)
    except Exception as e:
        logger.error(f"Reading {file_path}")
        raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_list", type=str, default="./filelists/train.txt", help="path to train list")
    parser.add_argument("--val_list", type=str, default="./filelists/val.txt", help="path to val list")
    parser.add_argument("--source_dir", type=str, default="./dataset/44k", help="path to source dir")
    parser.add_argument("--vol_aug", action="store_true", help="Whether to use volume embedding and volume augmentation")
    args = parser.parse_args()
    
    config_template = json.load(open("configs_template/config_template.json"))
    train = []
    val = []
    idx = 0
    spk_dict = {}
    spk_id = 0

    wavs = []

    for file_name in tqdm(os.listdir(os.path.join(args.source_dir))):
        if not file_name.endswith("wav"):
            continue
        if file_name.startswith("."):
            continue

        file_path = "/".join([args.source_dir, file_name])

        if not pattern.match(file_name):
            logger.warning("Detected non-ASCII file name: " + file_path)

        if get_wav_duration(file_path) < 0.3:
            logger.info("Skip too short audio: " + file_path)
            continue

        wavs.append(file_path)

    shuffle(wavs)
    train += wavs[2:]
    val += wavs[:2]

    shuffle(train)
    shuffle(val)

    logger.info("Writing " + args.train_list)
    with open(args.train_list, "w") as f:
        for fname in tqdm(train):
            wavpath = fname
            f.write(wavpath + "\n")

    logger.info("Writing " + args.val_list)
    with open(args.val_list, "w") as f:
        for fname in tqdm(val):
            wavpath = fname
            f.write(wavpath + "\n")

    logger.info("Writing to configs/config.json")
    with open("configs/config.json", "w") as f:
        json.dump(config_template, f, indent=2)
    logger.info("Writing to configs/diffusion.yaml")
