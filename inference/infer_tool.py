import gc
import io
import json
import logging
import os
import time
from pathlib import Path
import librosa
import numpy as np
import soundfile
import torch
import torchaudio
import utils
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
from modules.models import TrainModel
from . import slicer

logging.getLogger('matplotlib').setLevel(logging.WARNING)

progress = Progress(
    TextColumn("Running: "),
    BarColumn(), "[progress.percentage]{task.percentage:>3.1f}%",
    "•",
    MofNCompleteColumn(),
    "•",
    TimeElapsedColumn(),
    "|",
    TimeRemainingColumn(),
    "•",
    TextColumn("[progress.description]{task.description}"),
    transient=False
    )

def read_temp(file_name):
    if not os.path.exists(file_name):
        with open(file_name, "w") as f:
            f.write(json.dumps({"info": "temp_dict"}))
        return {}
    else:
        try:
            with open(file_name, "r") as f:
                data = f.read()
            data_dict = json.loads(data)
            if os.path.getsize(file_name) > 50 * 1024 * 1024:
                f_name = file_name.replace("\\", "/").split("/")[-1]
                print(f"clean {f_name}")
                for wav_hash in list(data_dict.keys()):
                    if int(time.time()) - int(data_dict[wav_hash]["time"]) > 14 * 24 * 3600:
                        del data_dict[wav_hash]
        except Exception as e:
            print(e)
            print(f"{file_name} error,auto rebuild file")
            data_dict = {"info": "temp_dict"}
        return data_dict

def format_wav(audio_path):
    if Path(audio_path).suffix == '.wav':
        return
    raw_audio, raw_sample_rate = librosa.load(audio_path, mono=True, sr=None)
    soundfile.write(Path(audio_path).with_suffix(".wav"), raw_audio, raw_sample_rate)

def mkdir(paths: list):
    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)

def pad_array(arr, target_length):
    current_length = arr.shape[0]
    if current_length >= target_length:
        return arr
    else:
        pad_width = target_length - current_length
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        padded_arr = np.pad(arr, (pad_left, pad_right), 'constant', constant_values=(0, 0))
        return padded_arr
    
def split_list_by_n(list_collection, n, pre=0):
    for i in range(0, len(list_collection), n):
        yield list_collection[i-pre if i-pre>=0 else i: i + n]

class Svc(object):
    def __init__(self, net_g_path, config_path,
                 device=None
                 ):
        self.net_g_path = net_g_path
        if device is None:
            self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.dev = torch.device(device)
        self.net_g_ms = None
        self.hps_ms = utils.get_hparams_from_file(config_path,True)
        self.target_sample = self.hps_ms.data.sampling_rate
        self.hop_size = self.hps_ms.data.hop_length

        self.load_model()

    def load_model(self):
        self.net_g_ms = TrainModel(
            self.hps_ms.data.hop_length,
            self.hps_ms.data.win_length,
            **self.hps_ms.model)
        
        _ = utils.load_checkpoint(self.net_g_path, self.net_g_ms, None)
        self.dtype = list(self.net_g_ms.parameters())[0].dtype
        if "half" in self.net_g_path and torch.cuda.is_available():
            _ = self.net_g_ms.half().eval().to(self.dev)
        else:
            _ = self.net_g_ms.eval().to(self.dev)

    def infer(self, raw_path):
        if isinstance(raw_path, str) or isinstance(raw_path, io.BytesIO):
            wav, sr = torchaudio.load(raw_path)
            if not hasattr(self,"audio_resample_transform") or self.audio_resample_transform.orig_freq != sr:
                self.audio_resample_transform = torchaudio.transforms.Resample(sr,self.target_sample)
            wav = self.audio_resample_transform(wav).to(self.dev)
        else:
            wav = raw_path

        with torch.no_grad():
            start = time.time()
            z, wav, (m, logs), commit_loss = self.net_g_ms(wav)
        return wav

    def unload_model(self):
        # unload model
        self.net_g_ms = self.net_g_ms.to("cpu")
        del self.net_g_ms
        if hasattr(self,"enhancer"): 
            self.enhancer.enhancer = self.enhancer.enhancer.to("cpu")
            del self.enhancer.enhancer
            del self.enhancer
        gc.collect()

    def slice_inference(self,
                        raw_audio_path,
                        slice_db,
                        pad_seconds=0.5,
                        clip_seconds=0,
                        lg_num=0,
                        lgr_num =0.75,
                        ):

        wav_path = Path(raw_audio_path).with_suffix('.wav')
        chunks = slicer.cut(wav_path, db_thresh=slice_db)
        audio_data, audio_sr = slicer.chunks2audio(wav_path, chunks)
        per_size = int(clip_seconds*audio_sr)
        lg_size = int(lg_num*audio_sr)
        lg_size_r = int(lg_size*lgr_num)
        lg_size_c_l = (lg_size-lg_size_r)//2
        lg_size_c_r = lg_size-lg_size_r-lg_size_c_l
        lg = np.linspace(0,1,lg_size_r) if lg_size!=0 else 0

        audio = []
        with progress:
            infer_task = progress.add_task("Infer", total=len(audio_data))
            for (slice_tag, data) in audio_data:
                length = int(np.ceil(len(data) / audio_sr * self.target_sample))
                if slice_tag:
                    _audio = np.zeros(length)
                    audio.extend(list(pad_array(_audio, length)))
                    progress.update(infer_task, advance=1, description=f"length=0s")
                    continue
                if per_size != 0:
                    datas = split_list_by_n(data, per_size,lg_size)
                else:
                    datas = [data]

                progress.update(infer_task, advance=1, description=f"length={round(len(data) / audio_sr, 3)}s")
                for k, dat in enumerate(datas):
                    per_length = int(np.ceil(len(dat) / audio_sr * self.target_sample)) if clip_seconds!=0 else length
                    pad_len = int(audio_sr * pad_seconds)
                    dat = np.concatenate([np.zeros([pad_len]), dat, np.zeros([pad_len])])
                    raw_path = io.BytesIO()
                    soundfile.write(raw_path, dat, audio_sr, format="wav")
                    raw_path.seek(0)
                    out_audio = self.infer(raw_path)[0,0,:]
                    _audio = out_audio.cpu().numpy()
                    pad_len = int(self.target_sample * pad_seconds)
                    _audio = _audio[pad_len:-pad_len]
                    _audio = pad_array(_audio, per_length)
                    if lg_size!=0 and k!=0:
                        lg1 = audio[-(lg_size_r+lg_size_c_r):-lg_size_c_r] if lgr_num != 1 else audio[-lg_size:]
                        lg2 = _audio[lg_size_c_l:lg_size_c_l+lg_size_r]  if lgr_num != 1 else _audio[0:lg_size]
                        lg_pre = lg1*(1-lg)+lg2*lg
                        audio = audio[0:-(lg_size_r+lg_size_c_r)] if lgr_num != 1 else audio[0:-lg_size]
                        audio.extend(lg_pre)
                        _audio = _audio[lg_size_c_l+lg_size_r:] if lgr_num != 1 else _audio[lg_size:]
                    audio.extend(list(_audio))
        return np.array(audio)