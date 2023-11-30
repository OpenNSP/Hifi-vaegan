import argparse
import logging
import soundfile
import torch
from inference import infer_tool
from inference.infer_tool import Svc

logging.getLogger('numba').setLevel(logging.WARNING)
chunks_dict = infer_tool.read_temp("inference/chunks_temp.json")

def main():
    parser = argparse.ArgumentParser(description='HiFi-VAEGAN Inference')

    parser.add_argument('-m', '--model_path', type=str, default="logs/VAEGAN/G_0.pth", help='model path')
    parser.add_argument('-c', '--config_path', type=str, default="logs/VAEGAN/config.json", help='config path')
    parser.add_argument('-n', '--clean_names', type=str, nargs='+', default=["test.wav"], help='List of file names, placed in the raw folder')

    parser.add_argument('-sd', '--slice_db', type=int, default=-40, help='Automatic slicing threshold, the larger the value, the fewer slices')
    parser.add_argument('-cl', '--clip', type=float, default=0, help='Audio forced slicing, default 0 is automatic slicing, unit is seconds')
    parser.add_argument('-lg', '--linear_gradient', type=float, default=0, help='The crossfade duration between two audio segments, unit is seconds')

    parser.add_argument('-lgr', '--linear_gradient_retain', type=float, default=0.75, help='The proportion of crossover length retained, range (0-1]')
    parser.add_argument('-p', '--pad_seconds', type=float, default=0.5, help='There are abnormal sounds at the beginning and end, and the pad needs to be muted for a short period of time, unit is seconds')
    parser.add_argument('-d', '--device', type=str, default=None, help='Inference device, None means automatic')
  
    args = parser.parse_args()

    slice_db = args.slice_db
    pad_seconds = args.pad_seconds
    clip = args.clip
    lg = args.linear_gradient
    lgr = args.linear_gradient_retain
    clean_name = args.clean_names[0]

    svc_model = Svc(args.model_path,
                    args.config_path,
                    args.device)
    
    infer_tool.mkdir(["raw", "results"])

    raw_audio_path = f"raw/{clean_name}"
    if "." not in raw_audio_path:
        raw_audio_path += ".wav"
    infer_tool.format_wav(raw_audio_path)

    kwarg = {
        "raw_audio_path" : raw_audio_path,
        "slice_db" : slice_db,
        "pad_seconds" : pad_seconds,
        "clip_seconds" : clip,
        "lg_num": lg,
        "lgr_num" : lgr
    }
    audio = svc_model.slice_inference(**kwarg)
    res_path = f'results/{clean_name}'
    soundfile.write(res_path, audio, svc_model.target_sample, format="wav")
    torch.cuda.empty_cache()
            
if __name__ == '__main__':
    main()
