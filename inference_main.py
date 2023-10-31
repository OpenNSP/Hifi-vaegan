import logging

import soundfile
from spkmix import spk_mix_map

from inference import infer_tool
from inference.infer_tool import Svc

logging.getLogger('numba').setLevel(logging.WARNING)
chunks_dict = infer_tool.read_temp("inference/chunks_temp.json")



def main():
    import argparse

    parser = argparse.ArgumentParser(description='VAEGAN TEST')

    parser.add_argument('-m', '--model_path', type=str, default="logs/44k/G_120.pth", help='模型路径')
    parser.add_argument('-c', '--config_path', type=str, default="logs/44k/config.json", help='配置文件路径')
    parser.add_argument('-cl', '--clip', type=float, default=0, help='音频强制切片，默认0为自动切片，单位为秒/s')
    parser.add_argument('-n', '--clean_names', type=str, nargs='+', default=["1.wav"], help='wav文件名列表，放在raw文件夹下')
    parser.add_argument('-lgr', '--linear_gradient_retain', type=float, default=0.75, help='自动音频切片后，需要舍弃每段切片的头尾。该参数设置交叉长度保留的比例，范围0-1,左开右闭')
    parser.add_argument('-lg', '--linear_gradient', type=float, default=0, help='两段音频切片的交叉淡入长度，如果强制切片后出现人声不连贯可调整该数值，如果连贯建议采用默认值0，单位为秒')
    parser.add_argument('-sd', '--slice_db', type=int, default=-40, help='默认-40，嘈杂的音频可以-30，干声保留呼吸可以-50')
    parser.add_argument('-p', '--pad_seconds', type=float, default=0.5, help='推理音频pad秒数，由于未知原因开头结尾会有异响，pad一小段静音段后就不会出现')
    parser.add_argument('-d', '--device', type=str, default=None, help='推理设备，None则为自动选择cpu和gpu')
  

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
    svc_model.clear_empty()
            
if __name__ == '__main__':
    main()
