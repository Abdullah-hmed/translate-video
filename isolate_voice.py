import argparse
import os
import sys
import tempfile
import shutil
import numpy as np
import soundfile as sf

SUBMODULE_DIR = os.path.join(os.path.dirname(__file__), "mel_band_roformer")
CONFIG_PATH   = os.path.join(SUBMODULE_DIR, "configs/config_vocals_mel_band_roformer.yaml")
MODEL_PATH    = os.path.join("weights/MelBandRoformer.ckpt")
MODEL_TYPE    = "mel_band_roformer"

sys.path.insert(0, SUBMODULE_DIR)

def convert_to_wav(input_path: str, tmp_dir: str) -> str:
    """Convert any audio format to WAV using ffmpeg, if needed."""
    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".wav":
        return input_path
    wav_path = os.path.join(tmp_dir, "input_converted.wav")
    ret = os.system(f'ffmpeg -y -i "{input_path}" "{wav_path}" -loglevel error')
    if ret != 0:
        raise RuntimeError(f"ffmpeg failed to convert {input_path}. Is ffmpeg installed?")
    return wav_path

def separate(
    input_path: str,
    config_path: str = CONFIG_PATH,
    model_path: str = MODEL_PATH,
    model_type: str = MODEL_TYPE,
    device_id: int = 0,
):
    import yaml
    import torch
    from ml_collections import ConfigDict
    from utils import demix_track, get_model_from_config

    with tempfile.TemporaryDirectory() as tmp_dir:
        wav_path = convert_to_wav(input_path, tmp_dir)

        with open(config_path) as f:
            config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))

        model = get_model_from_config(model_type, config)
        if model_path:
            print(f"Loading model from: {model_path}")
            model.load_state_dict(torch.load(model_path, map_location="cpu"))

        if device_id >= 0 and torch.cuda.is_available():
            device = torch.device(f"cuda:{device_id}")
            model = model.to(device)
            print(f"Running on GPU: cuda:{device_id}")
        else:
            device = torch.device("cpu")
            model = model.to(device)
            print("Running on CPU (slow).")

        model.eval()

        instruments = config.training.instruments
        if config.training.target_instrument is not None:
            instruments = [config.training.target_instrument]
        primary_stem = instruments[0]

        mix, sr = sf.read(wav_path)
        original_mono = len(mix.shape) == 1
        if original_mono:
            mix = np.stack([mix, mix], axis=-1)

        mixture = torch.tensor(mix.T, dtype=torch.float32)

        print("Running separation...")
        res, _ = demix_track(config, model, mixture, device, None)

        vocals = res[primary_stem].T
        instrumental = mix - vocals

        if original_mono:
            vocals = vocals[:, 0]
            instrumental = instrumental[:, 0]

    return vocals, instrumental, sr

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Separate stems from a single audio file.")
    parser.add_argument("--input",       required=True,       help="Path to input audio file")
    parser.add_argument("--output_dir",  required=True,       help="Directory to save output stems")
    parser.add_argument("--config_path", default=CONFIG_PATH, help="Path to model config YAML")
    parser.add_argument("--model_path",  default=MODEL_PATH,  help="Path to model checkpoint")
    parser.add_argument("--model_type",  default=MODEL_TYPE,  help="Model architecture type")
    parser.add_argument("--gpu",         type=int, default=0, help="GPU device id (-1 for CPU)")
    args = parser.parse_args()

    vocals, instrumental, sr = separate(
        input_path=args.input,
        config_path=args.config_path,
        model_path=args.model_path,
        model_type=args.model_type,
        device_id=args.gpu,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.input))[0]
    sf.write(os.path.join(args.output_dir, f"{base_name}_vocals.wav"),       vocals,       sr, subtype="FLOAT")
    sf.write(os.path.join(args.output_dir, f"{base_name}_instrumental.wav"), instrumental, sr, subtype="FLOAT")
    print(f"Saved vocals and instrumental to {args.output_dir}")