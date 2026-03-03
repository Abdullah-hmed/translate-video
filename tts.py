from pocket_tts import TTSModel
import scipy.io.wavfile
import os
import soundfile as sf
import librosa
import argparse

def trim_audio(input_path, output_path="source/trimmed_ref.wav", max_duration=12):
    audio, sr = librosa.load(input_path, sr=None)
    trimmed = audio[:int(max_duration * sr)]
    sf.write(output_path, trimmed, sr)
    return output_path


def pocket_tts(input_audio, text, filename="pocket_output.wav"):    
    tts_model = TTSModel.load_model()

    trimmed_ref = trim_audio(input_audio)

    voice_state = tts_model.get_state_for_audio_prompt(trimmed_ref)
    audio = tts_model.generate_audio(voice_state, text)
    # Audio is a 1D torch tensor containing PCM data.
    scipy.io.wavfile.write(filename, tts_model.sample_rate, audio.numpy())
    print(f"TTS generation completed. Output saved as {filename}")
    return filename

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate speech using Pocket TTS")
    parser.add_argument("--text", type=str, default="Hello! This is an audio generated using Pocket TTS, saved as an MP3 file.")
    args = parser.parse_args()

    filename = pocket_tts("source/temp_audio.wav", args.text, filename="source/pocket_output.wav")

    print(f"TTS generation completed. Output saved as {filename}")
    os.system(f"start {filename}")