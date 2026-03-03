from faster_whisper import WhisperModel
from utils import convert_to_wav
from tts import pocket_tts
import argparse, os
from pathlib import Path

model = WhisperModel("medium", device="cuda")

def transcribe_audio(audio_path):

    output_audio_path = convert_to_wav(audio_path, 'temp_audio.wav')

    segments, info = model.transcribe(output_audio_path, task="translate")
    print("Detected language:", info.language)

    # Concatenate all segments into a single string
    full_transcription = " ".join(segment.text for segment in segments)

    return output_audio_path, full_transcription

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Transcribe audio from a video file")
    parser.add_argument("audio_path", help="Path to the audio or video file")
    args = parser.parse_args()

    output_audio_path, transcription = transcribe_audio(args.audio_path)
    print(f"{transcription}")

    pocket_tts(output_audio_path, transcription, filename="source/translated_output.wav")
    
    Path('temp_audio.wav').unlink(missing_ok=True)

    os.system(f"start source/translated_output.wav")