from faster_whisper import WhisperModel
from utils import convert_to_wav
from translate import quick_translate
import argparse
from pathlib import Path

model = WhisperModel("small", device="cuda")

def transcribe_audio(audio_path):

    output_audio_path = convert_to_wav(audio_path, 'temp_audio.wav')

    segments, info = model.transcribe(output_audio_path)
    print("Detected language:", info.language)

    # Concatenate all segments into a single string
    full_transcription = " ".join(segment.text for segment in segments)

    Path('temp_audio.wav').unlink(missing_ok=True)

    return full_transcription

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Transcribe audio from a video file")
    parser.add_argument("audio_path", help="Path to the audio or video file")
    args = parser.parse_args()

    transcription = transcribe_audio(args.audio_path)
    print(f"Transcription: {transcription}")

    print(f"Translated: {quick_translate(transcription, target='en')}")