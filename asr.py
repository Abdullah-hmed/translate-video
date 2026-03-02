from faster_whisper import WhisperModel
from utils import convert_to_wav
from translate import quick_translate, mtranslate
from tts import pocket_tts
import argparse
from pathlib import Path

model = WhisperModel("small", device="cuda")

def transcribe_audio(audio_path):

    output_audio_path = convert_to_wav(audio_path, 'temp_audio.wav')

    segments, info = model.transcribe(output_audio_path)
    print("Detected language:", info.language)

    # Concatenate all segments into a single string
    full_transcription = " ".join(segment.text for segment in segments)

    # Path('temp_audio.wav').unlink(missing_ok=True)

    return output_audio_path, full_transcription

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Transcribe audio from a video file")
    parser.add_argument("audio_path", help="Path to the audio or video file")
    args = parser.parse_args()

    output_audio_path, transcription = transcribe_audio(args.audio_path)
    print(f"Transcription: {transcription}")

    translated_transcription = quick_translate(transcription, target='en')

    print(f"Translated: {translated_transcription}")

    pocket_tts(output_audio_path, translated_transcription, filename="source/translated_output.wav")
    
    Path('temp_audio.wav').unlink(missing_ok=True)