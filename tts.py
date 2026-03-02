from pocket_tts import TTSModel
import scipy.io.wavfile

def pocket_tts(input_audio, text, filename="pocket_output.wav"):    
    tts_model = TTSModel.load_model()
    voice_state = tts_model.get_state_for_audio_prompt(input_audio)
    audio = tts_model.generate_audio(voice_state, text)
    # Audio is a 1D torch tensor containing PCM data.
    scipy.io.wavfile.write(filename, tts_model.sample_rate, audio.numpy())
    print(f"TTS generation completed. Output saved as {filename}")
    return filename

if __name__ == "__main__":
    text = "Hello! This is Microsoft Edge TTS, saved as an MP3 file."
    filename = pocket_tts("source/temp_audio.wav", "Hello, this is me, Donald Trump, I have accepted Islam and I am now a Muslim, Alhamdulillah!")

    print(f"TTS generation completed. Output saved as {filename}")