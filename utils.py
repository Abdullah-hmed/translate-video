import subprocess
import wave

def convert_to_wav(input_audio_path, output_audio_path):
    
    command = ['ffmpeg', '-y', '-i', input_audio_path, output_audio_path]

    print("Converting audio to WAV format...")

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        print("FFmpeg failed!")
        print("Error output:")
        print(result.stderr)
        raise RuntimeError(f"FFmpeg conversion failed: {result.stderr}")
    else:
        print("Conversion successful!")
    return output_audio_path

def image_to_video(image_path, output_video_path, duration=5, fps=25):
    command = [
        'ffmpeg', '-y', '-loop', '1', '-i', image_path,
        '-c:v', 'libx264', '-t', str(duration), '-pix_fmt', 'yuv420p',
        '-vf', f'fps={fps}', output_video_path
    ]

    print("Creating video from image...")

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        print("FFmpeg failed!")
        print("Error output:")
        print(result.stderr)
        raise RuntimeError(f"FFmpeg image to video conversion failed: {result.stderr}")
    else:
        print("Video creation successful!")
    return output_video_path

def calc_audio_length(audio_path):
    with wave.open(audio_path, 'rb') as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        duration = frames / float(rate)
    return duration
