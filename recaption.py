from lipsync import LipSync
from pathlib import Path
import os
from gooey import Gooey, GooeyParser
from utils import convert_to_wav, image_to_video, calc_audio_length


def main(video_path, audio_path):
    os.makedirs('cache', exist_ok=True)

    # If the audio is not in WAV format, convert it
    if Path(audio_path).suffix.lower() != '.wav':
        convert_to_wav(audio_path, 'cache/converted_audio.wav')
        audio_path = 'cache/converted_audio.wav'

    # If the video is an image, convert it to a video
    if Path(video_path).suffix.lower() in ['.jpg', '.jpeg', '.png', '.jfif', '.webp']:
        audio_length = round(float(calc_audio_length(audio_path)))
        video_path = image_to_video(video_path, 'cache/temp_video.mp4', duration=audio_length, fps=25)

    os.makedirs('results', exist_ok=True)

    result_name = f'results/{Path(video_path).stem}_{Path(audio_path).stem}.mp4'

    print("Processing required video, will be saved as:", result_name)

    lip = LipSync(
        model='wav2lip',
        checkpoint_path='weights/wav2lip_gan.pth',
        nosmooth=True,
        device='cuda',
        cache_dir='cache',
        img_size=96,
        save_cache=True,
    )

    lip.sync(video_path, audio_path, result_name)

    # Clean up temporary files
    Path('cache/converted_audio.wav').unlink(missing_ok=True)
    Path('cache/temp_video.mp4').unlink(missing_ok=True)
    Path('cache/temp_video.mp4.pk').unlink(missing_ok=True)

    print("Lipsync process completed. Result saved at:", result_name)

@Gooey(program_name="Lipsync App", default_size=(700, 500))
def run():
    parser = GooeyParser(description="Lipsync helper script.")

    parser.add_argument(
        'video_path',
        metavar='Video Path',
        widget='FileChooser'
    )

    parser.add_argument(
        'audio_path',
        metavar='Audio Path',
        widget='FileChooser'
    )

    args = parser.parse_args()
    main(args.video_path, args.audio_path)


if __name__ == "__main__":
    run()