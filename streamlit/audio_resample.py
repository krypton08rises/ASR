import os
import subprocess
import numpy as np
import torch
import torchaudio
import librosa
import soundfile as sf
from pydub import AudioSegment


def resample_audio(input_file, output_file, target_sr=16000, method='torchaudio'):
    """
    Load, resample, and save an audio file using the specified method.

    :param input_file: Path to the input audio file
    :param output_file: Path to save the resampled audio file
    :param target_sr: Target sampling rate (default: 16000)
    :param method: Resampling method ('torchaudio', 'ffmpeg', 'librosa', or 'pydub')
    :return: None
    """
    if method == 'torchaudio':
        # Load audio file
        waveform, sample_rate = torchaudio.load(input_file)

        # Resample if necessary
        if sample_rate != target_sr:
            # resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
            waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=target_sr)

            # waveform = resampler(waveform)

        # Save resampled audio
        torchaudio.save(output_file, waveform, target_sr)

    elif method == 'ffmpeg':
        # Use FFmpeg for resampling
        cmd = [
            'ffmpeg',
            '-i', input_file,
            '-ar', str(target_sr),
            '-ac', '1',  # Convert to mono
            '-y',  # Overwrite output file if it exists
            output_file
        ]
        subprocess.run(cmd, check=True)

    elif method == 'librosa':

        # Load audio file
        y, sr = librosa.load(input_file, sr=None)

        # Resample if necessary
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

        # Save resampled audio
        sf.write(output_file, y, target_sr)

    elif method == 'pydub':
        # Load audio file
        audio = AudioSegment.from_file(input_file)

        # Resample and export
        audio = audio.set_frame_rate(target_sr).set_channels(1)
        audio.export(output_file, format=os.path.splitext(output_file)[1][1:])

    else:
        raise ValueError("Unsupported resampling method. Choose 'torchaudio', 'ffmpeg', 'librosa', or 'pydub'.")

    print(f"Audio resampled and saved to {output_file}")


# Example usage
if __name__ == "__main__":
    methods = ["torchaudio", "pydub", "librosa", "ffmpeg"]
    input_file = r"D:\Projects\Datasets\3langtest.wav"
    for meth in methods:
        output_file = r"D:\Projects\Datasets\3langtest_resampled{method}_audio.wav".format(method=meth)
        resample_audio(input_file, output_file, target_sr=16000, method=meth)