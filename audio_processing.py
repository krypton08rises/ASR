import torch
import whisper
import torchaudio as ta
from model_utils import get_processor, get_model, get_whisper_model_small, get_device
from config import SAMPLING_RATE, CHUNK_LENGTH_S

def detect_language(audio_file):
    whisper_model = get_whisper_model_small()
    trimmed_audio = whisper.pad_or_trim(audio_file.squeeze())
    mel = whisper.log_mel_spectrogram(trimmed_audio).to(whisper_model.device)
    _, probs = whisper_model.detect_language(mel)
    detected_lang = max(probs[0], key=probs[0].get)
    print(f"Detected language: {detected_lang}")
    return detected_lang

def process_long_audio(waveform, sampling_rate, task="transcribe", language=None):
    processor = get_processor()
    model = get_model()
    device = get_device()

    input_length = waveform.shape[1]
    chunk_length = int(CHUNK_LENGTH_S * sampling_rate)
    chunks = [waveform[:, i:i + chunk_length] for i in range(0, input_length, chunk_length)]

    results = []
    for chunk in chunks:
        input_features = processor(chunk[0], sampling_rate=sampling_rate, return_tensors="pt").input_features.to(device)

        with torch.no_grad():
            if task == "translate":
                forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task="translate")
                generated_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
            else:
                generated_ids = model.generate(input_features)

        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)
        results.extend(transcription)

        # Clear GPU cache
        torch.cuda.empty_cache()

    return " ".join(results)

def load_and_resample_audio(file):
    waveform, sampling_rate = ta.load(file)
    if sampling_rate != SAMPLING_RATE:
        waveform = ta.functional.resample(waveform, orig_freq=sampling_rate, new_freq=SAMPLING_RATE)
    return waveform