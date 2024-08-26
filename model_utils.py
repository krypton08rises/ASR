import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import whisper
from config import WHISPER_MODEL_SIZE

# Global variables to store models
whisper_processor = None
whisper_model = None
whisper_model_small = None

def load_models():
    global whisper_processor, whisper_model, whisper_model_small
    if whisper_processor is None:
        whisper_processor = WhisperProcessor.from_pretrained(f"openai/whisper-{WHISPER_MODEL_SIZE}")
    if whisper_model is None:
        whisper_model = WhisperForConditionalGeneration.from_pretrained(f"openai/whisper-{WHISPER_MODEL_SIZE}").to(get_device())
    if whisper_model_small is None:
        whisper_model_small = whisper.load_model(WHISPER_MODEL_SIZE)

def get_device():
    return "cuda:0" if torch.cuda.is_available() else "cpu"

def get_processor():
    global whisper_processor
    if whisper_processor is None:
        load_models()
    return whisper_processor

def get_model():
    global whisper_model
    if whisper_model is None:
        load_models()
    return whisper_model

def get_whisper_model_small():
    global whisper_model_small
    if whisper_model_small is None:
        load_models()
    return whisper_model_small