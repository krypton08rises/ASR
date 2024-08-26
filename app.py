import torch
import pickle
import streamlit as st

from io import BytesIO
from config import SAMPLING_RATE
from model_utils import load_models
from audio_processing import detect_language, process_long_audio, load_and_resample_audio

# Clear GPU cache
torch.cuda.empty_cache()

# Load models at startup
load_models()

# Title of the app
st.title("Audio Player with Live Transcription and Translation")

# Sidebar for file uploader and submit button
st.sidebar.header("Upload Audio Files")
uploaded_files = st.sidebar.file_uploader("Choose audio files", type=["mp3", "wav"], accept_multiple_files=True)
submit_button = st.sidebar.button("Submit")

# Session state to hold data
if 'audio_files' not in st.session_state:
    st.session_state.audio_files = []
    st.session_state.transcriptions = {}
    st.session_state.translations = {}
    st.session_state.detected_languages = []
    st.session_state.waveforms = []

# Process uploaded files
if submit_button and uploaded_files is not None:
    st.session_state.audio_files = uploaded_files
    st.session_state.detected_languages = []
    st.session_state.waveforms = []

    for uploaded_file in uploaded_files:
        waveform = load_and_resample_audio(BytesIO(uploaded_file.read()))
        st.session_state.waveforms.append(waveform)
        detected_language = detect_language(waveform)
        st.session_state.detected_languages.append(detected_language)

# Display uploaded files and options
if 'audio_files' in st.session_state and st.session_state.audio_files:
    for i, uploaded_file in enumerate(st.session_state.audio_files):
        st.write(f"**File name**: {uploaded_file.name}")
        st.audio(uploaded_file, format=uploaded_file.type)
        st.write(f"**Detected Language**: {st.session_state.detected_languages[i]}")

        col1, col2 = st.columns(2)

        with col1:
            if st.button(f"Transcribe {uploaded_file.name}"):
                with st.spinner("Transcribing..."):
                    transcription = process_long_audio(st.session_state.waveforms[i], SAMPLING_RATE)
                    st.session_state.transcriptions[i] = transcription

            if st.session_state.transcriptions.get(i):
                st.write("**Transcription**:")
                st.text_area("", st.session_state.transcriptions[i], height=200, key=f"transcription_{i}")
                st.markdown(f'<div style="text-align: right;"><a href="data:text/plain;charset=UTF-8,{st.session_state.transcriptions[i]}" download="transcription_{uploaded_file.name}.txt">Download Transcription</a></div>', unsafe_allow_html=True)

        with col2:
            if st.button(f"Translate {uploaded_file.name}"):
                with st.spinner("Translating..."):
                    with open('languages.pkl', 'rb') as f:
                        lang_dict = pickle.load(f)
                    detected_language_name = lang_dict[st.session_state.detected_languages[i]]

                    translation = process_long_audio(st.session_state.waveforms[i], SAMPLING_RATE, task="translate",
                                                     language=detected_language_name)
                    st.session_state.translations[i] = translation
            if st.session_state.translations.get(i):
                st.write("**Translation**:")
                st.text_area("", st.session_state.translations[i], height=200, key=f"translation_{i}")
                st.markdown(f'<div style="text-align: right;"><a href="data:text/plain;charset=UTF-8,{st.session_state.translations[i]}" download="translation_{uploaded_file.name}.txt">Download Translation</a></div>', unsafe_allow_html=True)