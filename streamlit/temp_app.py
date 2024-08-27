import streamlit as st
import pickle
from io import BytesIO
import pyperclip
from audio_processing import detect_language, process_long_audio, load_and_resample_audio
from model_utils import load_models
from config import SAMPLING_RATE
# from llm_utils import generate_answer, summarize_transcript

# Load models at startup
load_models()

# Title of the app
st.title("Audio Player with Live Transcription and Q&A")

# ... (previous code remains the same)

def copy_to_clipboard(text):
    pyperclip.copy(text)
    st.success("Copied to clipboard!")

# Display uploaded files and options
if 'audio_files' in st.session_state and st.session_state.audio_files:
    for i, uploaded_file in enumerate(st.session_state.audio_files):
        col1, col2 = st.columns([1, 3])

        with col1:
            st.write(f"**File name**: {uploaded_file.name}")
            st.audio(uploaded_file, format=uploaded_file.type)
            st.write(f"**Detected Language**: {st.session_state.detected_languages[i]}")

        with col2:
            if st.button(f"Transcribe {uploaded_file.name}"):
                with st.spinner("Transcribing..."):
                    transcription = process_long_audio(st.session_state.waveforms[i], SAMPLING_RATE)
                    st.session_state.transcriptions[i] = transcription

            if st.session_state.transcriptions.get(i):
                st.write("**Transcription**:")
                st.write(st.session_state.transcriptions[i])
                if st.button("Copy Transcription", key=f"copy_transcription_{i}"):
                    copy_to_clipboard(st.session_state.transcriptions[i])

                # ... (summarization and Q&A code remains the same)

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
                st.write(st.session_state.translations[i])
                if st.button("Copy Translation", key=f"copy_translation_{i}"):
                    copy_to_clipboard(st.session_state.translations[i])