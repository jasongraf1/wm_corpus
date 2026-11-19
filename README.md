# Multimodal corpus of West Midlands English

Workflow and code for processing of West Midlands Corpus data.

## Basic workflow

### Audio extraction, transcription and conversion

1. Extract the audio (native-rate and mono 16kHz WAV) using `ffmpeg`
2. Diarise the audio by speaker using `pyannote`
3. Transcribe with `faster whisper`
4. Merge diarised transcriptions
5. Generate multitiered `.eaf` file for annotation in ELAN

### Manual correction

Automated transcriptions must be manually checked and edited in ELAN.

### Alignment and segmentation

6. Create a custom dictionary for West Midlands
7. Generate transcriptions:
    a. Create MFA corpus of phonetically segmented `.TextGrid` files for each recording
    b. Create a plain text transcription for each recording

### Feature extraction

8. Extract vowel formants and normalise

