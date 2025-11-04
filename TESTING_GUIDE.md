# Testing Guide

This guide explains how to test the audio transcription and translation script.

## Prerequisites

### 1. Install Python Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install transformers torch librosa soundfile pyannote.audio pytest numpy
```

### 2. Hugging Face Access Token (Optional but Recommended)

The pyannote speaker diarization model may require authentication. Get your token:

1. Go to https://huggingface.co/settings/tokens
2. Create a new access token (read access is sufficient)
3. Set it as an environment variable:

```bash
export HUGGINGFACE_TOKEN=hf_your_token_here
```

Or on Windows:
```cmd
set HUGGINGFACE_TOKEN=hf_your_token_here
```

### 3. Audio File

Place your audio file named `call_recording.mp3` in the project root directory.

**Note:** The script expects MP3 format, but supports most common audio formats (WAV, FLAC, etc.) through librosa.

## Testing Methods

### Method 1: Run Unit Tests (Fast, No Audio File Needed)

Unit tests verify the core logic without requiring an actual audio file or downloading models:

```bash
pytest
```

Run with verbose output:
```bash
pytest -v
```

Run a specific test:
```bash
pytest tests/test_transcribe_translate.py::test_map_speakers_to_roles_handles_additional_speakers
```

### Method 2: Test the Main Script (Full Integration Test)

This requires a real audio file and will download models on first run:

```bash
python transcribe_translate.py
```

**Expected Output:**
1. Loading progress messages
2. Detected language
3. Translation progress
4. Speaker diarization progress
5. Final output with role tags:
   ```
   ==== Conversation (English Translation) ====
   [Customer] Hello, I need help with my order.
   [Customer Rep] Sure, I can help you with that.
   ...
   
   ==== Full English Translation (Unsegmented) ====
   Hello, I need help with my order. Sure, I can help you with that...
   ```

## System Requirements

- **Python:** 3.8 or higher
- **RAM:** At least 8GB recommended (models can use 4-6GB)
- **GPU:** Optional but recommended for faster processing (CUDA-compatible)
- **Disk Space:** ~3-5GB for model downloads (first run only)

## Troubleshooting

### Issue: "No module named 'transformers'"
**Solution:** Install dependencies: `pip install -r requirements.txt`

### Issue: "Audio file 'call_recording.mp3' not found"
**Solution:** Ensure the audio file exists in the same directory as the script

### Issue: "No transcription chunks were produced"
**Solution:** 
- Check audio file is valid and contains speech
- Ensure audio is not corrupted
- Try a different audio format (WAV, FLAC)

### Issue: "Speaker diarization returned no segments"
**Solution:**
- Check if audio has multiple speakers
- Verify Hugging Face token is set correctly
- Try a longer audio file (at least 10-15 seconds)

### Issue: Model download errors
**Solution:**
- Check internet connection
- Verify Hugging Face token if required
- Clear cache: `rm -rf ~/.cache/huggingface/`

### Issue: CUDA out of memory
**Solution:**
- Use CPU mode (script auto-detects)
- Or reduce batch size in pipeline configuration
- Use a smaller Whisper model (e.g., `whisper-medium` instead of `whisper-large-v3`)

## Testing Checklist

- [ ] Dependencies installed
- [ ] Hugging Face token set (if needed)
- [ ] Audio file `call_recording.mp3` in project root
- [ ] Unit tests pass: `pytest`
- [ ] Main script runs successfully: `python transcribe_translate.py`
- [ ] Output shows role tags [Customer] and [Customer Rep]
- [ ] Output shows English translation

## Quick Test Commands

```bash
# Install everything
pip install -r requirements.txt

# Set token (Linux/Mac)
export HUGGINGFACE_TOKEN=your_token_here

# Run unit tests
pytest -v

# Run main script
python transcribe_translate.py
```
