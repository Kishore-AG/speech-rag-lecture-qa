"""
Automatic Speech Recognition Module
Transcribes audio files to text using OpenAI's Whisper model.
Supports multiple audio formats: .wav, .mp3, .m4a, .flac, .ogg
"""

import os
import whisper
from config import ASR_CONFIG, AUDIO_DIR, TRANSCRIPT_DIR

def transcribe_audio():
    """
    Transcribe all supported audio files in AUDIO_DIR to text.
    Creates a text file for each audio file in TRANSCRIPT_DIR.
    Skips files that already have corresponding transcripts.
    """
    print(f"Loading Whisper model: {ASR_CONFIG['model']}...")
    model = whisper.load_model(ASR_CONFIG["model"], device=ASR_CONFIG["device"])
    print(f"Model loaded. Supported audio formats: {', '.join(ASR_CONFIG['supported_formats'])}\n")

    audio_files = [
        f for f in os.listdir(AUDIO_DIR)
        if any(f.lower().endswith(fmt) for fmt in ASR_CONFIG["supported_formats"])
    ]

    if not audio_files:
        print(f"No audio files found in {AUDIO_DIR}")
        return

    print(f"Found {len(audio_files)} audio file(s) to transcribe\n")

    for i, file in enumerate(audio_files, 1):
        audio_path = os.path.join(AUDIO_DIR, file)
        
        # Generate output filename (preserve original name, change extension to .txt)
        base_name = os.path.splitext(file)[0]
        output_file = os.path.join(TRANSCRIPT_DIR, f"{base_name}.txt")

        # Skip if transcript already exists
        if os.path.exists(output_file):
            print(f"[{i}/{len(audio_files)}] ✓ Transcript exists (skipped): {file}")
            continue

        print(f"[{i}/{len(audio_files)}] Transcribing: {file}")

        try:
            result = model.transcribe(
                audio_path,
                language=ASR_CONFIG["language"]  # None for auto-detect
            )

            transcript = result["text"]

            # Save transcript
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(transcript)

            print(f"  ✓ Saved: {output_file}\n")

        except Exception as e:
            print(f"  ✗ Error transcribing {file}: {str(e)}\n")

    print("Transcription complete.")


if __name__ == "__main__":
    transcribe_audio()