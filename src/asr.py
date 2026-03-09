import os
import whisper

AUDIO_DIR = "data/audio"
TRANSCRIPT_DIR = "data/transcripts"

os.makedirs(TRANSCRIPT_DIR, exist_ok=True)

def transcribe_audio():
    model = whisper.load_model("base")

    for file in os.listdir(AUDIO_DIR):
        if file.endswith(".wav"):
            audio_path = os.path.join(AUDIO_DIR, file)
            print(f"Transcribing: {file}")

            result = model.transcribe(audio_path)

            transcript = result["text"]

            output_file = os.path.join(
                TRANSCRIPT_DIR,
                file.replace(".wav", ".txt")
            )

            with open(output_file, "w", encoding="utf-8") as f:
                f.write(transcript)

            print(f"Saved transcript: {output_file}")


if __name__ == "__main__":
    transcribe_audio()