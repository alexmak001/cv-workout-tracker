import argparse
import os
import subprocess
import sys
import tempfile
import wave


def select_device(torch):
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def write_wav(path, wav_data, sample_rate):
    try:
        import numpy as np
    except Exception:
        np = None

    if np is not None and hasattr(wav_data, "dtype"):
        audio = np.asarray(wav_data).flatten()
        audio = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio * 32767.0).astype(np.int16)
        data = audio_int16.tobytes()
    else:
        audio = list(wav_data)
        audio = [max(-1.0, min(1.0, float(x))) for x in audio]
        data = b"".join(int(x * 32767.0).to_bytes(2, "little", signed=True) for x in audio)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(data)


def play_audio(path):
    if sys.platform == "darwin":
        return subprocess.call(["afplay", path])

    for cmd in ("aplay", "paplay"):
        if subprocess.call(["which", cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
            return subprocess.call([cmd, path])

    return 1


def main():
    parser = argparse.ArgumentParser(description="Chatterbox TTS speaker")
    parser.add_argument("text", nargs="?", help="Text to speak")
    parser.add_argument("--text", dest="text_kw", help="Text to speak")
    args = parser.parse_args()

    text = args.text_kw or args.text
    if not text:
        print("No text provided.", file=sys.stderr)
        return 2

    try:
        import torch
        from chatterbox.tts import ChatterboxTTS
    except Exception as exc:
        print(f"Failed to import chatterbox/torch: {exc}", file=sys.stderr)
        return 1

    device = select_device(torch)
    try:
        model = ChatterboxTTS.from_pretrained(device=device)
        wav = model.generate(text)
        sample_rate = getattr(model, "sample_rate", 24000)

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp_path = tmp.name
        tmp.close()

        write_wav(tmp_path, wav, sample_rate)
        result = play_audio(tmp_path)
    except Exception as exc:
        print(f"TTS failed: {exc}", file=sys.stderr)
        return 1
    finally:
        try:
            if "tmp_path" in locals() and os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except Exception:
            pass

    return 0 if result == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
