import argparse
import os
import subprocess
import sys
import tempfile


def select_device(torch):
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def play_audio(path):
    if sys.platform == "darwin":
        return subprocess.run(["afplay", path]).returncode

    for cmd in ("aplay", "paplay"):
        if subprocess.call(["which", cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
            return subprocess.run([cmd, path]).returncode

    return 1


def main():
    parser = argparse.ArgumentParser(description="Chatterbox TTS worker")
    parser.add_argument("--model", choices=["chatterbox", "turbo"], default="chatterbox")
    parser.add_argument("--audio-prompt", type=str, default=None)
    args = parser.parse_args()

    try:
        import torch
        import torchaudio
    except Exception as exc:
        print(f"Failed to import torch/torchaudio: {exc}", file=sys.stderr)
        return 1

    device = select_device(torch)
    model = None
    use_turbo = False
    if args.model == "turbo":
        try:
            from chatterbox.tts_turbo import ChatterboxTurboTTS

            model = ChatterboxTurboTTS.from_pretrained(device=device)
            use_turbo = True
            def generate(text):
                return model.generate(text, audio_prompt_path=args.audio_prompt)
        except Exception as exc:
            print(f"Turbo model failed, falling back to chatterbox: {exc}", file=sys.stderr)
            model = None

    if model is None:
        from chatterbox.tts import ChatterboxTTS

        model = ChatterboxTTS.from_pretrained(device=device)
        use_turbo = False
        def generate(text):
            return model.generate(text)

    print("READY", flush=True)

    for line in sys.stdin:
        text = line.strip()
        if not text:
            continue
        if text == "__quit__":
            break
        try:
            wav = generate(text)
            sample_rate = getattr(model, "sr", getattr(model, "sample_rate", 24000))
            if wav.ndim == 3:
                wav = wav.squeeze(0)
            if wav.ndim == 1:
                wav = wav.unsqueeze(0)
            assert wav.ndim == 2
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            tmp_path = tmp.name
            tmp.close()
            torchaudio.save(tmp_path, wav.cpu(), sample_rate)
            play_audio(tmp_path)
        except Exception as exc:
            if use_turbo:
                print(f"Turbo generate failed, falling back to chatterbox: {exc}", file=sys.stderr)
                try:
                    from chatterbox.tts import ChatterboxTTS

                    model = ChatterboxTTS.from_pretrained(device=device)
                    use_turbo = False
                    wav = model.generate(text)
                    sample_rate = getattr(model, "sr", getattr(model, "sample_rate", 24000))
                    if wav.ndim == 3:
                        wav = wav.squeeze(0)
                    if wav.ndim == 1:
                        wav = wav.unsqueeze(0)
                    assert wav.ndim == 2
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                    tmp_path = tmp.name
                    tmp.close()
                    torchaudio.save(tmp_path, wav.cpu(), sample_rate)
                    play_audio(tmp_path)
                except Exception as exc2:
                    print(f"TTS failed: {exc2}", file=sys.stderr)
            else:
                print(f"TTS failed: {exc}", file=sys.stderr)
        finally:
            try:
                if "tmp_path" in locals() and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
