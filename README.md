## Track workout with a camera

### Run

```bash
python -m app.main --auto
python -m app.main --mode pullup
python -m app.main --tts
python -m app.main --export --export-target homeassistant --export-destination dummy
```

### TTS Setup (Chatterbox)

```bash
python3.11 -m venv .venv_tts
source .venv_tts/bin/activate
pip install chatterbox-tts
```

### TTS Run

```bash
python -m app.main --auto --tts
python -m app.main --mode pullup --tts
```

### TTS Worker (Persistent)

```bash
python -m app.main --tts --tts-model turbo --video testVids/pull-ups-ex.MOV
python -m app.main --tts --tts-model turbo --tts-audio-prompt your_ref.wav --video testVids/pull-ups-ex.MOV
```
