class RepAnnouncer:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled

    def on_rep(self, exercise: str, rep_count: int):
        if not self.enabled:
            return
        print(f"[TTS] would say: {exercise.capitalize()} {rep_count}")
