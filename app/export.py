from typing import Optional


class Exporter:
    def __init__(self, enabled: bool = False, target: Optional[str] = None, destination: Optional[str] = None):
        self.enabled = enabled
        self.target = target
        self.destination = destination

    def on_rep(self, exercise: str, rep_count: int, timestamp: float):
        return


class NoOpExporter(Exporter):
    def __init__(self):
        super().__init__(enabled=False)


class HomeAssistantExporterStub(Exporter):
    def on_rep(self, exercise: str, rep_count: int, timestamp: float):
        if not self.enabled:
            return
        payload = {
            "exercise": exercise,
            "rep_count": rep_count,
            "timestamp": timestamp,
            "destination": self.destination,
        }
        print(f"[EXPORT] would POST to Home Assistant: {payload}")


class MQTTExporterStub(Exporter):
    def on_rep(self, exercise: str, rep_count: int, timestamp: float):
        if not self.enabled:
            return
        payload = {
            "exercise": exercise,
            "rep_count": rep_count,
            "timestamp": timestamp,
            "destination": self.destination,
        }
        print(f"[EXPORT] would publish to MQTT: {payload}")
