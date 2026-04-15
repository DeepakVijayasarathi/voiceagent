import json
import datetime
from pathlib import Path

_LEADS_FILE = Path(__file__).resolve().parent.parent.parent / "leads.json"


class DBService:

    def save(self, lead: dict, session_id: str = None):
        entry = {
            **lead,
            "session_id": session_id,
            "time": datetime.datetime.now().isoformat(),
        }
        with open(_LEADS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def get_all(self) -> list:
        leads = []
        try:
            with open(_LEADS_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        leads.append(json.loads(line))
        except FileNotFoundError:
            pass
        return leads
