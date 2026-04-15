import json
import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.tenant_service import TenantConfig

_LEADS_ROOT = Path(__file__).resolve().parent.parent.parent


def _leads_file(tenant_id: str = "default") -> Path:
    if tenant_id == "default":
        return _LEADS_ROOT / "leads.json"
    return _LEADS_ROOT / f"leads_{tenant_id}.json"


class DBService:

    def save(
        self,
        lead: dict,
        session_id: str = None,
        tenant_id: str = "default",
    ) -> None:
        entry = {
            **lead,
            "session_id": session_id,
            "tenant_id":  tenant_id,
            "time":       datetime.datetime.now().isoformat(),
        }
        with open(_leads_file(tenant_id), "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def get_all(self, tenant_id: str = "default") -> list:
        leads = []
        try:
            with open(_leads_file(tenant_id), "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        leads.append(json.loads(line))
        except FileNotFoundError:
            pass
        return leads
