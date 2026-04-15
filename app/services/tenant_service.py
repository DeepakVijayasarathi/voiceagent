"""
Tenant registry — manages per-tenant company config and knowledge base.

Each tenant has:
  • company info  (name, tagline, agent_name, services, location)
  • llm settings  (model, temperature, max_tokens)
  • an isolated KnowledgeBase instance
  • raw PDF text (stored so the KB can be rebuilt if needed)
  • a dedicated leads file (leads_<tenant_id>.json, or leads.json for "default")

agent.yaml structure (new multi-tenant format, fully backward-compatible):

    agent:              # default tenant — existing single-tenant config unchanged
      company: ...
      llm: ...

    tenants:            # optional — add extra tenants here
      acme_corp:
        company:
          name: "Acme Corp"
          tagline: "Your tech partner"
          agent_name: "Arun"
          services: [laptops, desktops, accessories]
          location: "Chennai"
        llm:
          model: gpt-4o
          temperature: 0.6
          max_tokens: 280
      xyz_mobiles:
        company:
          name: "XYZ Mobiles"
          tagline: "Best phones in town"
          agent_name: "Divya"
          services: [mobile phones, accessories, repairs]
          location: "Coimbatore"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from app.services.knowledge_service import KnowledgeBase

log = logging.getLogger(__name__)

_DEFAULT_TENANT = "default"
_LEADS_ROOT     = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# TenantConfig — one instance per tenant
# ---------------------------------------------------------------------------

@dataclass
class TenantConfig:
    tenant_id:      str
    company:        dict                                        # mutable — updated on PDF upload
    llm:            dict                                        # model, temperature, max_tokens
    knowledge_base: KnowledgeBase = field(default_factory=KnowledgeBase)
    pdf_text:       str           = ""                         # raw text from last uploaded PDF
    pdf_loaded:     bool          = False

    @property
    def leads_file(self) -> Path:
        """Path to this tenant's leads file."""
        if self.tenant_id == _DEFAULT_TENANT:
            return _LEADS_ROOT / "leads.json"
        return _LEADS_ROOT / f"leads_{self.tenant_id}.json"


# ---------------------------------------------------------------------------
# TenantRegistry — singleton, created once in main.py
# ---------------------------------------------------------------------------

class TenantRegistry:
    """
    Reads agent.yaml and exposes per-tenant TenantConfig objects.
    Tenants are created lazily for unknown IDs (cloned from the default tenant).
    """

    def __init__(self, yaml_config: dict) -> None:
        self._tenants: dict[str, TenantConfig] = {}

        # ── Default tenant from top-level "agent:" key (backward-compatible) ──
        agent_cfg = yaml_config.get("agent", {})
        self._tenants[_DEFAULT_TENANT] = self._build(_DEFAULT_TENANT, agent_cfg)

        # ── Additional tenants from "tenants:" key ────────────────────────────
        for tid, tcfg in yaml_config.get("tenants", {}).items():
            merged = self._merge_with_default(agent_cfg, tcfg)
            self._tenants[tid] = self._build(tid, merged)
            log.info("TenantRegistry: registered tenant '%s'", tid)

        log.info(
            "TenantRegistry: %d tenant(s) loaded: %s",
            len(self._tenants),
            list(self._tenants.keys()),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_with_default(default_cfg: dict, tenant_cfg: dict) -> dict:
        """
        Tenant config inherits LLM settings from the default agent config
        unless explicitly overridden.
        """
        merged = dict(default_cfg)
        merged["company"] = tenant_cfg.get("company", default_cfg.get("company", {}))
        if "llm" in tenant_cfg:
            merged["llm"] = {**default_cfg.get("llm", {}), **tenant_cfg["llm"]}
        return merged

    @staticmethod
    def _build(tenant_id: str, cfg: dict) -> TenantConfig:
        company_cfg  = cfg.get("company", {})
        services_raw = company_cfg.get("services", [])
        services_str = (
            ", ".join(services_raw)
            if isinstance(services_raw, list)
            else str(services_raw)
        )
        company = {
            "name":       company_cfg.get("name",       "Sales Agent"),
            "tagline":    company_cfg.get("tagline",    ""),
            "agent_name": company_cfg.get("agent_name", "Priya"),
            "services":   services_str,
            "location":   company_cfg.get("location",   ""),
        }
        llm = dict(cfg.get("llm", {}))
        return TenantConfig(tenant_id=tenant_id, company=company, llm=llm)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, tenant_id: str) -> TenantConfig:
        """
        Return the TenantConfig for tenant_id.
        If the tenant is unknown, auto-create it by cloning the default config.
        """
        if tenant_id not in self._tenants:
            default = self._tenants[_DEFAULT_TENANT]
            self._tenants[tenant_id] = TenantConfig(
                tenant_id=tenant_id,
                company=dict(default.company),
                llm=dict(default.llm),
            )
            log.info("TenantRegistry: auto-created tenant '%s' (cloned from default)", tenant_id)
        return self._tenants[tenant_id]

    def update_company(self, tenant_id: str, info: dict) -> None:
        """Merge extracted company info into a tenant's config (called after PDF upload)."""
        t = self.get(tenant_id)
        for key in ("name", "tagline", "agent_name", "services", "location"):
            val = info.get(key, "")
            if isinstance(val, str):
                val = val.strip()
            if val:
                t.company[key] = val
        t.pdf_loaded = True
        log.info(
            "TenantRegistry: company updated for tenant '%s': %s",
            tenant_id, t.company,
        )

    def list_tenants(self) -> list[str]:
        return list(self._tenants.keys())
