from __future__ import annotations

import json
import logging
import os
import re

from config import Config, get_config

log = logging.getLogger(__name__)

_DEFAULT_ABBREVS = {
    "chq": "cheque",
    "txn": "transaction",
    "amt": "amount",
    "bal": "balance",
    "acct": "account",
    "po": "purchase order",
    "grn": "goods received note",
    "inv": "invoice",
    "pmt": "payment",
    "dept": "department",
    "mgr": "manager",
    "qty": "quantity",
    "yoy": "year over year",
    "mom": "month over month",
    "kpi": "key performance indicator",
    "var": "variance",
}

_STOP_NOISE = {"please", "kindly", "could", "would", "can", "tell", "me", "show", "give", "about"}


class QueryPreprocessor:
    def __init__(self, config: Config | None = None):
        self.config = config or get_config()
        self._abbrevs = self._load_abbrevs()

    def _load_abbrevs(self) -> dict[str, str]:
        abbrev_path = os.path.join(os.path.dirname(__file__), "abbrev_map.json")
        if os.path.exists(abbrev_path):
            try:
                with open(abbrev_path, encoding="utf-8") as f:
                    return {**_DEFAULT_ABBREVS, **json.load(f)}
            except Exception as exc:
                log.warning("Could not load abbrev_map.json: %s", exc)
        return _DEFAULT_ABBREVS

    def preprocess(self, query: str) -> str:
        if not self.config.use_query_preprocessing:
            return query
        query = self._normalize_whitespace(query)
        query = self._lowercase(query)
        query = self._expand_abbreviations(query)
        query = self._remove_stop_noise(query)
        return query.strip()

    def _normalize_whitespace(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def _lowercase(self, text: str) -> str:
        return text.lower()

    def _expand_abbreviations(self, text: str) -> str:
        words = text.split()
        return " ".join(self._abbrevs.get(w, w) for w in words)

    def _remove_stop_noise(self, text: str) -> str:
        words = text.split()
        cleaned = [w for w in words if w not in _STOP_NOISE]
        return " ".join(cleaned) if cleaned else text
