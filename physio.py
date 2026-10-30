"""Načítání dat z PhysioNet. Balík WFDB se používá VÝHRADNĚ ke čtení souborů.

Data lze číst dvěma způsoby:
  * přímo z internetu (data_dir=None, streamování přes pn_dir),
  * z lokální kopie ve struktuře  <data_dir>/<databáze>/<záznam>.hea
    (např. po spuštění download.py) – rychlejší a spolehlivější.

Síťové čtení se při výpadku spojení automaticky opakuje (_retry).
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import numpy as np
import requests
import wfdb

# Anotace, které označují skutečný srdeční tep (ostatní jsou rytmové/šumové popisky)
BEAT_SYMBOLS = set("NLRBAaJSVrFejnE/fQ?")

NETWORK_ERRORS = (requests.exceptions.RequestException, ConnectionError, TimeoutError)


def utf8_stdout() -> None:
    """Výpisy s diakritikou nespadnou ani na Windows při přesměrování výstupu do souboru/roury."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def _retry(func, *args, attempts: int = 6, **kwargs):
    """Zavolá func s opakováním při výpadku sítě (přerušené stahování apod.)."""
    for attempt in range(1, attempts + 1):
        try:
            return func(*args, **kwargs)
        except NETWORK_ERRORS as exc:
            if attempt == attempts:
                raise
            wait = 2 * attempt
            print(f"  síťová chyba ({type(exc).__name__}), pokus {attempt}/{attempts - 1} znovu za {wait} s ...")
            time.sleep(wait)


def _where(record: str, db: str, data_dir: str | None) -> dict:
    if data_dir:
        return {"record_name": str(Path(data_dir) / db / record)}
    return {"record_name": record, "pn_dir": db}


def _natural_key(name: str) -> list:
    """Klíč pro přirozené řazení (charis2 před charis10)."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def list_records(db: str, data_dir: str | None = None) -> list[str]:
    """Seznam názvů záznamů v databázi (přirozeně seřazený)."""
    if data_dir:
        names = [p.stem for p in (Path(data_dir) / db).glob("*.hea")]
    else:
        names = _retry(wfdb.get_record_list, db)
    return sorted(names, key=_natural_key)


def read_header(record: str, db: str, data_dir: str | None = None) -> wfdb.Record:
    """Hlavička jednosegmentového záznamu (víceúsekové záznamy nejsou podporovány)."""
    hdr = _retry(wfdb.rdheader, **_where(record, db, data_dir))
    if not isinstance(hdr, wfdb.Record):
        raise TypeError(f"Záznam {record} je víceúsekový, což tento kód nepodporuje.")
    return hdr


def read_signal(record: str, db: str, channel: str, data_dir: str | None = None,
                seconds: float | None = None) -> tuple[np.ndarray, float]:
    """Načte jeden kanál (podle části názvu, bez ohledu na velikost písmen).

    Vrací (signál ve fyzikálních jednotkách, skutečná vzorkovací frekvence kanálu).
    Chybějící vzorky jsou v signálu označeny NaN. Kanály s různým počtem vzorků
    na rámec (drivedb, charisdb) se čtou v tzv. expanded režimu.
    """
    where = _where(record, db, data_dir)
    hdr = read_header(record, db, data_dir)
    sig_names = list(hdr.sig_name or [])
    if not hdr.fs:
        raise ValueError(f"Hlavička {record} neobsahuje vzorkovací frekvenci.")
    try:
        idx = next(i for i, n in enumerate(sig_names) if channel.lower() in n.lower())
    except StopIteration:
        raise KeyError(f"Kanál '{channel}' nebyl v {record} nalezen (kanály: {sig_names})")
    sampto = int(seconds * hdr.fs) if seconds else None
    rec = _retry(wfdb.rdrecord, **where, channels=[idx], sampto=sampto, smooth_frames=False)
    if not isinstance(rec, wfdb.Record) or rec.e_p_signal is None:
        raise ValueError(f"Signál {record}/{channel} se nepodařilo načíst.")
    spf = (hdr.samps_per_frame or [1] * len(sig_names))[idx] or 1
    return np.asarray(rec.e_p_signal[0], dtype=float), float(hdr.fs * spf)


def read_beat_annotations(record: str, db: str, data_dir: str | None = None,
                          sampto: int | None = None, extension: str = "atr") -> np.ndarray:
    """Pozice (vzorky) referenčních anotací srdečních tepů."""
    ann = _retry(wfdb.rdann, **_where(record, db, data_dir), extension=extension, sampto=sampto)
    keep = np.array([s in BEAT_SYMBOLS for s in ann.symbol], dtype=bool)
    return np.asarray(ann.sample)[keep]


def read_comments(record: str, db: str, data_dir: str | None = None) -> list[str]:
    """Komentáře z hlavičkového souboru (u VOICED obsahují diagnózu)."""
    return list(read_header(record, db, data_dir).comments or [])