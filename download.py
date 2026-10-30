"""Stažení databází PhysioNet do lokální složky (paralelně, odolné proti výpadkům sítě).

Příklad:  python download.py drivedb voiced nsrdb charisdb
Poté spouštějte skripty s  --data-dir data  (o řád rychlejší než streamování, kde se každý
záznam čte po kouscích s velkou latencí serveru).

* Soubory se stahují souběžně (--workers), latence serveru (~5 s na požadavek) se tak překrývá.
* U velkých databází (nsrdb, charisdb: jednotky GB) se z .dat stáhne jen úvodní část, kterou
  skripty potřebují (HTTP Range, --minutes). Hlavička zůstává původní, WFDB čte jen do sampto.
* Stahování lze kdykoli přerušit a spustit znovu, už stažené soubory se přeskočí
  (soubor se zapisuje přes dočasné jméno, takže nedokončený se nepovažuje za stažený).
"""
import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import wfdb

import physio

BASE = "https://physionet.org/files/{db}/1.0.0/{file}"
EXTRA = {"nsrdb": [".atr"]}
# kolik minut od začátku záznamu stáhnout (None = celý soubor); skripty potřebují 30 min nsrdb a 60 min charisdb
DEFAULT_MINUTES = {"nsrdb": 35, "charisdb": 61}
BYTES_PER_SAMPLE = {"16": 2, "212": 1.5, "32": 4, "8": 1, "80": 1}


def fetch(url: str, path: Path, nbytes: int | None = None, attempts: int = 6) -> str:
    if path.exists():
        return "přeskočeno"
    headers = {"Range": f"bytes=0-{nbytes - 1}"} if nbytes else {}
    for attempt in range(1, attempts + 1):
        try:
            r = requests.get(url, headers=headers, timeout=120)
            r.raise_for_status()
            tmp = path.with_suffix(path.suffix + ".part")
            tmp.write_bytes(r.content[:nbytes] if nbytes else r.content)
            tmp.replace(path)
            return f"{len(r.content) / 1e6:.1f} MB"
        except Exception as exc:  # síťové chyby mají různé typy
            if attempt == attempts:
                return f"CHYBA {type(exc).__name__} (spusťte znovu)"
            time.sleep(3 * attempt)


def dat_bytes(header: Path, minutes: float | None) -> int | None:
    """Počet bajtů .dat pro prvních `minutes` minut (jednosouborové záznamy bez offsetu)."""
    if minutes is None:
        return None
    hdr = wfdb.rdheader(str(header.with_suffix("")))
    if len(set(hdr.file_name)) != 1 or any(hdr.byte_offset or []):
        return None   # neobvyklé uložení -> raději celý soubor
    spf = hdr.samps_per_frame or [1] * hdr.n_sig
    per_frame = sum(BYTES_PER_SAMPLE[f] * s for f, s in zip(hdr.fmt, spf))
    return int(round(per_frame * hdr.fs * 60 * minutes + 0.5))


def download(db: str, root: Path, limit: int | None, minutes: float | None, workers: int):
    target = root / db
    target.mkdir(parents=True, exist_ok=True)
    records = physio.list_records(db)[:limit]
    print(f"{db}: {len(records)} záznamů -> {target}" + (f" (prvních {minutes} min)" if minutes else ""), flush=True)
    with ThreadPoolExecutor(workers) as pool:
        # 1) hlavičky a anotace (malé), 2) data s velikostí podle hlavičky
        small = [rec + ext for rec in records for ext in [".hea"] + EXTRA.get(db, [])]
        for name, res in zip(small, pool.map(lambda f: fetch(BASE.format(db=db, file=f), target / f), small)):
            if res.startswith("CHYBA"):
                print(f"  {name}: {res}", flush=True)
        jobs = [(rec + ".dat", dat_bytes(target / f"{rec}.hea", minutes))
                for rec in records if (target / f"{rec}.hea").exists()]
        done = 0
        for (name, _), res in zip(jobs, pool.map(lambda j: fetch(BASE.format(db=db, file=j[0]), target / j[0], j[1]), jobs)):
            done += 1
            if res.startswith("CHYBA") or done % 20 == 0 or done == len(jobs):
                print(f"  [{done}/{len(jobs)}] {name}: {res}", flush=True)


if __name__ == "__main__":
    physio.utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("databases", nargs="+", choices=["drivedb", "charisdb", "nsrdb", "voiced"])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--limit", type=int, default=None, help="Stáhnout jen prvních N záznamů")
    parser.add_argument("--minutes", type=float, default=None,
                        help="Stáhnout jen prvních N minut .dat (výchozí: nsrdb 35, charisdb 61, ostatní celé)")
    parser.add_argument("--full", action="store_true", help="Stáhnout celé soubory i u nsrdb/charisdb")
    parser.add_argument("--workers", type=int, default=16, help="Počet souběžných stahování")
    args = parser.parse_args()
    for db in args.databases:
        minutes = None if args.full else (args.minutes or DEFAULT_MINUTES.get(db))
        download(db, Path(args.data_dir), args.limit, minutes, args.workers)
    print("Hotovo. Chyby lze dořešit opětovným spuštěním (stažené soubory se přeskočí).")
