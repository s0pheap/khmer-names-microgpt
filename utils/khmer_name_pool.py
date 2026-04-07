"""
Generates Khmer name combinations from hardcoded surname/given-name pools,
converts them to Latin romanization, and saves to data/extra_names.csv
in the same format as khmer_names.csv:
  no, surname, givenName, romanizedSurname, romanizedGiven

Run: python utils/khmer_name_pool.py
"""
import csv
import multiprocessing as mp
import os
import random
from pathlib import Path

from khmer_latin_name_transformer import to_latin

LIMIT    = 500
OUT_PATH = Path(__file__).parent.parent / "data" / "extra_names.csv"

GIVEN_NAMES = [
    "បទុម", "ដារា", "ឡេង", "ណារុង", "និមល", "ភួង", "សារឿន", "ស្រី", "សន", "បុប្ផា",
    "ចិន្ដា", "ឈួន", "កុសល", "គុន្ធា", "ឡាយ", "លំអង", "ម៉ាលី", "ម៉ី", "ម៉ាប់", "ផល្លា",
    "ភារៈ", "ពិរុណ", "ពៅ", "រិទ្ធិ", "រី", "សំណាង", "សារិទ្ធ", "សុផល", "សម្បត្តិ", "សីហា",
    "ចាន់ត្រា", "ទេវី", "មុនី", "សុភ័ក្រ", "វិចិត្រ", "សោភា", "រតនា", "មករា", "បញ្ញា", "មាលា",
    "កញ្ញា", "ធារី", "សុខា", "បូរី", "សុវណ្ណ", "ចរិយា", "សុជាតា", "សេរី", "វិសាល", "មង្គល",
    "ភីរម្យ", "វាសនា", "ណារិន", "គន្ធី", "រស្មី", "ធានី", "សោភ័ណ", "ពិសិដ្ឋ", "ធីតា", "សីលា",
    "មនោ", "ភារម្យ", "រតនៈ", "សូភ័ណ", "វឌ្ឍនៈ", "សក្ដិ", "អរុណ", "សុរិយា", "សោភ័ណ្ឌ", "មេសា",
    "ឧសភា", "មិថុនា", "កក្កដា", "ធ្នូ", "កុម្ភៈ", "តុលា", "វិច្ឆិកា", "ចិត្រា", "សិរី", "ជ័យ",
    "ជំនឿ", "ភក្ដី", "មេត្ដា", "ករុណា", "មុទិតា", "ឧបេក្ខា", "សន្ដិ", "បុត្រ", "កល្យាណ", "សុផាត",
    "បុប្ផានី", "ទេព", "វង្ស", "តារា", "ច័ន្ទ", "សូរ្យ",
]

SURNAMES = [
    "ឱម", "ឱក", "ឯក", "អៀវ", "អៀម", "អ៊ុយ", "អ៊ុច", "អុង", "អ៊ុំ", "អិម",
    "អាង", "ឡុង", "ឡាយ", "ហូ", "ហ៊ុន", "ហុង", "សៅ", "សោម", "សេន", "សេង",
    "សៀង", "សឿង", "សួន", "ស៊ូ", "សូ", "ស៊ុយ", "សុន", "សុង", "សុខ", "សឺន",
    "លឹម", "ដុស", "សោន", "គង់", "ស៊ាន", "តាំង", "ជឹម", "អ៊ឹង", "ទូច", "កែវ",
    "ហែម", "តូច", "ស៊ឹម", "មាស", "យិន", "ឡោក", "ញ៉េប", "ឆាយ", "ពៅ", "គីម",
    "ង៉ែត", "ឌី", "ប៉ែន", "ព្រំ", "អ៊ិត", "ប៊ុន", "ចាន់", "ជួប", "ដួង", "ទៀង",
    "នួន", "ផាន់", "តេង", "ម៉ៅ", "ទិត្យ", "ឈឹម", "តាំ", "ណាំ", "នូ", "ព្រាប",
    "ភូ", "មឿន", "យស", "រស់", "រាជ", "លី", "វង្ស", "វ៉ាន់", "ស៊ាង", "សៀង",
    "អ៊ូ", "ខៀវ", "សោ", "គួច", "ឃួន", "ជ័យ", "ជា", "ជិន", "ឈិន", "ញ៉ឹក",
    "ដួង", "ថេង", "ថោង", "នេត", "ប៊ូ", "ប្រាក់", "ផល", "ផាត់", "ពាន", "មុំ",
    "យាន", "យុត", "សាយ",
]


def convert(pair: tuple[str, str]) -> tuple[str, str, str, str] | None:
    """Convert a (surname, givenName) Khmer pair to (surname, givenName, romanizedSurname, romanizedGiven)."""
    surname, given = pair
    try:
        rom_surname = to_latin(surname)
        rom_given   = to_latin(given)
        if rom_surname and rom_given:
            return surname, given, rom_surname.strip().lower(), rom_given.strip().lower()
    except Exception:
        pass
    return None


if __name__ == "__main__":
    random.seed(42)

    # Generate unique combinations and sample
    all_combos = list({(s, g) for s in SURNAMES for g in GIVEN_NAMES})
    random.shuffle(all_combos)
    selected = all_combos[:LIMIT]

    print(f"Converting {len(selected)} name combinations to Latin…")
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(convert, selected)

    rows = [r for r in results if r is not None]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["no", "surname", "givenName", "romanizedSurname", "romanizedGiven"])
        for i, (surname, given, rom_surname, rom_given) in enumerate(rows, 1):
            writer.writerow([i, surname, given, rom_surname, rom_given])

    print(f"Saved {len(rows)} rows → {OUT_PATH}")
