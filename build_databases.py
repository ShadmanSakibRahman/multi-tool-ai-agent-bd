"""
Downloads the 3 HuggingFace datasets and writes each to its own SQLite DB.

After running:
  data/institutions.db   table: institutions
  data/hospitals.db      table: hospitals
  data/restaurants.db    table: restaurants
"""

import os
import sqlite3
from pathlib import Path

from datasets import load_dataset

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)


# Column rename maps. Source HF column -> nice snake_case name.
INSTITUTIONS_COLS = {
    "INSTITUTE NAME": ("institute_name", "TEXT"),
    "EIIN": ("eiin", "INTEGER"),
    "INSTITUTE_TYPE": ("institute_type", "TEXT"),
    "DIVISION_ID": ("division_id", "INTEGER"),
    "DIVISION": ("division", "TEXT"),
    "DISTRICT_ID": ("district_id", "INTEGER"),
    "DISTRICT": ("district", "TEXT"),
    "THANA_ID": ("thana_id", "INTEGER"),
    "THANA": ("thana", "TEXT"),
    "UNION_ID": ("union_id", "INTEGER"),
    "UNION_NAME": ("union_name", "TEXT"),
    "MAUZA_ID": ("mauza_id", "INTEGER"),
    "MAUZA_NAME": ("mauza_name", "TEXT"),
    "AREA_STATUS": ("area_status", "TEXT"),
    "GEOGRPYCAL_STATUS": ("geographical_status", "TEXT"),
    "ADDRESS": ("address", "TEXT"),
    "POST": ("post", "TEXT"),
    "MANAGEMENT_TYPE": ("management_type", "TEXT"),
    "MOBILE": ("mobile", "TEXT"),
    "STUDENT_TYPE": ("student_type", "TEXT"),
    "EDUCATION_LEVEL": ("education_level", "TEXT"),
    "AFFILIATION": ("affiliation", "TEXT"),
    "MPO_STATUS": ("mpo_status", "TEXT"),
}

HOSPITALS_COLS = {
    "Id": ("id", "INTEGER"),
    "Name": ("name", "TEXT"),
    "Name (Bangla)": ("name_bangla", "TEXT"),
    "Code": ("code", "INTEGER"),
    "Agency": ("agency", "TEXT"),
    "Type": ("type", "TEXT"),
    "Division": ("division", "TEXT"),
    "District": ("district", "TEXT"),
    "City Corporation": ("city_corporation", "TEXT"),
    "Upazila": ("upazila", "TEXT"),
    "Paurasava": ("paurasava", "TEXT"),
    "Union": ("union_name", "TEXT"),
    "Private": ("private", "INTEGER"),
}

RESTAURANTS_COLS = {
    "place_id": ("place_id", "TEXT"),
    "name": ("name", "TEXT"),
    "latitude": ("latitude", "REAL"),
    "longitude": ("longitude", "REAL"),
    "rating": ("rating", "REAL"),
    "number_of_reviews": ("number_of_reviews", "REAL"),
    "affluence": ("affluence", "REAL"),
    "address": ("address", "TEXT"),
}


def clean_value(val):
    """Strip whitespace on strings. None -> None."""
    if val is None:
        return None
    if isinstance(val, str):
        v = val.strip()
        return v if v else None
    return val


def build_db(repo_id: str, db_filename: str, table_name: str, col_map: dict):
    """Download HF dataset and write to SQLite."""
    db_path = DATA_DIR / db_filename
    if db_path.exists():
        db_path.unlink()

    print(f"\n[{table_name}] downloading {repo_id} ...")
    ds = load_dataset(repo_id)
    split_name = list(ds.keys())[0]
    split = ds[split_name]
    print(f"[{table_name}] {len(split)} rows")

    src_cols = list(col_map.keys())
    dst_defs = [f'"{col_map[c][0]}" {col_map[c][1]}' for c in src_cols]
    dst_names = [col_map[c][0] for c in src_cols]

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(f'CREATE TABLE "{table_name}" ({", ".join(dst_defs)})')

    placeholders = ", ".join(["?"] * len(src_cols))
    insert_sql = (
        f'INSERT INTO "{table_name}" ({", ".join(dst_names)}) VALUES ({placeholders})'
    )

    rows = []
    for row in split:
        rows.append(tuple(clean_value(row.get(c)) for c in src_cols))
        if len(rows) >= 5000:
            cur.executemany(insert_sql, rows)
            rows = []
    if rows:
        cur.executemany(insert_sql, rows)

    conn.commit()
    n = cur.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]
    conn.close()
    print(f"[{table_name}] wrote {n} rows to {db_path}")


def main():
    build_db(
        "Mahadih534/Institutional-Information-of-Bangladesh",
        "institutions.db",
        "institutions",
        INSTITUTIONS_COLS,
    )
    build_db(
        "Mahadih534/all-bangladeshi-hospitals",
        "hospitals.db",
        "hospitals",
        HOSPITALS_COLS,
    )
    build_db(
        "Mahadih534/Bangladeshi-Restaurant-Data",
        "restaurants.db",
        "restaurants",
        RESTAURANTS_COLS,
    )
    print("\nAll three databases written to data/ directory.")


if __name__ == "__main__":
    main()
