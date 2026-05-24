"""Sanity checks on the 3 DBs - row counts, sample rows, a couple useful aggregations."""
import sqlite3
from pathlib import Path

DATA = Path(__file__).parent.parent / "data"


def show(db_file, queries):
    print(f"\n========== {db_file} ==========")
    conn = sqlite3.connect(DATA / db_file)
    cur = conn.cursor()
    for label, sql in queries:
        print(f"\n-- {label}\n   SQL: {sql}")
        try:
            rows = cur.execute(sql).fetchall()
            for r in rows[:10]:
                print("  ", r)
            if len(rows) > 10:
                print(f"   ... and {len(rows) - 10} more rows")
        except Exception as e:
            print("   ERROR:", e)
    conn.close()


show("institutions.db", [
    ("schema", "PRAGMA table_info(institutions)"),
    ("count", "SELECT COUNT(*) FROM institutions"),
    ("distinct institute types", "SELECT institute_type, COUNT(*) FROM institutions GROUP BY institute_type"),
    ("distinct management types", "SELECT management_type, COUNT(*) FROM institutions GROUP BY management_type"),
    ("government institutions in Rajshahi", "SELECT COUNT(*) FROM institutions WHERE district = 'RAJSHAHI' AND management_type = 'GOVERNMENT'"),
    ("government institutions in Rajshahi (division)", "SELECT COUNT(*) FROM institutions WHERE division = 'RAJSHAHI' AND management_type = 'GOVERNMENT'"),
])

show("hospitals.db", [
    ("schema", "PRAGMA table_info(hospitals)"),
    ("count", "SELECT COUNT(*) FROM hospitals"),
    ("distinct types (top 20)", "SELECT type, COUNT(*) c FROM hospitals GROUP BY type ORDER BY c DESC LIMIT 20"),
    ("hospitals in Dhaka district", "SELECT COUNT(*) FROM hospitals WHERE district = 'Dhaka'"),
    ("sample dhaka hospitals", "SELECT name, type FROM hospitals WHERE district = 'Dhaka' LIMIT 5"),
])

show("restaurants.db", [
    ("schema", "PRAGMA table_info(restaurants)"),
    ("count", "SELECT COUNT(*) FROM restaurants"),
    ("restaurants in Chattogram (address LIKE)", "SELECT COUNT(*) FROM restaurants WHERE LOWER(address) LIKE '%chattogram%' OR LOWER(address) LIKE '%chittagong%'"),
    ("restaurants whose name contains biryani", "SELECT name, address, rating FROM restaurants WHERE LOWER(name) LIKE '%biryani%' OR LOWER(name) LIKE '%biriyani%' LIMIT 5"),
    ("avg rating", "SELECT ROUND(AVG(rating), 2) FROM restaurants WHERE rating > 0"),
])
