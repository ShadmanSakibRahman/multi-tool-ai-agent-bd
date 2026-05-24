"""Quick look at the 3 HuggingFace datasets so we can design SQLite schemas."""

from datasets import load_dataset

DATASETS = [
    ("Mahadih534/Institutional-Information-of-Bangladesh", "institutions"),
    ("Mahadih534/all-bangladeshi-hospitals", "hospitals"),
    ("Mahadih534/Bangladeshi-Restaurant-Data", "restaurants"),
]

for repo_id, label in DATASETS:
    print("=" * 70)
    print(f"DATASET: {repo_id}  ->  {label}")
    print("=" * 70)
    ds = load_dataset(repo_id)
    print("Splits:", list(ds.keys()))
    split_name = list(ds.keys())[0]
    split = ds[split_name]
    print(f"Rows in '{split_name}':", len(split))
    print("Columns:", split.column_names)
    print("Features:")
    for col, feat in split.features.items():
        print(f"  - {col}: {feat}")
    print("First 2 rows:")
    for i, row in enumerate(split.select(range(min(2, len(split))))):
        print(f"  Row {i}:", {k: (str(v)[:80] + "..." if isinstance(v, str) and len(v) > 80 else v) for k, v in row.items()})
    print()
