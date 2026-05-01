import pandas as pd
import os

METADATA_KEYWORDS = [
    "brandskuid", "item/variant/addon", "description",
    "cdn", "veg/non veg", "in stock", "yyyy-mm-dd", "hh:mm:ss"
]

REQUIRED_COLUMNS = {"Price", "Markup Price", "Category", "Brand SKU ID Type", "Item"}


def detect_freeze_index(df):
    for i in df.index:
        try:
            row_text = " ".join(str(x).lower() for x in df.loc[i].values if pd.notna(x))
        except Exception:
            continue
        if any(k in row_text for k in METADATA_KEYWORDS):
            return i + 1
    return 0


def load_csv(path=None, label="CSV"):
    if path and os.path.exists(path):
        df = pd.read_csv(path, dtype=str)
    else:
        while True:
            path = input(f"Enter full path of {label}: ").strip().replace('"', "")
            if os.path.exists(path):
                df = pd.read_csv(path, dtype=str)
                break
            print("File not found.\n")

    df.columns = df.columns.str.strip()
    df = df.reset_index(drop=True)
    df = df.apply(lambda col: col.map(lambda x: x.strip() if isinstance(x, str) else x))
    return df


def load_menu_csv(path=None):
    df = load_csv(path=path, label="MENU CSV")

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Invalid menu CSV. Missing columns: {', '.join(missing)}")

    if "Update Required ?" not in df.columns:
        df["Update Required ?"] = ""

    freeze_idx = detect_freeze_index(df)
    return df, freeze_idx


def load_reference_csv(path=None):
    df = load_csv(path=path, label="REFERENCE CSV")

    if not any("item" in c.lower() for c in df.columns):
        raise ValueError("Reference CSV must contain an Item column.")

    return df