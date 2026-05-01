import re
import pandas as pd
from datetime import datetime
from rapidfuzz import fuzz

AUTO_THRESHOLD = 97

CANONICAL_MAP = {
    "gravy": "curry", "masala": "curry", "korma": "curry", "salan": "curry",
    "makhani": "curry", "kadai": "curry", "handi": "curry",
    "pulao": "rice", "fried rice": "rice", "jeera rice": "rice", "steamed rice": "rice",
    "roti": "bread", "naan": "bread", "paratha": "bread", "kulcha": "bread",
    "puri": "bread", "bhatura": "bread", "roomali": "bread", "chapati": "bread",
    "fries": "fries", "french fries": "fries", "chips": "fries", "wedges": "fries",
    "soda": "soft drink", "cola": "soft drink", "coke": "soft drink",
    "pepsi": "soft drink", "lemonade": "soft drink", "nimbu pani": "soft drink",
    "shikanji": "soft drink", "lassi": "lassi", "chaas": "lassi", "buttermilk": "lassi",
    "chkn": "chicken", "cken": "chicken",
    "paneer": "paneer", "cottage cheese": "paneer",
    "egg": "egg", "anda": "egg", "eggs": "egg",
    "dal": "dal", "daal": "dal", "lentils": "dal", "dhal": "dal",
    "tikka": "tikka", "tika": "tikka",
    "shawarma": "shawarma", "shawarmah": "shawarma",
    "kebab": "kebab", "kabab": "kebab", "kabob": "kebab",
    "biryani": "biryani", "biriyani": "biryani", "briyani": "biryani",
    "manchurian": "manchurian", "manchuria": "manchurian",
    "noodles": "noodles", "hakka": "noodles", "chowmein": "noodles", "chow mein": "noodles",
    "dimsum": "dumpling", "dim sum": "dumpling", "momos": "dumpling", "momo": "dumpling",
    "pizza": "pizza", "piza": "pizza",
    "pasta": "pasta", "penne": "pasta", "spaghetti": "pasta", "fettuccine": "pasta",
    "burger": "burger", "buger": "burger",
    "sandwich": "sandwich", "sandwitch": "sandwich", "sub": "sandwich", "wrap": "sandwich",
    "roll": "roll", "kathi roll": "roll",
    "gulab jamun": "gulab jamun", "gulabjamun": "gulab jamun",
    "halwa": "halwa", "kheer": "kheer", "payasam": "kheer", "phirni": "kheer",
    "ice cream": "ice cream", "icecream": "ice cream",
    "idli": "idli", "idly": "idli",
    "dosa": "dosa", "dosai": "dosa",
    "vada": "vada", "wada": "vada",
    "sambar": "sambar", "sambhar": "sambar",
    "salad": "salad", "bowl": "bowl",
    "smoothie": "smoothie", "shake": "shake", "milkshake": "shake",
    "sm": "small", "med": "medium", "lg": "large", "rg": "regular",
    "hf": "half", "fl": "full",
    "soup": "soup", "shorba": "soup",
    "chaat": "chaat", "raita": "raita", "curd": "curd", "dahi": "curd",
    "panipuri": "panipuri", "golgappa": "panipuri", "puchka": "panipuri",
    "thali": "thali", "combo": "combo", "meal": "combo",
    "steak": "steak",
    "sushi": "sushi", "ramen": "ramen", "udon": "udon", "tempura": "tempura",
    "taco": "taco", "burrito": "burrito", "quesadilla": "quesadilla", "nachos": "nachos",
    "hummus": "hummus", "falafel": "falafel",
    "croissant": "croissant", "bagel": "bagel",
}


def strip_ids(text):
    """Remove (id) patterns like (12345) from display names."""
    if pd.isna(text):
        return ""
    return re.sub(r"\s*\(.*?\)\s*", " ", str(text)).strip()


def normalize(text):
    if pd.isna(text) or str(text).strip() == "":
        return ""
    s = str(text).lower().strip()
    s = re.sub(r"\[.*?\]", "", s)
    s = re.sub(r"\(.*?\)", "", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    for key, val in CANONICAL_MAP.items():
        s = re.sub(r"\b" + re.escape(key) + r"\b", val, s)
    return s


def bigrams(text):
    words = normalize(text).split()
    return set(zip(words, words[1:])) if len(words) >= 2 else set()


def score_item(ref_item, menu_item):
    if not ref_item or not menu_item:
        return 0
    ref_n = normalize(ref_item)
    menu_n = normalize(menu_item)
    s1 = fuzz.token_sort_ratio(ref_n, menu_n)
    s2 = fuzz.token_set_ratio(ref_n, menu_n)
    s3 = fuzz.ratio(ref_n, menu_n)
    ref_bg = bigrams(ref_n)
    menu_bg = bigrams(menu_n)
    if ref_bg or menu_bg:
        common = len(ref_bg & menu_bg)
        total = len(ref_bg | menu_bg)
        bigram_score = (common / total) * 100 if total else 0
    else:
        bigram_score = 100 if ref_n == menu_n else 0
    return max(s1, s2, s3, bigram_score)


def score_field(ref_val, menu_val):
    if not ref_val:
        return None
    menu_norm = normalize(menu_val)
    if not menu_norm:
        return 0
    return fuzz.token_sort_ratio(normalize(ref_val), menu_norm)


def match_items(menu_df, ref_df):
    """
    Match each ref CSV row to the best menu CSV row.
    - Respects 'Add on (y/n)' column in ref CSV: if 'y', search Addon column in menu.
    - Returns (auto_matches, hitl_queue).
    - auto_matches: score >= AUTO_THRESHOLD.
    - hitl_queue: everything below threshold, with top 5 candidates shown to user.
    """
    menu_df = menu_df.copy().reset_index(drop=True)
    ref_df = ref_df.copy().reset_index(drop=True)

    # --- detect ref columns ---
    ref_item_col = next(
        (c for c in ref_df.columns if "item" in c.lower() and "group" not in c.lower() and "addon" not in c.lower()),
        None
    )
    if ref_item_col is None:
        raise ValueError("Could not find Item column in reference CSV.")

    ref_cat_col = next((c for c in ref_df.columns if "category" in c.lower() and "sub" not in c.lower()), None)
    ref_subcat_col = next((c for c in ref_df.columns if "subcategory" in c.lower()), None)
    ref_variant_col = next((c for c in ref_df.columns if "variant" in c.lower()), None)
    ref_addon_flag_col = next(
        (c for c in ref_df.columns if "add" in c.lower() and ("on" in c.lower() or "addon" in c.lower())),
        None
    )

    # --- detect menu columns ---
    menu_item_col = next((c for c in menu_df.columns if c.strip().lower() == "item"), None)
    menu_addon_col = next((c for c in menu_df.columns if c.strip().lower() == "addon"), None)
    menu_cat_col = next((c for c in menu_df.columns if "category" in c.lower() and "sub" not in c.lower()), None)
    menu_subcat_col = next((c for c in menu_df.columns if "subcategory" in c.lower()), None)
    menu_variant_col = next(
        (c for c in menu_df.columns
         if "variant" in c.lower() and "group" not in c.lower()
         and "food" not in c.lower() and "avail" not in c.lower()
         and c.lower().endswith("l1")),
        None
    )
    menu_price_col = "Price" if "Price" in menu_df.columns else None
    menu_sku_type_col = next((c for c in menu_df.columns if "sku" in c.lower() and "type" in c.lower()), None)

    if menu_item_col is None:
        raise ValueError("Could not find Item column in menu CSV.")

    # pre-build normalised item names for menu items and addons separately
    menu_items_norm = [normalize(str(v)) for v in menu_df[menu_item_col].tolist()]
    menu_addons_norm = (
        [normalize(str(v)) for v in menu_df[menu_addon_col].tolist()]
        if menu_addon_col else []
    )

    auto_matches = []
    hitl_queue = []

    for r_idx, r_row in ref_df.iterrows():
        ref_item_raw = str(r_row[ref_item_col])
        ref_item_norm = normalize(ref_item_raw)
        if not ref_item_norm:
            continue

        ref_cat = normalize(r_row[ref_cat_col]) if ref_cat_col else ""
        ref_subcat = normalize(r_row[ref_subcat_col]) if ref_subcat_col else ""
        ref_variant = normalize(r_row[ref_variant_col]) if ref_variant_col else ""

        # determine if this ref row is flagged as an addon
        is_addon_row = False
        if ref_addon_flag_col:
            flag_val = str(r_row[ref_addon_flag_col]).strip().lower()
            is_addon_row = flag_val in ("y", "yes", "1", "true")

        # choose which column pool to search
        if is_addon_row and menu_addons_norm:
            search_pool = menu_addons_norm
            search_label = "addon"
        else:
            search_pool = menu_items_norm
            search_label = "item"

        best_score = -1
        best_menu_idx = -1
        candidate_indices = []

        for m_idx, pool_val_norm in enumerate(search_pool):
            if not pool_val_norm:
                continue

            item_score = score_item(ref_item_norm, pool_val_norm)
            if item_score < 60:
                continue

            total_score = item_score
            weight = 1.0

            if ref_cat and menu_cat_col:
                cs = score_field(ref_cat, str(menu_df[menu_cat_col].iloc[m_idx]))
                if cs is not None:
                    total_score += cs * 0.3
                    weight += 0.3

            if ref_subcat and menu_subcat_col:
                ss = score_field(ref_subcat, str(menu_df[menu_subcat_col].iloc[m_idx]))
                if ss is not None:
                    total_score += ss * 0.2
                    weight += 0.2

            if ref_variant and menu_variant_col:
                vs = score_field(ref_variant, str(menu_df[menu_variant_col].iloc[m_idx]))
                if vs is not None:
                    total_score += vs * 0.2
                    weight += 0.2

            composite = total_score / weight

            if composite > best_score:
                best_score = composite
                best_menu_idx = m_idx

            candidate_indices.append((m_idx, item_score))

        if best_menu_idx == -1:
            hitl_queue.append({
                "ref_index": r_idx,
                "ref_item": ref_item_raw,
                "ref_cat": ref_cat,
                "ref_subcat": ref_subcat,
                "ref_variant": ref_variant,
                "is_addon": is_addon_row,
                "search_label": search_label,
                "candidates": [],
                "score": 0,
            })
            continue

        candidate_display = []
        for ci, sc in sorted(candidate_indices, key=lambda x: x[1], reverse=True)[:5]:
            addon_display = strip_ids(menu_df[menu_addon_col].iloc[ci]) if menu_addon_col else ""
            item_display = strip_ids(menu_df[menu_item_col].iloc[ci])
            candidate_display.append({
                "menu_index": int(menu_df.index[ci]),
                "menu_item": item_display if not is_addon_row else addon_display,
                "menu_cat": strip_ids(str(menu_df[menu_cat_col].iloc[ci])) if menu_cat_col else "",
                "menu_subcat": strip_ids(str(menu_df[menu_subcat_col].iloc[ci])) if menu_subcat_col else "",
                "menu_variant": strip_ids(str(menu_df[menu_variant_col].iloc[ci])) if menu_variant_col else "",
                "menu_price": str(menu_df[menu_price_col].iloc[ci]) if menu_price_col else "",
                "row_type": str(menu_df[menu_sku_type_col].iloc[ci]) if menu_sku_type_col else "",
                "score": round(sc),
            })

        if best_score >= AUTO_THRESHOLD:
            auto_matches.append({
                "ref_index": r_idx,
                "menu_index": int(menu_df.index[best_menu_idx]),
                "item": ref_item_raw,
                "score": round(best_score),
                "auto": True,
                "is_addon": is_addon_row,
            })
        else:
            hitl_queue.append({
                "ref_index": r_idx,
                "ref_item": ref_item_raw,
                "ref_cat": ref_cat,
                "ref_subcat": ref_subcat,
                "ref_variant": ref_variant,
                "is_addon": is_addon_row,
                "search_label": search_label,
                "candidates": candidate_display,
                "score": round(best_score),
            })

    return auto_matches, hitl_queue


def find_addon_rows(menu_df, item_name):
    """
    Given a matched item name, find all rows in menu where
    the Addon column contains that item (fuzzy match >= 90).
    Returns list of actual df indices.
    """
    addon_col = next((c for c in menu_df.columns if c.strip().lower() == "addon"), None)
    if not addon_col:
        return []

    item_norm = normalize(item_name)
    matches = []
    for idx, val in menu_df[addon_col].items():
        if normalize(str(val)) and fuzz.token_sort_ratio(item_norm, normalize(str(val))) >= 90:
            matches.append(idx)
    return matches


def process_matches(menu_df, ref_df, confirmed_matches, mode="slash", addon_indices=None):
    """
    Apply pricing to all confirmed matches.
    Audit log: one grouped entry per operation (not one row per item).
    """
    df = menu_df.copy()
    changed_rows = []

    ref_item_col = next(
        (c for c in ref_df.columns if "item" in c.lower() and "group" not in c.lower() and "addon" not in c.lower()),
        None
    )
    base_col = next((c for c in ref_df.columns if "base" in c.lower() and "price" in c.lower()), None)
    if base_col is None:
        base_col = next((c for c in ref_df.columns if "base" in c.lower()), None)
    revised_col = next((c for c in ref_df.columns if "revised" in c.lower()), None)

    def apply_pricing(idx, ref_base, ref_revised, reason, ref_item_name):
        old_price = pd.to_numeric(df.at[idx, "Price"], errors="coerce")
        old_markup = pd.to_numeric(df.at[idx, "Markup Price"], errors="coerce") if "Markup Price" in df.columns else None

        if mode == "slash":
            if pd.notna(ref_base) and pd.notna(ref_revised):
                df.at[idx, "Markup Price"] = ref_base
                df.at[idx, "Price"] = ref_revised
            elif pd.isna(ref_base) and pd.notna(ref_revised):
                df.at[idx, "Markup Price"] = ref_revised
            elif pd.notna(ref_base) and pd.isna(ref_revised):
                df.at[idx, "Markup Price"] = ref_base
            markup_val = pd.to_numeric(df.at[idx, "Markup Price"], errors="coerce")
            if pd.notna(markup_val) and markup_val == 0:
                df.at[idx, "Markup Price"] = None
        elif mode == "replace":
            if pd.notna(ref_revised):
                df.at[idx, "Price"] = ref_revised
            # always clear Markup Price for direct replace — no slashing construct
            df.at[idx, "Markup Price"] = None

        df.at[idx, "Update Required ?"] = "Yes"

        changed_rows.append({
            "Ref Item": ref_item_name,
            "Menu Item": df.at[idx, "Item"] if "Item" in df.columns else "",
            "Category": df.at[idx, "Category"] if "Category" in df.columns else "",
            "Subcategory": df.at[idx, "Subcategory"] if "Subcategory" in df.columns else "",
            "Old Base Price": old_markup,
            "New Base Price": df.at[idx, "Markup Price"] if "Markup Price" in df.columns else "",
            "Old Selling Price": old_price,
            "New Selling Price": df.at[idx, "Price"],
            "Reason": reason,
        })

    for m in confirmed_matches:
        r_idx = m["ref_index"]
        menu_idx = m["menu_index"]
        r = ref_df.loc[r_idx]

        ref_base = pd.to_numeric(r[base_col], errors="coerce") if base_col else None
        ref_revised = pd.to_numeric(r[revised_col], errors="coerce") if revised_col else None
        ref_item_name = str(r[ref_item_col]) if ref_item_col else ""

        apply_pricing(menu_idx, ref_base, ref_revised, "auto match" if m.get("auto") else "user confirmed", ref_item_name)

        if addon_indices and menu_idx in addon_indices:
            for addon_idx in addon_indices[menu_idx]:
                apply_pricing(addon_idx, ref_base, ref_revised, "addon propagation", ref_item_name)

    # ── build grouped audit: one entry per operation, items listed inside ──
    if not changed_rows:
        return df, pd.DataFrame()

    items_affected = list({r["Menu Item"] for r in changed_rows if r["Menu Item"]})
    audit_entry = pd.DataFrame([{
        "Operation": mode,
        "Items Affected": len(changed_rows),
        "Item List": ", ".join(items_affected),
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Detail": str(changed_rows),   # full detail preserved for rollback
    }])

    # also keep full detail rows for rollback under a separate key
    detail_df = pd.DataFrame(changed_rows)

    return df, audit_entry, detail_df