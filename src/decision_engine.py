import re
import pandas as pd
import numpy as np
from datetime import datetime
from rapidfuzz import fuzz

AUTO_THRESHOLD = 90

CANONICAL_MAP = {
    "gravy": "curry", "masala": "curry", "korma": "curry", "salan": "curry",
    "makhani": "curry", "kadai": "curry", "handi": "curry",
    "pulao": "rice", "fried rice": "rice", "jeera rice": "rice", "steamed rice": "rice",
    "roti": "bread", "naan": "bread", "paratha": "bread", "kulcha": "bread",
    "puri": "bread", "bhatura": "bread", "roomali": "bread", "chapati": "bread",
    "fries": "fries", "french fries": "fries", "chips": "fries", "wedges": "fries",
    "soda": "soft drink", "cola": "soft drink", "coke": "soft drink",
    "pepsi": "soft drink", "lemonade": "soft drink", "nimbu pani": "soft drink",
    "chkn": "chicken", "cken": "chicken",
    "paneer": "paneer", "cottage cheese": "paneer",
    "egg": "egg", "anda": "egg", "eggs": "egg",
    "dal": "dal", "daal": "dal", "lentils": "dal", "dhal": "dal",
    "tikka": "tikka", "tika": "tikka",
    "shawarma": "shawarma", "shawarmah": "shawarma",
    "kebab": "kebab", "kabab": "kebab", "kabob": "kebab",
    "biryani": "biryani", "biriyani": "biryani", "briyani": "biryani",
    "manchurian": "manchurian", "manchuria": "manchurian",
    "noodles": "noodles", "hakka": "noodles", "chowmein": "noodles",
    "dimsum": "dumpling", "dim sum": "dumpling", "momos": "dumpling", "momo": "dumpling",
    "pizza": "pizza", "piza": "pizza",
    "pasta": "pasta", "penne": "pasta", "spaghetti": "pasta",
    "burger": "burger", "buger": "burger",
    "sandwich": "sandwich", "sandwitch": "sandwich", "sub": "sandwich", "wrap": "sandwich",
    "roll": "roll", "kathi roll": "roll",
    "gulab jamun": "gulabjamun", "gulabjamun": "gulabjamun",
    "halwa": "halwa", "kheer": "kheer", "payasam": "kheer",
    "ice cream": "icecream", "icecream": "icecream",
    "idli": "idli", "idly": "idli",
    "dosa": "dosa", "dosai": "dosa",
    "vada": "vada", "wada": "vada",
    "sambar": "sambar", "sambhar": "sambar",
    "salad": "salad", "bowl": "bowl",
    "smoothie": "smoothie", "shake": "shake", "milkshake": "shake",
    "sm": "small", "med": "medium", "lg": "large", "rg": "regular",
    "hf": "half", "fl": "full",
    "soup": "soup", "chaat": "chaat", "raita": "raita", "curd": "curd", "dahi": "curd",
    "panipuri": "panipuri", "golgappa": "panipuri", "puchka": "panipuri",
    "thali": "thali", "combo": "combo", "meal": "combo",
    "sushi": "sushi", "ramen": "ramen", "taco": "taco", "burrito": "burrito",
    "hummus": "hummus", "falafel": "falafel",
    "croissant": "croissant", "bagel": "bagel",
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "piece": "pc", "pieces": "pc", "pcs": "pc",
    "pack": "pack", "portion": "portion", "serving": "serving",
}


def strip_ids(text):
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


def score_pair(ref_text, menu_text):
    rn = normalize(ref_text)
    mn = normalize(menu_text)
    if not rn or not mn:
        return 0

    s_ratio = fuzz.ratio(rn, mn)
    s_sort = fuzz.token_sort_ratio(rn, mn)
    s_set = fuzz.token_set_ratio(rn, mn)
    s_partial = fuzz.partial_ratio(rn, mn)

    base = max(s_ratio, s_sort, s_set, s_partial)

    ref_tokens = set(rn.split())
    menu_tokens = set(mn.split())
    if ref_tokens != menu_tokens and (ref_tokens < menu_tokens or menu_tokens < ref_tokens):
        extra = len(ref_tokens.symmetric_difference(menu_tokens))
        penalty = min(extra * 4, 18)
        base = max(0, base - penalty)

    return base


def match_items(menu_df, ref_df):
    menu_df = menu_df.copy().reset_index(drop=True)
    ref_df = ref_df.copy().reset_index(drop=True)

    ref_item_col = next(
        (c for c in ref_df.columns if "item" in c.lower()
         and "group" not in c.lower() and "addon" not in c.lower()), None)
    if ref_item_col is None:
        raise ValueError("Could not find Item column in reference CSV.")

    ref_cat_col = next((c for c in ref_df.columns if "category" in c.lower() and "sub" not in c.lower()), None)
    ref_subcat_col = next((c for c in ref_df.columns if "subcategory" in c.lower()), None)
    ref_variant_col = next((c for c in ref_df.columns if "variant" in c.lower()), None)
    ref_addon_flag_col = next(
        (c for c in ref_df.columns if "add" in c.lower() and ("on" in c.lower() or "addon" in c.lower())), None)

    menu_item_col = next((c for c in menu_df.columns if c.strip().lower() == "item"), None)
    menu_addon_col = next((c for c in menu_df.columns if c.strip().lower() == "addon"), None)
    menu_cat_col = next((c for c in menu_df.columns if "category" in c.lower() and "sub" not in c.lower()), None)
    menu_subcat_col = next((c for c in menu_df.columns if "subcategory" in c.lower()), None)
    menu_variant_col = next(
        (c for c in menu_df.columns if "variant" in c.lower() and "group" not in c.lower()
         and "food" not in c.lower() and "avail" not in c.lower() and c.lower().endswith("l1")), None)
    menu_price_col = "Price" if "Price" in menu_df.columns else None
    menu_sku_type_col = next((c for c in menu_df.columns if "sku" in c.lower() and "type" in c.lower()), None)

    if menu_item_col is None:
        raise ValueError("Could not find Item column in menu CSV.")

    menu_items = [str(menu_df.at[i, menu_item_col]) for i in menu_df.index]
    menu_addons = [str(menu_df.at[i, menu_addon_col]) if menu_addon_col else "" for i in menu_df.index]
    menu_variants = [str(menu_df.at[i, menu_variant_col]) if menu_variant_col else "" for i in menu_df.index]
    menu_sku_types = [str(menu_df.at[i, menu_sku_type_col]).strip() if menu_sku_type_col else "" for i in menu_df.index]

    def get_price(m_idx):
        if menu_price_col:
            v = menu_df.at[m_idx, menu_price_col]
            try:
                return float(v)
            except Exception:
                return None
        return None

    score_matrix = {}

    for r_idx in ref_df.index:
        ref_item_raw = str(ref_df.at[r_idx, ref_item_col])

        is_addon = False
        if ref_addon_flag_col:
            flag = str(ref_df.at[r_idx, ref_addon_flag_col]).strip().lower()
            is_addon = flag in ("y", "yes", "1", "true")

        ref_cat = normalize(ref_df.at[r_idx, ref_cat_col]) if ref_cat_col else ""
        ref_subcat = normalize(ref_df.at[r_idx, ref_subcat_col]) if ref_subcat_col else ""
        ref_variant_raw = str(ref_df.at[r_idx, ref_variant_col]).strip() if ref_variant_col else ""
        ref_variant = normalize(ref_variant_raw)
        has_variant = ref_variant not in ("", "nan", "none")

        for m_idx in menu_df.index:
            sku_type = menu_sku_types[m_idx]

            # strict SKU type gating
            if is_addon:
                if sku_type != "Addon":
                    continue
            elif has_variant:
                if sku_type not in ("Variant", ""):
                    continue
            else:
                # plain item: skip Variant and Addon rows
                if sku_type in ("Variant", "Addon"):
                    continue

            if is_addon:
                item_score = score_pair(ref_item_raw, menu_addons[m_idx])
            elif has_variant:
                # combine item name context + variant name match
                item_score_name = score_pair(ref_item_raw, menu_items[m_idx])
                variant_score = score_pair(ref_variant_raw, menu_variants[m_idx])
                item_score = item_score_name * 0.4 + variant_score * 0.6
            else:
                item_score = score_pair(ref_item_raw, menu_items[m_idx])

            if item_score < 38:
                continue

            total = item_score
            weight = 1.0

            if ref_cat and menu_cat_col:
                cs = fuzz.token_sort_ratio(ref_cat, normalize(str(menu_df.at[m_idx, menu_cat_col])))
                total += cs * 0.18
                weight += 0.18

            if ref_subcat and menu_subcat_col:
                ss = fuzz.token_sort_ratio(ref_subcat, normalize(str(menu_df.at[m_idx, menu_subcat_col])))
                total += ss * 0.1
                weight += 0.1

            composite = total / weight
            score_matrix[(r_idx, m_idx)] = (composite, item_score, is_addon)

    all_pairs = sorted(score_matrix.items(), key=lambda x: x[1][0], reverse=True)

    assigned_menu = {}
    assigned_ref = {}

    for (r_idx, m_idx), (composite, item_score, is_addon) in all_pairs:
        if r_idx in assigned_ref or m_idx in assigned_menu:
            continue
        if composite >= AUTO_THRESHOLD:
            assigned_ref[r_idx] = (m_idx, composite, is_addon)
            assigned_menu[m_idx] = r_idx

    auto_matches = []
    hitl_queue = []

    for r_idx in ref_df.index:
        ref_item_raw = str(ref_df.at[r_idx, ref_item_col])
        ref_cat = normalize(ref_df.at[r_idx, ref_cat_col]) if ref_cat_col else ""
        ref_subcat = normalize(ref_df.at[r_idx, ref_subcat_col]) if ref_subcat_col else ""
        ref_variant_raw = str(ref_df.at[r_idx, ref_variant_col]).strip() if ref_variant_col else ""
        ref_variant = normalize(ref_variant_raw)
        has_variant = ref_variant not in ("", "nan", "none")

        is_addon = False
        if ref_addon_flag_col:
            flag = str(ref_df.at[r_idx, ref_addon_flag_col]).strip().lower()
            is_addon = flag in ("y", "yes", "1", "true")

        if r_idx in assigned_ref:
            m_idx, composite, is_addon_m = assigned_ref[r_idx]
            auto_matches.append({
                "ref_index": r_idx,
                "menu_index": int(menu_df.index[m_idx]),
                "item": ref_item_raw,
                "score": round(composite),
                "auto": True,
                "is_addon": is_addon_m,
                "menu_sku_type": menu_sku_types[m_idx],
            })
        else:
            candidates_raw = [
                (m_idx, score_matrix[(r_idx, m_idx)][0], score_matrix[(r_idx, m_idx)][1])
                for m_idx in menu_df.index
                if (r_idx, m_idx) in score_matrix
            ]
            candidates_raw.sort(key=lambda x: x[1], reverse=True)
            top = candidates_raw[:6]

            candidate_display = []
            for m_idx, composite, item_score in top:
                if is_addon:
                    item_display = strip_ids(menu_addons[m_idx])
                elif has_variant:
                    item_display = f"{strip_ids(menu_items[m_idx])} · {strip_ids(menu_variants[m_idx])}"
                else:
                    item_display = strip_ids(menu_items[m_idx])

                price_raw = get_price(m_idx)
                price_str = str(int(price_raw)) if price_raw is not None and not np.isnan(price_raw) and price_raw == int(price_raw) else (str(round(price_raw, 2)) if price_raw is not None else "")

                candidate_display.append({
                    "menu_index": int(menu_df.index[m_idx]),
                    "menu_item": item_display,
                    "menu_cat": strip_ids(str(menu_df.at[m_idx, menu_cat_col])) if menu_cat_col else "",
                    "menu_subcat": strip_ids(str(menu_df.at[m_idx, menu_subcat_col])) if menu_subcat_col else "",
                    "menu_variant": strip_ids(menu_variants[m_idx]),
                    "menu_price": price_str,
                    "menu_sku_type": menu_sku_types[m_idx],
                    "score": round(composite),
                })

            hitl_queue.append({
                "ref_index": r_idx,
                "ref_item": ref_item_raw,
                "ref_cat": ref_cat,
                "ref_subcat": ref_subcat,
                "ref_variant": ref_variant,
                "is_addon": is_addon,
                "search_label": "addon" if is_addon else ("variant" if has_variant else "item"),
                "candidates": candidate_display,
                "score": round(top[0][1]) if top else 0,
            })

    return auto_matches, hitl_queue


def find_addon_rows(menu_df, item_name):
    addon_col = next((c for c in menu_df.columns if c.strip().lower() == "addon"), None)
    if not addon_col:
        return []
    item_norm = normalize(item_name)
    return [
        idx for idx, val in menu_df[addon_col].items()
        if normalize(str(val)) and fuzz.token_sort_ratio(item_norm, normalize(str(val))) >= 88
    ]


def process_matches(menu_df, ref_df, confirmed_matches, mode="slash", addon_indices=None):
    df = menu_df.copy()

    ref_item_col = next(
        (c for c in ref_df.columns if "item" in c.lower()
         and "group" not in c.lower() and "addon" not in c.lower()), None)
    base_col = next((c for c in ref_df.columns if "base" in c.lower() and "price" in c.lower()), None)
    if base_col is None:
        base_col = next((c for c in ref_df.columns if "base" in c.lower()), None)
    revised_col = next((c for c in ref_df.columns if "revised" in c.lower()), None)
    update_col = next(
        (c for c in df.columns if c.strip().lower().startswith("update required")),
        "Update Required ?")

    changed_rows = []

    def safe_float(val):
        if val is None:
            return None
        try:
            f = float(str(val).strip())
            return None if np.isnan(f) else f
        except Exception:
            return None

    def safe_price(val):
        f = safe_float(val)
        if f is None:
            return None
        if pd.isna(f):
            return None
        return f

    def apply_pricing(idx, ref_base, ref_revised, reason, ref_item_name):
        old_price = safe_price(df.at[idx, "Price"])
        old_markup = safe_price(df.at[idx, "Markup Price"]) if "Markup Price" in df.columns else None

        if mode == "slash":
            if ref_base is not None and ref_revised is not None:
                df.at[idx, "Markup Price"] = ref_base
                df.at[idx, "Price"] = ref_revised
            elif ref_base is None and ref_revised is not None:
                df.at[idx, "Markup Price"] = old_price if old_price else np.nan
                df.at[idx, "Price"] = ref_revised
            elif ref_base is not None and ref_revised is None:
                df.at[idx, "Markup Price"] = ref_base
            new_markup = safe_price(df.at[idx, "Markup Price"])
            if new_markup == 0:
                df.at[idx, "Markup Price"] = np.nan

        elif mode == "replace":
            if ref_revised is not None:
                df.at[idx, "Price"] = ref_revised
            df.at[idx, "Markup Price"] = np.nan

        df.at[idx, update_col] = "Yes"

        changed_rows.append({
            "Ref Item": ref_item_name,
            "Menu Item": str(df.at[idx, "Item"]) if "Item" in df.columns else "",
            "Category": str(df.at[idx, "Category"]) if "Category" in df.columns else "",
            "Old Selling Price": old_price,
            "New Selling Price": safe_price(df.at[idx, "Price"]),
            "Old Base Price": old_markup,
            "New Base Price": safe_price(df.at[idx, "Markup Price"]) if "Markup Price" in df.columns else None,
            "Reason": reason,
        })

    for m in confirmed_matches:
        r_idx = m["ref_index"]
        menu_idx = m["menu_index"]
        r = ref_df.loc[r_idx]

        ref_base = safe_float(r[base_col]) if base_col and str(r.get(base_col, "")).strip() not in ("", "nan") else None
        ref_revised = safe_float(r[revised_col]) if revised_col and str(r.get(revised_col, "")).strip() not in ("", "nan") else None
        ref_item_name = str(r[ref_item_col]) if ref_item_col else ""

        apply_pricing(menu_idx, ref_base, ref_revised, "auto match" if m.get("auto") else "user confirmed", ref_item_name)

        if addon_indices and menu_idx in addon_indices:
            for addon_idx in addon_indices[menu_idx]:
                apply_pricing(addon_idx, ref_base, ref_revised, "addon propagation", ref_item_name)

    # build summary: every ref row, matched or not
    detail_df = pd.DataFrame(changed_rows) if changed_rows else pd.DataFrame()
    matched_ref_indices = {m["ref_index"] for m in confirmed_matches}
    matched_map = {m["ref_index"]: m["menu_index"] for m in confirmed_matches}

    summary_rows = []
    for r_idx in ref_df.index:
        r = ref_df.loc[r_idx]
        ref_name = str(r[ref_item_col]) if ref_item_col else ""
        ref_revised_val = str(r[revised_col]).strip() if revised_col and pd.notna(r.get(revised_col)) else ""
        ref_base_val = str(r[base_col]).strip() if base_col and pd.notna(r.get(base_col)) else ""

        if r_idx in matched_ref_indices:
            menu_idx = matched_map[r_idx]
            menu_item = str(df.at[menu_idx, "Item"]) if "Item" in df.columns else ""
            new_price = safe_price(df.at[menu_idx, "Price"])
            new_markup = safe_price(df.at[menu_idx, "Markup Price"]) if "Markup Price" in df.columns else None
            summary_rows.append({
                "Ref Item": ref_name,
                "Ref Base Price": ref_base_val,
                "Ref Revised Price": ref_revised_val,
                "Matched To": menu_item,
                "New Menu Price": str(int(new_price)) if new_price and new_price == int(new_price) else (str(new_price) if new_price else ""),
                "New Markup Price": str(int(new_markup)) if new_markup and new_markup == int(new_markup) else (str(new_markup) if new_markup else ""),
                "Status": "Matched",
            })
        else:
            summary_rows.append({
                "Ref Item": ref_name,
                "Ref Base Price": ref_base_val,
                "Ref Revised Price": ref_revised_val,
                "Matched To": "",
                "New Menu Price": "",
                "New Markup Price": "",
                "Status": "Not matched — update manually",
            })

    audit_entry = pd.DataFrame(summary_rows)
    return df, audit_entry, detail_df
