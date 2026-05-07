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
    "gulab jamun": "gulab jamun", "gulabjamun": "gulab jamun",
    "halwa": "halwa", "kheer": "kheer", "payasam": "kheer",
    "ice cream": "ice cream", "icecream": "ice cream",
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


def token_set(text):
    return set(normalize(text).split())


def score_pair(ref_text, menu_text):
    """
    Multi-signal similarity score between two item name strings.
    Returns 0-100. Uses ratio, token_sort_ratio, token_set_ratio.
    Penalises cases where one string is a strict subset of the other
    (e.g. 'Rasmalai' vs 'Mango Rasmalai') to prefer exact matches.
    """
    rn = normalize(ref_text)
    mn = normalize(menu_text)

    if not rn or not mn:
        return 0

    s_ratio = fuzz.ratio(rn, mn)
    s_sort = fuzz.token_sort_ratio(rn, mn)
    s_set = fuzz.token_set_ratio(rn, mn)

    base = max(s_ratio, s_sort, s_set)

    ref_tokens = set(rn.split())
    menu_tokens = set(mn.split())

    if ref_tokens != menu_tokens:
        if ref_tokens < menu_tokens or menu_tokens < ref_tokens:
            extra = len(ref_tokens.symmetric_difference(menu_tokens))
            penalty = min(extra * 8, 30)
            base = max(0, base - penalty)

    return base


def match_items(menu_df, ref_df):
    """
    Context-aware matching:
    1. Compute all (ref_row, menu_row) scores upfront.
    2. Use a greedy globally-optimal assignment: best score first,
       each menu row can only be claimed by one ref row.
    3. Anything >= AUTO_THRESHOLD and uncontested -> auto match.
    4. Everything else -> HITL queue with top candidates.
    """
    menu_df = menu_df.copy().reset_index(drop=True)
    ref_df = ref_df.copy().reset_index(drop=True)

    ref_item_col = next(
        (c for c in ref_df.columns if "item" in c.lower()
         and "group" not in c.lower() and "addon" not in c.lower()),
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

    menu_items = [str(menu_df.at[i, menu_item_col]) for i in menu_df.index]
    menu_addons = [str(menu_df.at[i, menu_addon_col]) if menu_addon_col else "" for i in menu_df.index]

    score_matrix = {}

    for r_idx in ref_df.index:
        ref_item_raw = str(ref_df.at[r_idx, ref_item_col])

        is_addon = False
        if ref_addon_flag_col:
            flag = str(ref_df.at[r_idx, ref_addon_flag_col]).strip().lower()
            is_addon = flag in ("y", "yes", "1", "true")

        search_pool = menu_addons if (is_addon and menu_addon_col) else menu_items
        ref_cat = normalize(ref_df.at[r_idx, ref_cat_col]) if ref_cat_col else ""
        ref_subcat = normalize(ref_df.at[r_idx, ref_subcat_col]) if ref_subcat_col else ""
        ref_variant = normalize(ref_df.at[r_idx, ref_variant_col]) if ref_variant_col else ""

        for m_idx in menu_df.index:
            pool_val = search_pool[m_idx]
            item_score = score_pair(ref_item_raw, pool_val)
            if item_score < 50:
                continue

            total = item_score
            weight = 1.0

            if ref_cat and menu_cat_col:
                cs = fuzz.token_sort_ratio(ref_cat, normalize(str(menu_df.at[m_idx, menu_cat_col])))
                total += cs * 0.25
                weight += 0.25

            if ref_subcat and menu_subcat_col:
                ss = fuzz.token_sort_ratio(ref_subcat, normalize(str(menu_df.at[m_idx, menu_subcat_col])))
                total += ss * 0.15
                weight += 0.15

            if ref_variant and menu_variant_col:
                vs = fuzz.token_sort_ratio(ref_variant, normalize(str(menu_df.at[m_idx, menu_variant_col])))
                total += vs * 0.15
                weight += 0.15

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
        ref_variant = normalize(ref_df.at[r_idx, ref_variant_col]) if ref_variant_col else ""

        if r_idx in assigned_ref:
            m_idx, composite, is_addon = assigned_ref[r_idx]
            auto_matches.append({
                "ref_index": r_idx,
                "menu_index": int(menu_df.index[m_idx]),
                "item": ref_item_raw,
                "score": round(composite),
                "auto": True,
                "is_addon": is_addon,
            })
        else:
            is_addon = False
            if ref_addon_flag_col:
                flag = str(ref_df.at[r_idx, ref_addon_flag_col]).strip().lower()
                is_addon = flag in ("y", "yes", "1", "true")

            search_pool = menu_addons if (is_addon and menu_addon_col) else menu_items

            candidates_raw = [
                (m_idx, score_matrix[(r_idx, m_idx)][0], score_matrix[(r_idx, m_idx)][1])
                for m_idx in menu_df.index
                if (r_idx, m_idx) in score_matrix
            ]
            candidates_raw.sort(key=lambda x: x[1], reverse=True)
            top = candidates_raw[:5]

            candidate_display = []
            for m_idx, composite, item_score in top:
                pool_val = search_pool[m_idx]
                item_display = strip_ids(pool_val) if is_addon else strip_ids(menu_df.at[m_idx, menu_item_col])
                candidate_display.append({
                    "menu_index": int(menu_df.index[m_idx]),
                    "menu_item": item_display,
                    "menu_cat": strip_ids(str(menu_df.at[m_idx, menu_cat_col])) if menu_cat_col else "",
                    "menu_subcat": strip_ids(str(menu_df.at[m_idx, menu_subcat_col])) if menu_subcat_col else "",
                    "menu_variant": strip_ids(str(menu_df.at[m_idx, menu_variant_col])) if menu_variant_col else "",
                    "menu_price": str(menu_df.at[m_idx, menu_price_col]) if menu_price_col else "",
                    "score": round(composite),
                })

            hitl_queue.append({
                "ref_index": r_idx,
                "ref_item": ref_item_raw,
                "ref_cat": ref_cat,
                "ref_subcat": ref_subcat,
                "ref_variant": ref_variant,
                "is_addon": is_addon,
                "search_label": "addon" if is_addon else "item",
                "candidates": candidate_display,
                "score": round(top[0][1]) if top else 0,
            })

    return auto_matches, hitl_queue


def find_addon_rows(menu_df, item_name):
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
    df = menu_df.copy()
    changed_rows = []

    ref_item_col = next(
        (c for c in ref_df.columns if "item" in c.lower()
         and "group" not in c.lower() and "addon" not in c.lower()),
        None
    )
    base_col = next((c for c in ref_df.columns if "base" in c.lower() and "price" in c.lower()), None)
    if base_col is None:
        base_col = next((c for c in ref_df.columns if "base" in c.lower()), None)
    revised_col = next((c for c in ref_df.columns if "revised" in c.lower()), None)
    update_col = next(
        (c for c in df.columns if c.strip().lower().startswith("update required")),
        "Update Required ?"
    )

    def apply_pricing(idx, ref_base, ref_revised, reason, ref_item_name):
        old_price = df.at[idx, "Price"]
        old_markup = df.at[idx, "Markup Price"] if "Markup Price" in df.columns else None

        if mode == "slash":
            if pd.notna(ref_base) and pd.notna(ref_revised):
                df.at[idx, "Markup Price"] = ref_base
                df.at[idx, "Price"] = ref_revised
            elif pd.isna(ref_base) and pd.notna(ref_revised):
                df.at[idx, "Markup Price"] = ref_revised
            elif pd.notna(ref_base) and pd.isna(ref_revised):
                df.at[idx, "Markup Price"] = ref_base
            try:
                markup_val = float(df.at[idx, "Markup Price"])
                if markup_val == 0:
                    df.at[idx, "Markup Price"] = None
            except Exception:
                pass
        elif mode == "replace":
            if pd.notna(ref_revised):
                df.at[idx, "Price"] = ref_revised
            df.at[idx, "Markup Price"] = None

        df.at[idx, update_col] = "Yes"

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

        try:
            ref_base = float(r[base_col]) if base_col and str(r[base_col]).strip() not in ("", "nan") else None
        except Exception:
            ref_base = None
        try:
            ref_revised = float(r[revised_col]) if revised_col and str(r[revised_col]).strip() not in ("", "nan") else None
        except Exception:
            ref_revised = None

        ref_item_name = str(r[ref_item_col]) if ref_item_col else ""

        apply_pricing(menu_idx, ref_base, ref_revised, "auto match" if m.get("auto") else "user confirmed", ref_item_name)

        if addon_indices and menu_idx in addon_indices:
            for addon_idx in addon_indices[menu_idx]:
                apply_pricing(addon_idx, ref_base, ref_revised, "addon propagation", ref_item_name)

    if not changed_rows:
        return df, pd.DataFrame(), pd.DataFrame()

    detail_df = pd.DataFrame(changed_rows)
    items_affected = list({r["Menu Item"] for r in changed_rows if r["Menu Item"]})
    audit_entry = pd.DataFrame([{
        "Operation": mode,
        "Items Affected": len(changed_rows),
        "Item List": ", ".join(items_affected),
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }])

    return df, audit_entry, detail_df
