import re
import math
import pandas as pd
import numpy as np
from rapidfuzz import fuzz
from collections import Counter

AUTO_THRESHOLD = 82
CANDIDATE_MIN  = 62
ITEM_NAME_GATE = 55
FOOD_GROUP_PENALTY = 35

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
    "chocolate": "chocolate", "choco": "chocolate",
    "vanilla": "vanilla", "strawberry": "strawberry",
    "mango": "mango", "banana": "banana", "oreo": "oreo",
}

FOOD_GROUPS = [
    {"burger", "chicken burger", "veg burger", "patty"},
    {"pizza", "calzone"},
    {"pasta", "spaghetti", "penne", "fettuccine", "lasagna"},
    {"biryani", "pulao", "fried rice", "rice"},
    {"noodles", "chowmein", "hakka"},
    {"momos", "dumpling", "dimsum"},
    {"shake", "milkshake", "smoothie", "icecream"},
    {"soup"},
    {"bread", "roti", "naan", "paratha", "puri", "bhatura"},
    {"curry", "gravy", "dal"},
    {"salad", "bowl"},
    {"dosa", "idli", "vada", "sambar"},
    {"shawarma", "kebab", "roll", "wrap", "sandwich"},
    {"sushi", "ramen"},
    {"taco", "burrito", "hummus", "falafel"},
    {"soft drink", "soda", "juice"},
    {"chocolate", "vanilla", "strawberry"},
]


def strip_ids(text):
    if pd.isna(text):
        return ""
    s = str(text).strip()
    s = re.sub(r"^\s*\(\s*[\w\-]+\s*\)\s*", "", s)
    return s.strip()


def normalize(text):
    if pd.isna(text) or str(text).strip() == "":
        return ""
    s = str(text).lower().strip()
    s = re.sub(r"^\s*\[\s*[\w\-]+\s*\]\s*", "", s)
    s = re.sub(r"^\s*\(\s*[\w\-]+\s*\)\s*", "", s)
    s = re.sub(r"[\[\]()]", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    for key, val in sorted(CANONICAL_MAP.items(), key=lambda x: -len(x[0])):
        s = re.sub(r"\b" + re.escape(key) + r"\b", val, s)
    return s


def _tokens(text):
    return set(normalize(text).split())


def _food_group(token_set):
    for gi, group in enumerate(FOOD_GROUPS):
        if token_set & group:
            return gi
    return -1


def build_idf(menu_df, item_col):
    doc_freq = Counter()
    docs = menu_df[item_col].dropna().tolist()
    N = max(len(docs), 1)
    for doc in docs:
        for tok in _tokens(str(doc)):
            doc_freq[tok] += 1
    return {tok: math.log((N + 1) / (freq + 1)) + 1 for tok, freq in doc_freq.items()}, N


def _idf_weighted_jaccard(ref_tokens, menu_tokens, idf):
    inter = ref_tokens & menu_tokens
    union = ref_tokens | menu_tokens
    if not union:
        return 0.0
    num = sum(idf.get(t, 1.0) for t in inter)
    den = sum(idf.get(t, 1.0) for t in union)
    return (num / den) * 100 if den else 0.0


def score_pair(ref_text, menu_text, idf=None):
    rn = normalize(ref_text)
    mn = normalize(menu_text)
    if not rn or not mn:
        return 0

    s_sort    = fuzz.token_sort_ratio(rn, mn)
    s_set     = fuzz.token_set_ratio(rn, mn)
    s_partial = fuzz.partial_ratio(rn, mn)
    base      = max(s_sort, s_set, s_partial)

    if idf:
        ref_tok  = set(rn.split())
        menu_tok = set(mn.split())
        idf_j    = _idf_weighted_jaccard(ref_tok, menu_tok, idf)
        base     = base * 0.7 + idf_j * 0.3

    ref_tok  = set(rn.split())
    menu_tok = set(mn.split())
    extra    = len(ref_tok.symmetric_difference(menu_tok))
    penalty  = min(extra * 3, 15)
    return max(0, base - penalty)


def _composite_score(
    ref_text, menu_text,
    ref_variant_raw, menu_variant,
    ref_cat, menu_cat,
    ref_subcat, menu_subcat,
    has_variant, idf=None
):
    item_score = score_pair(ref_text, menu_text, idf=idf)
    if item_score < ITEM_NAME_GATE:
        return None, item_score

    ref_group  = _food_group(_tokens(ref_text))
    menu_group = _food_group(_tokens(menu_text))
    cross_penalty = FOOD_GROUP_PENALTY if (
        ref_group != -1 and menu_group != -1 and ref_group != menu_group
    ) else 0

    composite = item_score - cross_penalty
    if composite < ITEM_NAME_GATE:
        return None, item_score

    weight = 1.0
    total  = composite

    if has_variant:
        v_score = score_pair(ref_variant_raw, menu_variant, idf=None)
        total  += v_score * 0.6
        weight += 0.6

    if ref_cat and menu_cat:
        cs      = fuzz.token_sort_ratio(ref_cat, normalize(menu_cat))
        total  += cs * 0.15
        weight += 0.15

    if ref_subcat and menu_subcat:
        ss      = fuzz.token_sort_ratio(ref_subcat, normalize(menu_subcat))
        total  += ss * 0.08
        weight += 0.08

    return total / weight, item_score


def effective_current_price(price, markup_price):
    def _f(v):
        try:
            f = float(str(v).strip())
            return f if not math.isnan(f) else None
        except Exception:
            return None
    vals = [x for x in (_f(price), _f(markup_price)) if x is not None]
    return max(vals) if vals else None


def match_items(menu_df, ref_df):
    menu_df = menu_df.copy().reset_index(drop=True)
    ref_df  = ref_df.copy().reset_index(drop=True)

    ref_item_col = next(
        (c for c in ref_df.columns
         if "item" in c.lower() and "group" not in c.lower() and "addon" not in c.lower()),
        None)
    if ref_item_col is None:
        raise ValueError("Could not find Item column in reference CSV.")

    ref_cat_col      = next((c for c in ref_df.columns if "category" in c.lower() and "sub" not in c.lower()), None)
    ref_subcat_col   = next((c for c in ref_df.columns if "subcategory" in c.lower()), None)
    ref_variant_col  = next((c for c in ref_df.columns if "variant" in c.lower()), None)
    ref_addon_flag_col = next(
        (c for c in ref_df.columns if "add" in c.lower() and ("on" in c.lower() or "addon" in c.lower())),
        None)

    menu_item_col   = next((c for c in menu_df.columns if c.strip().lower() == "item"), None)
    menu_addon_col  = next((c for c in menu_df.columns if c.strip().lower() == "addon"), None)
    menu_cat_col    = next((c for c in menu_df.columns if "category" in c.lower() and "sub" not in c.lower()), None)
    menu_subcat_col = next((c for c in menu_df.columns if "subcategory" in c.lower()), None)

    menu_vg1_col = next(
        (c for c in menu_df.columns if "variant group" in c.lower() and "l1" in c.lower()),
        None)

    menu_v1_col = next(
        (c for c in menu_df.columns
         if re.search(r"va[ir]+an[t]?s?\s*l1", c.lower())
         and "group" not in c.lower()
         and "food" not in c.lower()
         and "avail" not in c.lower()),
        None)

    menu_desc_col = next(
        (c for c in menu_df.columns
         if "description" in c.lower() or c.strip().lower() == "item description"),
        None)

    menu_markup_col = "Markup Price" if "Markup Price" in menu_df.columns else None
    menu_price_col  = "Price" if "Price" in menu_df.columns else None
    menu_sku_col    = next((c for c in menu_df.columns if "sku" in c.lower() and "type" in c.lower()), None)

    if menu_item_col is None:
        raise ValueError("Could not find Item column in menu CSV.")

    def _col_vals(col):
        if col:
            return [str(menu_df.at[i, col]) for i in menu_df.index]
        return [""] * len(menu_df)

    menu_items     = _col_vals(menu_item_col)
    menu_addons    = _col_vals(menu_addon_col)
    menu_cats      = _col_vals(menu_cat_col)
    menu_subcats   = _col_vals(menu_subcat_col)
    menu_vg1s      = _col_vals(menu_vg1_col)
    menu_v1s       = _col_vals(menu_v1_col)
    menu_descs     = _col_vals(menu_desc_col)
    menu_sku_types = _col_vals(menu_sku_col)

    def _safe_price_str(col, m_idx):
        if col is None:
            return ""
        try:
            v = float(menu_df.at[m_idx, col])
            if math.isnan(v):
                return ""
            return str(int(v)) if v == int(v) else str(round(v, 2))
        except Exception:
            return ""

    idf, _ = build_idf(menu_df, menu_item_col)

    score_matrix = {}

    for r_idx in ref_df.index:
        ref_item_raw = str(ref_df.at[r_idx, ref_item_col])
        is_addon = False
        if ref_addon_flag_col:
            flag     = str(ref_df.at[r_idx, ref_addon_flag_col]).strip().lower()
            is_addon = flag in ("y", "yes", "1", "true")

        ref_cat         = normalize(ref_df.at[r_idx, ref_cat_col])    if ref_cat_col    else ""
        ref_subcat      = normalize(ref_df.at[r_idx, ref_subcat_col]) if ref_subcat_col else ""
        ref_variant_raw = str(ref_df.at[r_idx, ref_variant_col]).strip() if ref_variant_col else ""
        ref_variant     = normalize(ref_variant_raw)
        has_variant     = ref_variant not in ("", "nan", "none")

        for m_idx in menu_df.index:
            sku_type = menu_sku_types[m_idx].strip()

            if is_addon:
                if sku_type != "Addon":
                    continue
            elif has_variant:
                if sku_type not in ("Variant", ""):
                    continue
            else:
                if sku_type in ("Variant", "Addon"):
                    continue

            ref_text  = ref_item_raw
            menu_text = menu_addons[m_idx] if is_addon else menu_items[m_idx]

            composite, item_score = _composite_score(
                ref_text, menu_text,
                ref_variant_raw if has_variant else "",
                menu_v1s[m_idx],
                ref_cat, menu_cats[m_idx],
                ref_subcat, menu_subcats[m_idx],
                has_variant,
                idf=idf,
            )

            if composite is None or composite < CANDIDATE_MIN:
                continue

            score_matrix[(r_idx, m_idx)] = (composite, item_score, is_addon)

    all_pairs     = sorted(score_matrix.items(), key=lambda x: x[1][0], reverse=True)
    assigned_menu = {}
    assigned_ref  = {}

    for (r_idx, m_idx), (composite, item_score, is_addon) in all_pairs:
        if r_idx in assigned_ref or m_idx in assigned_menu:
            continue
        if composite >= AUTO_THRESHOLD:
            assigned_ref[r_idx]  = (m_idx, composite, is_addon)
            assigned_menu[m_idx] = r_idx

    auto_matches = []
    hitl_queue   = []

    for r_idx in ref_df.index:
        ref_item_raw    = str(ref_df.at[r_idx, ref_item_col])
        ref_cat         = normalize(ref_df.at[r_idx, ref_cat_col])    if ref_cat_col    else ""
        ref_subcat      = normalize(ref_df.at[r_idx, ref_subcat_col]) if ref_subcat_col else ""
        ref_variant_raw = str(ref_df.at[r_idx, ref_variant_col]).strip() if ref_variant_col else ""
        ref_variant     = normalize(ref_variant_raw)
        has_variant     = ref_variant not in ("", "nan", "none")
        is_addon        = False
        if ref_addon_flag_col:
            flag     = str(ref_df.at[r_idx, ref_addon_flag_col]).strip().lower()
            is_addon = flag in ("y", "yes", "1", "true")

        if r_idx in assigned_ref:
            m_idx, composite, is_addon_m = assigned_ref[r_idx]
            auto_matches.append({
                "ref_index":     r_idx,
                "menu_index":    int(menu_df.index[m_idx]),
                "item":          ref_item_raw,
                "score":         round(composite),
                "auto":          True,
                "is_addon":      is_addon_m,
                "menu_sku_type": menu_sku_types[m_idx],
                "menu_price":    _safe_price_str(menu_price_col, m_idx),
                "menu_markup":   _safe_price_str(menu_markup_col, m_idx),
            })
        else:
            candidates_raw = [
                (m_idx, score_matrix[(r_idx, m_idx)][0], score_matrix[(r_idx, m_idx)][1])
                for m_idx in menu_df.index
                if (r_idx, m_idx) in score_matrix
            ]
            candidates_raw.sort(key=lambda x: x[1], reverse=True)

            candidate_display = []
            for m_idx, composite, item_score in candidates_raw:
                if is_addon:
                    item_display = strip_ids(menu_addons[m_idx])
                elif has_variant:
                    item_display = f"{strip_ids(menu_items[m_idx])} · {strip_ids(menu_v1s[m_idx])}"
                else:
                    item_display = strip_ids(menu_items[m_idx])

                candidate_display.append({
                    "menu_index":            int(menu_df.index[m_idx]),
                    "menu_item":             item_display,
                    "menu_sku_type":         menu_sku_types[m_idx],
                    "menu_cat":              strip_ids(menu_cats[m_idx]),
                    "menu_subcat":           strip_ids(menu_subcats[m_idx]),
                    "menu_variant_group_l1": strip_ids(menu_vg1s[m_idx]),
                    "menu_variant_l1":       strip_ids(menu_v1s[m_idx]),
                    "menu_description":      strip_ids(menu_descs[m_idx]),
                    "menu_variant":          strip_ids(menu_v1s[m_idx]),
                    "menu_price":            _safe_price_str(menu_price_col, m_idx),
                    "menu_markup":           _safe_price_str(menu_markup_col, m_idx),
                    "score":                 round(composite),
                })

            hitl_queue.append({
                "ref_index":    r_idx,
                "ref_item":     ref_item_raw,
                "ref_cat":      ref_cat,
                "ref_subcat":   ref_subcat,
                "ref_variant":  ref_variant,
                "is_addon":     is_addon,
                "search_label": "addon" if is_addon else ("variant" if has_variant else "item"),
                "candidates":   candidate_display,
                "score":        round(candidates_raw[0][1]) if candidates_raw else 0,
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


def detect_existing_slashing_on_rows(df, menu_indices):
    slashed = []
    for idx in menu_indices:
        try:
            p  = float(str(df.at[idx, "Price"]).strip())
            mk = float(str(df.at[idx, "Markup Price"]).strip())
            if not math.isnan(p) and not math.isnan(mk) and mk > p:
                slashed.append(idx)
        except Exception:
            pass
    return slashed


def process_matches(menu_df, ref_df, confirmed_matches, mode="slash",
                    addon_indices=None, slash_base_strategy=None):
    df = menu_df.copy()

    ref_item_col = next(
        (c for c in ref_df.columns
         if "item" in c.lower() and "group" not in c.lower() and "addon" not in c.lower()),
        None)
    base_col = next(
        (c for c in ref_df.columns if "base" in c.lower() and "price" in c.lower()),
        None)
    if base_col is None:
        base_col = next((c for c in ref_df.columns if "base" in c.lower()), None)
    revised_col = next((c for c in ref_df.columns if "revised" in c.lower()), None)
    update_col  = next(
        (c for c in df.columns if c.strip().lower().startswith("update required")),
        "Update Required ?")

    changed_rows = []

    def safe_float(val):
        if val is None:
            return None
        try:
            f = float(str(val).strip())
            return None if math.isnan(f) else f
        except Exception:
            return None

    def safe_price(val):
        return safe_float(val)

    def _fmt(v):
        if v is None:
            return ""
        try:
            return str(int(v)) if float(v) == int(float(v)) else str(round(float(v), 2))
        except Exception:
            return str(v)

    def apply_pricing(idx, ref_base, ref_revised, reason, ref_item_name):
        old_price  = safe_price(df.at[idx, "Price"])
        old_markup = safe_price(df.at[idx, "Markup Price"]) if "Markup Price" in df.columns else None
        old_effective = effective_current_price(old_price, old_markup)

        if mode == "slash":
            existing_slashed = (
                old_price is not None and old_markup is not None and old_markup > old_price
            )

            if existing_slashed:
                resolved_markup = effective_current_price(old_price, old_markup)
            else:
                resolved_markup = old_price

            if ref_base is not None:
                if slash_base_strategy == "higher":
                    final_markup = max(resolved_markup or 0, ref_base)
                elif slash_base_strategy == "lower":
                    final_markup = min(resolved_markup or 0, ref_base)
                elif slash_base_strategy == "ref":
                    final_markup = ref_base
                elif slash_base_strategy == "menu":
                    final_markup = resolved_markup
                else:
                    final_markup = ref_base
            else:
                final_markup = resolved_markup

            if final_markup and final_markup > 0:
                df.at[idx, "Markup Price"] = final_markup
            else:
                df.at[idx, "Markup Price"] = np.nan

            if ref_revised is not None:
                df.at[idx, "Price"] = ref_revised

        elif mode == "replace":
            if ref_revised is not None:
                df.at[idx, "Price"] = ref_revised
            df.at[idx, "Markup Price"] = np.nan

        df.at[idx, update_col] = "Yes"

        new_price_val  = safe_price(df.at[idx, "Price"])
        new_markup_val = safe_price(df.at[idx, "Markup Price"]) if "Markup Price" in df.columns else None
        new_effective  = effective_current_price(new_price_val, new_markup_val)

        price_increased = (
            old_price is not None and new_price_val is not None
            and new_price_val > old_price
        )

        changed_rows.append({
            "Ref Item":            ref_item_name,
            "Menu Item":           str(df.at[idx, "Item"]) if "Item" in df.columns else "",
            "Category":            str(df.at[idx, "Category"]) if "Category" in df.columns else "",
            "Old Selling Price":   old_price,
            "New Selling Price":   new_price_val,
            "Old Base Price":      old_markup,
            "New Base Price":      new_markup_val,
            "Old Effective Price": old_effective,
            "New Effective Price": new_effective,
            "Price Increased":     price_increased,
            "Reason":              reason,
        })

        return old_price, new_price_val, price_increased

    increase_flags = {}

    for m in confirmed_matches:
        r_idx    = m["ref_index"]
        menu_idx = m["menu_index"]
        r        = ref_df.loc[r_idx]

        ref_base    = safe_float(r[base_col])    if base_col    and str(r.get(base_col,    "")).strip() not in ("", "nan") else None
        ref_revised = safe_float(r[revised_col]) if revised_col and str(r.get(revised_col, "")).strip() not in ("", "nan") else None
        ref_item_name = str(r[ref_item_col]) if ref_item_col else ""

        old_p, new_p, increased = apply_pricing(
            menu_idx, ref_base, ref_revised,
            "auto match" if m.get("auto") else "user confirmed",
            ref_item_name)
        increase_flags[r_idx] = increased

        if addon_indices and menu_idx in addon_indices:
            for addon_idx in addon_indices[menu_idx]:
                apply_pricing(addon_idx, ref_base, ref_revised, "addon propagation", ref_item_name)

    detail_df       = pd.DataFrame(changed_rows) if changed_rows else pd.DataFrame()
    matched_ref_idx = {m["ref_index"] for m in confirmed_matches}
    matched_map     = {m["ref_index"]: m["menu_index"] for m in confirmed_matches}

    summary_rows = []
    for r_idx in ref_df.index:
        r            = ref_df.loc[r_idx]
        ref_name     = str(r[ref_item_col]) if ref_item_col else ""
        ref_rev_val  = str(r[revised_col]).strip() if revised_col and pd.notna(r.get(revised_col)) else ""
        ref_base_val = str(r[base_col]).strip()    if base_col    and pd.notna(r.get(base_col))    else ""

        if r_idx in matched_ref_idx:
            menu_idx   = matched_map[r_idx]
            menu_item  = str(df.at[menu_idx, "Item"]) if "Item" in df.columns else ""
            new_price  = safe_price(df.at[menu_idx, "Price"])
            new_markup = safe_price(df.at[menu_idx, "Markup Price"]) if "Markup Price" in df.columns else None
            eff_price  = effective_current_price(new_price, new_markup)
            price_increased = increase_flags.get(r_idx, False)
            summary_rows.append({
                "Ref Item Name":           ref_name,
                "Ref Base Price":          ref_base_val,
                "Ref Revised Price":       ref_rev_val,
                "Matched Menu Item":       menu_item,
                "New Menu Price":          _fmt(new_price),
                "New Markup Price":        _fmt(new_markup),
                "Effective Current Price": _fmt(eff_price),
                "Status":                  "Matched",
                "⚠ Price Increased":      "YES — REVIEW" if price_increased else "",
            })
        else:
            summary_rows.append({
                "Ref Item Name":           ref_name,
                "Ref Base Price":          ref_base_val,
                "Ref Revised Price":       ref_rev_val,
                "Matched Menu Item":       "",
                "New Menu Price":          "",
                "New Markup Price":        "",
                "Effective Current Price": "",
                "Status":                  "Not matched — update manually",
                "⚠ Price Increased":      "",
            })

    return df, pd.DataFrame(summary_rows), detail_df
