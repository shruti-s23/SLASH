import pandas as pd
from rapidfuzz import fuzz

# -------------------------
# NORMALIZATION
# -------------------------
def norm(x):
    if pd.isna(x):
        return ""
    return str(x).lower().strip()


# -------------------------
# EFFECTIVE CURRENT PRICE
# -------------------------
def effective_current_price(price, markup_price):
    """
    Effective Current Price = MAX(Price, Markup Price)
    Handles NaN / None safely.
    """
    p = pd.to_numeric(price, errors="coerce")
    mk = pd.to_numeric(markup_price, errors="coerce")

    if pd.notna(p) and pd.notna(mk):
        return max(p, mk)
    if pd.notna(p):
        return p
    if pd.notna(mk):
        return mk
    return None


# -------------------------
# CANDIDATE SCORING
# -------------------------
def _field(d, key):
    return norm(d.get(key, ""))


def score_candidate(ref_row, menu_row):
    """
    Score a menu row as a candidate for a reference row.
    Prioritizes:
      1. Item Name similarity
      2. Brand SKU Type match
      3. Variant Group L1 match
      4. Variants L1 similarity
      5. Description similarity (if available)
      6. Other variant-identifying attributes
    Returns a score (0-100+) or None if the candidate is not viable.
    """

    # ---- ITEM NAME (primary, mandatory) ----
    ref_item = _field(ref_row, "Item Name") or _field(ref_row, "Item")
    menu_item = _field(menu_row, "Item")

    if not ref_item or not menu_item:
        return None

    name_score = fuzz.token_sort_ratio(ref_item, menu_item)

    # Hard gate: item name must be reasonably similar, otherwise reject
    # immediately (prevents unrelated items like "Aloo Paratha" matching
    # against "Ajwain Paratha").
    if name_score < 80:
        return None

    score = name_score

    # ---- BRAND SKU TYPE ----
    ref_type = _field(ref_row, "Brand SKU ID Type") or _field(ref_row, "Type")
    menu_type = _field(menu_row, "Brand SKU ID Type")

    if ref_type and menu_type:
        if ref_type == menu_type:
            score += 15
        else:
            score -= 25  # different SKU type => strongly discourage

    # ---- VARIANT GROUP L1 ----
    ref_vg1 = _field(ref_row, "Variant Group L1")
    menu_vg1 = _field(menu_row, "Variant Group L1")

    if ref_vg1 and menu_vg1:
        if ref_vg1 == menu_vg1:
            score += 8
        else:
            score -= 15

    # ---- VARIANTS L1 (the actual variant value, e.g. "2 pcs") ----
    ref_v1 = _field(ref_row, "Variants L1") or _field(ref_row, "Variant")
    menu_v1 = _field(menu_row, "Vaiants L1") or _field(menu_row, "Variants L1")

    if ref_v1 and menu_v1:
        v1_score = fuzz.token_sort_ratio(ref_v1, menu_v1)
        if ref_v1 == menu_v1:
            score += 20
        elif v1_score >= 85:
            score += 10
        else:
            # Distinguishes "2 pcs" vs "4 pcs" — penalise mismatches hard
            score -= 30
    elif ref_v1 and not menu_v1:
        # Reference specifies a variant but candidate has none — slight penalty
        score -= 5
    elif menu_v1 and not ref_v1:
        # Candidate has a variant the reference didn't specify — slight penalty
        score -= 5

    # ---- DESCRIPTION (if available) ----
    ref_desc = _field(ref_row, "Item Description") or _field(ref_row, "Description")
    menu_desc = _field(menu_row, "Item Description") or _field(menu_row, "Description")

    if ref_desc and menu_desc:
        desc_score = fuzz.token_sort_ratio(ref_desc, menu_desc)
        score += (desc_score - 50) / 5  # mild influence: -10 to +10

    # ---- CATEGORY / SUBCATEGORY (secondary, optional) ----
    ref_cat = _field(ref_row, "Category")
    menu_cat = _field(menu_row, "Category")
    if ref_cat and menu_cat:
        if ref_cat == menu_cat:
            score += 5
        else:
            score -= 10

    ref_subcat = _field(ref_row, "Subcategory")
    menu_subcat = _field(menu_row, "Subcategory")
    if ref_subcat and menu_subcat:
        if ref_subcat == menu_subcat:
            score += 5
        else:
            score -= 10

    return score


# -------------------------
# CANDIDATE GENERATION (for HITL / review)
# -------------------------
CANDIDATE_DISPLAY_COLS_MENU = [
    "Brand SKU ID Type",
    "Item",
    "Variant Group L1",
    "Vaiants L1",
    "Item Description",
]

CANDIDATE_MIN_SCORE = 60  # below this, a menu row is not considered a candidate at all


def get_candidates(ref_row, menu_df, top_n=5):
    """
    Returns a DataFrame of the top-N most relevant menu rows for the given
    reference row, with only genuinely similar items surfaced (filtered by
    CANDIDATE_MIN_SCORE), including all variant-identifying display columns.
    """
    scored = []

    for m_idx, m in menu_df.iterrows():
        s = score_candidate(ref_row, m)
        if s is None:
            continue
        if s < CANDIDATE_MIN_SCORE:
            continue
        scored.append((s, m_idx))

    scored.sort(key=lambda x: x[0], reverse=True)
    scored = scored[:top_n]

    rows = []
    for s, m_idx in scored:
        m = menu_df.loc[m_idx]
        row = {
            "menu_index": m_idx,
            "Score": round(s, 1),
        }
        for col in CANDIDATE_DISPLAY_COLS_MENU:
            row[col] = m.get(col, "")
        rows.append(row)

    return pd.DataFrame(rows)


# -------------------------
# CORE MATCHING ENGINE
# -------------------------
def match_items(menu_df, ref_df):
    """
    For each reference row, find the single best matching menu row using
    the multi-field candidate score. Only genuinely similar items are
    accepted as matches.
    """

    matches = []

    for r_idx, r in ref_df.iterrows():

        best_score = None
        best_idx = None

        for m_idx, m in menu_df.iterrows():
            s = score_candidate(r, m)
            if s is None:
                continue
            if s < CANDIDATE_MIN_SCORE:
                continue

            if best_score is None or s > best_score:
                best_score = s
                best_idx = m_idx

        if best_idx is None:
            continue

        ref_item_name = _field(r, "Item Name") or _field(r, "Item")

        matches.append({
            "ref_index": r_idx,
            "menu_index": best_idx,
            "item": ref_item_name,
            "score": round(best_score, 1),
        })

    return matches


# -------------------------
# APPLY UPDATES (WITH FLAGGING)
# -------------------------
def process_matches(menu_df, ref_df, matches, require_hitl=False):

    df = menu_df.copy()

    for m in matches:

        r = ref_df.loc[m["ref_index"]]
        menu_idx = m["menu_index"]

        ref_price = pd.to_numeric(
            r.get("Revised Price", r.get("Revised price")), errors="coerce"
        )

        current_price = df.at[menu_idx, "Price"]
        current_markup = df.at[menu_idx, "Markup Price"] if "Markup Price" in df.columns else None

        # Effective current price = MAX(Price, Markup Price)
        eff_current = effective_current_price(current_price, current_markup)

        flag = "OK"

        if eff_current is not None and pd.notna(ref_price):

            if eff_current > ref_price:
                flag = "PRICE_HIGH"
            elif eff_current < ref_price:
                flag = "PRICE_LOW"

        # HITL gate (optional usage)
        if require_hitl and flag != "OK":
            df.at[menu_idx, "HITL Required"] = "Yes"
            df.at[menu_idx, "Match Flag"] = flag
            continue

        # APPLY UPDATE
        df.at[menu_idx, "Price"] = ref_price
        df.at[menu_idx, "Update Required ?"] = "Yes"
        df.at[menu_idx, "Match Flag"] = flag

    return df


# -------------------------
# POST-MATCHING PREVIEW SUMMARY
# -------------------------
def build_match_summary(menu_df, ref_df, matches):
    """
    Builds the post-matching preview summary:
      - Ref Item Name
      - Ref Base Price
      - Matched Menu Item
      - Effective Current Price = MAX(Price, Markup Price)
    """

    rows = []

    for m in matches:

        r = ref_df.loc[m["ref_index"]]
        menu_idx = m["menu_index"]
        menu_row = menu_df.loc[menu_idx]

        ref_item_name = _field(r, "Item Name") or _field(r, "Item")
        ref_base_price = pd.to_numeric(
            r.get("Base Price", r.get("Revised Price")), errors="coerce"
        )

        menu_item_name = menu_row.get("Item", "")
        current_price = menu_row.get("Price", None)
        current_markup = menu_row.get("Markup Price", None)

        eff_current_price = effective_current_price(current_price, current_markup)

        rows.append({
            "Ref Item Name": ref_item_name,
            "Ref Base Price": ref_base_price,
            "Matched Menu Item": menu_item_name,
            "Effective Current Price": eff_current_price,
        })

    return pd.DataFrame(rows)
