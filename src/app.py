import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from decision_engine import match_items, process_matches, find_addon_rows, strip_ids

st.set_page_config(page_title="Price Revision", layout="wide")

REQUIRED_MENU_COLUMNS = {"Price", "Category", "Brand SKU ID Type", "Item"}
METADATA_KEYWORDS = [
    "brandskuid", "item/variant/addon", "description",
    "cdn", "veg/non veg", "in stock", "yyyy-mm-dd", "hh:mm:ss"
]


# ─── NUMERIC HELPERS ──────────────────────────────────────────────────────────
# Price and Markup Price are stored as float64 throughout.
# NaN = no value. Never use strings for prices internally.

def to_float(val):
    if val is None:
        return np.nan
    if isinstance(val, float):
        return val
    if isinstance(val, (int, np.integer)):
        return float(val)
    s = str(val).strip()
    if s in ("", "nan", "None", "NaN", "none", "<NA>"):
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan


def fmt_price(val):
    f = to_float(val)
    if np.isnan(f):
        return ""
    return str(int(f)) if f == int(f) else str(round(f, 2))


def normalise_price_col(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.apply(to_float), errors="coerce").astype("float64")


# ─── SESSION SAVE / LOAD ──────────────────────────────────────────────────────
# Prices stored as float64. Backup dict (plain Python) guards against pickle corruption.

def save_df(df: pd.DataFrame) -> None:
    df = df.copy()
    for col in ["Price", "Markup Price"]:
        if col in df.columns:
            df[col] = normalise_price_col(df[col])
    # store every non-nan price as plain Python (int, str, float) tuples
    # tuples of native Python types are 100% pickle-stable
    backup = []
    for col in ["Price", "Markup Price"]:
        if col in df.columns:
            for idx in df.index:
                v = df.at[idx, col]
                if isinstance(v, float) and not np.isnan(v):
                    backup.append((int(idx), col, float(v)))
    st.session_state.menu_df = df
    st.session_state._price_backup = backup


def load_df() -> pd.DataFrame:
    df = st.session_state.menu_df.copy()
    # force both price columns to float64 first
    for col in ["Price", "Markup Price"]:
        if col in df.columns:
            df[col] = normalise_price_col(df[col])
    # restore every value from backup — overwrite, not just fill NaN
    # this is the only source of truth for prices
    backup = st.session_state.get("_price_backup", [])
    for (idx, col, val) in backup:
        if idx in df.index and col in df.columns:
            df.at[idx, col] = float(val)
    return df


# ─── MISC HELPERS ─────────────────────────────────────────────────────────────

def detect_freeze_index(df):
    for i in df.index:
        try:
            row_text = " ".join(str(x).lower() for x in df.loc[i].values if pd.notna(x))
        except Exception:
            continue
        if any(k in row_text for k in METADATA_KEYWORDS):
            return i + 1
    return 0


def clean_label(val):
    return strip_ids(val)


def find_update_col(df):
    return next(
        (c for c in df.columns if c.strip().lower().startswith("update required")),
        "Update Required ?"
    )


def section(title):
    st.markdown(f"### {title}")
    st.markdown("---")


def validate_ref_csv(df: pd.DataFrame) -> tuple[bool, str]:
    cols_lower = [c.strip().lower() for c in df.columns]
    has_item = any(
        "item" in c and "group" not in c and "addon" not in c
        for c in cols_lower
    )
    has_revised = any("revised" in c for c in cols_lower)
    if not has_item:
        return False, "Missing an 'Item Name' column."
    if not has_revised:
        return False, "Missing a 'Revised Price' column."
    return True, ""


def render_preview(label: str):
    df = st.session_state.get(f"_preview_df_{label}")
    if df is None:
        return
    display = df.copy()
    for col in ["Price", "Markup Price"]:
        if col in display.columns:
            display[col] = display[col].apply(fmt_price)
    display = display.fillna("")
    n = len(display)
    expanded = st.session_state.get(f"_preview_exp_{label}", False)
    st.dataframe(display if expanded else display.head(8), use_container_width=True)
    if n > 8:
        lbl = f"▲ Collapse" if expanded else f"▼ Show all {n} rows"
        if st.button(lbl, key=f"_prev_btn_{label}"):
            st.session_state[f"_preview_exp_{label}"] = not expanded
            st.rerun()


def store_preview(label: str, df: pd.DataFrame):
    st.session_state[f"_preview_df_{label}"] = df.copy()


# ─── SESSION DEFAULTS ─────────────────────────────────────────────────────────

DEFAULTS = {
    "menu_df": None, "freeze_idx": 0, "original_name": None,
    "last_file_name": None, "_price_backup": {},
    "ref_df": None, "last_ref_name": None,
    "auto_matches": [], "hitl_queue": [], "hitl_cursor": 0,
    "hitl_history": [],
    "confirmed_matches": [], "addon_indices": {},
    "audit_log": None, "slash_snapshot": None,
    "slash_removal_done": False, "remove_slash_only_done": False,
    "flat_discount_done": False, "ref_apply_done": False,
    "ref_apply_count": 0, "last_ref_mode": None,
    "ref_uploader_key": 0, "matching_ran": False,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─── UPLOAD ────────────────────────────────────────────────────────────────────

st.title("Price Revision")
st.markdown(" ")
section("① Upload Menu CSV")

menu_file = st.file_uploader("Upload MENU CSV", type=["csv"], key="menu_uploader")

if menu_file is None and st.session_state.last_file_name is not None:
    for k in list(st.session_state.keys()):
        if k != "menu_uploader":
            del st.session_state[k]
    st.rerun()

if menu_file is not None and menu_file.name != st.session_state.last_file_name:
    raw = pd.read_csv(menu_file, dtype=str)
    raw.columns = raw.columns.str.strip()
    raw = raw.apply(lambda col: col.map(lambda x: x.strip() if isinstance(x, str) else x))

    missing = REQUIRED_MENU_COLUMNS - set(raw.columns)
    if missing:
        st.error(f"Invalid menu CSV — missing columns: {', '.join(missing)}")
        st.stop()

    for col, default in [
        ("Update Required ?", ""), ("Markup Price", ""),
        ("Start Date", ""), ("Start Time", ""),
        ("Revert Date", ""), ("Revert Time", ""),
    ]:
        if col not in raw.columns:
            raw[col] = default

    # clear previous Yes flags so re-uploads start clean
    raw["Update Required ?"] = raw["Update Required ?"].apply(
        lambda x: "" if str(x).strip() == "Yes" else x
    )

    freeze_idx_new = detect_freeze_index(raw)

    # build slashing snapshot before numeric normalisation
    snap_p = pd.to_numeric(raw["Price"], errors="coerce")
    snap_m = pd.to_numeric(raw.get("Markup Price", pd.Series(dtype=float)), errors="coerce")
    snap_mask = snap_p.notna() & snap_m.notna() & (snap_m > snap_p)
    snap_mask.iloc[:freeze_idx_new] = False
    if snap_mask.sum() > 0:
        snap_rows = raw[snap_mask].copy()
        snap_rows["_p"] = snap_p[snap_mask]
        snap_rows["_m"] = snap_m[snap_mask]
        snap_rows["Slashing %"] = ((1 - snap_rows["_p"] / snap_rows["_m"]) * 100).round(2).astype(str) + "%"
        snap_rows = snap_rows.drop(columns=["_p", "_m"])
        st.session_state.slash_snapshot = {
            "count": int(snap_mask.sum()),
            "sample_price": float(snap_p[snap_mask].iloc[0]),
            "sample_markup": float(snap_m[snap_mask].iloc[0]),
            "rows": snap_rows,
        }
    else:
        st.session_state.slash_snapshot = None

    # reset all state
    for k in ["auto_matches", "hitl_queue", "confirmed_matches", "addon_indices", "hitl_history"]:
        st.session_state[k] = [] if k != "addon_indices" else {}
    st.session_state.hitl_cursor = 0
    st.session_state.audit_log = None
    st.session_state.ref_df = None
    st.session_state.last_ref_name = None
    for k in ["slash_removal_done", "remove_slash_only_done", "flat_discount_done",
              "ref_apply_done", "matching_ran"]:
        st.session_state[k] = False
    st.session_state.ref_apply_count = 0
    st.session_state.freeze_idx = freeze_idx_new
    st.session_state.original_name = os.path.splitext(menu_file.name)[0]
    st.session_state.last_file_name = menu_file.name

    save_df(raw)
    store_preview("initial", load_df().iloc[freeze_idx_new:])

if st.session_state.menu_df is None:
    st.info("Upload a menu CSV to get started.")
    st.stop()

freeze_idx = st.session_state.freeze_idx
df_view = load_df()
working_view = df_view.iloc[freeze_idx:]
available_types = sorted(working_view["Brand SKU ID Type"].dropna().unique().tolist())
item_count = working_view[working_view["Brand SKU ID Type"] == "Item"]["Item"].nunique()

c1, c2, c3 = st.columns(3)
c1.metric("Working Rows", len(working_view))
c2.metric("Unique Items", item_count)
c3.metric("Row Types", " · ".join(available_types) if available_types else "N/A")
render_preview("initial")
st.markdown(" ")


# ─── SLASHING DETECTION ────────────────────────────────────────────────────────

section("② Slashing Detection")
snap = st.session_state.slash_snapshot

if snap is None:
    st.success("No existing slashing detected.")
else:
    try:
        pct = round((1 - snap["sample_price"] / snap["sample_markup"]) * 100, 2)
        st.warning(
            f"Existing Discount on **{snap['count']} rows** — approx **{pct}% off** "
            f"(e.g. ₹{int(snap['sample_price'])} selling / ₹{int(snap['sample_markup'])} base)"
        )
    except Exception:
        st.warning(f"Existing Discount on {snap['count']} rows.")

    with st.expander("View slashed rows"):
        show_cols = [c for c in ["Category", "Subcategory", "Item", "Price", "Markup Price", "Slashing %"]
                     if c in snap["rows"].columns]
        snap_display = snap["rows"][show_cols].reset_index(drop=True).copy()
        for col in ["Price", "Markup Price"]:
            if col in snap_display.columns:
                snap_display[col] = pd.to_numeric(snap_display[col], errors="coerce").apply(fmt_price)
        st.dataframe(snap_display, use_container_width=True)

    if not st.session_state.slash_removal_done:
        remove_choice = st.radio(
            "Remove existing slashing?",
            ["No — keep as is", "Yes — restore original prices"],
            key="remove_slash_radio",
        )
        if remove_choice == "Yes — restore original prices":
            if st.button("Confirm removal", key="confirm_removal_btn"):
                df_rm = load_df()
                update_col_rm = find_update_col(df_rm)
                slashed = [
                    i for i in df_rm.index[freeze_idx:]
                    if pd.notna(df_rm.at[i, "Price"])
                    and pd.notna(df_rm.at[i, "Markup Price"])
                    and df_rm.at[i, "Markup Price"] > df_rm.at[i, "Price"]
                ]
                for i in slashed:
                    df_rm.at[i, "Price"] = df_rm.at[i, "Markup Price"]
                    df_rm.at[i, "Markup Price"] = np.nan
                    df_rm.at[i, update_col_rm] = "Yes"
                save_df(df_rm)
                st.session_state.slash_removal_done = True
                # store preview as fresh numeric df — flat discount will load_df() fresh
                store_preview("post-slash-removal", load_df().iloc[freeze_idx:])
                st.rerun()
    else:
        st.success(f"✓ Slashing removed from {snap['count']} rows.")
        render_preview("post-slash-removal")

st.markdown(" ")


# ─── OPERATION ────────────────────────────────────────────────────────────────

section("③ Operation")
operation = st.selectbox(
    "What would you like to do?",
    ["Apply flat % discount", "Use reference CSV", "Remove existing slashing only"],
    key="operation_select",
)
st.markdown(" ")


# ── FLAT DISCOUNT ─────────────────────────────────────────────────────────────

if operation == "Apply flat % discount":

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("**Discount %**")
        discount = st.number_input("Discount", min_value=1.0, max_value=99.0, value=20.0,
                                   step=0.5, key="discount_pct", label_visibility="collapsed")

        st.markdown(" ")
        st.markdown("**Scope** — row types to apply discount on")
        scope_selected = st.multiselect("Scope", options=available_types, default=available_types,
                                        key="scope_ms", label_visibility="collapsed")

        st.markdown(" ")
        use_min_price = st.toggle("Set minimum price condition", value=False, key="min_price_toggle")
        min_price = 0.0
        if use_min_price:
            min_price = st.number_input("Skip rows priced ≤ ₹", min_value=0.0, value=99.0,
                                        step=1.0, key="min_price_input")

    with col_right:
        st.markdown("**Apply to**")
        df_scope = load_df().iloc[freeze_idx:]

        all_cats_raw = sorted(df_scope["Category"].dropna().unique().tolist())
        cat_map = {clean_label(c): c for c in all_cats_raw}
        sel_cats_display = st.multiselect("Categories (blank = all)", options=list(cat_map.keys()),
                                          default=[], key="sel_cats_ms", placeholder="All categories")
        sel_cats_raw = [cat_map[c] for c in sel_cats_display] if sel_cats_display else all_cats_raw

        sel_subcats_raw, sel_items_raw = [], []
        subcats_in_scope = sorted(df_scope[df_scope["Category"].isin(sel_cats_raw)]["Subcategory"].dropna().unique())
        if subcats_in_scope:
            sub_map = {clean_label(s): s for s in subcats_in_scope}
            sel_subs_display = st.multiselect("Subcategories (blank = all)", options=list(sub_map.keys()),
                                              default=[], key="sel_subs_ms", placeholder="All subcategories")
            sel_subcats_raw = [sub_map[s] for s in sel_subs_display] if sel_subs_display else list(subcats_in_scope)

            items_in_scope = sorted(df_scope[
                df_scope["Category"].isin(sel_cats_raw) &
                df_scope["Subcategory"].isin(sel_subcats_raw) &
                (df_scope["Brand SKU ID Type"] == "Item")
            ]["Item"].dropna().unique())
            if items_in_scope:
                item_map = {clean_label(i): i for i in items_in_scope}
                sel_items_display = st.multiselect("Items (blank = all)", options=list(item_map.keys()),
                                                   default=[], key="sel_items_ms", placeholder="All items")
                sel_items_raw = [item_map[i] for i in sel_items_display] if sel_items_display else list(items_in_scope)

    st.markdown(" ")
    if st.button("Apply Discount", key="apply_flat_btn", type="primary"):
        # always load fresh numeric df — mutually exclusive from slashing state
        df_apply = load_df()
        update_col = find_update_col(df_apply)
        factor = (100 - discount) / 100
        applied = skipped_min = 0

        for i in df_apply.index[freeze_idx:]:
            sku_type = str(df_apply.at[i, "Brand SKU ID Type"] or "").strip()
            if scope_selected and sku_type in ("Item", "Variant", "Addon") and sku_type not in scope_selected:
                continue
            if sel_cats_raw and str(df_apply.at[i, "Category"] or "").strip() not in sel_cats_raw:
                continue
            if sel_subcats_raw and str(df_apply.at[i, "Subcategory"] or "").strip() not in sel_subcats_raw:
                continue
            if sel_items_raw and str(df_apply.at[i, "Item"] or "").strip() not in sel_items_raw:
                continue

            price_val = df_apply.at[i, "Price"]
            if not isinstance(price_val, (int, float)) or np.isnan(price_val) or price_val <= 0:
                continue
            if min_price > 0 and price_val <= min_price:
                skipped_min += 1
                continue

            df_apply.at[i, "Markup Price"] = price_val
            df_apply.at[i, "Price"] = round(price_val * factor)
            df_apply.at[i, update_col] = "Yes"
            applied += 1

        save_df(df_apply)
        st.session_state.flat_discount_done = True
        store_preview("post-discount", load_df().iloc[freeze_idx:])
        msg = f"{discount}% discount applied to {applied} rows."
        if skipped_min:
            msg += f" ({skipped_min} skipped — price ≤ ₹{int(min_price)})"
        st.success(msg)
        st.rerun()

    if st.session_state.get("flat_discount_done"):
        render_preview("post-discount")


# ── REFERENCE CSV ─────────────────────────────────────────────────────────────

elif operation == "Use reference CSV":

    col_mode, col_template = st.columns([3, 1])
    with col_mode:
        ref_mode = st.radio("Mode:", ["Slash Prices", "Update Prices Directly"],
                            key="ref_mode_radio", horizontal=True)
    with col_template:
        template_csv = (
            "Category (optional),Subcategory (optional),Item Name,Variant (optional),"
            "Base Price (optional),Revised Price,Add on (y/n)\n"
            "Woodfired Pastas,Woodfired - Grilled Chicken Pasta,"
            "Woodfired - Grilled Chicken White Sauce Pasta with Truffle Oil,Penne,795,590,n\n"
        )
        st.download_button("⬇ Download Template", data=template_csv.encode("utf-8"),
                           file_name="SLASH-TEMPLATE.csv", mime="text/csv", key="template_dl")

    mode = "slash" if ref_mode == "Slash Prices" else "replace"
    st.markdown(" ")

    def _reset_ref_state():
        for k in ["auto_matches", "hitl_queue", "confirmed_matches", "addon_indices", "hitl_history"]:
            st.session_state[k] = [] if k != "addon_indices" else {}
        st.session_state.hitl_cursor = 0
        for k in ["ref_apply_done", "matching_ran"]:
            st.session_state[k] = False
        st.session_state.ref_apply_count = 0
        st.session_state.ref_df = None
        st.session_state.last_ref_name = None
        for k in ["_preview_df_post-ref-update", "_preview_exp_post-ref-update"]:
            st.session_state.pop(k, None)

    if st.session_state.last_ref_mode != mode:
        if st.session_state.last_ref_mode is not None:
            _reset_ref_state()
            st.session_state.ref_uploader_key += 1
        st.session_state.last_ref_mode = mode

    ref_file = st.file_uploader("Upload Reference CSV", type=["csv"],
                                key=f"ref_uploader_{st.session_state.ref_uploader_key}")

    if ref_file is None and st.session_state.last_ref_name is not None:
        _reset_ref_state()

    if ref_file is not None:
        if ref_file.name != st.session_state.last_ref_name:
            ref_df_raw = pd.read_csv(ref_file, dtype=str)
            ref_df_raw.columns = ref_df_raw.columns.str.strip()
            ref_df_raw = ref_df_raw.apply(lambda col: col.map(lambda x: x.strip() if isinstance(x, str) else x))
            ref_df_raw = ref_df_raw.reset_index(drop=True)

            valid, err_msg = validate_ref_csv(ref_df_raw)
            if not valid:
                st.error(f"Invalid reference CSV — {err_msg} Download the template above for the correct format.")
                st.stop()

            _reset_ref_state()
            st.session_state.ref_df = ref_df_raw.copy()
            st.session_state.last_ref_name = ref_file.name

        ref_df = st.session_state.ref_df.copy()

        # Reference CSV preview — expandable inline
        n_ref = len(ref_df)
        ref_exp = st.session_state.get("_ref_preview_exp", False)
        with st.expander("Reference CSV preview", expanded=False):
            st.dataframe(ref_df if ref_exp else ref_df.head(8), use_container_width=True)
            if n_ref > 8:
                lbl = "▲ Collapse" if ref_exp else f"▼ Show all {n_ref} rows"
                if st.button(lbl, key="ref_prev_toggle"):
                    st.session_state["_ref_preview_exp"] = not ref_exp
                    st.rerun()

        st.markdown(" ")

        if st.button("Run Matching", key="run_match_btn", type="primary"):
            menu_working = load_df().iloc[freeze_idx:].copy().reset_index(drop=True)
            auto_m, hitl_q = match_items(menu_working, ref_df)
            st.session_state.auto_matches = auto_m
            st.session_state.hitl_queue = hitl_q
            st.session_state.hitl_cursor = 0
            st.session_state.hitl_history = []
            st.session_state.confirmed_matches = list(auto_m)
            st.session_state.addon_indices = {}
            st.session_state.matching_ran = True
            st.rerun()

        if st.session_state.matching_ran:
            n_auto = len(st.session_state.auto_matches)
            n_hitl = len(st.session_state.hitl_queue)
            col_a, col_b = st.columns(2)
            col_a.metric("Auto-matched", n_auto)
            col_b.metric("Needs review", n_hitl)
            st.markdown(" ")

            if n_auto > 0:
                menu_prev = load_df().iloc[freeze_idx:].reset_index(drop=True)
                ref_price_col = next((c for c in ref_df.columns if "revised" in c.lower()), None)
                auto_rows = []
                for m in st.session_state.auto_matches:
                    try:
                        matched_item = clean_label(str(menu_prev.at[m["menu_index"], "Item"]))
                        menu_price = fmt_price(menu_prev.at[m["menu_index"], "Price"])
                    except Exception:
                        matched_item, menu_price = "—", ""
                    ref_price = ""
                    if ref_price_col:
                        try:
                            ref_price = str(ref_df.iloc[m["ref_index"]][ref_price_col]).strip()
                        except Exception:
                            pass
                    auto_rows.append({
                        "Reference Item": m["item"],
                        "Ref Revised Price (₹)": ref_price,
                        "Matched To": matched_item,
                        "Menu Price (₹)": menu_price,
                        "SKU Type": m.get("menu_sku_type", ""),
                    })
                with st.expander(f"View {n_auto} auto-matched rows"):
                    st.dataframe(pd.DataFrame(auto_rows), use_container_width=True)

            queue = st.session_state.hitl_queue
            cursor = st.session_state.hitl_cursor
            ref_price_col = next((c for c in ref_df.columns if "revised" in c.lower()), None)

            if cursor < len(queue):
                item = queue[cursor]
                with st.expander(f"Review {cursor + 1} of {len(queue)}", expanded=True):
                    st.progress(cursor / len(queue), text=f"{cursor + 1} / {len(queue)}")
                    st.markdown(" ")

                    badge = "🔖 Addon" if item.get("is_addon") else "🍽 Item"
                    ref_price_display = ""
                    if ref_price_col:
                        try:
                            ref_price_display = str(ref_df.iloc[item["ref_index"]][ref_price_col]).strip()
                        except Exception:
                            pass
                    price_note = f" — ₹{ref_price_display}" if ref_price_display else ""
                    st.markdown(f"**{badge}** `{item['ref_item']}`{price_note}")

                    meta = []
                    if item.get("ref_cat"): meta.append(f"Cat: `{item['ref_cat']}`")
                    if item.get("ref_subcat"): meta.append(f"Sub: `{item['ref_subcat']}`")
                    if item.get("ref_variant"): meta.append(f"Variant: `{item['ref_variant']}`")
                    if meta:
                        st.markdown("  ·  ".join(meta))

                    candidates = item.get("candidates", [])
                    if not candidates:
                        st.info("No candidates found.")
                        col_skip, col_undo = st.columns([1, 1])
                        if col_skip.button("Skip →", key=f"skip_nc_{cursor}"):
                            st.session_state.hitl_history.append(("skip", cursor, None))
                            st.session_state.hitl_cursor += 1
                            st.rerun()
                    else:
                        st.caption("Candidates ranked by confidence — top is most likely.")
                        options = ["— Skip this item —"] + [
                            "  ·  ".join(filter(None, [
                                c["menu_item"] if c.get("menu_item") not in ("", "nan", None) else None,
                                f"[{c['menu_sku_type']}]" if c.get("menu_sku_type") not in ("", "nan", None) else None,
                                c["menu_cat"] if c.get("menu_cat") not in ("", "nan", None) else None,
                                c["menu_subcat"] if c.get("menu_subcat") not in ("", "nan", None) else None,
                                c["menu_variant"] if c.get("menu_variant") not in ("", "nan", None) else None,
                                f"₹{c['menu_price']}" if c.get("menu_price") not in ("", "nan", None) else None,
                            ]))
                            for c in candidates
                        ]
                        choice = st.radio("Select correct match:", options, key=f"hitl_{cursor}")

                        is_addon_item = item.get("is_addon", False)
                        apply_all = False
                        if is_addon_item:
                            apply_all = st.checkbox(
                                "Apply same pricing wherever this appears as an addon",
                                key=f"apply_all_{cursor}")

                        col_confirm, col_skip, col_undo = st.columns([2, 2, 1])

                        if col_confirm.button("Confirm ✓", key=f"confirm_{cursor}", type="primary"):
                            if choice != "— Skip this item —":
                                ci = options.index(choice) - 1
                                c = candidates[ci]
                                entry = {
                                    "ref_index": item["ref_index"],
                                    "menu_index": c["menu_index"],
                                    "item": item["ref_item"],
                                    "auto": False,
                                    "is_addon": item.get("is_addon", False),
                                }
                                st.session_state.confirmed_matches.append(entry)
                                st.session_state.hitl_history.append(("confirm", cursor, entry))
                                if apply_all:
                                    norm_name = item["ref_item"].lower().strip()
                                    for fut in queue[cursor + 1:]:
                                        if fut["ref_item"].lower().strip() == norm_name:
                                            for fc in fut.get("candidates", []):
                                                if fc["menu_item"].lower().strip() == c["menu_item"].lower().strip():
                                                    extra = {
                                                        "ref_index": fut["ref_index"],
                                                        "menu_index": fc["menu_index"],
                                                        "item": fut["ref_item"],
                                                        "auto": False,
                                                        "is_addon": fut.get("is_addon", False),
                                                    }
                                                    st.session_state.confirmed_matches.append(extra)
                                                    break
                            else:
                                st.session_state.hitl_history.append(("skip", cursor, None))
                            st.session_state.hitl_cursor += 1
                            st.rerun()

                        if col_skip.button("Skip →", key=f"skip_{cursor}"):
                            st.session_state.hitl_history.append(("skip", cursor, None))
                            st.session_state.hitl_cursor += 1
                            st.rerun()

                        # Rollback — undo last action
                        history = st.session_state.get("hitl_history", [])
                        if history and col_undo.button("↩ Undo", key=f"undo_{cursor}"):
                            last_action, last_cursor, last_entry = history.pop()
                            if last_action == "confirm" and last_entry and last_entry in st.session_state.confirmed_matches:
                                st.session_state.confirmed_matches.remove(last_entry)
                            st.session_state.hitl_cursor = last_cursor
                            st.session_state.hitl_history = history
                            st.rerun()

            else:
                if len(queue) > 0:
                    st.success(f"Review complete — {len(st.session_state.confirmed_matches)} matches confirmed.")
                elif n_auto > 0:
                    st.success(f"All {n_auto} items auto-matched.")

                st.markdown(" ")

                menu_full = load_df()
                addon_col_exists = any(c.strip().lower() == "addon" for c in menu_full.columns)
                if addon_col_exists and not st.session_state.ref_apply_done:
                    items_with_addons = {}
                    for m in st.session_state.confirmed_matches:
                        if not m.get("is_addon"):
                            addon_idx_list = find_addon_rows(menu_full, m["item"])
                            if addon_idx_list:
                                items_with_addons[m["menu_index"]] = (m["item"], addon_idx_list)
                    if items_with_addons:
                        st.info(f"{len(items_with_addons)} matched item(s) also appear as addons. Apply same pricing?")
                        apply_to_addons = st.radio("Apply to addon rows?",
                                                   ["No", "Yes — apply to all addon occurrences"],
                                                   key="addon_propagation_radio", horizontal=True)
                        st.session_state.addon_indices = {
                            menu_idx: addon_list
                            for menu_idx, (_, addon_list) in items_with_addons.items()
                        } if apply_to_addons.startswith("Yes") else {}

                if not st.session_state.ref_apply_done:
                    if st.button("Apply All Confirmed Matches", key="apply_confirmed_btn", type="primary"):
                        df_apply = load_df()
                        working_index_list = df_apply.iloc[freeze_idx:].index.tolist()
                        confirmed_mapped = []
                        for m in st.session_state.confirmed_matches:
                            try:
                                actual_idx = working_index_list[m["menu_index"]]
                                confirmed_mapped.append({**m, "menu_index": actual_idx})
                            except IndexError:
                                continue

                        addon_idx_mapped = {}
                        for orig_idx, addon_list in st.session_state.addon_indices.items():
                            try:
                                actual_orig = working_index_list[orig_idx]
                                addon_idx_mapped[actual_orig] = addon_list
                            except IndexError:
                                continue

                        updated_df, audit_df, detail_df = process_matches(
                            df_apply, ref_df, confirmed_mapped,
                            mode=mode, addon_indices=addon_idx_mapped
                        )
                        save_df(updated_df)
                        st.session_state.audit_log = audit_df
                        store_preview("post-ref-update", load_df().iloc[freeze_idx:])
                        st.session_state.ref_apply_done = True
                        st.session_state.ref_apply_count = len(confirmed_mapped)
                        st.rerun()

                if st.session_state.ref_apply_done:
                    st.success(f"✓ Pricing applied to {st.session_state.ref_apply_count} rows.")
                    render_preview("post-ref-update")


# ── REMOVE SLASHING ONLY ──────────────────────────────────────────────────────

elif operation == "Remove existing slashing only":
    df_r = load_df()
    slashed_r = [
        i for i in df_r.index[freeze_idx:]
        if pd.notna(df_r.at[i, "Price"]) and pd.notna(df_r.at[i, "Markup Price"])
        and df_r.at[i, "Markup Price"] > df_r.at[i, "Price"]
    ]

    if not st.session_state.get("remove_slash_only_done"):
        if not slashed_r:
            st.info("No slashing detected.")
        else:
            sp, sm = df_r.at[slashed_r[0], "Price"], df_r.at[slashed_r[0], "Markup Price"]
            pct = round((1 - sp / sm) * 100, 2)
            st.warning(f"Slashing active on {len(slashed_r)} rows (~{pct}% off).")
            if st.button("Remove All Slashing", key="remove_slash_only_btn", type="primary"):
                update_col_r = find_update_col(df_r)
                for i in slashed_r:
                    df_r.at[i, "Price"] = df_r.at[i, "Markup Price"]
                    df_r.at[i, "Markup Price"] = np.nan
                    df_r.at[i, update_col_r] = "Yes"
                save_df(df_r)
                st.session_state.remove_slash_only_done = True
                store_preview("post-removal", load_df().iloc[freeze_idx:])
                st.rerun()

    if st.session_state.get("remove_slash_only_done"):
        st.success("✓ Slashing removed.")
        render_preview("post-removal")


# ─── DATES ────────────────────────────────────────────────────────────────────

st.markdown(" ")
section("④ Start & Revert Date / Time (Optional)")

use_start = st.toggle("Set a Start Date & Time", value=False, key="use_start_toggle")
use_revert = st.toggle("Set a Revert Date & Time", value=False, key="use_revert_toggle")

if use_start or use_revert:
    dt1, dt2 = st.columns(2)
    s_date = s_time = r_date = r_time = ""
    if use_start:
        with dt1:
            st.markdown("**Start**")
            sd = st.date_input("Start Date", key="start_date_input")
            st_t = st.time_input("Start Time", key="start_time_input", step=60)
            s_date, s_time = sd.strftime("%Y-%m-%d"), st_t.strftime("%H:%M:%S")
    if use_revert:
        with dt2:
            st.markdown("**Revert**")
            rd = st.date_input("Revert Date", key="revert_date_input")
            rt = st.time_input("Revert Time", key="revert_time_input", step=60)
            r_date, r_time = rd.strftime("%Y-%m-%d"), rt.strftime("%H:%M:%S")

    if st.button("Apply Dates to Updated Rows", key="apply_dates_btn", type="primary"):
        df_dated = load_df()
        update_col_d = find_update_col(df_dated)
        count = 0
        for i in df_dated.index[freeze_idx:]:
            if str(df_dated.at[i, update_col_d]).strip() == "Yes":
                if use_start:
                    if "Start Date" in df_dated.columns: df_dated.at[i, "Start Date"] = s_date
                    if "Start Time" in df_dated.columns: df_dated.at[i, "Start Time"] = s_time
                if use_revert:
                    if "Revert Date" in df_dated.columns: df_dated.at[i, "Revert Date"] = r_date
                    if "Revert Time" in df_dated.columns: df_dated.at[i, "Revert Time"] = r_time
                count += 1
        save_df(df_dated)
        st.success(f"Dates applied to {count} rows.")


# ─── DOWNLOAD ─────────────────────────────────────────────────────────────────

st.markdown(" ")
section("⑤ Download Output")

final_df = load_df().copy()
for col in ["Price", "Markup Price"]:
    if col in final_df.columns:
        final_df[col] = final_df[col].apply(fmt_price)
final_df = final_df.fillna("")

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
name = st.session_state.original_name

col_d1, col_d2 = st.columns(2)
col_d1.download_button(
    "⬇ Download Updated Menu CSV",
    data=final_df.to_csv(index=False).encode("utf-8"),
    file_name=f"{name}_processed_{ts}.csv", mime="text/csv",
)

audit = st.session_state.audit_log
if audit is not None and not audit.empty:
    col_d2.download_button(
        "⬇ Download Matching Summary",
        data=audit.to_csv(index=False).encode("utf-8"),
        file_name=f"{name}_summary_{ts}.csv", mime="text/csv",
    )
