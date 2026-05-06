import streamlit as st
import pandas as pd
import os
import sys
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from decision_engine import match_items, process_matches, find_addon_rows, strip_ids

st.set_page_config(page_title="Price Revision", layout="wide")

REQUIRED_COLUMNS = {"Price", "Category", "Brand SKU ID Type", "Item"}

METADATA_KEYWORDS = [
    "brandskuid", "item/variant/addon", "description",
    "cdn", "veg/non veg", "in stock", "yyyy-mm-dd", "hh:mm:ss"
]

                                                                                

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

def show_preview(df, label="Preview"):
    """Show preview and store df so the expand toggle works across reruns."""
    st.session_state[f"_preview_df_{label}"] = df
    _render_preview(label)

def _render_preview(label):
    df = st.session_state.get(f"_preview_df_{label}")
    if df is None:
        return
    n = len(df)
    expanded = st.session_state.get(f"_preview_expanded_{label}", False)
    st.dataframe(df if expanded else df.head(8), use_container_width=True)
    if n > 8:
        btn_label = f"▲ Collapse {label}" if expanded else f"▼ Show all {n} rows — {label}"
        if st.button(btn_label, key=f"_preview_btn_{label}"):
            st.session_state[f"_preview_expanded_{label}"] = not expanded
            st.rerun()

def section(title):
    st.markdown(f"### {title}")
    st.markdown("---")

                                                                                

for k, v in {
    "menu_df": None,
    "freeze_idx": 0,
    "original_name": None,
    "last_file_name": None,
    "ref_df": None,
    "last_ref_name": None,
    "auto_matches": [],
    "hitl_queue": [],
    "hitl_cursor": 0,
    "confirmed_matches": [],
    "addon_indices": {},
    "audit_log": None,
    "slash_snapshot": None,
    "slash_removal_done": False,
    "remove_slash_only_done": False,
    "flat_discount_done": False,
    "ref_apply_done": False,
    "ref_apply_count": 0,
    "last_ref_mode": None,
    "ref_uploader_key": 0,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

                                                                                

st.title("Price Revision")
st.markdown(" ")

section("① Upload Menu CSV")

menu_file = st.file_uploader("Upload MENU CSV", type=["csv"], key="menu_uploader")

                                 
if menu_file is None and st.session_state.last_file_name is not None:
    for k in list(st.session_state.keys()):
        if k not in ("menu_uploader",):
            del st.session_state[k]
    st.rerun()

if menu_file is not None and menu_file.name != st.session_state.last_file_name:
    raw = pd.read_csv(menu_file, dtype=str)
    raw.columns = raw.columns.str.strip()
    raw = raw.apply(lambda col: col.map(lambda x: x.strip() if isinstance(x, str) else x))

    missing = REQUIRED_COLUMNS - set(raw.columns)
    if missing:
        st.error(f"Invalid menu CSV — missing columns: {', '.join(missing)}")
        st.stop()

    if "Update Required ?" not in raw.columns:
        raw["Update Required ?"] = ""
    if "Markup Price" not in raw.columns:
        raw["Markup Price"] = ""

    for k in ["auto_matches", "hitl_queue", "confirmed_matches", "addon_indices"]:
        st.session_state[k] = [] if k != "addon_indices" else {}
    st.session_state.hitl_cursor = 0
    st.session_state.audit_log = None
    st.session_state.ref_df = None
    st.session_state.last_ref_name = None
    st.session_state.slash_removal_done = False
    st.session_state.remove_slash_only_done = False
    st.session_state.flat_discount_done = False
    st.session_state.ref_apply_done = False
    st.session_state.ref_apply_count = 0

    freeze_idx_new = detect_freeze_index(raw)
    st.session_state.menu_df = raw.copy()
    st.session_state.freeze_idx = freeze_idx_new
    st.session_state.original_name = os.path.splitext(menu_file.name)[0]
    st.session_state.last_file_name = menu_file.name

                                                                
    snap_df = raw.copy()
    snap_df["Price"] = pd.to_numeric(snap_df["Price"], errors="coerce")
    snap_df["Markup Price"] = pd.to_numeric(snap_df["Markup Price"], errors="coerce")
    snap_working = snap_df.iloc[freeze_idx_new:]
    snap_mask = (
        snap_working["Price"].notna()
        & snap_working["Markup Price"].notna()
        & (snap_working["Markup Price"] > snap_working["Price"])
    )
    if snap_mask.sum() > 0:
        snap_rows = snap_working[snap_mask].copy()
        snap_rows["Slashing %"] = (
            (1 - snap_rows["Price"] / snap_rows["Markup Price"]) * 100
        ).round(2).astype(str) + "%"
        st.session_state.slash_snapshot = {
            "count": int(snap_mask.sum()),
            "sample_price": float(snap_working[snap_mask].iloc[0]["Price"]),
            "sample_markup": float(snap_working[snap_mask].iloc[0]["Markup Price"]),
            "rows": snap_rows,
        }
    else:
        st.session_state.slash_snapshot = None

if st.session_state.menu_df is None:
    st.info("Upload a menu CSV to get started.")
    st.stop()

df = st.session_state.menu_df.copy()
freeze_idx = st.session_state.freeze_idx
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
df["Markup Price"] = pd.to_numeric(df["Markup Price"], errors="coerce")
working = df.iloc[freeze_idx:].copy()

available_types = sorted(working["Brand SKU ID Type"].dropna().unique().tolist())
item_count = working[working["Brand SKU ID Type"] == "Item"]["Item"].nunique()

c1, c2, c3 = st.columns(3)
c1.metric("Working Rows", len(working))
c2.metric("Unique Items", item_count)
c3.metric("Row Types", " · ".join(available_types) if available_types else "N/A")

show_preview(df, "Menu CSV")
st.markdown(" ")

                                                                                

section("② Slashing Detection")

snap = st.session_state.slash_snapshot

if snap is None:
    st.success("No existing slashing detected.")
else:
    try:
        pct = round((1 - snap["sample_price"] / snap["sample_markup"]) * 100, 2)
        st.warning(
            f"Existing Discount detected on **{snap['count']} rows** — "
            f"approx **{pct}% off** "
            f"(e.g. ₹{int(snap['sample_price'])} selling / ₹{int(snap['sample_markup'])} base)"
        )
    except Exception:
        st.warning(f"Existing Discount detected on {snap['count']} rows.")

    with st.expander("View slashed rows"):
        display_cols = ["Category", "Subcategory", "Item", "Price", "Markup Price", "Slashing %"]
        show_cols = [c for c in display_cols if c in snap["rows"].columns]
        st.dataframe(snap["rows"][show_cols].reset_index(drop=True), use_container_width=True)

    if not st.session_state.slash_removal_done:
        remove_choice = st.radio(
            "Remove existing slashing?",
            ["No — keep as is", "Yes — restore original prices"],
            key="remove_slash_radio",
        )
        if remove_choice == "Yes — restore original prices":
            if st.button("Confirm removal", key="confirm_removal_btn"):
                df_rm = st.session_state.menu_df.copy()
                df_rm["Price"] = pd.to_numeric(df_rm["Price"], errors="coerce")
                df_rm["Markup Price"] = pd.to_numeric(df_rm["Markup Price"], errors="coerce")
                w_rm = df_rm.iloc[freeze_idx:]
                mask_rm = (
                    w_rm["Price"].notna()
                    & w_rm["Markup Price"].notna()
                    & (w_rm["Markup Price"] > w_rm["Price"])
                )
                update_col_rm = next(
                    (c for c in df_rm.columns if c.strip().lower().startswith("update required")),
                    "Update Required ?"
                )
                idx_rm = w_rm[mask_rm].index
                df_rm.loc[idx_rm, "Price"] = df_rm.loc[idx_rm, "Markup Price"]
                df_rm.loc[idx_rm, "Markup Price"] = None
                df_rm.loc[idx_rm, update_col_rm] = "Yes"
                st.session_state.menu_df = df_rm.copy()
                st.session_state.slash_removal_done = True
                                                                           
                st.session_state[f"_preview_df_post-slash-removal preview"] = df_rm.iloc[freeze_idx:].copy()
                st.rerun()
    else:
        st.success(f"✓ Slashing removed from {snap['count']} rows.")
        _render_preview("post-slash-removal preview")

st.markdown(" ")

                                                                                

section("③ Operation")

operation = st.selectbox(
    "What would you like to do?",
    [
        "Apply flat % discount",
        "Use reference CSV",
        "Remove existing slashing only",
    ],
    key="operation_select",
)
st.markdown(" ")

                                                                                

if operation == "Apply flat % discount":

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("**Discount %**")
        discount = st.number_input(
            "Discount", min_value=1.0, max_value=99.0, value=20.0, step=0.5,
            key="discount_pct", label_visibility="collapsed"
        )

        st.markdown(" ")
        st.markdown("**Scope** — row types to apply discount on")
        scope_selected = st.multiselect(
            "Scope", options=available_types, default=available_types,
            key="scope_ms", label_visibility="collapsed",
        )

        st.markdown(" ")
        use_min_price = st.toggle("Set minimum price condition", value=False, key="min_price_toggle")
        min_price = 0.0
        if use_min_price:
            min_price = st.number_input(
                "Skip rows priced ≤ ₹", min_value=0.0, value=99.0, step=1.0,
                key="min_price_input"
            )

    with col_right:
        st.markdown("**Apply to**")

        ws = st.session_state.menu_df.copy()
        ws["Price"] = pd.to_numeric(ws["Price"], errors="coerce")
        ws = ws.iloc[freeze_idx:]

                                    
        all_cats_raw = sorted(ws["Category"].dropna().unique().tolist())
        all_cats_display = [clean_label(c) for c in all_cats_raw]
        cat_map = dict(zip(all_cats_display, all_cats_raw))

        sel_cats_display = st.multiselect(
            "Categories (leave blank = all)",
            options=all_cats_display, default=[],
            key="sel_cats_ms", placeholder="All categories",
        )
        sel_cats_raw = [cat_map[c] for c in sel_cats_display] if sel_cats_display else all_cats_raw

                                                                     
        sel_subcats_raw = []
        sel_items_raw = []

        subcats_in_scope = sorted(
            ws[ws["Category"].isin(sel_cats_raw)]["Subcategory"].dropna().unique().tolist()
        )

        if subcats_in_scope:
            all_subs_display = [clean_label(s) for s in subcats_in_scope]
            sub_map = dict(zip(all_subs_display, subcats_in_scope))

            sel_subs_display = st.multiselect(
                "Subcategories (leave blank = all within chosen categories)",
                options=all_subs_display, default=[],
                key="sel_subs_ms", placeholder="All subcategories",
            )
            sel_subcats_raw = [sub_map[s] for s in sel_subs_display] if sel_subs_display else subcats_in_scope

                                                                     
            items_in_scope = sorted(
                ws[
                    ws["Category"].isin(sel_cats_raw)
                    & ws["Subcategory"].isin(sel_subcats_raw)
                    & (ws["Brand SKU ID Type"] == "Item")
                ]["Item"].dropna().unique().tolist()
            )

            if items_in_scope:
                all_items_display = [clean_label(i) for i in items_in_scope]
                item_map = dict(zip(all_items_display, items_in_scope))

                sel_items_display = st.multiselect(
                    "Items (leave blank = all within chosen subcategories)",
                    options=all_items_display, default=[],
                    key="sel_items_ms", placeholder="All items",
                )
                sel_items_raw = [item_map[i] for i in sel_items_display] if sel_items_display else items_in_scope

    st.markdown(" ")
    if st.button("Apply Discount", key="apply_flat_btn", type="primary"):
        df_apply = st.session_state.menu_df.copy()
        df_apply["Price"] = pd.to_numeric(df_apply["Price"], errors="coerce")
        df_apply["Markup Price"] = pd.to_numeric(df_apply["Markup Price"], errors="coerce")
        factor = (100 - discount) / 100

        update_col = next(
            (c for c in df_apply.columns if c.strip().lower().startswith("update required")),
            "Update Required ?"
        )

        working_idx = df_apply.index[freeze_idx:]
        mask = pd.Series(False, index=df_apply.index)
        mask.loc[working_idx] = True

        if scope_selected:
            mask = mask & df_apply["Brand SKU ID Type"].isin(scope_selected)
        if sel_cats_raw and len(sel_cats_raw) < len(
            df_apply["Category"].dropna().unique()
        ):
            mask = mask & df_apply["Category"].isin(sel_cats_raw)
        if sel_subcats_raw and len(sel_subcats_raw) < len(
            df_apply["Subcategory"].dropna().unique()
        ):
            mask = mask & df_apply["Subcategory"].isin(sel_subcats_raw)
        if sel_items_raw and len(sel_items_raw) < len(
            df_apply["Item"].dropna().unique()
        ):
            mask = mask & df_apply["Item"].isin(sel_items_raw)

        if min_price > 0:
            mask = mask & (df_apply["Price"] > min_price)

        mask = mask & df_apply["Price"].notna()

        for idx in df_apply[mask].index:
            original_price = float(df_apply.at[idx, "Price"])
            if pd.isna(original_price):
                continue

            existing_markup = df_apply.at[idx, "Markup Price"]
            if pd.isna(existing_markup) or existing_markup == 0:
                df_apply.at[idx, "Markup Price"] = original_price

            df_apply.at[idx, "Price"] = round(original_price * factor)
            df_apply.at[idx, update_col] = "Yes"

        st.session_state.menu_df = df_apply.copy()
        st.session_state.flat_discount_done = True
        show_preview(df_apply.iloc[freeze_idx:], "post-discount preview")
        st.success(f"{discount}% discount applied to {int(mask.sum())} rows.")
        st.rerun()

    if st.session_state.get("flat_discount_done"):
        _render_preview("post-discount preview")

                                                                                

elif operation == "Use reference CSV":

    col_mode, col_template = st.columns([3, 1])

    with col_mode:
        ref_mode = st.radio(
            "What to do with the reference CSV?",
            ["Slash Prices", "Update Prices Directly"],
            key="ref_mode_radio",
            horizontal=True,
        )

    with col_template:
        template_csv = (
            "Category,Subcategory,Item Name,Variant,Base Price,Revised Price,Add on (y/n)\n"
            "Woodfired Pastas,Woodfired - Grilled Chicken Pasta,Woodfired - Grilled Chicken White Sauce Pasta with Truffle Oil,Penne,795,590,n\n"
        )
        st.download_button(
            "⬇ Download Template",
            data=template_csv.encode("utf-8"),
            file_name="SLASH-TEMPLATE.csv",
            mime="text/csv",
            key="template_dl",
        )

    mode = "slash" if ref_mode == "Slash Prices" else "replace"
    st.markdown(" ")

    def _reset_ref_state():
        st.session_state.auto_matches = []
        st.session_state.hitl_queue = []
        st.session_state.hitl_cursor = 0
        st.session_state.confirmed_matches = []
        st.session_state.addon_indices = {}
        st.session_state.ref_apply_done = False
        st.session_state.ref_apply_count = 0
        st.session_state.ref_df = None
        st.session_state.last_ref_name = None
        st.session_state.pop("_preview_df_post-ref-update preview", None)
        st.session_state.pop("_preview_expanded_post-ref-update preview", None)

                                                                                          
    if st.session_state.last_ref_mode != mode:
        if st.session_state.last_ref_mode is not None:
            _reset_ref_state()
            st.session_state.ref_uploader_key += 1
        st.session_state.last_ref_mode = mode

    ref_file = st.file_uploader(
        "Upload Reference CSV", type=["csv"],
        key=f"ref_uploader_{st.session_state.ref_uploader_key}"
    )

    if ref_file is None:
        if st.session_state.last_ref_name is not None:
            st.session_state.last_ref_name = None
            st.session_state.ref_df = None
            _reset_ref_state()

    if ref_file is not None:
        if ref_file.name != st.session_state.last_ref_name:
            ref_df_raw = pd.read_csv(ref_file, dtype=str)
            ref_df_raw.columns = ref_df_raw.columns.str.strip()
            ref_df_raw = ref_df_raw.apply(lambda col: col.map(lambda x: x.strip() if isinstance(x, str) else x))
            ref_df_raw = ref_df_raw.reset_index(drop=True)
            st.session_state.ref_df = ref_df_raw.copy()
            st.session_state.last_ref_name = ref_file.name
            _reset_ref_state()
            st.session_state.ref_df = ref_df_raw.copy()
            st.session_state.last_ref_name = ref_file.name

        ref_df = st.session_state.ref_df.copy()

        with st.expander("Reference CSV preview"):
            st.dataframe(ref_df.head(8), use_container_width=True)

        st.markdown(" ")

        if st.button("Run Matching", key="run_match_btn", type="primary"):
            menu_working = st.session_state.menu_df.iloc[freeze_idx:].copy().reset_index(drop=True)
            auto_m, hitl_q = match_items(menu_working, ref_df)
            st.session_state.auto_matches = auto_m
            st.session_state.hitl_queue = hitl_q
            st.session_state.hitl_cursor = 0
            st.session_state.confirmed_matches = list(auto_m)
            st.session_state.addon_indices = {}
            st.rerun()

        if st.session_state.auto_matches or st.session_state.hitl_queue:
            n_auto = len(st.session_state.auto_matches)
            n_hitl = len(st.session_state.hitl_queue)

            col_a, col_b = st.columns(2)
            col_a.metric("Auto-matched", n_auto)
            col_b.metric("Needs review", n_hitl)
            st.markdown(" ")

            if n_auto > 0:
                menu_preview_df = st.session_state.menu_df.iloc[freeze_idx:].reset_index(drop=True)
                auto_rows = []
                for m in st.session_state.auto_matches:
                    try:
                        matched_item = clean_label(menu_preview_df.at[m["menu_index"], "Item"])
                    except Exception:
                        matched_item = "—"
                    auto_rows.append({
                        "Reference Item": m["item"],
                        "Matched To": matched_item,
                        "Type": "Addon" if m.get("is_addon") else "Item",
                    })
                with st.expander(f"View {n_auto} auto-matched rows"):
                    st.dataframe(pd.DataFrame(auto_rows), use_container_width=True)

            queue = st.session_state.hitl_queue
            cursor = st.session_state.hitl_cursor

            if cursor < len(queue):
                item = queue[cursor]
                st.markdown("---")

                st.progress(cursor / len(queue), text=f"Review {cursor + 1} of {len(queue)}")
                st.markdown(" ")

                badge = "🔖 Addon" if item.get("is_addon") else "🍽 Item"
                st.markdown(f"**{badge} — Reference:** `{item['ref_item']}`")

                meta_parts = []
                if item["ref_cat"]:
                    meta_parts.append(f"Category: `{item['ref_cat']}`")
                if item["ref_subcat"]:
                    meta_parts.append(f"Subcategory: `{item['ref_subcat']}`")
                if item["ref_variant"]:
                    meta_parts.append(f"Variant: `{item['ref_variant']}`")
                if meta_parts:
                    st.markdown("  ·  ".join(meta_parts))

                candidates = item.get("candidates", [])

                if not candidates:
                    st.info("No candidates found for this item.")
                    if st.button("Skip →", key=f"skip_nc_{cursor}"):
                        st.session_state.hitl_cursor += 1
                        st.rerun()
                else:
                    st.markdown(" ")
                    options = ["— Skip this item —"] + [
                        "  ·  ".join(filter(None, [
                            c["menu_item"],
                            c["menu_cat"] or None,
                            c["menu_subcat"] or None,
                            c["menu_variant"] or None,
                            f"₹{c['menu_price']}" if c["menu_price"] else None,
                        ]))
                        for c in candidates
                    ]
                    choice = st.radio("Select the correct match:", options, key=f"hitl_{cursor}")
                    apply_all = st.checkbox(
                        "Apply this decision to all future rows with the same item name",
                        key=f"apply_all_{cursor}",
                    )

                    col_confirm, col_skip = st.columns([1, 4])
                    if col_confirm.button("Confirm ✓", key=f"confirm_{cursor}", type="primary"):
                        if choice != "— Skip this item —":
                            ci = options.index(choice) - 1
                            c = candidates[ci]
                            st.session_state.confirmed_matches.append({
                                "ref_index": item["ref_index"],
                                "menu_index": c["menu_index"],
                                "item": item["ref_item"],
                                "auto": False,
                                "is_addon": item.get("is_addon", False),
                            })
                            if apply_all:
                                norm_name = item["ref_item"].lower().strip()
                                for fut in queue[cursor + 1:]:
                                    if fut["ref_item"].lower().strip() == norm_name:
                                        for fc in fut.get("candidates", []):
                                            if fc["menu_item"].lower().strip() == c["menu_item"].lower().strip():
                                                st.session_state.confirmed_matches.append({
                                                    "ref_index": fut["ref_index"],
                                                    "menu_index": fc["menu_index"],
                                                    "item": fut["ref_item"],
                                                    "auto": False,
                                                    "is_addon": fut.get("is_addon", False),
                                                })
                                                break
                        st.session_state.hitl_cursor += 1
                        st.rerun()

                    if col_skip.button("Skip →", key=f"skip_{cursor}"):
                        st.session_state.hitl_cursor += 1
                        st.rerun()

            elif st.session_state.confirmed_matches:
                if len(queue) > 0:
                    st.success(f"Review complete — {len(st.session_state.confirmed_matches)} total matches confirmed.")

                st.markdown(" ")

                                               
                menu_full = st.session_state.menu_df.copy()
                addon_col_exists = any(c.strip().lower() == "addon" for c in menu_full.columns)

                if addon_col_exists:
                    items_with_addons = {}
                    for m in st.session_state.confirmed_matches:
                        item_name = m["item"]
                        addon_idx_list = find_addon_rows(menu_full, item_name)
                        if addon_idx_list:
                            items_with_addons[m["menu_index"]] = (item_name, addon_idx_list)

                    if items_with_addons:
                        st.info(
                            f"**{len(items_with_addons)} matched item(s)** also appear as addons in the menu. "
                            "Apply the same pricing to those addon rows too?"
                        )
                        apply_to_addons = st.radio(
                            "Apply pricing to addon rows?",
                            ["No", "Yes — apply to all addon occurrences"],
                            key="addon_propagation_radio",
                            horizontal=True,
                        )
                        if apply_to_addons == "Yes — apply to all addon occurrences":
                            st.session_state.addon_indices = {
                                menu_idx: addon_list
                                for menu_idx, (_, addon_list) in items_with_addons.items()
                            }
                        else:
                            st.session_state.addon_indices = {}

                st.markdown(" ")
                if st.button("Apply All Confirmed Matches", key="apply_confirmed_btn", type="primary"):
                    df_apply = st.session_state.menu_df.copy()
                    df_apply["Price"] = pd.to_numeric(df_apply["Price"], errors="coerce")
                    df_apply["Markup Price"] = pd.to_numeric(df_apply["Markup Price"], errors="coerce")

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
                    st.session_state.menu_df = updated_df.copy()
                    st.session_state.audit_log = detail_df
                                                                    
                    st.session_state[f"_preview_df_post-ref-update preview"] = updated_df.iloc[freeze_idx:].copy()
                    st.session_state.ref_apply_done = True
                    st.session_state.ref_apply_count = len(confirmed_mapped)
                    st.rerun()

            if st.session_state.get("ref_apply_done"):
                st.success(f"✓ Pricing applied to {st.session_state.ref_apply_count} rows.")
                _render_preview("post-ref-update preview")

                                                                               

elif operation == "Remove existing slashing only":

    df_r = st.session_state.menu_df.copy()
    df_r["Price"] = pd.to_numeric(df_r["Price"], errors="coerce")
    df_r["Markup Price"] = pd.to_numeric(df_r["Markup Price"], errors="coerce")
    w_r = df_r.iloc[freeze_idx:]
    mask_r = (
        w_r["Price"].notna()
        & w_r["Markup Price"].notna()
        & (w_r["Markup Price"] > w_r["Price"])
    )

    if not st.session_state.get("remove_slash_only_done"):
        if int(mask_r.sum()) == 0:
            st.info("No slashing detected.")
        else:
            try:
                s = w_r[mask_r].iloc[0]
                pct = round((1 - s["Price"] / s["Markup Price"]) * 100, 2)
                st.warning(f"Slashing active on {int(mask_r.sum())} rows (~{pct}% off).")
            except Exception:
                st.warning(f"Slashing active on {int(mask_r.sum())} rows.")

            if st.button("Remove All Slashing", key="remove_slash_only_btn", type="primary"):
                idx_r = w_r[mask_r].index
                update_col_r = next(
                    (c for c in df_r.columns if c.strip().lower().startswith("update required")),
                    "Update Required ?"
                )
                df_r.loc[idx_r, "Price"] = df_r.loc[idx_r, "Markup Price"]
                df_r.loc[idx_r, "Markup Price"] = None
                df_r.loc[idx_r, update_col_r] = "Yes"
                st.session_state.menu_df = df_r.copy()
                st.session_state.remove_slash_only_done = True
                st.session_state[f"_preview_df_post-removal preview"] = df_r.iloc[freeze_idx:].copy()
                st.rerun()

    if st.session_state.get("remove_slash_only_done"):
        st.success(f"✓ Slashing removed.")
        _render_preview("post-removal preview")

                                                                                

st.markdown(" ")
section("④ Download Output")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
final_df = st.session_state.menu_df

col_d1, col_d2 = st.columns(2)

col_d1.download_button(
    "⬇ Download Updated Menu CSV",
    data=final_df.to_csv(index=False).encode("utf-8"),
    file_name=f"{st.session_state.original_name}_processed_{timestamp}.csv",
    mime="text/csv",
)

if st.session_state.audit_log is not None and not st.session_state.audit_log.empty:
    col_d2.download_button(
        "⬇ Download Change Summary",
        data=st.session_state.audit_log.to_csv(index=False).encode("utf-8"),
        file_name=f"{st.session_state.original_name}_summary_{timestamp}.csv",
        mime="text/csv",
    )
    out_dir = os.path.join(current_dir, "..", "output")
    os.makedirs(out_dir, exist_ok=True)
    st.session_state.audit_log.to_csv(os.path.join(out_dir, "audit_log.csv"), index=False)
