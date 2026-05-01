import streamlit as st
import pandas as pd
import sys
import os
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from decision_engine import match_items, process_matches
st.set_page_config(page_title="Price Revision Tool", layout="wide")
st.title("Price Revision Tool")

# -------------------------
# SESSION STATE
# -------------------------
if "menu_df" not in st.session_state:
    st.session_state.menu_df = None

if "freeze_idx" not in st.session_state:
    st.session_state.freeze_idx = 0


# -------------------------
# RESET WHEN FILE REMOVED
# -------------------------
def reset_state():
    st.session_state.menu_df = None
    st.session_state.freeze_idx = 0


# -------------------------
# METADATA DETECTION
# -------------------------
def detect_freeze_index(df):
    keywords = [
        "brandskuid",
        "item/variant/addon",
        "description",
        "cdn",
        "veg/non veg",
        "in stock",
        "yyyy-mm-dd",
        "hh:mm:ss"
    ]

    for i in df.index:
        try:
            row_text = " ".join(str(x).lower() for x in df.loc[i].values if pd.notna(x))
        except:
            continue

        if any(k in row_text for k in keywords):
            return i + 1

    return 0


# -------------------------
# UPLOAD
# -------------------------
st.header("Upload Menu CSV")

menu_file = st.file_uploader(
    "Upload MENU CSV",
    type=["csv"],
    accept_multiple_files=False,
    on_change=reset_state
)

if menu_file:

    df = pd.read_csv(menu_file, dtype=str)
    df.columns = df.columns.str.strip()

    df = df.apply(lambda col: col.map(lambda x: x.strip() if isinstance(x, str) else x))

    freeze_idx = detect_freeze_index(df)

    if 'Update Required ?' not in df.columns:
        df['Update Required ?'] = ''

    st.session_state.menu_df = df.copy()
    st.session_state.freeze_idx = freeze_idx

    st.subheader("Menu Preview")
    st.dataframe(df.head())

    # ITEM COUNT (WORKING AREA ONLY)
    if 'Item' in df.columns and 'Brand SKU ID Type' in df.columns:
        working_df = df.iloc[freeze_idx:]
        item_count = working_df[working_df['Brand SKU ID Type'] == 'Item']['Item'].nunique()
    else:
        item_count = "N/A"

    st.info(f"Rows: {df.shape[0]} | Items: {item_count}")
    st.info(f"Freeze till row: {freeze_idx}")


# -------------------------
# MAIN FLOW
# -------------------------
if st.session_state.menu_df is not None:

    df = st.session_state.menu_df.copy()
    freeze_idx = st.session_state.freeze_idx
    working_df = df.iloc[freeze_idx:]

    # -------------------------
    # SLASHING DETECTION
    # -------------------------
    st.header("Slashing Detection")

    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    df['Markup Price'] = pd.to_numeric(df['Markup Price'], errors='coerce')

    slashed_mask = (
        working_df['Price'].notna() &
        working_df['Markup Price'].notna() &
        (working_df['Markup Price'] > working_df['Price'])
    )

    if slashed_mask.sum() > 0:

        sample = working_df[slashed_mask].iloc[0]

        try:
            discount_pct = round((1 - (sample['Price'] / sample['Markup Price'])) * 100, 2)
            st.info(f"Detected slashing in {slashed_mask.sum()} rows | Example discount: {discount_pct}%")
        except:
            st.info(f"Detected slashing in {slashed_mask.sum()} rows")

        remove = st.radio("Remove existing slashing?", ["No", "Yes"])

        if remove == "Yes":
            idx = working_df[slashed_mask].index

            df.loc[idx, 'Price'] = df.loc[idx, 'Markup Price']
            df.loc[idx, 'Markup Price'] = None
            df.loc[idx, 'Update Required ?'] = 'Yes'

            st.success("Slashing removed")

    else:
        st.success("No slashing detected")

    st.session_state.menu_df = df

    # -------------------------
    # OPERATION
    # -------------------------
    st.header("Select Operation")

    operation = st.selectbox(
        "Choose operation",
        [
            "Apply flat % discount",
            "Use reference CSV",
            "Replace prices directly",
            "Remove existing slashing only"
        ]
    )

    # -------------------------
    # SCOPE (ONLY WORKING AREA)
    # -------------------------
    if 'Brand SKU ID Type' in df.columns:
        available_types = sorted(working_df['Brand SKU ID Type'].dropna().unique())
    else:
        available_types = []

    scope_selected = st.multiselect(
        "Select Scope",
        options=available_types,
        default=available_types
    )

    # -------------------------
    # FLAT DISCOUNT
    # -------------------------
    if operation == "Apply flat % discount":

        discount = st.number_input("Discount %", 1.0, 99.0, 20.0)

        if st.button("Apply Discount"):

            factor = (100 - discount) / 100

            mask = df.index.isin(working_df.index)

            if scope_selected:
                mask = mask & df['Brand SKU ID Type'].isin(scope_selected)

            df.loc[mask, 'Markup Price'] = df.loc[mask, 'Price']
            df.loc[mask, 'Price'] = (df.loc[mask, 'Price'] * factor).round(0)
            df.loc[mask, 'Update Required ?'] = 'Yes'

            st.session_state.menu_df = df
            st.success(f"Discount applied on {mask.sum()} rows")

    # -------------------------
    # REFERENCE CSV FLOW
    # -------------------------
    elif operation == "Use reference CSV":

        ref_file = st.file_uploader("Upload Reference CSV", type=["csv"])

        if ref_file:

            ref_df = pd.read_csv(ref_file, dtype=str)
            ref_df.columns = ref_df.columns.str.strip()

            st.subheader("Reference Preview")
            st.dataframe(ref_df.head())

            if st.button("Run Matching"):

                matches = match_items(working_df, ref_df)

                if not matches:
                    st.warning("No matches found")
                else:
                    df = process_matches(df, ref_df, matches)
                    st.session_state.menu_df = df

                    st.subheader("Updated Preview")
                    st.dataframe(df.head())

                    st.success(f"Matched {len(matches)} items")

    # -------------------------
    # DIRECT REPLACE
    # -------------------------
    elif operation == "Replace prices directly":

        ref_file = st.file_uploader("Upload Reference CSV", type=["csv"])

        if ref_file:

            ref_df = pd.read_csv(ref_file, dtype=str)
            ref_df.columns = ref_df.columns.str.strip()

            st.subheader("Reference Preview")
            st.dataframe(ref_df.head())

            if st.button("Apply Replace"):

                matches = match_items(working_df, ref_df)

                updated = 0

                for r_idx, m_indices, _ in matches:
                    for m_idx in m_indices:
                        actual_idx = working_df.index[m_idx]

                        df.at[actual_idx, 'Price'] = pd.to_numeric(
                            ref_df.iloc[r_idx]['Revised Price'],
                            errors='coerce'
                        )
                        df.at[actual_idx, 'Update Required ?'] = 'Yes'
                        updated += 1

                st.session_state.menu_df = df

                st.subheader("Updated Preview")
                st.dataframe(df.head())

                st.success(f"Updated {updated} rows")

    # -------------------------
    # DOWNLOAD
    # -------------------------
    st.header("Download Output")

    final_df = st.session_state.menu_df

    csv = final_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download CSV",
        data=csv,
        file_name=f"processed_{datetime.now().strftime('%H%M%S')}.csv"
    )