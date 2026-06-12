import streamlit as st
import pandas as pd
import sys
import os
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from decision_engine import (
    match_items,
    process_matches,
    get_candidates,
    build_match_summary,
    CANDIDATE_DISPLAY_COLS_MENU,
)
from slashing_detector import detect_existing_slashing
from rollback import rollback_changes

st.set_page_config(page_title="Price Revision", layout="wide")
st.title("Price Revision")

# -------------------------
# SESSION STATE
# -------------------------
if "menu_df" not in st.session_state:
    st.session_state.menu_df = None
if "original_name" not in st.session_state:
    st.session_state.original_name = None

# -------------------------
# UPLOAD
# -------------------------
st.header("Upload Menu CSV")

menu_file = st.file_uploader("Upload MENU CSV", type=["csv"])

if menu_file is None:
    st.stop()

if st.session_state.menu_df is None:
    df = pd.read_csv(menu_file, dtype=str)
    df.columns = df.columns.str.strip()

    df = df.apply(lambda col: col.map(lambda x: x.strip() if isinstance(x, str) else x))

    if "Price" not in df.columns:
        st.error("Invalid MENU CSV")
        st.stop()

    df["Update Required ?"] = ""

    st.session_state.menu_df = df
    st.session_state.original_name = menu_file.name

# -------------------------
# DATA
# -------------------------
df = st.session_state.menu_df.copy()
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
df["Markup Price"] = pd.to_numeric(df.get("Markup Price"), errors="coerce")

# -------------------------
# PREVIEW
# -------------------------
st.subheader("Menu Preview")

with st.expander("Expand Full CSV"):
    st.dataframe(df, use_container_width=True)

st.dataframe(df.head(), use_container_width=True)

# -------------------------
# OPERATION SELECT
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
# 1. FLAT DISCOUNT
# -------------------------
if operation == "Apply flat % discount":

    discount = st.number_input("Discount %", 1.0, 99.0, 20.0)

    if st.button("Apply Discount"):

        factor = (100 - discount) / 100

        df["Markup Price"] = df["Price"]
        df["Price"] = (df["Price"] * factor).round(0)
        df["Update Required ?"] = "Yes"

        st.session_state.menu_df = df

        st.success("Discount applied")

        st.dataframe(df.head())

        with st.expander("Full Updated CSV"):
            st.dataframe(df, use_container_width=True)

# -------------------------
# 2. REFERENCE MATCHING (NEW ENGINE)
# -------------------------
elif operation == "Use reference CSV":

    ref_file = st.file_uploader("Upload Reference CSV", type=["csv"], key="ref1")

    if ref_file:

        ref_df = pd.read_csv(ref_file, dtype=str)
        ref_df.columns = ref_df.columns.str.strip()

        st.subheader("Reference Preview")
        st.dataframe(ref_df.head())

        if st.button("Run Matching"):

            matches = match_items(df, ref_df)

            if not matches:
                st.warning("No matches found")
            else:

                # -------------------------
                # CANDIDATE REVIEW (variant-level differentiation)
                # -------------------------
                st.subheader("Candidate Review")

                for m in matches:

                    ref_row = ref_df.loc[m["ref_index"]]
                    ref_item_name = (
                        ref_row.get("Item Name")
                        or ref_row.get("Item")
                        or ""
                    )

                    with st.expander(
                        f"Ref: {ref_item_name}  →  Best match: "
                        f"{df.loc[m['menu_index'], 'Item']} "
                        f"(score {m['score']})"
                    ):
                        candidates = get_candidates(ref_row, df, top_n=5)

                        if candidates.empty:
                            st.write("No close candidates found.")
                        else:
                            display_cols = ["Score"] + CANDIDATE_DISPLAY_COLS_MENU
                            st.dataframe(
                                candidates[display_cols],
                                use_container_width=True,
                            )

                # -------------------------
                # POST-MATCHING PREVIEW SUMMARY
                # -------------------------
                st.subheader("Match Summary")

                summary_df = build_match_summary(df, ref_df, matches)
                st.dataframe(summary_df, use_container_width=True)

                df = process_matches(df, ref_df, matches)

                st.session_state.menu_df = df

                st.success(f"Matched {len(matches)} items")

                st.subheader("Updated Preview")
                st.dataframe(df.head())

                with st.expander("Full Updated CSV"):
                    st.dataframe(df, use_container_width=True)

# -------------------------
# 3. DIRECT REPLACE
# -------------------------
elif operation == "Replace prices directly":

    ref_file = st.file_uploader("Upload Reference CSV", type=["csv"], key="ref2")

    if ref_file:

        ref_df = pd.read_csv(ref_file, dtype=str)
        ref_df.columns = ref_df.columns.str.strip()

        if st.button("Apply Direct Replace"):

            matches = match_items(df, ref_df)

            updated = 0

            for m in matches:

                r_idx = m["ref_index"]
                m_idx = m["menu_index"]

                df.at[m_idx, "Price"] = pd.to_numeric(
                    ref_df.iloc[r_idx]["Revised Price"],
                    errors="coerce"
                )

                df.at[m_idx, "Update Required ?"] = "Yes"
                updated += 1

            st.session_state.menu_df = df

            st.success(f"Updated {updated} rows")

            st.dataframe(df.head())

# -------------------------
# 4. REMOVE SLASHING ONLY
# -------------------------
elif operation == "Remove existing slashing only":

    if st.button("Remove All Slashing"):

        mask = (
            df["Markup Price"].notna() &
            df["Price"].notna() &
            (df["Markup Price"] > df["Price"])
        )

        df.loc[mask, "Price"] = df.loc[mask, "Markup Price"]
        df.loc[mask, "Markup Price"] = None
        df.loc[mask, "Update Required ?"] = "Yes"

        st.session_state.menu_df = df

        st.success(f"Removed slashing from {mask.sum()} rows")

# -------------------------
# DOWNLOAD
# -------------------------
st.header("Download Output")

csv = st.session_state.menu_df.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download CSV",
    data=csv,
    file_name=f"{st.session_state.original_name}_processed_{datetime.now().strftime('%H%M%S')}.csv"
)
