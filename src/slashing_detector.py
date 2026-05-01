import pandas as pd


def detect_existing_slashing(menu_df, freeze_idx=2):
    menu_df = menu_df.copy()
    menu_df["Price"] = pd.to_numeric(menu_df["Price"], errors="coerce")
    menu_df["Markup Price"] = pd.to_numeric(menu_df["Markup Price"], errors="coerce")

    working_idx = menu_df.index[freeze_idx:]

    slashed_mask = (
        menu_df.loc[working_idx, "Price"].notna()
        & menu_df.loc[working_idx, "Markup Price"].notna()
        & (menu_df.loc[working_idx, "Markup Price"] > menu_df.loc[working_idx, "Price"])
    )

    count = slashed_mask.sum()

    if count == 0:
        return menu_df, False, 0

    sample = menu_df.loc[working_idx].loc[slashed_mask].iloc[0]
    price = sample["Price"]
    markup = sample["Markup Price"]

    discount_pct = 0
    if pd.notna(markup) and markup != 0:
        discount_pct = round((1 - price / markup) * 100, 2)

    return menu_df, True, discount_pct


def remove_slashing(menu_df, freeze_idx=2):
    menu_df = menu_df.copy()
    menu_df["Price"] = pd.to_numeric(menu_df["Price"], errors="coerce")
    menu_df["Markup Price"] = pd.to_numeric(menu_df["Markup Price"], errors="coerce")

    working_idx = menu_df.index[freeze_idx:]

    slashed_mask = (
        menu_df.loc[working_idx, "Price"].notna()
        & menu_df.loc[working_idx, "Markup Price"].notna()
        & (menu_df.loc[working_idx, "Markup Price"] > menu_df.loc[working_idx, "Price"])
    )

    idx = working_idx[slashed_mask]

    menu_df.loc[idx, "Price"] = menu_df.loc[idx, "Markup Price"]
    menu_df.loc[idx, "Markup Price"] = None
    menu_df.loc[idx, "Update Required ?"] = "Yes"

    return menu_df, len(idx)