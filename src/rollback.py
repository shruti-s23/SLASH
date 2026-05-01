import pandas as pd
import os


def rollback_changes(menu_df, freeze_idx=2, audit_path="output/audit_log.csv"):
    if not os.path.exists(audit_path):
        print("No audit log found.")
        return menu_df, 0

    audit_df = pd.read_csv(audit_path)

    if audit_df.empty:
        print("Audit log is empty.")
        return menu_df, 0

    menu_df = menu_df.copy()
    menu_df["Price"] = pd.to_numeric(menu_df["Price"], errors="coerce")
    menu_df["Markup Price"] = pd.to_numeric(menu_df["Markup Price"], errors="coerce")

    working_idx = menu_df.index[freeze_idx:]
    restored = 0

    for _, row in audit_df.iloc[::-1].iterrows():
        item_name = row.get("Menu Item")
        old_price = row.get("Old Selling Price")
        old_base = row.get("Old Base Price")

        mask = (menu_df["Item"] == item_name) & menu_df.index.isin(working_idx)

        if not mask.any():
            continue

        if pd.notna(old_price):
            menu_df.loc[mask, "Price"] = old_price

        if pd.notna(old_base):
            menu_df.loc[mask, "Markup Price"] = old_base
        else:
            menu_df.loc[mask, "Markup Price"] = None

        menu_df.loc[mask, "Update Required ?"] = "Yes"
        restored += 1

    return menu_df, restored