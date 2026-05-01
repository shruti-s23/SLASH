from input_handler import load_menu_csv, load_reference_csv
from slashing_detector import detect_existing_slashing
from intent_handler import get_user_intent
from flat_discount import apply_flat_discount
from reference_update import match_items, process_matches
from rollback import rollback_changes

import pandas as pd
import os


def direct_replace(menu_df, ref_df):

    working_idx = menu_df.index[2:]

    # -------------------------
    # SAFE NUMERIC CONVERSION
    # -------------------------
    menu_df['Price'] = pd.to_numeric(menu_df['Price'], errors='coerce').astype(float)
    menu_df['Markup Price'] = pd.to_numeric(menu_df['Markup Price'], errors='coerce').astype(float)
    ref_df['Revised Price'] = pd.to_numeric(ref_df['Revised Price'], errors='coerce').astype(float)

    # -------------------------
    # NORMALIZE
    # -------------------------
    menu_df['Item_norm'] = menu_df['Item'].astype(str).str.strip().str.lower()
    ref_df['Item_norm'] = ref_df['Item'].astype(str).str.strip().str.lower()

    if 'Variant' in menu_df.columns and 'Variant' in ref_df.columns:
        menu_df['Variant_norm'] = menu_df['Variant'].astype(str).str.strip().str.lower()
        ref_df['Variant_norm'] = ref_df['Variant'].astype(str).str.strip().str.lower()
    else:
        menu_df['Variant_norm'] = ""
        ref_df['Variant_norm'] = ""

    # -------------------------
    # MATCH KEY
    # -------------------------
    menu_df['match_key'] = menu_df['Item_norm'] + "|" + menu_df['Variant_norm']
    ref_df['match_key'] = ref_df['Item_norm'] + "|" + ref_df['Variant_norm']

    ref_map = ref_df.set_index('match_key')['Revised Price'].to_dict()

    preview = []
    flagged = []

    # -------------------------
    # BUILD PREVIEW
    # -------------------------
    for i in working_idx:

        key = menu_df.at[i, 'match_key']

        if key not in ref_map:
            continue

        revised_price = ref_map[key]
        markup_price = menu_df.at[i, 'Markup Price']
        current_price = menu_df.at[i, 'Price']
        item_name = menu_df.at[i, 'Item']

        # constraint
        if pd.notna(markup_price) and revised_price > markup_price:
            flagged.append({
                "Item": item_name,
                "Markup": markup_price,
                "Revised": revised_price
            })
            continue

        preview.append({
            "idx": i,
            "Item": item_name,
            "Old Price": current_price,
            "New Price": revised_price,
            "Markup": markup_price
        })

    preview_df = pd.DataFrame(preview)

    print("\n=========== PREVIEW ===========\n")
    print(preview_df)

    if flagged:
        print("\n=========== FLAGGED (REVISED > MARKUP) ===========\n")
        for f in flagged:
            print(f"{f['Item']} | Markup: {f['Markup']} | Revised: {f['Revised']}")

    # -------------------------
    # CONFIRM
    # -------------------------
    choice = input("\nApply changes? (y/n): ").strip().lower()
    if choice != "y":
        print("Aborted.")
        return menu_df, 0

    # -------------------------
    # APPLY
    # -------------------------
    updated_count = 0

    for row in preview:

        idx = row["idx"]

        old_markup = menu_df.at[idx, 'Markup Price']

        menu_df.at[idx, 'Price'] = round(row["New Price"])
        menu_df.at[idx, 'Update Required ?'] = 'Yes'

        # NEW RULE
        if pd.isna(old_markup):
            menu_df.at[idx, 'Markup Price'] = 0

        updated_count += 1

    # cleanup
    menu_df.drop(columns=['Item_norm', 'Variant_norm', 'match_key'], inplace=True, errors='ignore')

    return menu_df, updated_count


def main():

    print("\n=== PRICE SLASHING TOOL STARTED ===\n")

    menu_df = load_menu_csv()

    print("\nInitial Preview:\n")
    print(menu_df.head())

    menu_df, slashing_removed = detect_existing_slashing(menu_df)

    intent = get_user_intent()

    if intent == "flat_discount":
        menu_df = apply_flat_discount(menu_df)

    elif intent == "reference_csv":
        ref_df = load_reference_csv()
        matches = match_items(menu_df, ref_df)
        menu_df = process_matches(menu_df, ref_df, matches)

    elif intent == "remove_only":
        if slashing_removed:
            print("Slashing already removed earlier.\n")
        else:
            print("No action needed.\n")

    elif intent == "rollback":
        menu_df = rollback_changes(menu_df)

    elif intent == "direct_replace":
        ref_df = load_reference_csv()
        menu_df, match_count = direct_replace(menu_df, ref_df)
        print(f"\nDirect replace completed. Updated rows: {match_count}\n")

    print("\nFinal Preview:\n")
    print(menu_df.head())

    os.makedirs("output", exist_ok=True)

    output_path = "output/final_menu.csv"
    menu_df.to_csv(output_path, index=False)

    print(f"\nFinal file saved at: {output_path}\n")
    print("=== PROCESS COMPLETED ===\n")


if __name__ == "__main__":
    main()