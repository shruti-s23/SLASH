import pandas as pd


def apply_flat_discount(menu_df):
    print("\n--- Flat Discount Module ---\n")

    required_cols = ['Price', 'Markup Price', 'Brand SKU ID Type', 'Category']
    for col in required_cols:
        if col not in menu_df.columns:
            raise Exception(f"Missing required column: {col}")

    # -------------------------
    # SAFE NUMERIC CONVERSION
    # -------------------------
    menu_df['Price'] = pd.to_numeric(menu_df['Price'], errors='coerce').astype(float)
    menu_df['Markup Price'] = pd.to_numeric(menu_df['Markup Price'], errors='coerce').astype(float)

    menu_df['Category'] = menu_df['Category'].astype(str).str.strip()

    working_idx = menu_df.index[2:]  # skip first 2 rows

    # -------------------------
    # INPUT: DISCOUNT
    # -------------------------
    while True:
        try:
            discount = float(input("Enter discount % (e.g., 20 for 20%): "))
            if 0 < discount < 100:
                break
            print("Enter a value between 0 and 100.\n")
        except:
            print("Invalid input.\n")

    discount_factor = (100 - discount) / 100

    # -------------------------
    # INPUT: SCOPE
    # -------------------------
    print("\nSelect scope:")
    print("1. Items only")
    print("2. Items + Variants")
    print("3. Items + Variants + Addons")

    scope_map = {
        '1': ['Item'],
        '2': ['Item', 'Variant'],
        '3': ['Item', 'Variant', 'Addon']
    }

    while True:
        scope_choice = input("Enter choice (1/2/3): ").strip()
        if scope_choice in scope_map:
            allowed_types = scope_map[scope_choice]
            break
        print("Invalid input.\n")

    # -------------------------
    # INPUT: CATEGORY
    # -------------------------
    categories = sorted(menu_df['Category'].dropna().unique())

    print("\nAvailable Categories:")
    for i, cat in enumerate(categories):
        print(f"{i+1}. {cat}")
    print("0. Select ALL categories")

    while True:
        selection = input("\nEnter category numbers: ").strip()

        if selection == '0':
            selected_categories = categories
            break

        try:
            indices = [int(x.strip()) - 1 for x in selection.split(',')]
            selected_categories = [categories[i] for i in indices]
            break
        except:
            print("Invalid selection.\n")

    # -------------------------
    # APPLY DISCOUNT
    # -------------------------
    print("\nApplying discount...\n")

    mask = (
        menu_df['Brand SKU ID Type'].isin(allowed_types) &
        menu_df['Category'].isin(selected_categories) &
        menu_df['Price'].notna() &
        menu_df.index.isin(working_idx)
    )

    # store old markup for comparison
    old_markup = menu_df['Markup Price'].copy()

    # Move original → Markup ONLY if missing
    markup_missing = mask & menu_df['Markup Price'].isna()
    menu_df.loc[markup_missing, 'Markup Price'] = menu_df.loc[markup_missing, 'Price']

    # Apply discount + ROUND to nearest integer
    menu_df.loc[mask, 'Price'] = (
        menu_df.loc[mask, 'Price'].astype(float) * discount_factor
    ).round(0)

    # Mark update
    menu_df.loc[mask, 'Update Required ?'] = 'Yes'

    # -------------------------
    # NEW RULE: ONLY PRICE UPDATED → MARKUP = 0
    # -------------------------
    only_price_changed = (
        (menu_df['Update Required ?'] == 'Yes') &
        (old_markup == menu_df['Markup Price'])
    )

    menu_df.loc[only_price_changed, 'Markup Price'] = 0

    print(f"Flat discount applied. Rows updated: {mask.sum()}\n")

    return menu_df