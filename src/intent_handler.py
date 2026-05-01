def get_user_intent():

    print("\nSelect operation:\n")

    options = {
        "1": "flat_discount",
        "2": "reference_csv",
        "3": "direct_replace",
        "4": "remove_only",
        "5": "rollback"
    }

    descriptions = {
        "1": "Apply flat % discount",
        "2": "Use reference CSV",
        "3": "Replace prices directly",
        "4": "Remove existing slashing only",
        "5": "Rollback previous changes"
    }

    for key in options:
        print(f"{key}. {descriptions[key]}")

    print()

    while True:
        try:
            choice = input("Enter choice (1/2/3/4/5): ").strip()

            if choice in options:
                selected_intent = options[choice]
                print(f"\nSelected: {descriptions[choice]}\n")
                return selected_intent

            print("Invalid input. Please enter a valid option.\n")

        except KeyboardInterrupt:
            print("\nOperation cancelled by user.\n")
            return None

        except Exception:
            print("Unexpected error. Try again.\n")