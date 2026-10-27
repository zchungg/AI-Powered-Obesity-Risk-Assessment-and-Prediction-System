# Imports Path to create and manage file and folder paths.
from pathlib import Path

# Imports pandas to organize dataset information into a DataFrame and save it as a CSV file.
import pandas as pd


# Gets the directory containing this Python script.
ROOT = Path(__file__).resolve().parent

# Defines the data folder where the downloaded dataset will be stored.
DATA_DIR = ROOT / "data"

# Defines the complete path and filename of the output CSV dataset.
OUTPUT = DATA_DIR / "ObesityDataSet_raw_and_data_sinthetic.csv"


# Defines the main function that handles downloading and saving the dataset.
def main():
    # Creates the data folder if it does not exist; existing folders are left unchanged.
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Attempts to import the function needed to download datasets from the UCI repository.
    try:
        from ucimlrepo import fetch_ucirepo

    # Handles the situation where the required ucimlrepo package is not installed.
    except ImportError as exc:
        # Stops the program and tells the user how to install the missing dependency.
        raise SystemExit(
            "Missing dependency: ucimlrepo. Run `pip install ucimlrepo` first."
        ) from exc

    # Displays a message indicating that the UCI obesity dataset is being downloaded.
    print("Downloading UCI dataset #544 ...")

    # Downloads dataset number 544 from the UCI Machine Learning Repository.
    dataset = fetch_ucirepo(id=544)

    # Copies the input features, such as age, height, weight, and lifestyle information.
    features = dataset.data.features.copy()

    # Copies the target column containing the obesity classification labels.
    target = dataset.data.targets.copy()

    # Combines the input features and target labels into one DataFrame.
    df = pd.concat([features, target], axis=1)

    # Saves the combined dataset as a CSV file without including pandas row indices.
    df.to_csv(OUTPUT, index=False)

    # Displays the number of downloaded rows and the location of the saved file.
    print(f"Saved {len(df)} rows to {OUTPUT}")


# Checks whether this file is being executed directly rather than imported by another module.
if __name__ == "__main__":
    # Runs the main function only when this script is executed directly.
    main()