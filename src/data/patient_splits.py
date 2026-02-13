from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path("src/data")
META_PATH = DATA_DIR / "metadata.csv"

SPLITS_DIR = Path("splits")
SPLITS_DIR.mkdir(exist_ok=True)

SEED = 175
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

def main():
    df = pd.read_csv(META_PATH)

    df["patient_id"] = df["patient_id"].astype(str)

    patients = sorted(df["patient_id"].unique().tolist())

    rng = np.random.default_rng(SEED)
    rng.shuffle(patients)

    n = len(patients)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)

    train_patients = patients[:n_train]
    val_patients = patients[n_train:n_train + n_val]
    test_patients = patients[n_train + n_val:]

    (SPLITS_DIR / "train_patients.txt").write_text("\n".join(train_patients))
    (SPLITS_DIR / "val_patients.txt").write_text("\n".join(val_patients))
    (SPLITS_DIR / "test_patients.txt").write_text("\n".join(test_patients))

    print(f"Unique patients: {n}")
    print(f"Train patients: {len(train_patients)}")
    print(f"Val patients:   {len(val_patients)}")
    print(f"Test patients:  {len(test_patients)}")

    def count_rows(pats):
        return int(df[df["patient_id"].isin(pats)].shape[0])

    print("\nRows (images) by split:")
    print("Train rows:", count_rows(train_patients))
    print("Val rows:  ", count_rows(val_patients))
    print("Test rows: ", count_rows(test_patients))

    overlap = set(train_patients) & set(val_patients) | set(train_patients) & set(test_patients) | set(val_patients) & set(test_patients)
    print("\nOverlap patients across splits:", len(overlap))

if __name__ == "__main__":
    main()
