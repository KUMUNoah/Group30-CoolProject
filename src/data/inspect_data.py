from pathlib import Path
import pandas as pd

DATA_DIR = Path("src/data")
IMG_DIRS = [DATA_DIR / "imgs_part_1", DATA_DIR / "imgs_part_2", DATA_DIR / "imgs_part_3"]
META_PATH = DATA_DIR / "metadata.csv"

def collect_image_paths():
    paths = []
    for d in IMG_DIRS:
        if not d.exists():
            print(f"[WARN] Missing folder: {d}")
            continue
        paths += list(d.rglob("*.png"))
        paths += list(d.rglob("*.jpg"))
        paths += list(d.rglob("*.jpeg"))
    return paths

def main():
    df = pd.read_csv(META_PATH)
    print("Loaded metadata rows:", len(df))
    print("Columns:", list(df.columns))

    image_paths = collect_image_paths()
    print("Found image files:", len(image_paths))

    stem_to_path = {p.stem: p for p in image_paths}

    df["img_key"] = df["img_id"].astype(str).map(lambda s: Path(s).stem)

    df["has_image"] = df["img_key"].isin(stem_to_path.keys())
    covered = int(df["has_image"].sum())
    print(f"Rows with matching image file: {covered} / {len(df)} ({covered/len(df)*100:.1f}%)")

    print("\nClass counts (diagnostic):")
    print(df["diagnostic"].value_counts(dropna=False))

    print("\nExample rows without matching image (up to 10):")
    print(df.loc[~df["has_image"], ["img_id", "img_key", "patient_id", "diagnostic"]].head(10))

if __name__ == "__main__":
    main()
