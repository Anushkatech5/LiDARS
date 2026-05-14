import glob
import pandas as pd
from pathlib import Path

ARTIFACTS = Path("artifacts")
OUT = ARTIFACTS / "images_all.csv"
IMG_ROOT = Path("data/selected_by_zip")

def main():
    csvs = sorted(glob.glob(str(ARTIFACTS / "padchest_zip*_labels.csv")))
    if not csvs:
        raise FileNotFoundError("No artifacts/padchest_zip*_labels.csv found")

    frames = []
    for f in csvs:
        df = pd.read_csv(f, low_memory=False)
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
        # filepath = data/selected_by_zip/{ImageDir}_selectedimages/{ImageID}
        df["filepath"] = df.apply(
            lambda r: str(IMG_ROOT / f"{int(r['ImageDir'])}_selectedimages" / str(r["ImageID"])),
            axis=1
        )
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(OUT, index=False)
    print("Saved:", OUT, "rows=", len(all_df))

if __name__ == "__main__":
    main()