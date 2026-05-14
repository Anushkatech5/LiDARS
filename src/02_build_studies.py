import ast
import pandas as pd
from pathlib import Path

INP = Path("artifacts/images_all.csv")
OUT = Path("artifacts/studies.csv")

def safe_list(x):
    if pd.isna(x):
        return []
    if isinstance(x, list):
        return x
    s = str(x)
    try:
        v = ast.literal_eval(s)
        return v if isinstance(v, list) else []
    except:
        return []

def main():
    df = pd.read_csv(INP, low_memory=False)

    # Parse labels from string list -> python list
    df["labels_list"] = df["Labels"].apply(safe_list)

    # StudyDate_DICOM is usually yyyymmdd (int); convert to datetime
    # Use errors='coerce' so weird rows won't crash
    df["study_date"] = pd.to_datetime(df["StudyDate_DICOM"].astype(str), format="%Y%m%d", errors="coerce")

    # IMPORTANT: group by PatientID+StudyID
    # image_paths: list of filepaths
    g = df.groupby(["PatientID", "StudyID"], dropna=False)

    studies = g.agg(
        patient_id=("PatientID", "first"),
        study_id=("StudyID", "first"),
        study_date=("study_date", "min"),
        image_paths=("filepath", lambda x: list(x)),
        labels_list=("labels_list", lambda x: sorted(set([lab for row in x for lab in row])))
    ).reset_index(drop=True)

    # Keep only studies that have >=1 image path that exists
    # (we don’t check file existence here to stay fast; optional check later)
    studies.to_csv(OUT, index=False)
    print("Saved:", OUT, "studies=", len(studies))

if __name__ == "__main__":
    main()
