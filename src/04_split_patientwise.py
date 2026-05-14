import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit

INP = Path("artifacts/studies.csv")

OUT_TRAIN = Path("artifacts/studies_train.csv")
OUT_VAL   = Path("artifacts/studies_val.csv")
OUT_TEST  = Path("artifacts/studies_test.csv")

def main():
    df = pd.read_csv(INP, low_memory=False)

    groups = df["patient_id"].astype(str)

    # 80/10/10
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, tmp_idx = next(gss1.split(df, groups=groups))

    train = df.iloc[train_idx].copy()
    tmp = df.iloc[tmp_idx].copy()

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_idx, test_idx = next(gss2.split(tmp, groups=tmp["patient_id"].astype(str)))

    val = tmp.iloc[val_idx].copy()
    test = tmp.iloc[test_idx].copy()

    train.to_csv(OUT_TRAIN, index=False)
    val.to_csv(OUT_VAL, index=False)
    test.to_csv(OUT_TEST, index=False)

    print("Saved:", OUT_TRAIN, "rows=", len(train))
    print("Saved:", OUT_VAL, "rows=", len(val))
    print("Saved:", OUT_TEST, "rows=", len(test))

if __name__ == "__main__":
    main()