import ast
import pandas as pd

PRED = "artifacts/delta_reports_test.csv"
GT   = "artifacts/test_pairs_with_delta.csv"

def parse_list(x):
    return ast.literal_eval(x) if isinstance(x, str) else x

def parse_pred(cell):
    if pd.isna(cell) or str(cell).strip() == "":
        return set()
    return set([x.strip() for x in str(cell).split(";") if x.strip()])

def vec_to_set(vec, label_cols):
    return set([label_cols[i] for i,v in enumerate(vec) if int(v)==1])

def prf(tp, fp, fn):
    p = tp/(tp+fp+1e-9)
    r = tp/(tp+fn+1e-9)
    f = 2*p*r/(p+r+1e-9)
    return p,r,f

def main():
    pred = pd.read_csv(PRED)
    gt = pd.read_csv(GT, low_memory=False)

    # parse gt columns
    for c in ["delta_new","delta_resolved","delta_stable_present","label_cols"]:
        gt[c] = gt[c].apply(parse_list)
    label_cols = gt.iloc[0]["label_cols"]

    # parse predicted sets
    pred_new = pred["new"].apply(parse_pred)
    pred_res = pred["resolved"].apply(parse_pred)
    pred_stp = pred["stable_present"].apply(parse_pred)

    # build true sets
    true_new = gt["delta_new"].apply(lambda v: vec_to_set(v, label_cols))
    true_res = gt["delta_resolved"].apply(lambda v: vec_to_set(v, label_cols))
    true_stp = gt["delta_stable_present"].apply(lambda v: vec_to_set(v, label_cols))

    def score(name, pred_sets, true_sets):
        tp=fp=fn=0
        for ps, ts in zip(pred_sets, true_sets):
            tp += len(ps & ts)
            fp += len(ps - ts)
            fn += len(ts - ps)
        p,r,f = prf(tp,fp,fn)
        print(f"{name}: P={p:.3f} R={r:.3f} F1={f:.3f} | TP={tp} FP={fp} FN={fn}")

    score("NEW", pred_new, true_new)
    score("RESOLVED", pred_res, true_res)
    score("STABLE_PRESENT", pred_stp, true_stp)

if __name__ == "__main__":
    main()