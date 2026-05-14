import ast
import numpy as np
import pandas as pd

GT = "artifacts/test_pairs_with_delta.csv"
PRED = "artifacts/delta_reports_test.csv"
N_BOOT = 1000
SEED = 42

def parse_list(x):
    return ast.literal_eval(x) if isinstance(x, str) else x

def parse_pred(cell):
    if pd.isna(cell) or str(cell).strip() == "":
        return set()
    return set([x.strip() for x in str(cell).split(";") if x.strip()])

def vec_to_set(vec, label_cols):
    return set([label_cols[i] for i,v in enumerate(vec) if int(v)==1])

def micro_f1(pred_sets, true_sets):
    tp=fp=fn=0
    for ps, ts in zip(pred_sets, true_sets):
        tp += len(ps & ts)
        fp += len(ps - ts)
        fn += len(ts - ps)
    p = tp/(tp+fp+1e-9)
    r = tp/(tp+fn+1e-9)
    f = 2*p*r/(p+r+1e-9)
    return p,r,f

def main():
    np.random.seed(SEED)

    pred = pd.read_csv(PRED)
    gt = pd.read_csv(GT, low_memory=False)

    for c in ["delta_new","delta_resolved","delta_stable_present","label_cols"]:
        gt[c] = gt[c].apply(parse_list)
    label_cols = gt.iloc[0]["label_cols"]

    pred_new = pred["new"].apply(parse_pred).tolist()
    pred_res = pred["resolved"].apply(parse_pred).tolist()
    pred_stp = pred["stable_present"].apply(parse_pred).tolist()

    true_new = gt["delta_new"].apply(lambda v: vec_to_set(v, label_cols)).tolist()
    true_res = gt["delta_resolved"].apply(lambda v: vec_to_set(v, label_cols)).tolist()
    true_stp = gt["delta_stable_present"].apply(lambda v: vec_to_set(v, label_cols)).tolist()

    N = len(gt)
    heads = {
        "NEW": (pred_new, true_new),
        "RESOLVED": (pred_res, true_res),
        "STABLE_PRESENT": (pred_stp, true_stp)
    }

    print(f"Bootstrap CI with N={N}, samples={N_BOOT}")
    for name, (ps, ts) in heads.items():
        f1s = []
        for _ in range(N_BOOT):
            idx = np.random.randint(0, N, size=N)
            ps_b = [ps[i] for i in idx]
            ts_b = [ts[i] for i in idx]
            _, _, f1 = micro_f1(ps_b, ts_b)
            f1s.append(f1)

        f1s = np.array(f1s)
        lo, mid, hi = np.percentile(f1s, [2.5, 50, 97.5])
        print(f"{name}: F1 median={mid:.3f} | 95% CI [{lo:.3f}, {hi:.3f}]")

if __name__ == "__main__":
    main()