# import ast
# import re
# import pandas as pd
#
# # Defaults match your evaluation script expectations
# DEFAULT_PRED = "D:/ImagingInformatics/artifacts/delta_reports_test.csv"
# DEFAULT_GT   = "D:/ImagingInformatics/artifacts/test_pairs_with_delta.csv"
#
# WOW_KEYWORDS = [
#     "effusion", "pneumonia", "cardiomegaly"
# ]
#
# def parse_list(x):
#     return ast.literal_eval(x) if isinstance(x, str) else x
#
# def parse_pred_items(cell):
#     """
#     Handles both formats:
#       - "y_effusion;y_pneumonia"
#       - "y_effusion(0.82); y_pneumonia(0.77)"  -> keeps score if present
#     Returns list of (label, score|None)
#     """
#     if pd.isna(cell) or str(cell).strip() == "":
#         return []
#     items = []
#     for raw in str(cell).split(";"):
#         s = raw.strip()
#         if not s:
#             continue
#         m = re.match(r"^(.*?)(?:\((0\.\d+|1\.0+)\))?$", s)
#         if m:
#             lbl = m.group(1).strip()
#             sc = float(m.group(2)) if m.group(2) is not None else None
#             items.append((lbl, sc))
#         else:
#             items.append((s, None))
#     return items
#
# def items_to_set(items):
#     return set([lbl for (lbl, _) in items])
#
# def vec_to_set(vec, label_cols):
#     return set([label_cols[i] for i, v in enumerate(vec) if int(v) == 1])
#
# def f1_sample(pred_set, true_set):
#     tp = len(pred_set & true_set)
#     fp = len(pred_set - true_set)
#     fn = len(true_set - pred_set)
#     denom = (2*tp + fp + fn)
#     return (2*tp / denom) if denom > 0 else 1.0  # both empty -> perfect
#
# def pretty(lbl):
#     # y_pleural_effusion -> pleural effusion
#     x = str(lbl).strip()
#     x = re.sub(r"\(.*\)$", "", x)  # remove trailing (score) if any
#     x = x.replace("y_", "")
#     x = x.replace("_", " ")
#     return x
#
# def top_findings(items, max_k=3):
#     """
#     Choose best K by score if available; else alphabetical.
#     Return "a, b, c" or "a, b, ..." if more exist.
#     """
#     if not items:
#         return "—"
#     scored = [(lbl, sc) for (lbl, sc) in items]
#     if any(sc is not None for _, sc in scored):
#         scored.sort(key=lambda t: (-1 if t[1] is None else -t[1], t[0]))
#     else:
#         scored.sort(key=lambda t: t[0])
#
#     labels = [pretty(lbl) for (lbl, _) in scored]
#     if len(labels) <= max_k:
#         return ", ".join(labels)
#     return ", ".join(labels[:max_k]) + ", ..."
#
# def dt_bucket(dt):
#     # tweak buckets if you want
#     if dt <= 7:
#         return "short"
#     if dt <= 30:
#         return "medium"
#     if dt <= 180:
#         return "long"
#     return "very_long"
#
# def main(pred_path=DEFAULT_PRED, gt_path=DEFAULT_GT, out_path="D:/ImagingInformatics/artifacts/qual_table_rows_20.txt"):
#     pred = pd.read_csv(pred_path)
#     gt = pd.read_csv(gt_path, low_memory=False)
#
#     # GT parsing (same columns your eval uses)
#     for c in ["delta_new", "delta_resolved", "delta_stable_present", "label_cols"]:
#         gt[c] = gt[c].apply(parse_list)
#     label_cols = gt.iloc[0]["label_cols"]
#
#     # Pred parsing (same columns your eval uses: new/resolved/stable_present)
#     pred_new_items = pred["new"].apply(parse_pred_items)
#     pred_res_items = pred["resolved"].apply(parse_pred_items)
#     pred_stp_items = pred["stable_present"].apply(parse_pred_items)
#
#     pred_new_set = pred_new_items.apply(items_to_set)
#     pred_res_set = pred_res_items.apply(items_to_set)
#     pred_stp_set = pred_stp_items.apply(items_to_set)
#
#     true_new_set = gt["delta_new"].apply(lambda v: vec_to_set(v, label_cols))
#     true_res_set = gt["delta_resolved"].apply(lambda v: vec_to_set(v, label_cols))
#     true_stp_set = gt["delta_stable_present"].apply(lambda v: vec_to_set(v, label_cols))
#
#     # Pair id for table: prefer explicit column if present; else row index + 1
#     if "pair_id" in gt.columns:
#         pair_ids = gt["pair_id"].astype(str)
#     else:
#         pair_ids = pd.Series((gt.index + 1).astype(str))
#
#     # dt
#     dt_days = gt["dt_days"] if "dt_days" in gt.columns else pd.Series([0]*len(gt))
#
#     rows = []
#     for i in range(len(gt)):
#         pn, pr, ps = pred_new_set.iloc[i], pred_res_set.iloc[i], pred_stp_set.iloc[i]
#         tn, tr, ts = true_new_set.iloc[i], true_res_set.iloc[i], true_stp_set.iloc[i]
#
#         f1n = f1_sample(pn, tn)
#         f1r = f1_sample(pr, tr)
#         f1s = f1_sample(ps, ts)
#         base = (f1n + f1r + f1s) / 3.0
#
#         # bonuses to make the qualitative table “wow” + diverse Δt
#         all_pred_labels = " ".join([pretty(x) for x in (pn | pr | ps)]).lower()
#         wow_bonus = 0.05 if any(k in all_pred_labels for k in WOW_KEYWORDS) else 0.0
#         dtv = int(dt_days.iloc[i]) if pd.notna(dt_days.iloc[i]) else 0
#         dt_bonus = 0.02 if (dtv <= 7 or dtv >= 180) else 0.0
#
#         # penalty: empty prediction when GT has any delta
#         gt_nonempty = (len(tn) + len(tr) + len(ts)) > 0
#         pred_empty = (len(pn) + len(pr) + len(ps)) == 0
#         empty_penalty = 0.08 if (gt_nonempty and pred_empty) else 0.0
#
#         score = base + wow_bonus + dt_bonus - empty_penalty
#
#         rows.append({
#             "i": i,
#             "pair": pair_ids.iloc[i],
#             "dt": dtv,
#             "bucket": dt_bucket(dtv),
#             "score": score,
#             "N": len(pn),
#             "R": len(pr),
#             "S": len(ps),
#             "NEW_findings": top_findings(pred_new_items.iloc[i]),
#             "RES_findings": top_findings(pred_res_items.iloc[i]),
#             "STP_findings": top_findings(pred_stp_items.iloc[i]),
#         })
#
#     df = pd.DataFrame(rows).sort_values("score", ascending=False)
#
#     # pick 5 per bucket (5*4=20); backfill if a bucket has fewer
#     picked = []
#     for b in ["short", "medium", "long", "very_long"]:
#         sub = df[df["bucket"] == b].head(5)
#         picked.append(sub)
#     picked = pd.concat(picked).drop_duplicates(subset=["i"])
#
#     if len(picked) < 20:
#         backfill = df[~df["i"].isin(picked["i"])].head(20 - len(picked))
#         picked = pd.concat([picked, backfill])
#
#     picked = picked.head(20).copy()
#
#     # output in your Gemini [PASTE_ROWS_HERE] format:
#     # Pair | Δt | N_count | R_count | S_count | NEW_findings | RES_findings | STP_findings
#     lines = []
#     for _, r in picked.iterrows():
#         dt_str = f"{int(r['dt'])} d"
#         line = " | ".join([
#             str(r["pair"]),
#             dt_str,
#             str(r["N"]),
#             str(r["R"]),
#             str(r["S"]),
#             str(r["NEW_findings"]),
#             str(r["RES_findings"]),
#             str(r["STP_findings"]),
#         ])
#         lines.append(line)
#
#     with open(out_path, "w", encoding="utf-8") as f:
#         f.write("\n".join(lines))
#
#     print("Wrote:", out_path)
#     print("\n--- Paste these into [PASTE_ROWS_HERE] ---\n")
#     print("\n".join(lines))
#
# if __name__ == "__main__":
#     main()

## FOR TOP 5 WITH THREE ROWS FILLED
# import pandas as pd
# import ast
# 
# PAIRS_CSV = "D:/ImagingInformatics/artifacts/test_pairs_with_delta.csv"
# df = pd.read_csv(PAIRS_CSV, low_memory=False)
# 
# print("Columns in CSV:", df.columns.tolist())
# 
# # Use patient_id as the identifier
# pair_col = "patient_id"
# 
# # dt column
# dt_col = "dt_days"
# 
# # Parse delta columns into Python lists
# for c in ["delta_new", "delta_resolved", "delta_stable_present", "label_cols"]:
#     df[c] = df[c].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
# 
# # Compute counts
# df["N_count"] = df["delta_new"].apply(lambda v: sum(v) if isinstance(v, list) else 0)
# df["R_count"] = df["delta_resolved"].apply(lambda v: sum(v) if isinstance(v, list) else 0)
# df["S_count"] = df["delta_stable_present"].apply(lambda v: sum(v) if isinstance(v, list) else 0)
# 
# # Convert dt to numeric
# df[dt_col] = pd.to_numeric(df[dt_col], errors="coerce")
# 
# # Filter rows with at least one finding in each category and dt <= 30
# cand = df[
#     (df["N_count"] > 0) & (df["R_count"] > 0) & (df["S_count"] > 0) &
#     (df[dt_col] <= 30)
# ].copy()
# 
# # Pick 5 with shortest Δt, then highest stable count
# cand = cand.sort_values([dt_col, "S_count"], ascending=[True, False]).head(5)
# 
# print("Suggested PAIR_IDS:", cand[pair_col].tolist())
# print(cand[[pair_col, dt_col, "N_count", "R_count", "S_count"]])

##pick_qual_cases.py
