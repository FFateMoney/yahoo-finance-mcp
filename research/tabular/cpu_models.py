"""F138 LightGBM (expanding), F139 LightGBM (rolling 2 years), F142 linear sign-weighted baseline.
Fixed hyper-parameters from 回测/表格模型_预登记.md. Writes research/tabular/out/pred_<name>.npy (one value per panel row,
NaN outside the out-of-sample blocks). Usage: python research/tabular/cpu_models.py
"""
import time

import lightgbm as lgb
import numpy as np

from wf import daily_ic, folds, load_panel, save_pred

P = load_panel()
X, y, di = P["Xf"], P["y"], P["di"]
Xs = np.column_stack([X, P["sec"].astype(np.float32)])
names = list(P["feat"]) + ["sector"]
LGB = dict(objective="regression", learning_rate=0.05, num_leaves=31, min_data_in_leaf=2000, feature_fraction=0.8,
           bagging_fraction=0.8, bagging_freq=1, lambda_l2=0.0, verbose=-1, num_threads=4, seed=1)


def run_lgb(name, window):
    pred, imp = np.full(len(y), np.nan), np.zeros(len(names))
    for tr, va, pr in folds(P, window):
        t = time.time()
        dtr = lgb.Dataset(Xs[tr], y[tr], feature_name=names, categorical_feature=["sector"])
        dva = lgb.Dataset(Xs[va], y[va], reference=dtr)
        m = lgb.train(LGB, dtr, 500, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])
        pred[pr] = m.predict(Xs[pr], num_iteration=m.best_iteration)
        imp += m.feature_importance("gain")
        print(f"{name} fold {P['dates'][di[pr[0]]]} train {len(tr)} trees {m.best_iteration} {time.time() - t:.0f}s", flush=True)
    save_pred(name, pred)
    top = sorted(zip(imp / imp.sum(), names), reverse=True)[:10]
    print(name, "top gain:", ", ".join(f"{n} {g:.0%}" for g, n in top))


def run_linear():
    pred = np.full(len(y), np.nan)
    for tr, va, pr in folds(P):
        tv = np.concatenate([tr, va])
        sign = np.array([np.sign(np.nanmean(daily_ic_fast(X[tv, j], y[tv], di[tv]))) for j in range(X.shape[1])])
        Z = (X[pr] - 0.5) * sign
        pred[pr] = np.nanmean(Z, axis=1)
        print("linear fold", P["dates"][di[pr[0]]], "signs", "".join("+" if s > 0 else "-" for s in sign), flush=True)
    save_pred("linear", pred)


def daily_ic_fast(x, yy, d):
    ok = np.isfinite(x)
    x, yy, d = x[ok], yy[ok], d[ok]
    u, inv = np.unique(d, return_inverse=True)
    n = np.bincount(inv)
    mx, my = np.bincount(inv, x) / n, np.bincount(inv, yy) / n
    xc, yc = x - mx[inv], yy - my[inv]
    return np.bincount(inv, xc * yc) / np.sqrt(np.bincount(inv, xc * xc) * np.bincount(inv, yc * yc))


if __name__ == "__main__":
    run_linear()
    run_lgb("lgb_expanding", None)
    run_lgb("lgb_rolling2y", 504)
