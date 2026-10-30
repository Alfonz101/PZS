"""Seminární práce II: klasifikace hlasových záznamů /a/ (databáze VOICED).

Postup:
  1) načtení 208 záznamů a diagnóz z hlavičkových souborů (expertní anotace),
  2) výpočet příznaků vlastními algoritmy (voice_features.py),
  3) fáze 1: zdravý x patologický,
     fáze 2: typ poruchy u patologických (malé skupiny sloučeny do "other"),
  4) kaskáda fáze 1 -> fáze 2 hodnocená stratifikovanou křížovou validací,
  5) tabulky úspěšnosti, matice záměn, PCA vizualizace, ukázky signálu/spektra/kepstra.

Použití:
  python sem2_voiced_classification.py [--data-dir data] [--reuse]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (balanced_accuracy_score, classification_report, confusion_matrix, roc_auc_score)
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import physio
from dsp import magnitude_spectrum, real_cepstrum
from voice_features import extract_features, stable_segment

DB = "voiced"
SEED = 42
MIN_GROUP_SIZE = 10          # skupiny menší než toto se sloučí do "other"
HEALTHY, PATHOLOGICAL = "healthy", "pathological"
N_REPEATS = 10               # opakování křížové validace pro odhad rozptylu úspěšnosti
# české popisky do grafů a tabulek (interně se pracuje s anglickými názvy z hlaviček)
CZ = {"healthy": "zdravý", "pathological": "patologický", "hyperkinetic dysphonia": "hyperkinetická dysfonie",
      "hypokinetic dysphonia": "hypokinetická dysfonie", "reflux laryngitis": "reflux laryngitis", "other": "ostatní"}


def cz(label: str) -> str:
    return CZ.get(label, label)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
# Překlepy v anotacích databáze (např. "hyperkineti dysphonia" u voice143)
DIAGNOSIS_FIXES = {"hyperkineti dysphonia": "hyperkinetic dysphonia"}


def parse_diagnosis(comments: list[str]) -> str | None:
    """Vyhledá diagnózu v komentářích hlavičky.

    Na jednom řádku může být víc polí ('<diagnosis>: healthy <medications>: none'),
    proto se text ořízne před dalším polem '<...>'.
    """
    for line in comments:
        match = re.search(r"<?diagnos\w*>?\s*:\s*([^<]+)", line, flags=re.IGNORECASE)
        if match:
            text = " ".join(match.group(1).lower().split())
            return DIAGNOSIS_FIXES.get(text, text)
    return None


def parse_sex(comments: list[str]) -> str | None:
    """Pohlaví z komentáře hlavičky ('<sex>: F') – jen pro kontrolu F0, není to příznak klasifikace."""
    for line in comments:
        match = re.search(r"<sex>\s*:\s*([MF])", line, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None


def build_feature_table(data_dir, cache: Path, reuse: bool) -> pd.DataFrame:
    if reuse and cache.exists():
        return pd.read_csv(cache)
    rows = []
    records = [r for r in physio.list_records(DB, data_dir) if r.startswith("voice")]
    print(f"Počet záznamů: {len(records)}")
    for i, rec in enumerate(records, 1):
        comments = physio.read_comments(rec, DB, data_dir)
        diagnosis = parse_diagnosis(comments)
        signal, fs = physio.read_signal(rec, DB, "voice", data_dir)
        rows.append({"record": rec, "diagnosis": diagnosis, "sex": parse_sex(comments), **extract_features(signal, fs)})
        print(f"[{i}/{len(records)}] {rec}: {diagnosis}")
    df = pd.DataFrame(rows)
    df.to_csv(cache, index=False)
    return df


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    if df["diagnosis"].isna().any():
        raise SystemExit("U některých záznamů nebyla nalezena diagnóza – zkontrolujte komentáře v .hea "
                         "a upravte parse_diagnosis().")
    df = df.copy()
    is_healthy = df["diagnosis"].str.contains("healthy|normal")
    df["binary"] = np.where(is_healthy, HEALTHY, PATHOLOGICAL)
    counts = df.loc[~is_healthy, "diagnosis"].value_counts()
    small = counts[counts < MIN_GROUP_SIZE].index
    df["group"] = np.where(is_healthy, HEALTHY, np.where(df["diagnosis"].isin(small), "other", df["diagnosis"]))
    return df


# --------------------------------------------------------------------------
# Klasifikace
# --------------------------------------------------------------------------
K_GRID = [5, 8, 12, "all"]   # prohledávaný prostor počtu příznaků ve vnitřní CV


def _param_grid(name: str) -> dict:
    return {
        "SVM (RBF)": {"selectkbest__k": K_GRID, "svc__C": [1, 3, 10]},
        "kNN (k=5)": {"selectkbest__k": K_GRID, "kneighborsclassifier__n_neighbors": [3, 5, 7]},
        "Random forest": {"selectkbest__k": K_GRID},
    }[name]


def make_model(name: str) -> GridSearchCV:
    """Pipeline uvnitř zanořeného GridSearchCV: k příznaků i hyperparametr klasifikátoru se ladí
    na trénovací části každého foldu vnější CV – žádný únik informace z testovací části."""
    clf = {
        "SVM (RBF)": SVC(C=3, gamma="scale", class_weight="balanced", random_state=SEED),
        "kNN (k=5)": KNeighborsClassifier(n_neighbors=5, weights="distance"),
        "Random forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED),
    }[name]
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), SelectKBest(f_classif, k=8), clf)
    return GridSearchCV(pipe, _param_grid(name), cv=3, scoring="balanced_accuracy", refit=True, n_jobs=-1)


MODEL_NAMES = ["SVM (RBF)", "kNN (k=5)", "Random forest"]


def safe_splits(y: np.ndarray, n_splits: int) -> int:
    """Počet foldů nesmí překročit počet vzorků nejmenší třídy."""
    return max(2, min(n_splits, int(pd.Series(y).value_counts().min())))


def cv_predict(model_name: str, X: pd.DataFrame, y: np.ndarray, n_splits: int) -> np.ndarray:
    pred = np.empty(len(y), dtype=object)
    n_splits = safe_splits(y, n_splits)
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=SEED)
    for train, test in cv.split(X, y):
        model = make_model(model_name).fit(X.iloc[train], y[train])
        pred[test] = model.predict(X.iloc[test])
    return pred


def repeated_scores(model_name: str, X: pd.DataFrame, y: np.ndarray, n_splits: int):
    """Přesnost a vyvážená přesnost z opakované stratifikované CV -> (pole acc, pole bacc), hodnota za opakování."""
    n_splits = safe_splits(y, n_splits)
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=N_REPEATS, random_state=SEED)
    acc, bacc = [], []
    pred = np.empty(len(y), dtype=object)
    for i, (train, test) in enumerate(cv.split(X, y)):
        pred[test] = make_model(model_name).fit(X.iloc[train], y[train]).predict(X.iloc[test])
        if (i + 1) % n_splits == 0:       # konec jednoho opakování = predikce pro všechny nahrávky
            acc.append(np.mean(pred == y)); bacc.append(balanced_accuracy_score(y, pred))
    return 100 * np.array(acc), 100 * np.array(bacc)


def compare_models(X, y, n_splits, title) -> tuple[str, pd.DataFrame]:
    """Srovnání modelů: jedno rozdělení (semínko SEED) a průměr ± sm. odchylka přes N_REPEATS opakování CV.

    Nejlepší model se vybírá podle průměrné vyvážené přesnosti z opakované CV (stabilnější než jedno rozdělení).
    """
    rows = []
    for name in MODEL_NAMES:
        pred = cv_predict(name, X, y, n_splits)
        acc, bacc = repeated_scores(name, X, y, n_splits)
        rows.append({"model": name, "accuracy_%": 100 * np.mean(pred == y),
                     "balanced_accuracy_%": 100 * balanced_accuracy_score(y, pred),
                     "accuracy_mean_%": acc.mean(), "accuracy_std_%": acc.std(),
                     "balanced_accuracy_mean_%": bacc.mean(), "balanced_accuracy_std_%": bacc.std()})
    table = pd.DataFrame(rows)
    print(f"\n{title}\n{table.round(2).to_string(index=False)}")
    return table.loc[table["balanced_accuracy_mean_%"].idxmax(), "model"], table


def feature_auc(X: pd.DataFrame, y_bin: np.ndarray) -> pd.DataFrame:
    """ROC AUC jednotlivých příznaků pro zdravý x patologický (max(AUC, 1-AUC), na směru nezáleží)."""
    target = (y_bin == PATHOLOGICAL).astype(int)
    rows = []
    for col in X:
        ok = X[col].notna().to_numpy()
        auc = roc_auc_score(target[ok], X.loc[ok, col])
        rows.append({"feature": col, "AUC": max(auc, 1 - auc), "higher_in": PATHOLOGICAL if auc >= 0.5 else HEALTHY})
    return pd.DataFrame(rows).sort_values("AUC", ascending=False).reset_index(drop=True)


def cascade_predict(model1: str, model2: str, X: pd.DataFrame, df: pd.DataFrame, n_splits: int) -> np.ndarray:
    """Kaskáda: fáze 1 (zdravý/patologický) -> fáze 2 (skupina poruchy), vše v rámci jedné CV."""
    y_bin, y_grp = df["binary"].to_numpy(), df["group"].to_numpy()
    pred = np.empty(len(df), dtype=object)
    cv = StratifiedKFold(safe_splits(y_grp, n_splits), shuffle=True, random_state=SEED)
    for train, test in cv.split(X, y_grp):
        stage1 = make_model(model1).fit(X.iloc[train], y_bin[train])
        path_train = train[y_bin[train] == PATHOLOGICAL]
        stage2 = make_model(model2).fit(X.iloc[path_train], y_grp[path_train])  # zanořená CV ladí k i hyperparametry
        p1 = stage1.predict(X.iloc[test])
        out = np.full(len(test), HEALTHY, dtype=object)
        sick = p1 == PATHOLOGICAL
        if sick.any():
            out[sick] = stage2.predict(X.iloc[test][sick])
        pred[test] = out
    return pred


# --------------------------------------------------------------------------
# Výstupy
# --------------------------------------------------------------------------
def plot_confusion(y, pred, labels, title, path: Path):
    cm = confusion_matrix(y, pred, labels=labels)
    fig, ax = plt.subplots(figsize=(1.3 * len(labels) + 3, 1.1 * len(labels) + 2.5))
    ax.imshow(cm, cmap="Blues")
    for i, j in np.ndindex(cm.shape):
        ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    short = [cz(l) for l in labels]
    ax.set_xticks(range(len(labels)), short, rotation=40, ha="right")
    ax.set_yticks(range(len(labels)), short)
    ax.set(xlabel="Predikce", ylabel="Anotace expertů", title=title)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_pca(X: pd.DataFrame, df: pd.DataFrame, path: Path):
    Z = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit_transform(X)
    pca = PCA(n_components=2).fit(Z)
    P = pca.transform(Z)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, col in zip(axes, ["binary", "group"]):
        for label in sorted(df[col].unique()):
            m = (df[col] == label).to_numpy()
            ax.scatter(P[m, 0], P[m, 1], s=22, alpha=0.75, label=f"{cz(label)} ({m.sum()})")
        ax.set(xlabel=f"PC1 ({100 * pca.explained_variance_ratio_[0]:.1f} %)",
               ylabel=f"PC2 ({100 * pca.explained_variance_ratio_[1]:.1f} %)",
               title="PCA příznaků: " + ("zdravý/patologický" if col == "binary" else "skupiny poruch"))
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_feature_boxplots(X: pd.DataFrame, df: pd.DataFrame, path: Path):
    names = ["f0_mean", "jitter_local", "shimmer_local", "hnr_mean", "cpp_mean", "spec_centroid"]
    names = [n for n in names if n in X]
    groups = sorted(df["group"].unique())
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, name in zip(axes.ravel(), names):
        ax.boxplot([X.loc[df["group"] == g, name].dropna() for g in groups], tick_labels=[cz(g).replace(" ", "\n", 1) for g in groups])
        ax.set_title(name); ax.tick_params(axis="x", labelsize=8); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_examples(df: pd.DataFrame, data_dir, path: Path):
    """Signál v čase, spektrum a kepstrum jednoho zdravého a jednoho patologického záznamu."""
    picks = [(df.loc[df["binary"] == HEALTHY, "record"].iloc[0], "zdravý"),
             (df.loc[df["binary"] == PATHOLOGICAL, "record"].iloc[0], "patologický")]
    fig, axes = plt.subplots(3, 2, figsize=(13, 9))
    for col, (rec, label) in enumerate(picks):
        x, fs = physio.read_signal(rec, DB, "voice", data_dir)
        t = np.arange(len(x)) / fs
        axes[0, col].plot(t, x, lw=0.4)
        axes[0, col].set(title=f"{rec} ({label}): signál v čase", xlabel="t [s]")
        seg = stable_segment(x - x.mean(), fs)[: int(0.1 * fs)]
        freq, spec = magnitude_spectrum(seg, fs, nfft=4096)
        axes[1, col].plot(freq, 20 * np.log10(spec + 1e-12), lw=0.7)
        axes[1, col].set(title="Spektrum (FFT, 100 ms)", xlabel="f [Hz]", ylabel="dB", xlim=(0, fs / 2))
        quef, ceps = real_cepstrum(seg[:512], fs)
        axes[2, col].plot(quef * 1000, ceps, lw=0.7)
        axes[2, col].set(title="Reálné kepstrum", xlabel="kvefrence [ms]", xlim=(0, 20))
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def per_group_table(y, pred, labels) -> pd.DataFrame:
    report = classification_report(y, pred, labels=labels, output_dict=True, zero_division=0)
    table = pd.DataFrame({l: report[l] for l in labels}).T
    table.index = [cz(l) for l in labels]
    table["recall"] *= 100; table["precision"] *= 100; table["f1-score"] *= 100
    table.columns = ["precision_%", "recall_% (úspěšnost)", "f1_%", "počet"]
    return table


def main():
    physio.utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out", default="outputs/sem2")
    parser.add_argument("--reuse", action="store_true", help="Použít uloženou tabulku příznaků")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    df = add_labels(build_feature_table(args.data_dir, out / "priznaky.csv", args.reuse))
    X = df.drop(columns=["record", "diagnosis", "sex", "binary", "group"], errors="ignore")
    if "sex" in df:   # kontrola věrohodnosti F0 (oktávové chyby by se projevily nízkou F0 u žen)
        f0 = df.groupby("sex")["f0_mean"].median()
        low = int(((df["sex"] == "F") & (df["f0_mean"] < 140)).sum())
        print(f"\nMedián F0: ženy {f0.get('F', np.nan):.0f} Hz, muži {f0.get('M', np.nan):.0f} Hz; "
              f"ženy s F0 < 140 Hz: {low} z {int((df['sex'] == 'F').sum())}; "
              f"medián sm. odchylky F0 v nahrávce {df['f0_std'].median():.1f} Hz")
    print("\nDiagnózy:\n", df["diagnosis"].value_counts().to_string())
    print("\nSkupiny po sloučení:\n", df["group"].value_counts().to_string())

    plot_examples(df, args.data_dir, out / "ukazka_signalu_spektra_kepstra.png")
    plot_pca(X, df, out / "pca.png")
    plot_feature_boxplots(X, df, out / "boxploty_priznaku.png")
    auc = feature_auc(X, df["binary"].to_numpy())
    auc.round(3).to_csv(out / "auc_priznaku.csv", index=False)
    print("\nNejlepší jednotlivé příznaky (AUC zdravý x patologický):\n", auc.head(8).round(3).to_string(index=False))
    X.groupby(df["group"].map(cz)).median().T.round(4).to_csv(out / "mediany_priznaku_po_skupinach.csv")
    print(f"Nahrávky s chybějícím jitterem/shimmerem: {int(X['jitter_local'].isna().sum())}")

    # fáze 1
    best1, t1 = compare_models(X, df["binary"].to_numpy(), args.folds, "FÁZE 1: zdravý x patologický")
    t1.round(2).to_csv(out / "faze1_modely.csv", index=False)
    pred1 = cv_predict(best1, X, df["binary"].to_numpy(), args.folds)
    labels1 = [HEALTHY, PATHOLOGICAL]
    plot_confusion(df["binary"], pred1, labels1, f"Fáze 1 – {best1}", out / "faze1_matice_zamen.png")
    per_group_table(df["binary"], pred1, labels1).round(2).to_csv(out / "faze1_uspesnost.csv")

    # fáze 2 (jen patologické)
    sick = df["binary"] == PATHOLOGICAL
    Xs, ys = X[sick].reset_index(drop=True), df.loc[sick, "group"].to_numpy()
    best2, t2 = compare_models(Xs, ys, args.folds, "FÁZE 2: typ poruchy (jen patologické)")  # k_best ladí zanořená CV
    t2.round(2).to_csv(out / "faze2_modely.csv", index=False)

    # kaskáda
    pred = cascade_predict(best1, best2, X, df, args.folds)
    labels = [HEALTHY] + sorted(df.loc[sick, "group"].unique())
    plot_confusion(df["group"], pred, labels, "Kaskáda: všechny skupiny", out / "kaskada_matice_zamen.png")
    table = per_group_table(df["group"], pred, labels)
    table.round(2).to_csv(out / "kaskada_uspesnost_po_skupinach.csv")
    pathological_ok = np.mean((pred[sick] != HEALTHY))
    print(f"\nKASKÁDA (modely: {best1} -> {best2})\n{table.round(1).to_string()}")
    print(f"\nCelková přesnost přes všechny skupiny: {100 * np.mean(pred == df['group']):.1f} %")
    print(f"Podíl patologických správně označených jako patologické: {100 * pathological_ok:.1f} %")
    print(f"Výstupy uloženy do {out}/")


if __name__ == "__main__":
    main()