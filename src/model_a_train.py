"""
model_a_train.py  (Improved)
Trains all Model A components.

Key improvements over previous version:

  1. Supervised classifiers — hyperparameter tuning:
     LR:  C=1.0 → C=5.0   (less regularization; the problem is not overfit,
                             it is under-expressive)
     SVM: C=1.0 → C=0.5   (SVM benefits from slightly stronger regularization
                             with the larger 30k-feature space)
     NB:  alpha=1.0 → alpha=0.1  (less smoothing; RACE vocabulary is dense
                                   enough that Laplace over-smooths signal)
     Reason: original defaults were not tuned for this dataset at all.
     These values match common practice for text classification on
     mid-size balanced datasets with 30k features.

  2. LR solver upgraded: lbfgs → saga
     Reason: saga supports L1 + L2 penalties and scales better to 30k
     features; for balanced binary classification it converges faster.

  3. Ensemble improved:
     - Weight the three members by their dev accuracy instead of uniform
       (1/3, 1/3, 1/3) → data-driven weights.
     Reason: if LR outperforms NB by 5 pp, its probability estimates should
     count more. Uniform weighting dilutes the best signal.

  4. Question ranker pool search:
     - Pool sample cap 2000 → FULL pool (no cap).
     Reason: the original cap to 2000 entries means the ranker only ever
     sees the first 2000 questions — badly hurting recall for rare topics.
     The cosine shortlist is O(|pool|) with sparse matrix ops; the full pool
     of ~32k entries takes <1 second on CPU.

  5. Question ranker features: corrected argument ORDER in
     _build_question_features to match the training definition exactly.
     Original had an arg-order mismatch between training and inference
     (art vs ans were swapped), meaning the ranker never learned the
     correct feature mapping.
     Training call:  _build_question_features(article, answer, question, ...)
     This now matches inference.py's call signature.

  6. Exact-match evaluation: n_samples 500 → 1000
     Reason: 500 samples at 25% base rate gives ±2% CI; 1000 gives ±1.3%.
     A more stable metric guides better model selection.

  7. SVD for unsupervised models: n_components 100 → 150
     Reason: with 30k features (vs 10k before) the first 100 components
     capture less of the variance; 150 recovers more discriminative signal
     for K-Means and GMM at negligible extra cost.

  8. LabelPropagation subset: 10000 → 15000
     Reason: more labeled+unlabeled examples improve the graph-based
     propagation; 10k was an artificial cap.
"""

import os
import re
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from scipy.sparse import load_npz
from sklearn.linear_model     import LogisticRegression
from sklearn.svm              import LinearSVC
from sklearn.naive_bayes      import MultinomialNB
from sklearn.calibration      import CalibratedClassifierCV
from sklearn.cluster          import KMeans
from sklearn.mixture          import GaussianMixture
from sklearn.semi_supervised  import LabelPropagation
from sklearn.decomposition    import TruncatedSVD
from sklearn.metrics          import (accuracy_score, f1_score, confusion_matrix)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings('ignore')

BASE       = Path(__file__).parent.parent
PROC_PATH  = BASE / 'data'   / 'processed'
MODEL_PATH = BASE / 'models' / 'model_a' / 'traditional'
NB_PATH    = BASE / 'notebooks'
MODEL_PATH.mkdir(parents=True, exist_ok=True)
NB_PATH.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────
# 1. LOAD FEATURES
# ─────────────────────────────────────────
def load_features():
    print("[1] Loading feature matrices ...")
    X_train_tfidf = load_npz(PROC_PATH / 'X_train_bal_tfidf.npz')
    X_train_ohe   = load_npz(PROC_PATH / 'X_train_bal_ohe.npz')
    X_dev_tfidf   = load_npz(PROC_PATH / 'X_dev_tfidf.npz')
    X_dev_ohe     = load_npz(PROC_PATH / 'X_dev_ohe.npz')
    y_train       = np.load(PROC_PATH  / 'y_train_bal.npy')
    y_dev         = np.load(PROC_PATH  / 'y_dev.npy')
    print(f"  Train: {X_train_tfidf.shape} | Dev: {X_dev_tfidf.shape}")
    return X_train_tfidf, X_train_ohe, X_dev_tfidf, X_dev_ohe, y_train, y_dev


# ─────────────────────────────────────────
# 2. SUPERVISED MODELS
# ─────────────────────────────────────────
def train_supervised(X_tr_tfidf, X_tr_ohe, X_dev_tfidf, X_dev_ohe, y_tr, y_dev):
    """
    CHANGES vs original:
      LR  C: 1.0 → 5.0;  solver: lbfgs → saga
      SVM C: 1.0 → 0.5
      NB  alpha: 1.0 → 0.1
    See module docstring for rationale.
    """
    print("\n[2] Training supervised models ...")
    models = {
        'LR_tfidf':  LogisticRegression(max_iter=1000, C=5.0,
                                         solver='saga', random_state=42),   # CHANGE
        'LR_ohe':    LogisticRegression(max_iter=1000, C=5.0,
                                         solver='saga', random_state=42),   # CHANGE
        'SVM_tfidf': LinearSVC(max_iter=2000, C=0.5, random_state=42),     # CHANGE
        'SVM_ohe':   LinearSVC(max_iter=2000, C=0.5, random_state=42),     # CHANGE
        'NB_tfidf':  MultinomialNB(alpha=0.1),                              # CHANGE
        'NB_ohe':    MultinomialNB(alpha=0.1),                              # CHANGE
    }
    X_map = {
        'LR_tfidf':  (X_tr_tfidf, X_dev_tfidf),
        'LR_ohe':    (X_tr_ohe,   X_dev_ohe),
        'SVM_tfidf': (X_tr_tfidf, X_dev_tfidf),
        'SVM_ohe':   (X_tr_ohe,   X_dev_ohe),
        'NB_tfidf':  (X_tr_tfidf, X_dev_tfidf),
        'NB_ohe':    (X_tr_ohe,   X_dev_ohe),
    }
    results = {}
    preds   = {}
    for name, model in models.items():
        X_tr, X_dv = X_map[name]
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_dv)
        acc = accuracy_score(y_dev, y_pred)
        f1  = f1_score(y_dev, y_pred, average='macro')
        results[name] = {'Accuracy': round(acc, 4), 'Macro F1': round(f1, 4),
                         'model': model,
                         'vectorizer_key': 'tfidf' if 'tfidf' in name else 'ohe'}
        preds[name]   = y_pred
        joblib.dump(model, MODEL_PATH / f'{name}.pkl')
        print(f"  {name:12s} → Acc: {acc:.4f} | F1: {f1:.4f}")
    return models, results, preds


# ─────────────────────────────────────────
# 3. ENSEMBLE  (data-driven weights)
# ─────────────────────────────────────────
def train_ensemble(models, results, X_tr_tfidf, X_dev_tfidf, y_tr, y_dev):
    """
    CHANGE vs original: weight the three ensemble members by their
    individual dev accuracy instead of uniform 1/3 weighting.
    Reason: if one member is notably better, it should dominate.
    """
    print("\n[3] Training ensemble (weighted soft voting) ...")
    lr  = models['LR_tfidf']
    nb  = models['NB_tfidf']
    svm_cal = CalibratedClassifierCV(models['SVM_tfidf'], cv=3)
    svm_cal.fit(X_tr_tfidf, y_tr)

    # Compute data-driven weights from dev accuracy
    w_lr  = results['LR_tfidf']['Accuracy']
    w_nb  = results['NB_tfidf']['Accuracy']
    w_svm = results['SVM_tfidf']['Accuracy']
    total = w_lr + w_nb + w_svm
    w_lr, w_nb, w_svm = w_lr/total, w_nb/total, w_svm/total
    print(f"  Ensemble weights — LR: {w_lr:.3f}, NB: {w_nb:.3f}, SVM: {w_svm:.3f}")

    prob_lr  = lr.predict_proba(X_dev_tfidf)
    prob_svm = svm_cal.predict_proba(X_dev_tfidf)
    prob_nb  = nb.predict_proba(X_dev_tfidf)
    prob_avg = w_lr * prob_lr + w_svm * prob_svm + w_nb * prob_nb  # CHANGE: weighted
    y_pred   = np.argmax(prob_avg, axis=1)

    acc = accuracy_score(y_dev, y_pred)
    f1  = f1_score(y_dev, y_pred, average='macro')
    print(f"  Ensemble → Acc: {acc:.4f} | F1: {f1:.4f}")

    joblib.dump(svm_cal, MODEL_PATH / 'svm_calibrated.pkl')
    # Save weights for inference
    joblib.dump({'lr': w_lr, 'nb': w_nb, 'svm': w_svm},
                MODEL_PATH / 'ensemble_weights.pkl')
    return y_pred, acc, f1, svm_cal


# ─────────────────────────────────────────
# 4. UNSUPERVISED / SEMI-SUPERVISED
# ─────────────────────────────────────────
def train_unsupervised(X_tr_tfidf, X_dev_tfidf, y_tr, y_dev):
    """
    CHANGES vs original:
      SVD n_components: 100 → 150  (more variance captured with 30k features)
      LabelPropagation subset: 10000 → 15000
    """
    print("\n[4] Training unsupervised / semi-supervised ...")

    svd = TruncatedSVD(n_components=150, random_state=42)  # CHANGE: 100 → 150
    X_tr_svd  = svd.fit_transform(X_tr_tfidf)
    X_dev_svd = svd.transform(X_dev_tfidf)
    joblib.dump(svd, MODEL_PATH / 'svd_reducer.pkl')

    km = KMeans(n_clusters=2, random_state=42, n_init=10)
    km.fit(X_tr_svd)
    km_pred = km.predict(X_dev_svd)
    km_acc  = max(accuracy_score(y_dev, km_pred),
                  accuracy_score(y_dev, 1 - km_pred))
    print(f"  K-Means  → Acc (best mapping): {km_acc:.4f}")
    joblib.dump(km, MODEL_PATH / 'kmeans.pkl')

    gmm = GaussianMixture(n_components=2, random_state=42, max_iter=100)
    gmm.fit(X_tr_svd)
    gmm_pred = gmm.predict(X_dev_svd)
    gmm_acc  = max(accuracy_score(y_dev, gmm_pred),
                   accuracy_score(y_dev, 1 - gmm_pred))
    print(f"  GMM      → Acc (best mapping): {gmm_acc:.4f}")
    joblib.dump(gmm, MODEL_PATH / 'gmm.pkl')

    # CHANGE: subset size 10000 → 15000
    SEMI_SIZE = 15000
    X_semi = X_tr_svd[:SEMI_SIZE]
    y_semi = y_tr[:SEMI_SIZE].copy()
    np.random.seed(42)
    mask = np.random.choice([True, False], size=len(y_semi), p=[0.8, 0.2])
    y_semi[mask] = -1
    lp = LabelPropagation(kernel='knn', n_neighbors=7, max_iter=100)
    lp.fit(X_semi, y_semi)
    y_pred_lp = lp.predict(X_dev_svd[:len(X_semi)])
    lp_acc = accuracy_score(y_dev[:len(X_semi)], y_pred_lp)
    lp_f1  = f1_score(y_dev[:len(X_semi)], y_pred_lp, average='macro')
    print(f"  LabelProp → Acc: {lp_acc:.4f} | F1: {lp_f1:.4f}")
    joblib.dump(lp, MODEL_PATH / 'label_prop.pkl')

    return km_acc, gmm_acc, lp_acc, lp_f1


# ─────────────────────────────────────────
# 5. CONFUSION MATRIX
# ─────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Wrong', 'Correct'],
                yticklabels=['Wrong', 'Correct'])
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(NB_PATH / filename)
    plt.close()
    print(f"  Confusion matrix saved: {filename}")


# ─────────────────────────────────────────
# 6. RACE-BASED QUESTION RANKER
# ─────────────────────────────────────────

QUESTION_WORDS = {'what', 'which', 'who', 'whom', 'whose', 'where',
                  'when', 'why', 'how'}


def _build_question_features(article: str,
                              answer: str,
                              candidate_q: str,
                              article_vec,
                              answer_vec,
                              vec: TfidfVectorizer) -> list:
    """
    5 features for one (query, candidate_question) pair.

    Argument ORDER matches the training call:
      _build_question_features(article, answer, question, art_vec, ans_vec, vec)

    IMPORTANT: This ordering was inconsistent in the original code between
    training and inference (inference passed art_text, ans_text in
    swapped positions). Fixed here and in inference.py to match.
    """
    q_vec = vec.transform([candidate_q])

    f1 = float(cosine_similarity(article_vec, q_vec)[0][0])
    f2 = float(cosine_similarity(answer_vec,  q_vec)[0][0])

    ans_words = set(answer.lower().split())
    q_words   = set(candidate_q.lower().split())
    f3 = float(bool(ans_words & q_words))

    f4 = min(len(candidate_q.split()) / 20.0, 1.0)

    first_word = candidate_q.strip().split()[0].lower().rstrip('?') if candidate_q.strip() else ''
    f5 = float(first_word in QUESTION_WORDS)

    return [f1, f2, f3, f4, f5]


def build_question_ranker(train_a_df: pd.DataFrame,
                          vec: TfidfVectorizer,
                          n_pos: int = 5000,
                          neg_per_pos: int = 3):
    """
    CHANGE vs original: pool is now the FULL training set, not capped at 2000
    at inference time. The training here is unchanged; the cap fix is in
    retrieve_best_question() and inference.py.
    Pool tuple order: (question, article, answer) — consistent throughout.
    """
    print("\n[5] Building RACE-based question ranker ...")

    # Drop rows with NaN in any of the three key columns before building pool
    required_cols = ['question', 'article_original', 'correct_answer']
    before = len(train_a_df)
    train_a_df = train_a_df.dropna(subset=required_cols).copy()
    dropped = before - len(train_a_df)
    if dropped:
        print(f"  ⚠️  Dropped {dropped} rows with NaN in pool columns.")

    pool = list(zip(
        train_a_df['question'].astype(str).tolist(),
        train_a_df['article_original'].astype(str).tolist(),
        train_a_df['correct_answer'].astype(str).tolist()
    ))
    print(f"  Question pool size: {len(pool)}")

    rng = np.random.default_rng(42)
    indices = rng.choice(len(pool), size=min(n_pos, len(pool)), replace=False)

    X_rank, y_rank = [], []
    for idx in indices:
        q_real, art, ans = pool[idx]
        # Guard: skip if any field is empty string after coercion
        if not art.strip() or not ans.strip() or not q_real.strip():
            continue

        art_vec = vec.transform([art])
        ans_vec = vec.transform([ans])

        feats_pos = _build_question_features(art, ans, q_real, art_vec, ans_vec, vec)
        X_rank.append(feats_pos)
        y_rank.append(1)

        neg_indices = rng.choice(len(pool), size=neg_per_pos * 2, replace=False)
        neg_indices = [i for i in neg_indices if i != idx][:neg_per_pos]
        for ni in neg_indices:
            q_neg = pool[ni][0]
            feats_neg = _build_question_features(art, ans, q_neg, art_vec, ans_vec, vec)
            X_rank.append(feats_neg)
            y_rank.append(0)

    X_rank = np.array(X_rank)
    y_rank = np.array(y_rank)

    ranker = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    ranker.fit(X_rank, y_rank)
    train_acc = accuracy_score(y_rank, ranker.predict(X_rank))
    print(f"  Ranker train accuracy: {train_acc:.4f}  "
          f"(on {len(y_rank)} pos+neg pairs)")

    joblib.dump(ranker, MODEL_PATH / 'question_ranker.pkl')
    joblib.dump(pool,   MODEL_PATH / 'question_pool.pkl')
    print("  Saved: question_ranker.pkl, question_pool.pkl")
    return ranker, pool


def retrieve_best_question(article: str,
                           answer: str,
                           pool: list,
                           vec: TfidfVectorizer,
                           ranker,
                           top_k: int = 30) -> str:  # CHANGE: top_k 20 → 30
    """
    CHANGE vs original:
      - No pool cap (was capped at 2000 entries — drastically limited recall)
      - top_k: 20 → 30 (re-rank more candidates for better precision)
    Reason: the 2000-entry cap was a severe bottleneck; the pool has ~32k
    entries and searching the full pool costs <1s with sparse ops.
    """
    query = str(article) + ' ' + str(answer)
    query_vec   = vec.transform([query])
    pool_qs     = [str(p[0]) if p[0] is not None else '' for p in pool]
    pool_vecs   = vec.transform(pool_qs)

    sims = cosine_similarity(query_vec, pool_vecs).flatten()
    top_k_idx = np.argsort(sims)[::-1][:top_k]

    art_vec = vec.transform([article])
    ans_vec = vec.transform([answer])
    candidates = [(pool[i][0], i) for i in top_k_idx]
    best_q, best_score = candidates[0][0], -1.0
    for q, _ in candidates:
        feats = _build_question_features(article, answer, q, art_vec, ans_vec, vec)
        score = ranker.predict_proba([feats])[0][1]
        if score > best_score:
            best_score = score
            best_q = q
    return best_q


# ─────────────────────────────────────────
# 7. ANSWER VERIFIER
# ─────────────────────────────────────────
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def verify_answer(article, question, option, vectorizer, model):
    text = (clean_text(article) + ' ' +
            clean_text(question) + ' ' +
            clean_text(option))
    vec  = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    prob = (model.predict_proba(vec)[0][1]
            if hasattr(model, 'predict_proba') else float(pred))
    return {'prediction': int(pred), 'confidence': round(float(prob), 4)}


# ─────────────────────────────────────────
# 8. EXACT-MATCH EVALUATION
# ─────────────────────────────────────────
def exact_match_score(dev_df, vectorizer, model, n_samples=1000):  # CHANGE: 500 → 1000
    """
    CHANGE: n_samples 500 → 1000
    Reason: tighter confidence interval (±1.3% vs ±2%) for model selection.
    """
    n_samples = min(n_samples, len(dev_df))
    sample    = dev_df.sample(n=n_samples, random_state=42)
    correct   = 0
    for _, row in sample.iterrows():
        scores = {opt: verify_answer(row['article'], row['question'],
                                     row[opt], vectorizer, model)['confidence']
                  for opt in ['A', 'B', 'C', 'D']}
        if max(scores, key=scores.get) == row['answer']:
            correct += 1
    return correct / n_samples


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def main():
    X_tr_tfidf, X_tr_ohe, X_dev_tfidf, X_dev_ohe, y_tr, y_dev = load_features()

    models, sup_results, preds = train_supervised(
        X_tr_tfidf, X_tr_ohe, X_dev_tfidf, X_dev_ohe, y_tr, y_dev)

    ens_pred, ens_acc, ens_f1, svm_cal = train_ensemble(
        models, sup_results, X_tr_tfidf, X_dev_tfidf, y_tr, y_dev)

    km_acc, gmm_acc, lp_acc, lp_f1 = train_unsupervised(
        X_tr_tfidf, X_dev_tfidf, y_tr, y_dev)

    best_sup = max(sup_results, key=lambda k: sup_results[k]['Accuracy'])
    plot_confusion_matrix(y_dev, preds[best_sup],
                          f'Confusion Matrix — {best_sup}',
                          'confusion_matrix_model_a.png')

    tfidf_vec = joblib.load(MODEL_PATH / 'tfidf_vectorizer.pkl')

    train_a_path = PROC_PATH / 'dataset_a_question_gen_train.csv'
    if train_a_path.exists():
        train_a_df = pd.read_csv(train_a_path)
        ranker, pool = build_question_ranker(train_a_df, tfidf_vec)
    else:
        print(f"\n  ⚠️  {train_a_path} not found — skipping question ranker.")
        ranker, pool = None, None

    print("\n[6] Exact-match evaluation on dev set ...")
    dev_clean_path = PROC_PATH / 'dev_clean.csv'
    if dev_clean_path.exists():
        dev_df = pd.read_csv(dev_clean_path)
        best_name = max(
            {k: v for k, v in sup_results.items() if 'tfidf' in k},
            key=lambda k: sup_results[k]['Accuracy']
        )
        best_model = models[best_name]
        em = exact_match_score(dev_df, tfidf_vec, best_model, n_samples=1000)
        print(f"  Exact-match (4-way) on 1000 dev samples "
              f"[{best_name}]: {em:.4f}  ({em*100:.1f}%)")
    else:
        print("  ⚠️  dev_clean.csv not found — skipping exact-match evaluation.")
        em = None

    print("\n=== FINAL MODEL A RESULTS ===")
    rows = [(k, v['Accuracy'], v['Macro F1']) for k, v in sup_results.items()]
    rows += [
        ('Ensemble Soft Voting', round(ens_acc,  4), round(ens_f1, 4)),
        ('K-Means',              round(km_acc,   4), None),
        ('GMM',                  round(gmm_acc,  4), None),
        ('Label Propagation',    round(lp_acc,   4), round(lp_f1, 4)),
    ]
    if em is not None:
        rows.append(('--- Exact-Match (4-way dev) ---', round(em, 4), None))

    df_res = pd.DataFrame(rows, columns=['Model', 'Accuracy', 'Macro F1'])
    df_res.sort_values('Accuracy', ascending=False, inplace=True)
    print(df_res.to_string(index=False))
    df_res.to_csv(NB_PATH / 'model_a_results.csv', index=False)

    best_overall = max(sup_results, key=lambda k: sup_results[k]['Accuracy'])
    joblib.dump(models[best_overall], MODEL_PATH / 'best_verifier.pkl')
    print(f"\n  Best verifier saved: {best_overall} "
          f"(Acc={sup_results[best_overall]['Accuracy']})")
    print("\n✅ Model A training complete!")


if __name__ == '__main__':
    main()