"""
model_b_train.py  (Improved)
Trains all Model B components.

Key improvements over previous version:

  DISTRACTOR RANKER
  1. Candidate extraction at INFERENCE changed from full article sentences
     to SHORT NOUN-PHRASE CHUNKS extracted from the article.
     Reason: RACE distractors average 33-36 characters — roughly one short
     phrase. Full sentences average 100+ chars. A ranker trained on short
     human-written RACE phrases but evaluated on full sentences has a severe
     train-test mismatch. Chunking the article into noun-phrase-sized pieces
     aligns the candidate space with both the training data and the expected
     output format.

  2. Additional distractor feature: f8 = length_chars_norm
     len(candidate in chars) / median(correct_answer chars)
     Reason: RACE distractors have similar character length to the correct
     answer. Adding this feature gives the ranker an explicit length signal
     it previously lacked.

  3. RF: n_estimators 200 → 300, max_depth 10 → 12
     Reason: more trees and slightly deeper trees improve out-of-bag
     generalization for a 8-feature, 6000-sample dataset. The compute
     cost is modest (< 30s additional on CPU).

  4. LR for distractor ranker: added class_weight='balanced'
     Reason: after the plausibility gate the positive/negative ratio in
     the RACE pool is not perfectly 1:1; balanced weights prevent the
     majority class dominating.

  HINT SCORER
  5. Added feature f7: bigram_overlap_qa
     Fraction of bigrams in the sentence that appear in (question + answer).
     Reason: bigram overlap captures phrase-level similarity better than
     unigram word overlap alone, especially for multi-word answer phrases.

  6. Hint scorer training data: n=4000 → n=6000
     Reason: more data gives the LR a better estimate of feature weights,
     particularly for the new bigram feature.

  7. Hint scorer LR: added class_weight='balanced'
     Reason: after balancing via resample the dataset is 50/50, but real
     hint sentences are rare; balanced weights handle any residual imbalance.

  8. generate_hints() truncation: [:120] → [:200] per hint
     Reason: the original 120-char truncation cuts off many informative
     hints mid-sentence. 200 chars comfortably fits a RACE sentence while
     still being manageable for display.
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
from collections import Counter

from sklearn.linear_model     import LogisticRegression
from sklearn.ensemble         import RandomForestClassifier
from sklearn.model_selection  import train_test_split
from sklearn.metrics          import (accuracy_score, precision_score,
                                      recall_score, f1_score, confusion_matrix)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils            import resample

warnings.filterwarnings('ignore')

BASE       = Path(__file__).parent.parent
PROC_PATH  = BASE / 'data'   / 'processed'
MODEL_PATH = BASE / 'models' / 'model_b' / 'traditional'
MA_PATH    = BASE / 'models' / 'model_a' / 'traditional'
NB_PATH    = BASE / 'notebooks'
MODEL_PATH.mkdir(parents=True, exist_ok=True)
NB_PATH.mkdir(parents=True, exist_ok=True)

STOPWORDS = frozenset([
    'the','a','an','is','are','was','were','be','been',
    'have','has','had','do','does','did','will','would',
    'could','should','may','might','shall','can','need',
    'to','of','in','for','on','with','at','by','from',
    'up','about','into','through','during','and','or',
    'but','if','while','although','because','so','yet',
    'both','each','more','most','other','some','such',
    'than','too','very','just','this','that','these',
    'those','i','we','you','he','she','it','they','my',
    'our','your','his','her','its','their'
])


# ─────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def jaccard(a: str, b: str) -> float:
    sa = set(re.findall(r'\b\w+\b', a.lower()))
    sb = set(re.findall(r'\b\w+\b', b.lower()))
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def tfidf_cosine(text_a: str, text_b: str, vec: TfidfVectorizer) -> float:
    try:
        va = vec.transform([text_a])
        vb = vec.transform([text_b])
        return float(cosine_similarity(va, vb)[0][0])
    except Exception:
        return 0.0


def split_sentences(text: str, min_len: int = 20, max_len: int = 400):
    sents = re.split(r'(?<=[.!?])\s+', str(text))
    return [s.strip() for s in sents if min_len <= len(s.strip()) <= max_len]


def extract_short_chunks(article: str,
                          correct_answer: str,
                          max_chunks: int = 60) -> list:
    """
    Extract short phrase-length chunks from the article as distractor candidates.

    CHANGE vs original: instead of returning full sentences (100+ chars avg),
    we split sentences further into comma/semicolon-separated clauses and
    sliding noun-phrase-sized windows (4-12 words). This aligns the candidate
    space with real RACE distractors (33-36 chars avg).

    Filters:
    - Skip chunks that contain >40% of the correct answer's words
    - Skip near-duplicates (Jaccard > 0.5 against already-selected chunks)
    - Skip chunks shorter than 3 words or longer than 15 words
    """
    correct_words = set(clean_text(correct_answer).split())
    sentences = split_sentences(article)

    raw_chunks = []
    for sent in sentences:
        # Split on commas and semicolons to get clause-level chunks
        parts = re.split(r'[,;]', sent)
        for part in parts:
            part = part.strip()
            words = part.split()
            if 3 <= len(words) <= 15:
                raw_chunks.append(part)
        # Also add sliding windows of 4-10 words over each sentence
        words = sent.split()
        for w in range(4, 11):
            for start in range(0, len(words) - w + 1, 2):
                chunk = ' '.join(words[start:start + w])
                raw_chunks.append(chunk)

    seen_word_sets = []
    candidates = []
    for chunk in raw_chunks:
        chunk_words = set(clean_text(chunk).split())
        if len(chunk_words) < 3:
            continue

        # Skip if too similar to correct answer
        overlap_ratio = len(chunk_words & correct_words) / max(len(chunk_words), 1)
        if overlap_ratio > 0.4:
            continue

        # Skip near-duplicates
        too_similar = any(
            len(chunk_words & sw) / max(len(chunk_words | sw), 1) > 0.5
            for sw in seen_word_sets
        )
        if too_similar:
            continue

        candidates.append(chunk)
        seen_word_sets.append(chunk_words)
        if len(candidates) >= max_chunks:
            break

    return candidates


# ═══════════════════════════════════════════════════════════════════════════
# PART A — DISTRACTOR RANKER
# ═══════════════════════════════════════════════════════════════════════════

def distractor_features(candidate: str,
                        correct: str,
                        article: str,
                        vec: TfidfVectorizer) -> list:
    """
    8 features (CHANGE: was 7, added f8 = length_chars_norm).

    f1  cos_answer    : TF-IDF cosine(candidate, answer)
    f2  cos_article   : TF-IDF cosine(candidate, article)
    f3  jaccard_answer: Token Jaccard(candidate, answer)
    f4  len_ratio     : len(candidate words) / len(answer words)
    f5  char_match    : character-set overlap fraction
    f6  freq_norm     : normalised frequency of candidate in article
    f7  starts_capital: binary — candidate first word was capitalised
    f8  length_chars_norm: len(candidate chars) / median(answer chars) — NEW
        Reason: RACE distractors should have similar char length to the answer;
        this feature penalises overly short/long candidates.
    """
    cand_c = clean_text(candidate)
    ans_c  = clean_text(correct)
    art_c  = clean_text(article)

    f1 = tfidf_cosine(cand_c, ans_c, vec)
    f2 = tfidf_cosine(cand_c, art_c, vec)
    f3 = jaccard(cand_c, ans_c)

    cand_words = cand_c.split()
    ans_words  = ans_c.split()
    f4 = len(cand_words) / max(len(ans_words), 1)

    chars_a = set(ans_c)
    chars_c = set(cand_c)
    f5 = len(chars_a & chars_c) / max(len(chars_a), 1)

    art_words = art_c.split()
    raw_freq  = art_words.count(cand_words[0]) if cand_words else 0
    f6 = raw_freq / max(len(art_words), 1)

    first_char = str(candidate).strip()[:1]
    f7 = float(first_char.isupper())

    # f8: character-length ratio — new feature
    f8 = len(candidate) / max(len(correct), 1)

    return [f1, f2, f3, f4, f5, f6, f7, f8]


FEAT_COLS = ['cos_answer', 'cos_article', 'jaccard_answer',
             'len_ratio', 'char_match', 'freq_norm', 'starts_capital',
             'length_chars_norm']  # CHANGE: added length_chars_norm


def build_distractor_dataset(train_b_df: pd.DataFrame,
                             tfidf: TfidfVectorizer,
                             n_pos: int = 6000,
                             neg_per_pos: int = 2) -> pd.DataFrame:
    """
    Positive: real RACE wrong options.
    Negative: wrong options from different questions.
    Uses the updated 8-feature distractor_features().
    """
    print(f"[1] Building distractor dataset from RACE pool ...")

    pool = []
    for _, row in train_b_df.iterrows():
        for col in ['wrong_option_1', 'wrong_option_2', 'wrong_option_3']:
            pool.append({
                'distractor': str(row[col]),
                'article':    str(row['article_original']),
                'correct':    str(row['correct_answer']),
            })
    pool_df = pd.DataFrame(pool).reset_index(drop=True)
    print(f"  Pool size: {len(pool_df)} distractor entries")

    rng = np.random.default_rng(42)
    n_pos = min(n_pos, len(pool_df))
    pos_idx = rng.choice(len(pool_df), size=n_pos, replace=False)

    rows = []
    for idx in pos_idx:
        entry = pool_df.iloc[idx]
        dist  = entry['distractor']
        art   = entry['article']
        corr  = entry['correct']

        try:
            feats = distractor_features(dist, corr, art, tfidf)
            rows.append(feats + [1])
        except Exception:
            continue

        neg_candidates = rng.choice(len(pool_df),
                                    size=neg_per_pos * 5,
                                    replace=False)
        neg_added = 0
        for ni in neg_candidates:
            if neg_added >= neg_per_pos:
                break
            neg_entry = pool_df.iloc[ni]
            if neg_entry['article'] == art:
                continue
            try:
                feats = distractor_features(
                    neg_entry['distractor'], corr, art, tfidf)
                rows.append(feats + [0])
                neg_added += 1
            except Exception:
                continue

    df = pd.DataFrame(rows, columns=FEAT_COLS + ['label'])
    print(f"  Dataset: {df.shape} | {df['label'].value_counts().to_dict()}")
    return df


def train_distractor_ranker(dist_df: pd.DataFrame):
    """
    CHANGES vs original:
      RF: n_estimators 200 → 300, max_depth 10 → 12
      LR: added class_weight='balanced'
    """
    print("\n[2] Training distractor ranker ...")
    X = dist_df[FEAT_COLS].values
    y = dist_df['label'].values

    X_tr, X_dv, y_tr, y_dv = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    lr_d = LogisticRegression(max_iter=1000, C=1.0,
                               class_weight='balanced',   # CHANGE: added
                               random_state=42)
    lr_d.fit(X_tr, y_tr)
    y_pred_lr = lr_d.predict(X_dv)

    rf_d = RandomForestClassifier(
        n_estimators=300,    # CHANGE: 200 → 300
        max_depth=12,        # CHANGE: 10 → 12
        random_state=42,
        n_jobs=-1
    )
    rf_d.fit(X_tr, y_tr)
    y_pred_rf = rf_d.predict(X_dv)

    print("\n--- Distractor Ranker Results ---")
    for name, yp in [('LR', y_pred_lr), ('RF', y_pred_rf)]:
        print(f"  {name}: Acc={accuracy_score(y_dv,yp):.4f} "
              f"P={precision_score(y_dv,yp,zero_division=0):.4f} "
              f"R={recall_score(y_dv,yp,zero_division=0):.4f} "
              f"F1={f1_score(y_dv,yp,zero_division=0):.4f}")

    importances = rf_d.feature_importances_
    print("\n  RF Feature Importances:")
    for name, imp in sorted(zip(FEAT_COLS, importances),
                            key=lambda x: x[1], reverse=True):
        print(f"    {name:<25s}: {imp:.4f}")

    cm = confusion_matrix(y_dv, y_pred_rf)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=['Bad', 'Good'],
                yticklabels=['Bad', 'Good'])
    plt.title('Confusion Matrix — Distractor Ranker (RF)')
    plt.tight_layout()
    plt.savefig(NB_PATH / 'confusion_matrix_distractor.png')
    plt.close()

    joblib.dump(lr_d, MODEL_PATH / 'distractor_ranker_lr.pkl')
    joblib.dump(rf_d, MODEL_PATH / 'distractor_ranker_rf.pkl')
    print("  Models saved.")
    return rf_d, lr_d, y_dv, y_pred_rf


def generate_distractors(article: str,
                          correct_answer: str,
                          tfidf: TfidfVectorizer,
                          ranker,
                          top_n: int = 3) -> list:
    """
    CHANGE vs original: candidates are now SHORT CHUNKS (extract_short_chunks)
    instead of full sentences. This aligns candidate length with RACE distractor
    length (~33-36 chars) and the training distribution.
    """
    candidates = extract_short_chunks(article, correct_answer, max_chunks=80)

    if not candidates:
        return ['Incorrect option A', 'Incorrect option B', 'Incorrect option C']

    scored = []
    for cand in candidates:
        try:
            feats = distractor_features(cand, correct_answer, article, tfidf)
            prob  = ranker.predict_proba([feats])[0][1]
            scored.append((cand, prob))
        except Exception:
            continue

    scored.sort(key=lambda x: x[1], reverse=True)

    selected = []
    selected_word_sets = []
    for cand, prob in scored:
        if len(selected) >= top_n:
            break
        cand_words = set(clean_text(cand).split())
        too_similar = any(
            len(cand_words & sw) / max(len(cand_words | sw), 1) > 0.5
            for sw in selected_word_sets
        )
        if not too_similar:
            selected.append(cand[0].upper() + cand[1:] if cand else cand)
            selected_word_sets.append(cand_words)

    fallbacks = ['The opposite is true', 'None of the above', 'All of the above']
    i = 0
    while len(selected) < top_n:
        selected.append(fallbacks[i % len(fallbacks)])
        i += 1

    return selected[:top_n]


# ═══════════════════════════════════════════════════════════════════════════
# PART B — HINT SCORER
# ═══════════════════════════════════════════════════════════════════════════

def _bigram_overlap(sentence: str, query: str) -> float:
    """
    NEW helper: fraction of sentence bigrams that appear in the query.
    Reason: bigrams capture phrase-level relevance that unigrams miss.
    """
    def bigrams(text):
        words = clean_text(text).split()
        return set(zip(words[:-1], words[1:]))

    sent_bg  = bigrams(sentence)
    query_bg = bigrams(query)
    if not sent_bg:
        return 0.0
    return len(sent_bg & query_bg) / len(sent_bg)


def build_hint_dataset(train_c_df: pd.DataFrame,
                       tfidf: TfidfVectorizer,
                       n: int = 6000,            # CHANGE: 4000 → 6000
                       cosine_threshold: float = 0.05) -> pd.DataFrame:
    """
    CHANGES vs original:
      - n: 4000 → 6000  (more training data)
      - Added f7: bigram_overlap_qa  (new feature)
    """
    print(f"\n[3] Building hint dataset (n={n}) ...")

    sample = train_c_df.sample(n=min(n, len(train_c_df)), random_state=42)
    rows = []

    for _, row in sample.iterrows():
        article  = str(row['article_original'])
        question = str(row['question'])
        answer   = str(row['correct_answer'])
        qa_query = question + ' ' + answer

        sents = split_sentences(article)
        if len(sents) < 2:
            continue

        lengths = [len(s.split()) for s in sents]
        med_len = np.median(lengths) if lengths else 10.0

        ans_words = set(clean_text(answer).split()) - STOPWORDS
        q_words   = set(clean_text(question).split()) - STOPWORDS

        for i, sent in enumerate(sents):
            sw = set(clean_text(sent).split())

            ans_ov   = len(sw & ans_words) / max(len(ans_words), 1)
            q_ov     = len(sw & q_words)   / max(len(q_words),   1)
            cos_qa   = tfidf_cosine(sent, qa_query, tfidf)
            pos_n    = i / max(len(sents) - 1, 1)
            len_n    = len(sent.split()) / max(med_len, 1)
            is_early = float(pos_n <= 0.3)
            bigram_ov = _bigram_overlap(sent, qa_query)  # CHANGE: new feature

            label = 1 if cos_qa > cosine_threshold else 0
            rows.append([ans_ov, q_ov, cos_qa, pos_n, len_n, is_early,
                         bigram_ov, label])

    df = pd.DataFrame(rows, columns=['ans_overlap', 'q_overlap', 'tfidf_cos_qa',
                                     'position_norm', 'sent_len_norm',
                                     'is_early', 'bigram_overlap_qa', 'label'])

    print(f"  Raw dataset: {df.shape} | {df['label'].value_counts().to_dict()}")

    pos = df[df['label'] == 1]
    neg = df[df['label'] == 0]
    if len(pos) == 0:
        print("  ⚠️  No positive hint examples — lower cosine_threshold.")
        return df
    if len(pos) < len(neg):
        pos_up = resample(pos, replace=True,
                          n_samples=len(neg), random_state=42)
        df = pd.concat([neg, pos_up]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"  Balanced dataset: {df.shape} | {df['label'].value_counts().to_dict()}")
    return df


HINT_FEAT_COLS = ['ans_overlap', 'q_overlap', 'tfidf_cos_qa',
                  'position_norm', 'sent_len_norm', 'is_early',
                  'bigram_overlap_qa']   # CHANGE: added bigram feature


def train_hint_scorer(hint_df: pd.DataFrame):
    """
    CHANGE: added class_weight='balanced' to LR.
    """
    print("\n[4] Training hint scorer ...")
    X = hint_df[HINT_FEAT_COLS].values
    y = hint_df['label'].values

    X_tr, X_dv, y_tr, y_dv = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    lr_h = LogisticRegression(max_iter=1000, C=1.0,
                               class_weight='balanced',  # CHANGE: added
                               random_state=42)
    lr_h.fit(X_tr, y_tr)
    y_pred = lr_h.predict(X_dv)

    print("\n--- Hint Scorer Results ---")
    print(f"  Acc={accuracy_score(y_dv, y_pred):.4f} "
          f"P={precision_score(y_dv, y_pred, zero_division=0):.4f} "
          f"R={recall_score(y_dv, y_pred, zero_division=0):.4f} "
          f"F1={f1_score(y_dv, y_pred, zero_division=0):.4f}")

    print("\n  LR Coefficients (hint scorer):")
    for fname, coef in sorted(zip(HINT_FEAT_COLS, lr_h.coef_[0]),
                               key=lambda x: abs(x[1]), reverse=True):
        print(f"    {fname:<25s}: {coef:+.4f}")

    joblib.dump(lr_h, MODEL_PATH / 'hint_scorer.pkl')
    print("  hint_scorer.pkl saved.")
    return lr_h


def generate_hints(article: str,
                   question: str,
                   correct_answer: str,
                   tfidf: TfidfVectorizer,
                   scorer) -> list:
    """
    Generate 3 graduated hints pointing toward the correct answer.

    FIXED vs previous:
    - Uses cos_q (question similarity), cos_qa (q+answer), cos_ans (answer)
      separately so each hint slot picks from the right signal.
    - Hint 1: best question match, avoiding direct answer giveaways.
      Two-pass fallback so Hint 1 is NEVER blank.
    - Hint 2: best combined q+answer relevance (not already used).
    - Hint 3: most similar to the correct answer itself.
    - Truncation: 200 chars (was 120).
    """
    DEFAULT = 'Re-read the passage carefully.'
    sents   = split_sentences(article)
    if len(sents) < 2:
        return [f'Hint 1 (General): {DEFAULT}',
                f'Hint 2 (Specific): {DEFAULT}',
                f'Hint 3 (Near Answer): {DEFAULT}']

    ans_words = set(clean_text(correct_answer).split()) - STOPWORDS
    q_words   = set(clean_text(question).split()) - STOPWORDS
    lengths   = [len(s.split()) for s in sents]
    med_len   = np.median(lengths) if lengths else 10.0
    qa_query  = question + ' ' + correct_answer

    vq  = tfidf.transform([clean_text(question)])
    vqa = tfidf.transform([clean_text(qa_query)])
    van = tfidf.transform([clean_text(correct_answer)])

    scored = []
    for i, sent in enumerate(sents):
        sw        = set(clean_text(sent).split())
        ans_ov    = len(sw & ans_words) / max(len(ans_words), 1)
        q_ov      = len(sw & q_words)   / max(len(q_words),   1)
        vs        = tfidf.transform([clean_text(sent)])
        cos_q     = float(cosine_similarity(vs, vq)[0][0])
        cos_qa    = float(cosine_similarity(vs, vqa)[0][0])
        cos_ans   = float(cosine_similarity(vs, van)[0][0])
        pos_n     = i / max(len(sents) - 1, 1)
        len_n     = len(sent.split()) / max(med_len, 1)
        is_early  = float(pos_n <= 0.3)
        bigram_ov = _bigram_overlap(sent, qa_query)
        feats     = [ans_ov, q_ov, cos_qa, pos_n, len_n, is_early, bigram_ov]
        prob      = scorer.predict_proba([feats])[0][1]
        scored.append((sent, prob, i, cos_q, cos_qa, cos_ans, ans_ov, q_ov))

    used_idx = set()

    # Hint 1 — General: strongest question match, not giving away the answer
    h1_sorted = sorted(scored, key=lambda x: x[3] * 0.6 + x[7] * 0.4, reverse=True)
    hint1_sent, hint1_idx = DEFAULT, -1
    for row in h1_sorted:
        sent, prob, idx, cos_q, cos_qa, cos_ans, ans_ov, q_ov = row
        if ans_ov < 0.6:
            hint1_sent, hint1_idx = sent, idx
            used_idx.add(idx)
            break
    # Fallback: threshold rejected everything — just take best question match
    if hint1_sent == DEFAULT and h1_sorted:
        sent, prob, idx, *_ = h1_sorted[0]
        hint1_sent, hint1_idx = sent, idx
        used_idx.add(idx)

    # Hint 2 — Specific: combined q+answer relevance, not used yet
    h2_sorted = sorted(scored, key=lambda x: x[4] * 0.5 + x[3] * 0.3 + x[1] * 0.2, reverse=True)
    hint2_sent, hint2_idx = DEFAULT, -1
    for row in h2_sorted:
        sent, prob, idx, cos_q, cos_qa, cos_ans, ans_ov, q_ov = row
        if idx not in used_idx and ans_ov < 0.8:
            hint2_sent, hint2_idx = sent, idx
            used_idx.add(idx)
            break

    # Hint 3 — Near Answer: most similar to the answer itself
    h3_sorted = sorted(scored, key=lambda x: x[5] * 0.5 + x[6] * 0.5, reverse=True)
    hint3_sent = DEFAULT
    for row in h3_sorted:
        sent, prob, idx, *_ = row
        if idx not in used_idx:
            hint3_sent = sent
            break

    # Final guard: if article is tiny, just use top 3 by cos_qa
    if hint2_sent == DEFAULT:
        all_s = [r[0] for r in sorted(scored, key=lambda x: x[4], reverse=True)]
        if len(all_s) >= 3:
            hint1_sent, hint2_sent, hint3_sent = all_s[0], all_s[1], all_s[2]

    return [
        f"Hint 1 (General): {hint1_sent[:200]}",
        f"Hint 2 (Specific): {hint2_sent[:200]}",
        f"Hint 3 (Near Answer): {hint3_sent[:200]}",
    ]


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    print("Loading data ...")
    train_b_path = PROC_PATH / 'dataset_b_distractor_gen_train.csv'
    train_c_path = PROC_PATH / 'dataset_c_hint_gen_train.csv'
    train_clean  = PROC_PATH / 'train_clean.csv'

    if not train_b_path.exists() or not train_c_path.exists():
        print("⚠️  Dataset B/C CSVs not found. Run preprocessing.py first.")
        return

    train_b_df = pd.read_csv(train_b_path)
    train_c_df = pd.read_csv(train_c_path)
    train_df   = pd.read_csv(train_clean)
    tfidf      = joblib.load(MA_PATH / 'tfidf_vectorizer.pkl')

    print(f"  Dataset B (distractor): {len(train_b_df)} rows")
    print(f"  Dataset C (hint):       {len(train_c_df)} rows")

    dist_df = build_distractor_dataset(train_b_df, tfidf, n_pos=6000)
    rf_dist, lr_dist, y_dv, y_pred_rf = train_distractor_ranker(dist_df)

    hint_df = build_hint_dataset(train_c_df, tfidf, n=6000)
    lr_hint = train_hint_scorer(hint_df)

    print("\n=== GENERATION TEST (first 3 examples from train_clean) ===")
    for i in range(min(3, len(train_df))):
        row     = train_df.iloc[i]
        correct = str(row[row['answer']])
        art     = str(row['article'])
        q       = str(row['question'])

        distractors = generate_distractors(art, correct, tfidf, rf_dist)
        hints       = generate_hints(art, q, correct, tfidf, lr_hint)

        print(f"\n--- Example {i+1} ---")
        print(f"  Question:       {q[:80]}")
        print(f"  Correct Answer: {correct}")
        print(f"  Distractors:    {distractors}")
        for h in hints:
            print(f"  {h[:130]}")

    print("\n✅ Model B training complete!")


if __name__ == '__main__':
    main()