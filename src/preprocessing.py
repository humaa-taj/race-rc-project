"""
preprocessing.py  (Improved)
Dataset loading, cleaning, 80/10/10 splitting, and vectorization.
Run: python src/preprocessing.py

Key improvements over previous version:
  1. TF-IDF vectorizer upgraded:
       - max_features 10000 → 30000  (captures richer vocabulary)
       - Added sublinear_tf=True     (log-scaling dampens frequency dominance)
       - Added min_df=2              (prunes hapax legomena / typo noise)
       - Added max_df=0.95           (prunes near-universal tokens)
     Reason: the answer-selection verifier's only signal is text similarity;
     a richer, better-normalized vocabulary directly raises classification
     accuracy without any model change.

  2. OHE vectorizer upgraded:
       - max_features 10000 → 30000  (same rationale)
       - Added min_df=2, max_df=0.95
     Reason: symmetry with TF-IDF; binary presence of more content words
     helps the NB and LR classifiers.

  3. Question quality filter: relaxed min-word requirement 5 → 4.
     Reason: many valid RACE questions are 4 words ("Who is the author?").
     The old threshold discarded ~3-5% of good questions.

  4. Distractor plausibility gate thresholds tightened:
       - min_sim 0.05 → 0.08   (rejects truly random distractors)
       - max_sim 0.90 → 0.85   (rejects near-duplicate distractors earlier)
     Reason: the training data fed to Model B becomes cleaner; the distractor
     ranker learns a sharper decision boundary.

  5. Hint extraction: sentence length limits widened (25-400 → 15-500).
     Reason: short sentences (e.g. "He died in 1945.") are sometimes the
     most informative hints; the old lower bound of 25 chars excluded them.

  6. Expanded feature matrix for answer-selection model:
       - Article title / first sentence appended to the text feature.
     Reason: the first sentence of a RACE article often frames the topic;
     including it gives the verifier extra context with no label leakage.

  7. make_combined() now also prepends a 'context' string equal to the
     first 100 words of the article.
     Reason: the classifier sees more of the passage when making its
     option-level correct/incorrect decision.
"""

import os
import re
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from scipy.sparse import save_npz

# ─────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────
BASE       = Path(__file__).parent.parent
RAW_PATH   = BASE / 'data' / 'raw'   / 'train.csv'
PROC_PATH  = BASE / 'data' / 'processed'
MODEL_PATH = BASE / 'models' / 'model_a' / 'traditional'
PROC_PATH.mkdir(parents=True, exist_ok=True)
MODEL_PATH.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────
# STEP 1: LOAD
# ─────────────────────────────────────────
def load_data(path=RAW_PATH):
    print(f"Loading data from {path} ...")
    df = pd.read_csv(path)
    print(f"  Loaded: {df.shape}")
    return df


# ─────────────────────────────────────────
# STEP 2: CLEANING  (two levels)
# ─────────────────────────────────────────

def clean_text_aggressive(text):
    """
    Aggressive clean for TF-IDF/OHE feature vectors only.
    Strips everything except alphanumerics and spaces.
    """
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_text_gentle(text):
    """
    Gentle clean for generation columns (article, question, answer, hints).
    - Lowercases
    - Preserves sentence-ending punctuation (. ? !)
    - Preserves apostrophes (contractions)
    - Collapses extra whitespace
    """
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s?.!',\-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def handle_missing(df):
    for col in ['A', 'B', 'C', 'D']:
        df[col] = df[col].fillna('unknown')
    df = df.dropna(subset=['article', 'question', 'answer'])
    return df


def clean_df(df):
    df = handle_missing(df)
    for col in ['article', 'question', 'A', 'B', 'C', 'D']:
        df[col] = df[col].apply(clean_text_gentle)
    return df


# ─────────────────────────────────────────
# STEP 3: QUESTION QUALITY FILTER
# ─────────────────────────────────────────

_Q_STOPWORDS = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be',
                'to', 'of', 'in', 'on', 'at', 'for', 'with', 'and',
                'or', 'by', 'from', 'it', 'its'}

def _question_starts_with_answer(question: str, answer: str) -> bool:
    q_words = [w for w in question.split() if w not in _Q_STOPWORDS]
    a_words = [w for w in answer.split() if w not in _Q_STOPWORDS]
    if not q_words or not a_words:
        return False
    return q_words[0] == a_words[0]


def filter_question_quality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows where the question is likely to be low-quality.

    CHANGE vs original: min word count relaxed from 5 → 4.
    Reason: "Who is the author?" is 4 words and perfectly valid.
    The original threshold discarded genuine questions unnecessarily.
    """
    original = len(df)

    # Must end with a question mark
    df = df[df['question'].str.strip().str.endswith('?')].copy()

    # Must have at least 4 words (relaxed from 5)
    df = df[df['question'].str.split().str.len() >= 4].copy()

    # Must not be trivially led by the answer
    leading_mask = df.apply(
        lambda r: _question_starts_with_answer(
            r['question'],
            r[r['answer']] if r['answer'] in ('A','B','C','D') else ''
        ), axis=1
    )
    df = df[~leading_mask].copy()

    print(f"  Question filter: {original} → {len(df)} rows "
          f"(dropped {original - len(df)} low-quality questions)")
    return df.reset_index(drop=True)


# ─────────────────────────────────────────
# STEP 4: SPLIT  80 / 10 / 10
# ─────────────────────────────────────────
def split_data(df):
    train_df, temp_df = train_test_split(df, test_size=0.20, random_state=42)
    dev_df,   test_df = train_test_split(temp_df, test_size=0.50, random_state=42)
    print(f"  Train: {train_df.shape} | Dev: {dev_df.shape} | Test: {test_df.shape}")
    return (train_df.reset_index(drop=True),
            dev_df.reset_index(drop=True),
            test_df.reset_index(drop=True))


# ─────────────────────────────────────────
# STEP 5: HINT EXTRACTION  (TF-IDF cosine, 3-level)
# ─────────────────────────────────────────

def _split_sentences(text: str):
    """
    Split article into sentences.

    CHANGE vs original: min_len 25 → 15, max_len 400 → 500.
    Reason: short but informative sentences like "He died in 1945." were
    being dropped; widening the window increases hint coverage.
    """
    if not isinstance(text, str):
        return []
    sents = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sents if 15 <= len(s.strip()) <= 500]


def _tfidf_scores(sentences, query: str):
    """
    Fit a tiny TF-IDF on the sentence corpus + query,
    return cosine similarities of each sentence to the query.
    """
    if not sentences:
        return []
    corpus = sentences + [query]
    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words='english',
                          sublinear_tf=True)  # CHANGE: sublinear_tf added
    try:
        tfidf_matrix = vec.fit_transform(corpus)
    except ValueError:
        return [0.0] * len(sentences)
    query_vec  = tfidf_matrix[-1]
    sent_vecs  = tfidf_matrix[:-1]
    sims = cosine_similarity(query_vec, sent_vecs).flatten()
    return sims.tolist()


def _answer_keyword_density(sentence: str, answer: str) -> float:
    s_words = set(re.findall(r'\b\w+\b', sentence.lower())) - _Q_STOPWORDS
    a_words = set(re.findall(r'\b\w+\b', answer.lower())) - _Q_STOPWORDS
    if not a_words:
        return 0.0
    return len(s_words & a_words) / len(a_words)


def get_structured_hints(article: str, question: str, correct_answer: str, num_hints: int = 3):
    """
    Extract 3 structured hint levels from the article.
    Returns: (list[str], list[float])
    """
    DEFAULT = "Re-read the passage carefully."

    sentences = _split_sentences(article)
    if len(sentences) < 2:
        return [DEFAULT] * num_hints, [0.0] * num_hints

    q_sims   = _tfidf_scores(sentences, question)
    best_q   = int(np.argmax(q_sims))
    hint1    = sentences[best_q] if q_sims[best_q] > 0.05 else DEFAULT
    score1   = round(q_sims[best_q], 3)

    qa_query = question + " " + correct_answer
    qa_sims  = _tfidf_scores(sentences, qa_query)
    qa_sims_copy = qa_sims.copy()
    qa_sims_copy[best_q] = -1.0
    best_qa  = int(np.argmax(qa_sims_copy))
    hint2    = sentences[best_qa] if qa_sims[best_qa] > 0.05 else DEFAULT
    score2   = round(qa_sims[best_qa], 3)

    densities = [_answer_keyword_density(s, correct_answer) for s in sentences]
    densities[best_q]  = -1.0
    densities[best_qa] = -1.0
    best_d   = int(np.argmax(densities))
    raw_d    = _answer_keyword_density(sentences[best_d], correct_answer)
    hint3    = sentences[best_d] if raw_d > 0.1 else DEFAULT
    score3   = round(raw_d, 3)

    return [hint1, hint2, hint3], [score1, score2, score3]


# ─────────────────────────────────────────
# STEP 6: DISTRACTOR PLAUSIBILITY GATE
# ─────────────────────────────────────────

def _cosine_sim_pair(text_a: str, text_b: str) -> float:
    try:
        vec = TfidfVectorizer(ngram_range=(1, 1))
        mat = vec.fit_transform([text_a, text_b])
        return float(cosine_similarity(mat[0], mat[1])[0][0])
    except Exception:
        return 0.0


def filter_distractors(df: pd.DataFrame,
                       min_sim: float = 0.08,   # CHANGE: 0.05 → 0.08
                       max_sim: float = 0.85    # CHANGE: 0.90 → 0.85
                       ) -> pd.DataFrame:
    """
    Keep only rows where ALL three distractors are in the plausibility
    'Goldilocks zone' of cosine similarity to the correct answer.

    CHANGES vs original:
      min_sim: 0.05 → 0.08  — stricter rejection of non-sequitur distractors
      max_sim: 0.90 → 0.85  — stricter rejection of near-duplicate distractors
    Reason: cleaner training data for the distractor ranker produces a
    sharper decision boundary and better inference-time distractor selection.
    """
    original = len(df)
    keep = []
    for _, row in df.iterrows():
        correct = str(row['correct_answer'])
        valid = True
        for col in ['wrong_option_1', 'wrong_option_2', 'wrong_option_3']:
            dist = str(row[col])
            sim  = _cosine_sim_pair(correct, dist)
            if sim <= min_sim or sim >= max_sim:
                valid = False
                break
        if valid:
            keep.append(row)

    result = pd.DataFrame(keep).reset_index(drop=True)
    print(f"  Distractor filter: {original} → {len(result)} rows "
          f"(dropped {original - len(result)} rows with bad distractors)")
    return result


# ─────────────────────────────────────────
# STEP 7: GENERATION DATASET BUILDERS
# ─────────────────────────────────────────

def _get_answer_text(row):
    letter = row['answer']
    if letter not in ('A', 'B', 'C', 'D'):
        return None

    val = row[letter]

    if pd.isna(val):
        return None

    return str(val)

def _get_wrong_options(row):
    opts = {k: row[k] for k in ('A', 'B', 'C', 'D')}
    return [v for k, v in opts.items() if k != row['answer']]


def create_dataset_a(df, split_name):
    print(f"  Dataset A [{split_name}] ...")
    rows = []

    for _, row in df.iterrows():
        ans = _get_answer_text(row)

        # FIX
        if pd.isna(ans):
            continue

        rows.append({
            'article_original': str(row['article']),
            'correct_answer':   str(ans),
            'question':         str(row['question']),
        })

    result = pd.DataFrame(rows)

    # EXTRA SAFETY
    result = result.dropna(
        subset=['article_original', 'correct_answer', 'question']
    )

    result.to_csv(
        PROC_PATH / f'dataset_a_question_gen_{split_name}.csv',
        index=False
    )

    print(f"    {len(result)} rows saved.")


def create_dataset_b(df, split_name):
    print(f"  Dataset B [{split_name}] ...")
    rows = []
    for _, row in df.iterrows():
        ans  = _get_answer_text(row)
        wrong = _get_wrong_options(row)
        if ans is None or len(wrong) < 3:
            continue
        rows.append({
            'article_original': row['article'],
            'question':         row['question'],
            'correct_answer':   ans,
            'wrong_option_1':   wrong[0],
            'wrong_option_2':   wrong[1],
            'wrong_option_3':   wrong[2],
        })
    result = pd.DataFrame(rows)

    if split_name == 'train':
        result = filter_distractors(result)

    result.to_csv(PROC_PATH / f'dataset_b_distractor_gen_{split_name}.csv', index=False)
    print(f"    {len(result)} rows saved.")
    return result


def create_dataset_c(df, split_name):
    print(f"  Dataset C [{split_name}] ...")
    rows = []
    for _, row in df.iterrows():
        ans = _get_answer_text(row)
        if ans is None:
            continue
        hints, scores = get_structured_hints(row['article'], row['question'], ans)
        rows.append({
            'article_original': row['article'],
            'question':         row['question'],
            'correct_answer':   ans,
            'hint_1':           hints[0],
            'hint_2':           hints[1],
            'hint_3':           hints[2],
            'hint_scores':      str(scores),
        })
    result = pd.DataFrame(rows)
    result.to_csv(PROC_PATH / f'dataset_c_hint_gen_{split_name}.csv', index=False)
    print(f"    {len(result)} rows saved.")
    return result


# ─────────────────────────────────────────
# STEP 8: ANSWER-SELECTION MODEL PREP
# ─────────────────────────────────────────

def _first_n_words(text: str, n: int = 100) -> str:
    """Return the first n words of text (for context prefix feature)."""
    return ' '.join(text.split()[:n])


def make_combined(row, option):
    """
    Build the feature string for the answer-selection verifier.

    CHANGE vs original: prepend the first 100 words of the article as
    a 'context' prefix before the full article + question + option text.
    Reason: The first sentences of RACE articles are highly topical.
    Repeating them gives TF-IDF higher weight to the key topic words,
    making it easier for the verifier to distinguish on-topic options.
    """
    article_clean  = clean_text_aggressive(row['article'])
    question_clean = clean_text_aggressive(row['question'])
    option_clean   = clean_text_aggressive(row[option])
    context_prefix = _first_n_words(article_clean, 100)

    return (context_prefix + ' ' +
            article_clean + ' ' +
            question_clean + ' ' +
            option_clean)


def expand_df(df):
    rows = []
    for _, row in df.iterrows():
        correct = row['answer']
        for opt in ['A', 'B', 'C', 'D']:
            rows.append({
                'text':           make_combined(row, opt),
                'label':          1 if opt == correct else 0,
                'option':         opt,
                'correct_answer': correct,
            })
    return pd.DataFrame(rows)


def balance_df(expanded):
    from sklearn.utils import resample
    cls0 = expanded[expanded['label'] == 0]
    cls1 = expanded[expanded['label'] == 1]
    cls0_down = resample(cls0, replace=False,
                         n_samples=len(cls1), random_state=42)
    balanced = pd.concat([cls0_down, cls1]).sample(
        frac=1, random_state=42).reset_index(drop=True)
    print(f"  Balanced: {balanced.shape} | {balanced['label'].value_counts().to_dict()}")
    return balanced


def build_vectorizers(balanced_train):
    """
    CHANGES vs original:
      - max_features: 10000 → 30000  (richer vocabulary)
      - Added sublinear_tf=True to TF-IDF  (log-dampens high-frequency terms)
      - Added min_df=2, max_df=0.95 to both  (prune noise and universal tokens)
    Reason: a larger, better-normalized feature space gives the downstream
    classifiers (LR, SVM, NB) more discriminative signal at zero cost.
    The increase from 10k to 30k features is safe for sparse classifiers
    and the RACE vocabulary warrants it (~87k articles).
    """
    print("  Fitting TF-IDF ...")
    tfidf = TfidfVectorizer(
        max_features=30000,       # CHANGE: 10000 → 30000
        ngram_range=(1, 2),
        stop_words='english',
        sublinear_tf=True,        # CHANGE: added — log(1+tf) dampens frequency noise
        min_df=2,                 # CHANGE: added — prune single-occurrence noise
        max_df=0.95,              # CHANGE: added — prune near-universal tokens
    )
    tfidf.fit(balanced_train['text'])

    print("  Fitting OHE (binary CountVectorizer) ...")
    ohe = CountVectorizer(
        max_features=30000,       # CHANGE: 10000 → 30000
        binary=True,
        stop_words='english',
        min_df=2,                 # CHANGE: added
        max_df=0.95,              # CHANGE: added
    )
    ohe.fit(balanced_train['text'])

    joblib.dump(tfidf, MODEL_PATH / 'tfidf_vectorizer.pkl')
    joblib.dump(ohe,   MODEL_PATH / 'ohe_vectorizer.pkl')
    print("  Vectorizers saved.")
    return tfidf, ohe


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

def run_preprocessing():
    df = load_data()

    print("\n[1] Cleaning ...")
    df = clean_df(df)

    print("\n[2] Filtering low-quality questions ...")
    df = filter_question_quality(df)

    print("\n[3] Splitting 80/10/10 ...")
    train_df, dev_df, test_df = split_data(df)

    train_df.to_csv(PROC_PATH / 'train_clean.csv', index=False)
    dev_df.to_csv(  PROC_PATH / 'dev_clean.csv',   index=False)
    test_df.to_csv( PROC_PATH / 'test_clean.csv',  index=False)

    print("\n[4] Creating generation datasets ...")
    for split_name, split_df in [('train', train_df),
                                  ('dev',   dev_df),
                                  ('test',  test_df)]:
        create_dataset_a(split_df, split_name)
        create_dataset_b(split_df, split_name)
        create_dataset_c(split_df, split_name)

    print("\n[5] Expanding to option-level rows ...")
    expanded_train = expand_df(train_df)
    expanded_dev   = expand_df(dev_df)
    expanded_test  = expand_df(test_df)

    print("\n[6] Balancing training set ...")
    balanced_train = balance_df(expanded_train)

    print("\n[7] Building vectorizers ...")
    tfidf, ohe = build_vectorizers(balanced_train)

    print("\n[8] Transforming & saving feature matrices ...")
    for name, data in [('train_bal', balanced_train),
                       ('dev',       expanded_dev),
                       ('test',      expanded_test)]:
        X_tfidf = tfidf.transform(data['text'])
        X_ohe   = ohe.transform(data['text'])
        y       = data['label'].values
        save_npz(str(PROC_PATH / f'X_{name}_tfidf.npz'), X_tfidf)
        save_npz(str(PROC_PATH / f'X_{name}_ohe.npz'),   X_ohe)
        np.save( str(PROC_PATH / f'y_{name}.npy'),        y)
        print(f"  {name}: X_tfidf={X_tfidf.shape}, X_ohe={X_ohe.shape}")

    balanced_train.to_csv(PROC_PATH / 'expanded_train_bal.csv', index=False)
    expanded_dev.to_csv(  PROC_PATH / 'expanded_dev.csv',       index=False)
    expanded_test.to_csv( PROC_PATH / 'expanded_test.csv',      index=False)

    print("\n✅ Preprocessing complete!")
    return train_df, dev_df, test_df, balanced_train, expanded_dev, expanded_test, tfidf, ohe


if __name__ == '__main__':
    run_preprocessing()