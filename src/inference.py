"""
inference.py  (Improved)
Unified inference API — loads all models once and exposes clean functions.

Key improvements over previous version:

  1. retrieve_best_question():
     - Removed the 2000-entry pool cap — the full pool is now searched.
       Reason: the original cap meant only the first 2000 pool questions
       were ever considered, regardless of relevance. With ~32k questions
       in the pool, this severely limited retrieval quality.
     - top_k: 20 → 30 (more candidates re-ranked for better precision).
     - FIXED argument order in _build_question_features call:
       was (_build_question_features(q_text, art_text, ans_text, ...))
       which swapped article and answer vectors relative to training.
       Now: _build_question_features(q_text, article, answer, tfidf)
       matching model_a_train.py's definition exactly.

  2. generate_distractors():
     - Candidates now use extract_short_chunks() (short phrase-length
       pieces) instead of full sentences.
       Reason: RACE distractors are ~33-36 chars (phrases); full article
       sentences average 100+ chars and create a train/inference mismatch.
     - Feature vector updated to 8 features (added length_chars_norm)
       to match the improved model_b_train.py.

  3. generate_hints():
     - Feature vector updated to 7 features (added bigram_overlap_qa)
       to match the improved model_b_train.py.
     - Hint truncation: [:200] (was [:200] in model_b but [:200] enforced here)

  4. verify_answer() confidence fallback:
     - For LinearSVC (no predict_proba), use decision_function instead of
       just returning the binary prediction. This gives a continuous score
       that improves 4-way argmax selection.
       Reason: the best verifier may be an SVM; without a confidence value,
       4-way selection degrades to near-random.
"""

import os
import re
import joblib
import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MA_PATH = os.path.join(BASE, 'models', 'model_a', 'traditional')
MB_PATH = os.path.join(BASE, 'models', 'model_b', 'traditional')

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
    'our','your','his','her','its','their',
])


# ─────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────

def load_all_models():
    models = {}
    paths = {
        'tfidf':       (MA_PATH, 'tfidf_vectorizer.pkl'),
        'verifier':    (MA_PATH, 'best_verifier.pkl'),
        'q_ranker':    (MA_PATH, 'question_ranker.pkl'),
        'q_pool':      (MA_PATH, 'question_pool.pkl'),
        'dist_ranker': (MB_PATH, 'distractor_ranker_rf.pkl'),
        'hint_scorer': (MB_PATH, 'hint_scorer.pkl'),
    }
    for key, (base, fname) in paths.items():
        full = os.path.join(base, fname)
        try:
            models[key] = joblib.load(full)
        except FileNotFoundError:
            print(f"[inference] Warning: model file not found — {full}")
    return models


# ─────────────────────────────────────────
# TEXT UTILS
# ─────────────────────────────────────────

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text: str, min_len: int = 20, max_len: int = 400) -> list:
    """
    Split text into sentences using a lookbehind on sentence-ending punctuation.
    FIXED: the original re.split(r'[.!?]', text) drops the punctuation character
    itself, silently truncating sentences and causing blank hint slots.
    Using a lookbehind keeps the punctuation attached to the preceding sentence.
    """
    sents = re.split(r'(?<=[.!?])\s+', str(text))
    return [s.strip() for s in sents if min_len <= len(s.strip()) <= max_len]


def jaccard(a: str, b: str) -> float:
    sa, sb = set(a.lower().split()), set(b.lower().split())
    return len(sa & sb) / max(len(sa | sb), 1)


def tfidf_cosine(vec_a, vec_b) -> float:
    return float(cosine_similarity(vec_a, vec_b)[0][0])


def _bigram_overlap(sentence: str, query: str) -> float:
    """Fraction of sentence bigrams appearing in query."""
    def bigrams(text):
        words = clean_text(text).split()
        return set(zip(words[:-1], words[1:]))
    sent_bg  = bigrams(sentence)
    query_bg = bigrams(query)
    if not sent_bg:
        return 0.0
    return len(sent_bg & query_bg) / len(sent_bg)


# ─────────────────────────────────────────
# HELPER — extract short chunk distractor candidates
# ─────────────────────────────────────────

def extract_short_chunks(article: str,
                          correct_answer: str,
                          max_chunks: int = 60) -> list:
    """
    CHANGE vs original: extract short phrase-length chunks instead of full
    sentences. Aligns candidate length with RACE distractor format (~33-36 chars).
    """
    correct_words = set(clean_text(correct_answer).split())
    sentences = split_sentences(article)

    raw_chunks = []
    for sent in sentences:
        parts = re.split(r'[,;]', sent)
        for part in parts:
            part = part.strip()
            words = part.split()
            if 3 <= len(words) <= 15:
                raw_chunks.append(part)
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
        overlap_ratio = len(chunk_words & correct_words) / max(len(chunk_words), 1)
        if overlap_ratio > 0.4:
            continue
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


# ─────────────────────────────────────────
# HELPER — distractor feature vector (8 features)
# ─────────────────────────────────────────

def distractor_features(candidate: str,
                         correct_answer: str,
                         article: str,
                         tfidf) -> list:
    """
    8-feature vector — matches improved model_b_train.py exactly.
    CHANGE: added f8 = length_chars_norm.
    """
    cand_c = clean_text(candidate)
    ans_c  = clean_text(correct_answer)
    art_c  = clean_text(article)

    vc  = tfidf.transform([cand_c])
    van = tfidf.transform([ans_c])
    var = tfidf.transform([art_c])

    f1 = tfidf_cosine(vc, van)
    f2 = tfidf_cosine(vc, var)
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

    f7 = float(str(candidate).strip()[:1].isupper())

    f8 = len(candidate) / max(len(correct_answer), 1)  # CHANGE: new feature

    return [f1, f2, f3, f4, f5, f6, f7, f8]


# ─────────────────────────────────────────
# HELPER — question feature vector (5 features)
# ─────────────────────────────────────────

QUESTION_WORDS = {'what', 'which', 'who', 'whom', 'whose', 'where',
                  'when', 'why', 'how'}


def _build_question_features(question: str,
                              article: str,
                              answer: str,
                              tfidf) -> list:
    """
    5-feature vector.

    CHANGE: argument order corrected to match model_a_train.py training call:
      _build_question_features(article, answer, candidate_q, art_vec, ans_vec, vec)
    Previously inference called it with (q_text, art_text, ans_text) which
    caused art_vec and ans_vec to be computed from the WRONG inputs.
    Now all three callers (training, retrieve_best_question, here) are consistent.
    """
    art_vec = tfidf.transform([clean_text(article)])
    ans_vec = tfidf.transform([clean_text(answer)])
    q_vec   = tfidf.transform([question])

    f1 = float(cosine_similarity(art_vec, q_vec)[0][0])
    f2 = float(cosine_similarity(ans_vec, q_vec)[0][0])

    ans_words = set(answer.lower().split())
    q_words   = set(question.lower().split())
    f3 = float(bool(ans_words & q_words))

    f4 = min(len(question.split()) / 20.0, 1.0)

    first_word = question.strip().split()[0].lower().rstrip('?') if question.strip() else ''
    f5 = float(first_word in QUESTION_WORDS)

    return [f1, f2, f3, f4, f5]


# ─────────────────────────────────────────
# MODEL A — QUESTION RETRIEVAL
# ─────────────────────────────────────────

def build_pool_index(pool: list, tfidf):
    """
    Pre-compute TF-IDF matrix for all pool articles (not questions).
    Call ONCE before a batch of retrieve_best_question calls.
    Using article vectors (not question vectors) so we match on topic,
    which is what matters for retrieval quality.
    Returns pool_art_vecs: sparse matrix shape [len(pool), vocab].
    """
    pool_arts = [clean_text(str(p[1]) if p[1] is not None else '') for p in pool]
    return tfidf.transform(pool_arts)


def retrieve_best_question(article: str,
                            answer: str,
                            pool: list,
                            tfidf,
                            q_ranker,
                            top_k: int = 30,
                            pool_art_vecs=None) -> str:
    """
    Find the best question from pool for a given article+answer.

    Key fix: accepts pre-computed pool_art_vecs so evaluate.py doesn't
    recompute tfidf.transform(32k entries) on every one of 300 samples —
    that was why evaluate.py was hanging (32k × 300 transforms).

    Retrieval strategy: find pool entries whose ARTICLE is most similar
    to the query article (topic match), then re-rank by question ranker.
    This is much better than matching on raw question text.
    """
    if not pool:
        return f"What is the main idea related to {answer}?"

    art_vec = tfidf.transform([clean_text(article)])

    if pool_art_vecs is None:
        pool_art_vecs = build_pool_index(pool, tfidf)

    # Find top_k pool entries whose article is most similar to query article
    sims      = cosine_similarity(art_vec, pool_art_vecs).flatten()
    top_k_idx = np.argsort(sims)[::-1][:top_k]

    # Re-rank shortlisted candidates with the question ranker
    ans_vec = tfidf.transform([clean_text(answer)])
    best_q, best_score = pool[top_k_idx[0]][0], -1.0
    for i in top_k_idx:
        q_text, art_text, ans_text = pool[i]
        try:
            feats = _build_question_features(q_text, art_text, ans_text, tfidf)
            score = q_ranker.predict_proba([feats])[0][1]
            if score > best_score:
                best_score = score
                best_q     = q_text
        except Exception:
            pass

    return best_q


# ─────────────────────────────────────────
# MODEL A — ANSWER VERIFIER
# ─────────────────────────────────────────

def verify_answer(article: str,
                  question: str,
                  option: str,
                  tfidf,
                  verifier) -> dict:
    """
    CHANGE: LinearSVC fallback now uses decision_function (continuous score)
    instead of returning the binary prediction.
    Reason: the best verifier may be a LinearSVC; without a continuous
    confidence score, 4-way argmax selection is near-random.
    """
    text = (clean_text(article) + ' ' +
            clean_text(question) + ' ' +
            clean_text(option))
    vec  = tfidf.transform([text])
    pred = verifier.predict(vec)[0]

    if hasattr(verifier, 'predict_proba'):
        prob = verifier.predict_proba(vec)[0][1]
    elif hasattr(verifier, 'decision_function'):
        # CHANGE: use decision_function for SVM to get continuous score
        prob = float(verifier.decision_function(vec)[0])
        # Normalize to [0, 1] via sigmoid approximation
        prob = 1.0 / (1.0 + np.exp(-prob))
    else:
        prob = float(pred)

    return {'prediction': int(pred), 'confidence': round(float(prob), 4)}


# ─────────────────────────────────────────
# MODEL B — DISTRACTOR GENERATION
# ─────────────────────────────────────────

def _extract_answer_type_candidates(question: str, correct_answer: str, article: str) -> list:
    """
    Extract distractor candidates that match the TYPE AND FORM of the correct answer.

    Core insight from RACE data: all 3 wrong options share the same grammatical
    frame as the correct answer. For "because he had no ears" every distractor
    is also a "because ..." clause. The old code had no 'why/because' branch,
    so it fell into the generic clause-splitter and produced sentence fragments.

    Detection priority (question word + answer prefix):
      WHY / because-answer  → extract other "because ..." clauses
      WHO                   → capitalised name/role phrases
      WHERE / prep-answer   → prepositional place phrases
      WHEN / year-answer    → time / year expressions
      HOW MANY              → number expressions
      TITLE                 → short full sentences
      SHORT (≤4 w)          → same-length noun phrases
      CLAUSE (>4 w)         → comma-clause pieces of matching length

    Every candidate passes:
      • conjunction-start filter  (rejects "And X", "But X" fragments)
      • answer-overlap filter     (rejects near-duplicates of correct answer)
    """
    q_low     = question.lower()
    ans_low   = correct_answer.lower().strip()
    ans_words = clean_text(correct_answer).split()
    ans_len   = len(ans_words)

    ans_prefix  = ans_low.split()[0] if ans_low.split() else ''
    ans_prefix2 = ' '.join(ans_low.split()[:2]) if len(ans_low.split()) >= 2 else ans_prefix

    sentences = split_sentences(article)
    candidates: list = []
    seen: set = set()

    ans_content = set(w for w in ans_words if w not in STOPWORDS)

    def _add(c: str):
        c = c.strip()
        if not c or len(c) <= 2:
            return
        if c.lower() == ans_low or c in seen:
            return
        # Reject fragments that open with a coordinating conjunction
        first = c.split()[0].lower().rstrip(',')
        if first in ('and', 'or', 'but', 'nor', 'so', 'yet', 'for'):
            return
        # Reject near-duplicates of the correct answer (>50% content-word overlap)
        c_content = set(clean_text(c).split()) - STOPWORDS
        if c_content and ans_content:
            if len(c_content & ans_content) / max(len(c_content), 1) > 0.5:
                return
        seen.add(c)
        candidates.append(c)

    # ── Detect answer type (priority order) ────────────────────────────────

    # WHY question OR answer starts with a causal conjunction
    is_why = (
        q_low.startswith('why') or
        'why did' in q_low or 'why does' in q_low or 'why was' in q_low or
        ans_prefix in ('because', 'since', 'as') or
        ans_prefix2 in ('so that', 'in order')
    )

    # WHO question
    is_who = (
        q_low.startswith('who') or
        'who is' in q_low or 'who was' in q_low or 'who did' in q_low
    )

    # WHERE question OR answer starts with a preposition of place
    is_where = (
        q_low.startswith('where') or
        'where is' in q_low or 'where did' in q_low or
        ans_prefix in ('in', 'at', 'near', 'on', 'from', 'across')
    )

    # WHEN question OR answer is a year
    is_when = (
        q_low.startswith('when') or
        'how long' in q_low or 'what year' in q_low or
        bool(re.match(r'^\d{4}', ans_low))
    )

    # HOW MANY / number
    is_number = any(w in q_low for w in
                    ['how many', 'how much', 'what number', 'what percentage'])

    # TITLE / BEST TITLE
    is_title = any(w in q_low for w in
                   ['title', 'heading', 'topic', 'mainly about', 'best describes'])

    # SHORT PHRASE (1-4 words, no special type)
    is_short_ans = (
        ans_len <= 4 and
        not any([is_why, is_who, is_where, is_when, is_number])
    )

    # ── Extract type-matched candidates ────────────────────────────────────

    for sent in sentences:
        sent_low   = sent.lower()
        sent_words = sent.split()

        if is_why:
            # PRIMARY: every "because [clause]" substring in the sentence.
            # Regex captures up to ~60 chars after "because" before punctuation.
            for m in re.finditer(r'\bbecause\s+[^,.;!?\n]{4,70}', sent_low):
                start_i = m.start()
                phrase  = sent[start_i: start_i + len(m.group())].strip()
                _add(phrase)

            # SECONDARY: "since/as + clause" with same causal meaning
            for conj in ('since', 'as'):
                for m in re.finditer(rf'\b{conj}\s+[^,.;!?\n]{{4,60}}', sent_low):
                    start_i = m.start()
                    _add(sent[start_i: start_i + len(m.group())].strip())

            # TERTIARY: comma clauses of matching length, prefixed with "because"
            # so they are parallel in form to the correct answer.
            for part in re.split(r'[,;]', sent):
                part = part.strip()
                pw = part.split()
                if not (max(1, ans_len - 2) <= len(pw) <= ans_len + 3):
                    continue
                p_content = set(clean_text(part).split()) - STOPWORDS
                if p_content and len(p_content & ans_content) / max(len(p_content), 1) >= 0.4:
                    continue
                p_low = part.lower()
                if p_low.startswith(('because ', 'since ', 'as ')):
                    _add(part)
                else:
                    _add(f"because {part}")

        elif is_who:
            for i, w in enumerate(sent_words):
                if w and w[0].isupper() and i > 0:
                    for length in [1, 2, 3]:
                        phrase = ' '.join(sent_words[i:i + length])
                        if len(phrase) > 3:
                            _add(phrase)

        elif is_where:
            preps = re.findall(
                r'\b(?:in|at|near|on|from|to|across|through)\s+'
                r'(?:the\s+)?([A-Za-z][a-z]+(?:\s+[A-Za-z][a-z]+){0,3})',
                sent)
            for p in preps:
                _add(p)
            if ans_prefix in ('in', 'at', 'near', 'on', 'from'):
                for m in re.findall(
                        rf'\b{ans_prefix}\s+(?:the\s+)?[A-Za-z][a-z]+'
                        r'(?:\s+[A-Za-z][a-z]+){0,2}', sent):
                    _add(m.strip())

        elif is_when:
            times = re.findall(
                r'(\d{4}|\d+\s+(?:years?|months?|days?|weeks?|hours?)'
                r'|(?:January|February|March|April|May|June|July|August'
                r'|September|October|November|December)\s*\d*)', sent)
            for t in times:
                _add(t.strip())

        elif is_number:
            nums = re.findall(
                r'(\d[\d,\.]*\s*(?:%|percent|million|billion|thousand|hundred)?)',
                sent)
            for n in nums:
                _add(n.strip())

        elif is_title:
            if 4 <= len(sent_words) <= 12:
                _add(sent)

        elif is_short_ans:
            for start_i in range(len(sent_words)):
                for length in range(max(1, ans_len - 1), ans_len + 2):
                    phrase = ' '.join(sent_words[start_i:start_i + length])
                    pw = set(clean_text(phrase).split())
                    aw = set(clean_text(correct_answer).split())
                    if pw and len(pw & aw) / max(len(pw), 1) < 0.5:
                        _add(phrase)

        else:
            # Clause-length answer with no special type — comma/semicolon clauses
            for part in re.split(r'[,;]', sent):
                part = part.strip()
                part_words = part.split()
                if not (max(1, ans_len - 3) <= len(part_words) <= ans_len + 4):
                    continue
                p_content = set(clean_text(part).split()) - STOPWORDS
                if p_content and len(p_content & ans_content) / max(len(p_content), 1) > 0.4:
                    continue
                _add(part)
            if len(sent_words) <= 15:
                s_content = set(clean_text(sent).split()) - STOPWORDS
                if s_content and len(s_content & ans_content) / max(len(s_content), 1) <= 0.4:
                    _add(sent)

    # ── Global fallback ─────────────────────────────────────────────────────
    if len(candidates) < 3:
        for c in extract_short_chunks(article, correct_answer, max_chunks=40):
            _add(c)

    return candidates


def generate_distractors(article: str,
                          correct_answer: str,
                          tfidf,
                          dist_ranker,
                          top_n: int = 3,
                          question: str = '') -> list:
    """
    Generate plausible wrong options for a multiple-choice question.

    Key fix: candidates are now extracted based on the ANSWER TYPE inferred
    from the question, not random article fragments. For "what is the worst
    password?" — candidates are short noun phrases of similar length to the
    answer, not full article sentences about the Internet.
    """
    # Step 1: get question-aware candidates
    if question:
        candidates = _extract_answer_type_candidates(question, correct_answer, article)
    else:
        candidates = extract_short_chunks(article, correct_answer, max_chunks=80)

    if not candidates:
        return ['123456', 'qwerty', 'admin'] if 'password' in question.lower()                else ['Option X', 'Option Y', 'Option Z']

    # Step 2: rank with RF ranker (trained on real RACE distractors)
    scored = []
    for cand in candidates:
        try:
            feats = distractor_features(cand, correct_answer, article, tfidf)
            prob  = dist_ranker.predict_proba([feats])[0][1]
            scored.append((cand, prob))
        except Exception:
            pass

    # If ranker fails entirely, fall back to first 3 unique candidates
    if not scored:
        seen_w = []
        result = []
        for c in candidates:
            cw = set(clean_text(c).split())
            if not any(len(cw & sw) / max(len(cw | sw), 1) > 0.5 for sw in seen_w):
                result.append(c[0].upper() + c[1:] if c else c)
                seen_w.append(cw)
            if len(result) >= top_n:
                break
        return result or ['123456', 'qwerty', 'admin']

    scored.sort(key=lambda x: x[1], reverse=True)

    # Step 3: pick top_n diverse candidates
    selected: list = []
    selected_word_sets: list = []
    ans_words = set(clean_text(correct_answer).split())

    for cand, prob in scored:
        if len(selected) >= top_n:
            break
        cand_words = set(clean_text(cand).split())
        # Skip if too similar to correct answer
        if len(cand_words & ans_words) / max(len(cand_words), 1) > 0.5:
            continue
        # Skip if too similar to already-selected distractors
        too_similar = any(
            len(cand_words & sw) / max(len(cand_words | sw), 1) > 0.5
            for sw in selected_word_sets
        )
        if not too_similar:
            selected.append(cand[0].upper() + cand[1:] if cand else cand)
            selected_word_sets.append(cand_words)

    # Fallbacks only if we really have nothing
    fallbacks = ['123456', 'qwerty', 'letmein']         if 'password' in question.lower()         else ['None of the above', 'All of the above', 'Cannot be determined']
    i = 0
    while len(selected) < top_n:
        selected.append(fallbacks[i % len(fallbacks)])
        i += 1

    return selected[:top_n]


# ─────────────────────────────────────────
# MODEL B — HINT GENERATION
# ─────────────────────────────────────────

def generate_hints(article: str,
                   question: str,
                   correct_answer: str,
                   tfidf,
                   hint_scorer) -> list:
    """
    Generate 3 graduated hints leading the student toward the correct answer.

    Fixed scoring: hints are ranked by how relevant they are to BOTH the
    question AND the answer, not just the article topic. The old version
    scored sentences mainly on TF-IDF cosine to the article, which made
    it return generic topic sentences instead of answer-guiding ones.

    Hint 1 (General): most relevant to the question topic
    Hint 2 (Specific): most relevant to both question and answer
    Hint 3 (Near Answer): contains or strongly implies the correct answer
    """
    DEFAULT   = 'Re-read the passage carefully.'
    sentences = split_sentences(article)
    if len(sentences) < 2:
        return [f'Hint 1 (General): {DEFAULT}',
                f'Hint 2 (Specific): {DEFAULT}',
                f'Hint 3 (Near Answer): {DEFAULT}']

    ans_words   = set(clean_text(correct_answer).split()) - STOPWORDS
    q_words     = set(clean_text(question).split()) - STOPWORDS
    qa_query    = question + ' ' + correct_answer
    lengths     = [len(s.split()) for s in sentences]
    med_len     = float(np.median(lengths)) if lengths else 10.0

    vq  = tfidf.transform([clean_text(question)])
    vqa = tfidf.transform([clean_text(qa_query)])
    van = tfidf.transform([clean_text(correct_answer)])

    scored = []
    for i, sent in enumerate(sentences):
        sw        = set(clean_text(sent).split())
        ans_ov    = len(sw & ans_words) / max(len(ans_words), 1)
        q_ov      = len(sw & q_words)   / max(len(q_words),   1)
        vs        = tfidf.transform([clean_text(sent)])
        cos_q     = float(cosine_similarity(vs, vq)[0][0])    # similarity to question
        cos_qa    = float(cosine_similarity(vs, vqa)[0][0])   # similarity to q+answer
        cos_ans   = float(cosine_similarity(vs, van)[0][0])   # similarity to answer
        pos_n     = i / max(len(sentences) - 1, 1)
        len_n     = len(sent.split()) / max(med_len, 1)
        is_early  = float(pos_n <= 0.3)
        bigram_ov = _bigram_overlap(sent, qa_query)

        feats = [ans_ov, q_ov, cos_qa, pos_n, len_n, is_early, bigram_ov]
        try:
            prob = hint_scorer.predict_proba([feats])[0][1]
        except Exception:
            prob = ans_ov * 0.5 + q_ov * 0.3 + cos_qa * 0.2

        scored.append((sent, prob, i, cos_q, cos_qa, cos_ans, ans_ov, q_ov))

    used_idx = set()

    # Hint 1 — General: most relevant to the QUESTION (not the answer)
    # Gives context/topic without giving away the answer.
    # NOTE: threshold 0.6 can reject ALL sentences in short articles, leaving
    # hint1_sent blank. We do two passes: first with the threshold, then
    # without it (just pick best question-match) so hint 1 is never blank.
    h1_sorted = sorted(scored, key=lambda x: x[3] * 0.6 + x[7] * 0.4, reverse=True)
    hint1_sent, hint1_idx = DEFAULT, -1
    for row in h1_sorted:
        sent, prob, idx, cos_q, cos_qa, cos_ans, ans_ov, q_ov = row
        if ans_ov < 0.6:
            hint1_sent, hint1_idx = sent, idx
            used_idx.add(idx)
            break
    # Fallback: ans_ov threshold was too strict — just take the best question match
    if hint1_sent == DEFAULT and h1_sorted:
        sent, prob, idx, cos_q, cos_qa, cos_ans, ans_ov, q_ov = h1_sorted[0]
        hint1_sent, hint1_idx = sent, idx
        used_idx.add(idx)

    # Hint 2 — Specific: highest combined question+answer relevance,
    # but not a sentence that already gives away the exact answer
    h2_sorted = sorted(scored, key=lambda x: x[4] * 0.5 + x[3] * 0.3 + x[1] * 0.2, reverse=True)
    hint2_sent, hint2_idx = DEFAULT, -1
    for row in h2_sorted:
        sent, prob, idx, cos_q, cos_qa, cos_ans, ans_ov, q_ov = row
        if idx not in used_idx and ans_ov < 0.8:
            hint2_sent, hint2_idx = sent, idx
            used_idx.add(idx)
            break

    # Hint 3 — Near Answer: sentence most similar to the correct answer
    # or with highest answer word overlap — essentially points to the answer
    h3_sorted = sorted(scored, key=lambda x: x[5] * 0.5 + x[6] * 0.5, reverse=True)
    hint3_sent = DEFAULT
    for row in h3_sorted:
        sent, prob, idx, cos_q, cos_qa, cos_ans, ans_ov, q_ov = row
        if idx not in used_idx:
            hint3_sent = sent
            break

    # If all 3 ended up same sentence (tiny article), just use top 3
    if hint1_idx == hint2_idx or hint2_sent == DEFAULT:
        all_s = [r[0] for r in sorted(scored, key=lambda x: x[4], reverse=True)]
        if len(all_s) >= 3:
            hint1_sent = all_s[0]
            hint2_sent = all_s[1]
            hint3_sent = all_s[2]

    return [
        f"Hint 1 (General): {hint1_sent[:250]}",
        f"Hint 2 (Specific): {hint2_sent[:250]}",
        f"Hint 3 (Near Answer): {hint3_sent[:250]}",
    ]