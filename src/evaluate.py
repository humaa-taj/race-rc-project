"""
app.py — Streamlit UI  (Improved)
4 Screens: Article Input | Quiz | Hints | Analytics
Run: streamlit run ui/app.py

Key fixes vs original:
  1. Random RACE Sample now uses the REAL question + REAL A/B/C/D options
     directly from the dataset — no generation needed, no distractor mismatch.
     The original wrongly regenerated distractors for RACE samples, replacing
     perfectly good human-written options with AI-generated fragments.

  2. Custom article path now searches the FULL pool (all 32k articles) for
     the most topically similar article using pre-cached TF-IDF vectors,
     then uses that entry's real question and answer. The original only
     searched pool[:500] and matched on raw text, missing most topics.

  3. Pool article index is pre-cached at startup (build_pool_index) so all
     similarity searches are fast (<1s) instead of recomputing 32k transforms.

  4. Distractor generation for custom articles now calls generate_distractors
     properly and shows all 4 options shuffled (correct answer not always A).

  5. Analytics table now shows ACTUAL training results, not hardcoded wrong values.
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import time
import random
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

from src.inference import (
    load_all_models,
    build_pool_index,
    clean_text,
    retrieve_best_question,
    verify_answer,
    generate_distractors,
    generate_hints,
)

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="AI Quiz System — RACE",
    page_icon="🧠",
    layout="wide"
)

BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_PATH = os.path.join(BASE, 'data', 'processed')
RAW_PATH  = os.path.join(PROC_PATH, 'train_clean.csv')

# ─────────────────────────────────────────
# LOAD MODELS + PRE-CACHE POOL INDEX
# ─────────────────────────────────────────
@st.cache_resource
def get_models():
    m = load_all_models()
    # Pre-cache pool article vectors so all similarity searches are instant
    if m.get('q_pool') and m.get('tfidf'):
        m['pool_art_vecs'] = build_pool_index(m['q_pool'], m['tfidf'])
    else:
        m['pool_art_vecs'] = None
    return m

@st.cache_data
def load_race_df():
    try:
        return pd.read_csv(RAW_PATH)
    except Exception:
        return pd.DataFrame()

models  = get_models()
race_df = load_race_df()

# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def get_keywords(text, top_n=5):
    stopwords = {'the','a','an','is','are','was','were','be','been','have',
                 'has','had','do','does','did','will','would','could','should',
                 'may','might','shall','can','to','of','in','for','on','with',
                 'at','by','from','and','or','but','this','that','it','its'}
    words = re.findall(r'\b[a-z]{4,}\b', text.lower())
    words = [w for w in words if w not in stopwords]
    return [w for w, _ in Counter(words).most_common(top_n)]


def find_best_pool_match(article_text: str, pool: list, pool_art_vecs, tfidf, top_k: int = 5):
    """
    Find the pool entry whose article is most similar to article_text.
    Returns (question, article, answer, pool_idx).
    Uses pre-cached pool_art_vecs — instant on CPU.
    """
    art_vec = tfidf.transform([clean_text(article_text)])
    sims    = cosine_similarity(art_vec, pool_art_vecs).flatten()
    top_idx = np.argsort(sims)[::-1][:top_k]
    # Pick the best match
    best = top_idx[0]
    q, art, ans = pool[best]
    return str(q), str(art), str(ans)


def shuffle_options(correct_answer: str, distractors: list):
    """
    Return options dict {A/B/C/D: text} with correct answer at a random position.
    Returns (options_dict, correct_key).
    """
    all_opts = [correct_answer] + distractors[:3]
    while len(all_opts) < 4:
        all_opts.append("None of the above")
    random.shuffle(all_opts)
    keys = ['A', 'B', 'C', 'D']
    opts = {keys[i]: all_opts[i] for i in range(4)}
    correct_key = keys[all_opts.index(correct_answer)]
    return opts, correct_key


# ─────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────
defaults = {
    'article': '', 'question': '', 'options': {},
    'correct_answer': '', 'hints': [],
    'hints_shown': 0, 'answered': False,
    'selected_opt': None, 'session_log': [],
    'inference_times': [], 'source': ''
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
st.sidebar.title("🧠 AI Quiz System")
st.sidebar.markdown("**FAST NUCES — AI Lab 2026**")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "📄 Article Input",
    "❓ Quiz",
    "💡 Hints",
    "📊 Analytics"
])
st.sidebar.markdown("---")
if st.session_state.source:
    st.sidebar.info(f"Source: {st.session_state.source}")
st.sidebar.warning("⚠️ AI-generated content. Errors possible.")

# ─────────────────────────────────────────
# PAGE 1 — ARTICLE INPUT
# ─────────────────────────────────────────
if page == "📄 Article Input":
    st.title("📄 Article Input")

    col1, col2 = st.columns([3, 1])

    with col2:
        st.markdown("### Quick Load")
        st.markdown("Load a real RACE reading comprehension question with its original options and hints.")

        if st.button("🎲 Random RACE Sample", type="primary"):
            if race_df.empty:
                st.error("train_clean.csv not found.")
            else:
                row = race_df.sample(1).iloc[0]
                correct_opt  = row['answer']          # e.g. 'B'
                correct_text = str(row[correct_opt])  # actual text

                with st.spinner("Generating hints from passage..."):
                    start = time.time()
                    # Use REAL RACE options directly — do NOT regenerate distractors
                    # The original app was replacing perfectly good human-written
                    # RACE options with short AI-generated phrase fragments
                    opts = {
                        'A': str(row['A']),
                        'B': str(row['B']),
                        'C': str(row['C']),
                        'D': str(row['D']),
                    }
                    hints = generate_hints(
                        row['article'], row['question'], correct_text,
                        models.get('tfidf'), models.get('hint_scorer'))
                    elapsed = time.time() - start

                st.session_state.update({
                    'article':        str(row['article']),
                    'question':       str(row['question']),
                    'correct_answer': correct_opt,   # 'A', 'B', 'C', or 'D'
                    'options':        opts,
                    'hints':          hints,
                    'hints_shown':    0,
                    'answered':       False,
                    'selected_opt':   None,
                    'source':         'RACE dataset (real question & options)',
                })
                st.session_state.inference_times.append(elapsed)
                st.success(f"✅ Real RACE sample loaded ({elapsed:.2f}s) — go to ❓ Quiz!")

        st.markdown("---")
        st.markdown("**Or paste your own article below ↙ and click Submit.**")

    with col1:
        article_input = st.text_area(
            "Paste your reading passage here:",
            value=st.session_state.article,
            height=300,
            placeholder="Paste any English reading passage here. The system will find the most relevant question from the RACE training pool and generate plausible wrong options..."
        )

    if st.button("🚀 Submit & Generate Quiz", type="primary"):
        if not article_input.strip():
            st.error("⚠️ Please paste an article or load a random sample.")
        else:
            pool         = models.get('q_pool') or []
            tfidf        = models.get('tfidf')
            pool_art_vecs = models.get('pool_art_vecs')

            with st.spinner("Finding best matching question from RACE pool..."):
                start = time.time()

                if pool and tfidf is not None and pool_art_vecs is not None:
                    # Search FULL pool (32k articles) using pre-cached index
                    question, _, correct_answer = find_best_pool_match(
                        article_input, pool, pool_art_vecs, tfidf, top_k=10)
                else:
                    # Fallback
                    keywords   = get_keywords(article_input, top_n=3)
                    question   = f"What is the passage mainly about regarding {', '.join(keywords)}?"
                    correct_answer = keywords[0] if keywords else "the topic"

            with st.spinner("Generating distractors and hints..."):
                distractors = generate_distractors(
                    article_input, correct_answer,
                    tfidf, models.get('dist_ranker'))

                hints = generate_hints(
                    article_input, question, correct_answer,
                    tfidf, models.get('hint_scorer'))

                # Shuffle so correct answer isn't always A
                opts, correct_key = shuffle_options(correct_answer, distractors)
                elapsed = time.time() - start

            st.session_state.update({
                'article':        article_input,
                'question':       question,
                'correct_answer': correct_key,
                'options':        opts,
                'hints':          hints,
                'hints_shown':    0,
                'answered':       False,
                'selected_opt':   None,
                'source':         'Custom article (AI-generated question & options)',
            })
            st.session_state.inference_times.append(elapsed)
            st.success(f"✅ Quiz generated in {elapsed:.2f}s — go to ❓ Quiz!")
            st.info(f"**Matched question:** {question}")

# ─────────────────────────────────────────
# PAGE 2 — QUIZ
# ─────────────────────────────────────────
elif page == "❓ Quiz":
    st.title("❓ Quiz")

    if not st.session_state.article:
        st.warning("⚠️ Go to **📄 Article Input** first and load or submit an article.")
    else:
        st.markdown("### 📖 Passage")
        with st.expander("Read the passage", expanded=True):
            article = st.session_state.article
            st.write(article[:1500] + "..." if len(article) > 1500 else article)

        st.markdown("### ❓ Question")
        st.info(st.session_state.question)

        st.markdown("### 🔘 Choose your answer:")
        selected = st.radio(
            "Options:",
            options=list(st.session_state.options.keys()),
            format_func=lambda x: f"**{x}:** {st.session_state.options[x]}",
            index=None,
            key="answer_radio"
        )

        if st.button("✅ Check Answer", type="primary"):
            if selected is None:
                st.error("⚠️ Please select an option first!")
            else:
                with st.spinner("Verifying..."):
                    start  = time.time()
                    result = verify_answer(
                        st.session_state.article,
                        st.session_state.question,
                        st.session_state.options[selected],
                        models.get('tfidf'),
                        models.get('verifier')
                    )
                    elapsed = time.time() - start

                correct = st.session_state.correct_answer
                st.session_state.answered     = True
                st.session_state.selected_opt = selected
                st.session_state.inference_times.append(elapsed)
                st.session_state.session_log.append({
                    'Question':   st.session_state.question[:60],
                    'Selected':   selected,
                    'Correct':    correct,
                    'Match':      selected == correct,
                    'Confidence': result['confidence'],
                    'Latency(s)': round(elapsed, 3)
                })

                correct_text = st.session_state.options.get(correct, '')
                if selected == correct:
                    st.success(f"🎉 Correct! **{correct}: {correct_text}**")
                    st.balloons()
                else:
                    st.error(
                        f"❌ Wrong. You chose **{selected}: {st.session_state.options[selected]}**\n\n"
                        f"Correct answer: **{correct}: {correct_text}**")

                st.markdown(f"**Model Confidence:** {result['confidence']:.2%}")

# ─────────────────────────────────────────
# PAGE 3 — HINTS
# ─────────────────────────────────────────
elif page == "💡 Hints":
    st.title("💡 Hint Panel")

    if not st.session_state.article:
        st.warning("⚠️ Go to **📄 Article Input** first.")
    else:
        st.info("Hints are graduated — General → Specific → Near Answer. "
                "Reveal Answer appears only after all hints are shown.")

        hints = st.session_state.hints
        shown = st.session_state.hints_shown

        for i in range(min(shown, len(hints))):
            with st.expander(f"💡 Hint {i+1}", expanded=True):
                # Strip the "Hint N (Label):" prefix for cleaner display
                hint_text = hints[i]
                hint_text = re.sub(r'^Hint \d+ \([^)]+\):\s*', '', hint_text)
                st.write(hint_text)

        if shown < len(hints):
            if st.button(f"Show Hint {shown + 1}"):
                st.session_state.hints_shown += 1
                st.rerun()
        else:
            st.success("✅ All hints shown!")
            if st.button("🎯 Reveal Answer", type="primary"):
                correct  = st.session_state.correct_answer
                ans_text = st.session_state.options.get(correct, '')
                st.success(f"The correct answer is **{correct}: {ans_text}**")

# ─────────────────────────────────────────
# PAGE 4 — ANALYTICS
# ─────────────────────────────────────────
elif page == "📊 Analytics":
    st.title("📊 Analytics Dashboard")

    log   = st.session_state.session_log
    times = st.session_state.inference_times

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Questions Attempted", len(log))
    with c2:
        if log:
            acc = sum(1 for l in log if l['Match']) / len(log)
            st.metric("Session Accuracy", f"{acc:.1%}")
        else:
            st.metric("Session Accuracy", "N/A")
    with c3:
        if times:
            st.metric("Avg Inference Time", f"{np.mean(times):.3f}s")
        else:
            st.metric("Avg Inference Time", "N/A")

    st.markdown("---")

    st.markdown("### 📈 Model A — Actual Training Results")
    df_a = pd.DataFrame({
        'Model':    ['LR+TF-IDF','LR+OHE','SVM+TF-IDF','SVM+OHE',
                     'NB+TF-IDF','NB+OHE','Ensemble','K-Means','GMM','Label Prop'],
        'Binary Acc': [0.4592,0.4775,0.4601,0.4803,0.4320,0.4384,0.4504,0.5916,0.5421,0.4015],
        'Macro F1':   [0.4437,0.4569,0.4444,0.4584,0.4247,0.4295,0.4377,None,None,0.3996]
    })
    st.dataframe(df_a, use_container_width=True)
    st.caption("Binary accuracy: is this option correct? (50% = random). 4-way exact-match: 20.1%.")

    st.markdown("### 📈 Model B — Distractor & Hint Results")
    df_b = pd.DataFrame({
        'Component':  ['Distractor Ranker (RF)', 'Hint Scorer (LR)'],
        'Accuracy':   ['96.0%', '99.4%'],
        'F1':         ['0.940', '0.994'],
        'Key Feature': ['jaccard_answer (0.45)', 'tfidf_cos_qa (+75.3)']
    })
    st.dataframe(df_b, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📋 Session Log")
    if log:
        df_log = pd.DataFrame(log)
        st.dataframe(df_log, use_container_width=True)
        st.download_button("⬇️ Export Session CSV",
                           df_log.to_csv(index=False),
                           "session_log.csv", "text/csv")
    else:
        st.info("No session data yet. Answer some questions first!")

    st.markdown("### ⏱️ Inference Latency")
    if times:
        st.line_chart(times)
    else:
        st.info("No inference data yet.")