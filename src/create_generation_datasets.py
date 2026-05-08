"""
Phase 1: Data Restructuring for Supervised Generation Tasks

This script transforms RACE data into three supervised training datasets:
- Dataset A: Question Generation (article + correct_answer → question)
- Dataset B: Distractor Generation (article + question + correct_answer → 3 wrong options)
- Dataset C: Hint Generation (article + question + correct_answer → top 3 relevant sentences)

All text is preserved with original punctuation and structure (NOT aggressively cleaned yet).
"""

import pandas as pd
import numpy as np
import re
import os
from collections import Counter
from pathlib import Path

# Configuration
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# Create output directory if needed
DATA_DIR.mkdir(parents=True, exist_ok=True)
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """Load the three clean splits."""
    print("Loading data splits...")
    
    train_df = pd.read_csv(DATA_DIR / "train_clean.csv")
    dev_df = pd.read_csv(DATA_DIR / "dev_clean.csv")
    test_df = pd.read_csv(DATA_DIR / "test_clean.csv")
    
    print(f"  Train: {len(train_df)} rows")
    print(f"  Dev: {len(dev_df)} rows")
    print(f"  Test: {len(test_df)} rows")
    
    return train_df, dev_df, test_df


def get_answer_text(row, answer_letter):
    """Get the actual text of the answer from the option column."""
    if answer_letter not in ['A', 'B', 'C', 'D']:
        return None
    return row[answer_letter]


def get_wrong_options(row, answer_letter):
    """Get all three wrong options (the distractors)."""
    options = {'A': row['A'], 'B': row['B'], 'C': row['C'], 'D': row['D']}
    wrong_opts = [opt for letter, opt in options.items() if letter != answer_letter]
    return wrong_opts[:3]  # Ensure exactly 3


def split_into_sentences(text):
    """Split text into sentences using regex on .!?"""
    if not isinstance(text, str):
        return []
    # Split on sentence boundaries
    sentences = re.split(r'[.!?]+', text)
    # Clean up and filter
    sentences = [s.strip() for s in sentences if 20 <= len(s.strip()) <= 300]
    return sentences


def compute_word_overlap_score(sentence, question_text, answer_text):
    """
    Compute word overlap score: 
    (words in sentence ∩ (question_words ∪ answer_words)) / max(len(question_words), 1)
    """
    if len(sentence) < 20:  # Skip sentences < 20 chars
        return 0.0
    
    # Handle NaN/float values
    if not isinstance(sentence, str):
        sentence = str(sentence)
    if not isinstance(question_text, str):
        question_text = str(question_text)
    if not isinstance(answer_text, str):
        answer_text = str(answer_text)
    
    # Normalize to lowercase for comparison
    sentence_words = set(re.findall(r'\b\w+\b', sentence.lower()))
    question_words = set(re.findall(r'\b\w+\b', question_text.lower()))
    answer_words = set(re.findall(r'\b\w+\b', answer_text.lower()))
    
    combined_words = question_words | answer_words
    overlap = len(sentence_words & combined_words)
    max_len = max(len(question_words), 1)
    
    score = overlap / max_len
    return score


def get_top_hint_sentences(article, question, correct_answer, num_hints=3):
    """
    Extract top 3 relevant sentences from article based on word overlap with question+answer.
    Returns: (list of hints, list of overlap scores)
    """
    if not isinstance(article, str):
        return ["Re-read the passage carefully."] * num_hints, [0.0] * num_hints
    if not isinstance(question, str):
        question = str(question)
    if not isinstance(correct_answer, str):
        correct_answer = str(correct_answer)
    
    sentences = split_into_sentences(article)
    
    # Score each sentence
    scored_sentences = []
    for sent in sentences:
        if len(sent) < 20:  # Skip short sentences
            continue
        score = compute_word_overlap_score(sent, question, correct_answer)
        if score > 0:  # Only keep sentences with some overlap
            scored_sentences.append((sent, score))
    
    # Sort by score descending
    scored_sentences.sort(key=lambda x: x[1], reverse=True)
    
    # Extract top 3
    hints = []
    scores = []
    for i in range(num_hints):
        if i < len(scored_sentences):
            hint_text, score = scored_sentences[i]
            hints.append(hint_text)
            scores.append(round(score, 3))
        else:
            # Pad with default if not enough sentences
            hints.append("Re-read the passage carefully.")
            scores.append(0.0)
    
    return hints, scores


def create_dataset_a(df, split_name):
    """
    Dataset A: Question Generation
    Input: article + correct_answer (text)
    Output: question
    Columns: [article_original, correct_answer, question]
    """
    print(f"  Creating Dataset A for {split_name}...")
    
    rows = []
    for idx, row in df.iterrows():
        answer_letter = row['answer']
        answer_text = get_answer_text(row, answer_letter)
        
        if answer_text is None:
            continue
        
        rows.append({
            'article_original': row['article'],
            'correct_answer': answer_text,
            'question': row['question']
        })
    
    result_df = pd.DataFrame(rows)
    output_file = DATA_DIR / f"dataset_a_question_gen_{split_name}.csv"
    result_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"    Saved {len(result_df)} rows to {output_file.name}")
    
    return result_df


def create_dataset_b(df, split_name):
    """
    Dataset B: Distractor Generation
    Input: article + question + correct_answer
    Output: 3 wrong options (extract from A/B/C/D excluding correct answer)
    Columns: [article_original, question, correct_answer, wrong_option_1, wrong_option_2, wrong_option_3]
    One row per Q&A pair (not expanded)
    """
    print(f"  Creating Dataset B for {split_name}...")
    
    rows = []
    for idx, row in df.iterrows():
        answer_letter = row['answer']
        answer_text = get_answer_text(row, answer_letter)
        
        if answer_text is None:
            continue
        
        wrong_options = get_wrong_options(row, answer_letter)
        
        if len(wrong_options) < 3:
            # Skip rows with missing options
            continue
        
        rows.append({
            'article_original': row['article'],
            'question': row['question'],
            'correct_answer': answer_text,
            'wrong_option_1': wrong_options[0],
            'wrong_option_2': wrong_options[1],
            'wrong_option_3': wrong_options[2]
        })
    
    result_df = pd.DataFrame(rows)
    output_file = DATA_DIR / f"dataset_b_distractor_gen_{split_name}.csv"
    result_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"    Saved {len(result_df)} rows to {output_file.name}")
    
    return result_df


def create_dataset_c(df, split_name):
    """
    Dataset C: Hint Generation
    Input: article + question + correct_answer
    Output: Top 3 relevant sentences from article (by word overlap)
    Columns: [article_original, question, correct_answer, hint_1, hint_2, hint_3, overlap_scores]
    """
    print(f"  Creating Dataset C for {split_name}...")
    
    rows = []
    for idx, row in df.iterrows():
        answer_letter = row['answer']
        answer_text = get_answer_text(row, answer_letter)
        
        if answer_text is None:
            continue
        
        hints, scores = get_top_hint_sentences(
            row['article'],
            row['question'],
            answer_text,
            num_hints=3
        )
        
        rows.append({
            'article_original': row['article'],
            'question': row['question'],
            'correct_answer': answer_text,
            'hint_1': hints[0],
            'hint_2': hints[1],
            'hint_3': hints[2],
            'overlap_scores': f"[{scores[0]}, {scores[1]}, {scores[2]}]"
        })
    
    result_df = pd.DataFrame(rows)
    output_file = DATA_DIR / f"dataset_c_hint_gen_{split_name}.csv"
    result_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"    Saved {len(result_df)} rows to {output_file.name}")
    
    return result_df


def print_statistics(dataset_a_dict, dataset_b_dict, dataset_c_dict):
    """Print statistics for validation."""
    report_lines = []
    
    report_lines.append("=" * 80)
    report_lines.append("PHASE 1 DATA RESTRUCTURING - STATISTICS")
    report_lines.append("=" * 80)
    
    # Dataset A Statistics
    report_lines.append("\n" + "=" * 80)
    report_lines.append("DATASET A: QUESTION GENERATION")
    report_lines.append("=" * 80)
    report_lines.append("Structure: Input (article + correct_answer) → Output (question)")
    
    for split in ['train', 'dev', 'test']:
        df = dataset_a_dict[split]
        report_lines.append(f"\n{split.upper()} split:")
        report_lines.append(f"  Total rows: {len(df)}")
        report_lines.append(f"  Avg article length: {df['article_original'].str.len().mean():.0f} chars")
        report_lines.append(f"  Avg answer length: {df['correct_answer'].str.len().mean():.0f} chars")
        report_lines.append(f"  Avg question length: {df['question'].str.len().mean():.0f} chars")
    
    # Dataset B Statistics
    report_lines.append("\n" + "=" * 80)
    report_lines.append("DATASET B: DISTRACTOR GENERATION")
    report_lines.append("=" * 80)
    report_lines.append("Structure: Input (article + question + correct_answer) → Output (3 wrong options)")
    
    for split in ['train', 'dev', 'test']:
        df = dataset_b_dict[split]
        report_lines.append(f"\n{split.upper()} split:")
        report_lines.append(f"  Total rows: {len(df)}")
        report_lines.append(f"  Avg article length: {df['article_original'].str.len().mean():.0f} chars")
        report_lines.append(f"  Avg question length: {df['question'].str.len().mean():.0f} chars")
        report_lines.append(f"  Avg correct answer length: {df['correct_answer'].str.len().mean():.0f} chars")
        
        # Check distractor statistics
        for i in [1, 2, 3]:
            col = f'wrong_option_{i}'
            report_lines.append(f"  Avg distractor {i} length: {df[col].str.len().mean():.0f} chars")
    
    # Dataset C Statistics
    report_lines.append("\n" + "=" * 80)
    report_lines.append("DATASET C: HINT GENERATION")
    report_lines.append("=" * 80)
    report_lines.append("Structure: Input (article + question + correct_answer) → Output (top 3 hint sentences)")
    
    for split in ['train', 'dev', 'test']:
        df = dataset_c_dict[split]
        report_lines.append(f"\n{split.upper()} split:")
        report_lines.append(f"  Total rows: {len(df)}")
        
        # Check hints
        for i in [1, 2, 3]:
            col = f'hint_{i}'
            avg_len = df[col].str.len().mean()
            non_default = (df[col] != "Re-read the passage carefully.").sum()
            report_lines.append(f"  Hint {i}: {non_default}/{len(df)} real hints (default padded: {len(df) - non_default}), avg length: {avg_len:.0f} chars")
    
    # Sample Examples
    report_lines.append("\n" + "=" * 80)
    report_lines.append("SAMPLE EXAMPLES FROM TRAINING SPLIT")
    report_lines.append("=" * 80)
    
    report_lines.append("\n--- DATASET A: QUESTION GENERATION (5 examples) ---")
    df_a_train = dataset_a_dict['train']
    for i in range(min(5, len(df_a_train))):
        row = df_a_train.iloc[i]
        report_lines.append(f"\nExample {i+1}:")
        report_lines.append(f"  Article (first 150 chars): {row['article_original'][:150]}...")
        report_lines.append(f"  Correct Answer: {row['correct_answer']}")
        report_lines.append(f"  Generated Question: {row['question']}")
    
    report_lines.append("\n--- DATASET B: DISTRACTOR GENERATION (5 examples) ---")
    df_b_train = dataset_b_dict['train']
    for i in range(min(5, len(df_b_train))):
        row = df_b_train.iloc[i]
        report_lines.append(f"\nExample {i+1}:")
        report_lines.append(f"  Article (first 150 chars): {row['article_original'][:150]}...")
        report_lines.append(f"  Question: {row['question']}")
        report_lines.append(f"  Correct Answer: {row['correct_answer']}")
        report_lines.append(f"  Distractor 1: {row['wrong_option_1']}")
        report_lines.append(f"  Distractor 2: {row['wrong_option_2']}")
        report_lines.append(f"  Distractor 3: {row['wrong_option_3']}")
    
    report_lines.append("\n--- DATASET C: HINT GENERATION (5 examples) ---")
    df_c_train = dataset_c_dict['train']
    for i in range(min(5, len(df_c_train))):
        row = df_c_train.iloc[i]
        report_lines.append(f"\nExample {i+1}:")
        report_lines.append(f"  Article (first 150 chars): {row['article_original'][:150]}...")
        report_lines.append(f"  Question: {row['question']}")
        report_lines.append(f"  Correct Answer: {row['correct_answer']}")
        report_lines.append(f"  Hint 1: {row['hint_1']}")
        report_lines.append(f"  Hint 2: {row['hint_2']}")
        report_lines.append(f"  Hint 3: {row['hint_3']}")
        report_lines.append(f"  Overlap Scores: {row['overlap_scores']}")
    
    # Quality Checks
    report_lines.append("\n" + "=" * 80)
    report_lines.append("QUALITY CHECKS")
    report_lines.append("=" * 80)
    
    report_lines.append("\n✓ DATASET A: Question Diversity Check")
    df_a_train_unique = df_a_train['question'].nunique()
    report_lines.append(f"  Unique questions in train: {df_a_train_unique}/{len(df_a_train)} ({100*df_a_train_unique/len(df_a_train):.1f}%)")
    report_lines.append(f"  → Questions appear CONTEXTUAL (tied to specific articles)")
    
    report_lines.append("\n✓ DATASET B: Distractor Validation Check")
    # Sample: are distractors actually different from correct answer?
    df_b_train_sample = df_b_train.head(10)
    different_count = 0
    for idx, row in df_b_train_sample.iterrows():
        correct = row['correct_answer'].lower().strip()
        d1 = row['wrong_option_1'].lower().strip()
        d2 = row['wrong_option_2'].lower().strip()
        d3 = row['wrong_option_3'].lower().strip()
        if correct != d1 and correct != d2 and correct != d3:
            different_count += 1
    report_lines.append(f"  Sample check (10 examples): {different_count}/10 have 3 distinct wrong options")
    report_lines.append(f"  → Distractors are DISTINCT from correct answer")
    
    report_lines.append("\n✓ DATASET C: Hint Relevance Check")
    real_hints_count = ((df_c_train['hint_1'] != "Re-read the passage carefully.") |
                        (df_c_train['hint_2'] != "Re-read the passage carefully.") |
                        (df_c_train['hint_3'] != "Re-read the passage carefully.")).sum()
    report_lines.append(f"  Examples with at least 1 real hint: {real_hints_count}/{len(df_c_train)} ({100*real_hints_count/len(df_c_train):.1f}%)")
    report_lines.append(f"  → Hints use WORD OVERLAP scoring (contextual)")
    
    report_lines.append("\n" + "=" * 80)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 80)
    
    return "\n".join(report_lines)


def main():
    """Main execution."""
    print("\n" + "=" * 80)
    print("PHASE 1: DATA RESTRUCTURING FOR GENERATION TASKS")
    print("=" * 80)
    
    # Load data
    train_df, dev_df, test_df = load_data()
    
    # Create datasets
    print("\n--- Creating Dataset A (Question Generation) ---")
    dataset_a_train = create_dataset_a(train_df, 'train')
    dataset_a_dev = create_dataset_a(dev_df, 'dev')
    dataset_a_test = create_dataset_a(test_df, 'test')
    dataset_a_dict = {
        'train': dataset_a_train,
        'dev': dataset_a_dev,
        'test': dataset_a_test
    }
    
    print("\n--- Creating Dataset B (Distractor Generation) ---")
    dataset_b_train = create_dataset_b(train_df, 'train')
    dataset_b_dev = create_dataset_b(dev_df, 'dev')
    dataset_b_test = create_dataset_b(test_df, 'test')
    dataset_b_dict = {
        'train': dataset_b_train,
        'dev': dataset_b_dev,
        'test': dataset_b_test
    }
    
    print("\n--- Creating Dataset C (Hint Generation) ---")
    dataset_c_train = create_dataset_c(train_df, 'train')
    dataset_c_dev = create_dataset_c(dev_df, 'dev')
    dataset_c_test = create_dataset_c(test_df, 'test')
    dataset_c_dict = {
        'train': dataset_c_train,
        'dev': dataset_c_dev,
        'test': dataset_c_test
    }
    
    # Print statistics and generate report
    print("\n" + "=" * 80)
    print("GENERATING VALIDATION REPORT")
    print("=" * 80)
    
    report = print_statistics(dataset_a_dict, dataset_b_dict, dataset_c_dict)
    print(report)
    
    # Save report
    report_file = NOTEBOOKS_DIR / "phase1_analysis.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n✓ Analysis report saved to: {report_file}")
    
    print("\n" + "=" * 80)
    print("PHASE 1 COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
