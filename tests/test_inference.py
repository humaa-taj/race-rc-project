"""
test_inference.py
Basic unit tests for inference functions.
Run: python tests/test_inference.py
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Support both 'src/inference' and bare 'inference' layouts
from src.inference import clean_text, get_answer_from_passage, extract_candidates
from src.model_a_train import generate_questions

def test_clean_text():
    assert clean_text("Hello, World!") == "hello world"
    assert clean_text("  spaces  ") == "spaces"
    print("✅ test_clean_text passed")

def test_generate_questions():
    article = "Tom was a student who loved reading books every day at school."
    answer  = "student"
    qs = generate_questions(article, answer)
    assert len(qs) > 0
    assert all(isinstance(q, str) for q in qs)
    print(f"✅ test_generate_questions passed — {len(qs)} questions generated")

def test_extract_candidates():
    article = "The cat sat on the mat. The dog ran through the park every morning."
    answer  = "cat"
    cands   = extract_candidates(article, answer, top_n=5)
    assert len(cands) > 0
    assert all(isinstance(c[0], str) for c in cands)
    print(f"✅ test_extract_candidates passed — {len(cands)} candidates found")

def test_get_answer_from_passage_uses_raw_article():
    """Verify that get_answer_from_passage extracts proper nouns from raw (uncleaned) text."""
    article = "William Shakespeare was an English playwright born in Stratford-upon-Avon in 1564."
    answer, _ = get_answer_from_passage(article)   # must pass RAW article, NOT clean_text(article)
    # Proper noun detection requires original capitalisation
    assert answer != "the topic", "No meaningful answer extracted — check raw article is passed"
    assert isinstance(answer, str) and len(answer) > 0
    print(f"✅ test_get_answer_from_passage_uses_raw_article passed — answer: '{answer}'")

if __name__ == '__main__':
    test_clean_text()
    test_generate_questions()
    test_extract_candidates()
    test_get_answer_from_passage_uses_raw_article()
    print("\n✅ All tests passed!")