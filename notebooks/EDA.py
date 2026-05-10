# EDA.py — Run this as a script or copy cells into a Jupyter/Colab notebook
# Exploratory Data Analysis on RACE Dataset

import os, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(BASE, 'data', 'raw', 'train.csv')
NB_PATH  = os.path.join(BASE, 'notebooks')
os.makedirs(NB_PATH, exist_ok=True)

# Load
df = pd.read_csv(RAW_PATH)
print("Shape:", df.shape)
print("Columns:", list(df.columns))

# Missing values
print("\nMissing values:\n", df.isnull().sum())

# Answer distribution
plt.figure(figsize=(6,4))
df['answer'].value_counts().sort_index().plot(kind='bar', color=['#4C72B0','#DD8452','#55A868','#C44E52'])
plt.title('Answer Label Distribution')
plt.xlabel('Answer')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(NB_PATH, 'answer_distribution.png'))
plt.show()
print(df['answer'].value_counts())

# Article & question length
df['article_len']  = df['article'].apply(lambda x: len(str(x).split()))
df['question_len'] = df['question'].apply(lambda x: len(str(x).split()))

fig, axes = plt.subplots(1, 2, figsize=(12,4))
axes[0].hist(df['article_len'],  bins=50, color='steelblue', edgecolor='black')
axes[0].set_title('Article Length (words)')
axes[1].hist(df['question_len'], bins=30, color='salmon',    edgecolor='black')
axes[1].set_title('Question Length (words)')
plt.tight_layout()
plt.savefig(os.path.join(NB_PATH, 'length_distributions.png'))
plt.show()
print("Article length stats:\n",  df['article_len'].describe())
print("Question length stats:\n", df['question_len'].describe())

# Question types
def get_q_type(q):
    q = str(q).lower().strip()
    if q.startswith('_') or 'blank' in q: return 'Fill-in-blank'
    for w in ['what','which','how','who','where','when','why']:
        if w in q: return w.capitalize()
    return 'Other'

df['q_type'] = df['question'].apply(get_q_type)
plt.figure(figsize=(8,4))
df['q_type'].value_counts().plot(kind='bar', color='teal', edgecolor='black')
plt.title('Question Type Distribution')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(NB_PATH, 'question_types.png'))
plt.show()
print(df['q_type'].value_counts())

# Summary stats
summary = pd.DataFrame({
    'Metric': ['Total Samples','Avg Article Length','Avg Question Length',
               'Most Common Answer','Unique Q Types','Missing Values'],
    'Value':  [len(df), round(df['article_len'].mean(),1),
               round(df['question_len'].mean(),1),
               df['answer'].value_counts().idxmax(),
               df['q_type'].nunique(), df.isnull().sum().sum()]
})
print("\n", summary.to_string(index=False))
summary.to_csv(os.path.join(NB_PATH, 'summary_stats.csv'), index=False)
print("\n✅ EDA complete!")
