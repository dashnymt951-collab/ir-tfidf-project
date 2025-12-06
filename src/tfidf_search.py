"""
Simple TF-IDF based search engine
Usage:
    python src/tfidf_search.py --data_dir data/docs --queries data/queries.json
Outputs:
    - Demonstration of queries and evaluation printed to console.
"""
import os
import json
import argparse
import re
from typing import List, Dict

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure required NLTK data is downloaded (first run)
nltk_packages = ['punkt', 'stopwords', 'wordnet', 'omw-1.4']
for pkg in nltk_packages:
    try:
        nltk.data.find(pkg)
    except LookupError:
        nltk.download(pkg.split('/')[-1])

STOPWORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    lem = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(lem)

class TFIDFSearch:
    def __init__(self, ngram=(1,2), min_df=1, max_features=None, sublinear_tf=True):
        self.vectorizer = TfidfVectorizer(ngram_range=ngram, min_df=min_df, max_features=max_features, sublinear_tf=sublinear_tf)
        self.doc_ids = []
        self.docs = []
        self.tfidf_matrix = None

    def fit(self, doc_ids: List[str], raw_docs: List[str]):
        self.doc_ids = doc_ids
        self.docs = [preprocess(d) for d in raw_docs]
        self.tfidf_matrix = self.vectorizer.fit_transform(self.docs)

    def query(self, q: str, top_k: int = 10):
        q_proc = preprocess(q)
        q_vec = self.vectorizer.transform([q_proc])
        sim = cosine_similarity(q_vec, self.tfidf_matrix).flatten()
        ranked_idx = sim.argsort()[::-1][:top_k]
        return [(self.doc_ids[i], float(sim[i])) for i in ranked_idx]

def load_docs(folder: str):
    docs = []
    ids = []
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith('.txt'):
            continue
        path = os.path.join(folder, fname)
        with open(path, 'r', encoding='utf-8') as f:
            docs.append(f.read().strip())
        ids.append(fname)
    return ids, docs

def evaluate(searcher: TFIDFSearch, queries: Dict[str, List[str]], k=10):
    precisions = []
    recalls = []
    for q, relevant in queries.items():
        res = searcher.query(q, top_k=k)
        retrieved = [doc for doc, _ in res]
        true_pos = sum(1 for r in retrieved if r in relevant)
        prec = true_pos / len(retrieved) if retrieved else 0.0
        rec = true_pos / len(relevant) if relevant else 0.0
        precisions.append(prec)
        recalls.append(rec)
    return {
        "precision@{}".format(k): sum(precisions)/len(precisions) if precisions else 0.0,
        "recall@{}".format(k): sum(recalls)/len(recalls) if recalls else 0.0
    }

def main(data_dir, queries_path):
    ids, docs = load_docs(data_dir)
    searcher = TFIDFSearch(ngram=(1,2), min_df=1)
    searcher.fit(ids, docs)

    with open(queries_path, 'r', encoding='utf-8') as f:
        queries = json.load(f)

    print("Loaded {} documents, {} queries".format(len(ids), len(queries)))

    for q in queries.keys():
        print("\\nQuery:", q)
        results = searcher.query(q, top_k=5)
        for rank, (doc_id, score) in enumerate(results, start=1):
            print(f"{rank:02d}. {doc_id} (score={score:.4f})")

    eval5 = evaluate(searcher, queries, k=5)
    eval10 = evaluate(searcher, queries, k=10)
    print("\\nEvaluation:")
    print(json.dumps({"eval5": eval5, "eval10": eval10}, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/docs', help='Folder with .txt documents')
    parser.add_argument('--queries', dest='queries', default='data/queries.json', help='Queries JSON (ground truth)')
    args = parser.parse_args()
    main(args.data_dir, args.queries)
