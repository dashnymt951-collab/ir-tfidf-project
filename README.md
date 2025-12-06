# IR TF-IDF Project

Repository with a simple TF-IDF search engine for the course project.

## Structure

```
ir_tfidf_project/
├─ data/
│  └─ docs/          # .txt documents
│  └─ queries.json    # queries + ground truth
├─ src/
│  └─ tfidf_search.py
├─ requirements.txt
└─ README.md
```

## Usage

1. Create a virtualenv and install requirements:

```
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

2. Run the demo:

```
python src/tfidf_search.py --data_dir data/docs --queries data/queries.json
```

The script will print query results and evaluation (precision@5, recall@5 and precision@10, recall@10).

Replace `data/docs` with your own dataset if available.
