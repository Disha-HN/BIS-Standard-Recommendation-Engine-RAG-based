#  BIS Standard Recommendation Engine (RAG-based)

An AI-powered system that maps real-world product descriptions to relevant **Bureau of Indian Standards (BIS)** using a **Retrieval-Augmented Generation (RAG)** pipeline.

---

## 🚀 Overview

Selecting the correct BIS standard for a product can be complex and time-consuming.
This project solves that by:

* Understanding **natural language product descriptions**
* Retrieving relevant standards from **BIS SP 21**
* Ranking them based on **semantic relevance**
* Producing **accurate, explainable recommendations**

---

## 🏗️ Architecture

```
User Input → Query Expansion → Retrieval (BM25 + FAISS)
           → Ranking → Output (Top BIS Standards)
```

---

## ⚙️ Tech Stack

* **Python**
* **FAISS** (vector similarity search)
* **BM25** (keyword-based retrieval)
* **Sentence Transformers (MiniLM)**
* **pdfplumber** (data extraction)

---

## 📂 Project Structure

```
.
├── src/
│   ├── ingestion.py
│   ├── retriever.py
│   ├── indexer.py
│   ├── generator.py
│   └── query_expander.py
│
├── scripts/
│   ├── diagnose.py
│   └── merge_for_eval.py
│
├── inference.py
├── eval_script.py
├── requirements.txt
├── public_test_set.json
├── sample_output.json
└── README.md
```

---

## 🧪 How it Works

1. User provides a product description
   *Example:*

   > “We manufacture 33 Grade Ordinary Portland Cement”

2. System:

   * Expands query context
   * Searches BIS dataset
   * Retrieves top relevant standards

3. Output:

```json
[
  {
    "standard": "IS 269:2015",
    "title": "Ordinary Portland Cement - Specification",
    "relevance": "High"
  }
]
```

---

## ▶️ Run Locally

```bash
git clone https://github.com/your-username/bis-standard-recommendation-engine.git
cd bis-standard-recommendation-engine
pip install -r requirements.txt
python inference.py
```

---

## 📊 Key Features

* 🔍 Hybrid Retrieval (**BM25 + Semantic Search**)
* ⚡ Fast and scalable (FAISS indexing)
* 🧠 Context-aware query expansion
* 🎯 High relevance ranking
* 📦 Lightweight and reproducible

---

## 🧩 Example Use Cases

* Manufacturing compliance
* Product certification assistance
* Regulatory mapping automation
* Industry-level standard lookup

---

## 👩‍💻 Author

**Disha H N**

---

## ⭐ Acknowledgements

* BIS (Bureau of Indian Standards)
* Sentence Transformers
* Open-source retrieval libraries

---

## 🌟 Future Improvements

* Fine-tuned domain-specific embeddings
* UI dashboard for interactive queries
* Multi-domain standard support
* Explainable AI outputs

---

> Built for **BIS × SS Hackathon 2026**
