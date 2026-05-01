# 🤖 Support Triage Agent v2.0

A high-performance, RAG-enabled terminal agent designed to autonomously triage support tickets using **Groq (Llama-3.3-70b)** and **TF-IDF Retrieval**.

---

## 🏗️ Architecture Overview

The system is built on a modular, decoupled architecture to ensure maintainability and testability:

- **`corpus_loader.py`**: A recursive document scanner that ingests markdown documentation from local directories into a searchable corpus.
- **`retriever.py`**: The "Search Engine." It implements a TF-IDF vectorizer and cosine similarity to find the most relevant documentation snippets for any given support issue.
- **`agent.py`**: The "Brain." It interfaces with the Groq API (Llama-3.3-70B) to classify tickets, detect security threats, and generate responses. It features a robust exponential backoff retry mechanism.
- **`main.py`**: The "Orchestrator." It manages the end-to-end workflow, handles CSV I/O, enforces multi-tier rate limiting, and provides a **Rich Terminal UI**.

---

## 🧠 How RAG Works Here

This project utilizes **Retrieval-Augmented Generation (RAG)** to provide grounded, factual responses:

1.  **Indexing**: On startup, `corpus_loader` reads all markdown files. `retriever` then converts these documents into a TF-IDF matrix.
2.  **Retrieval**: When a ticket arrives, the system searches the matrix for the Top-3 most relevant snippets based on keyword similarity.
3.  **Augmentation**: These snippets, along with a "Confidence Score," are injected into a specialized system prompt.
4.  **Generation**: The LLM uses this context to decide if it can safely answer the ticket or if it must escalate to a human.

---

## 🛡️ Safety & Security Features

Engineering standards for security are baked into the core of the agent:

- **Malicious Prompt Detection**: The agent is instructed to detect and flag "jailbreak" attempts, system instruction extraction, or manipulative language.
- **Language Constraint**: To prevent hallucination in translation, the agent detects the ticket language. All non-English tickets are automatically escalated for human verification.
- **Empty Field Handling**: Robust sanitization handles `NaN` values and empty strings, preventing runtime crashes.
- **Rate Limit Resilience**: The system implements a three-tier safeguard:
    1.  2-second delay between every request.
    2.  60-second batch sleep every 10 tickets.
    3.  Exponential backoff for `429 Too Many Requests` errors.

---

## 🚩 Escalation Philosophy

The agent is designed with a **"Safety First"** mindset. A ticket is **escalated** if:
- It involves fraud, billing disputes, or account security.
- The retrieval confidence is too low (< 0.15), indicating the docs don't have the answer.
- A security threat is detected.
- The ticket is non-English.
- The issue is complex and requires senior oversight.

Every escalation includes a **Justification** containing a one-line summary and a detailed reasoning for auditability.

---

## 🛠️ Tech Stack

- **LLM**: Groq Llama-3.3-70b-versatile (Incredible speed & reasoning).
- **Retrieval**: Scikit-Learn (TF-IDF Vectorization).
- **UI**: Rich (Modern terminal tables, progress bars, and panels).
- **Data**: Pandas (High-speed CSV processing).

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install pandas scikit-learn tqdm groq rich openai
```

### 2. Set Environment Variables
```bash
# Windows
set GROQ_API_KEY=your_key_here
```

### 3. Run the Agent
```bash
python code/main.py
```

---

## ⚖️ Design Decisions & Tradeoffs

- **TF-IDF vs. Vector DB**: We chose TF-IDF because the current corpus (HackerRank/Visa docs) is highly keyword-specific. A full Vector Database (like Chroma/Pinecone) would add unnecessary infrastructure overhead for this scale.
- **Groq vs. Local**: While Ollama was supported, Groq was selected as the default for the production README because the 70B Llama model provides significantly better reasoning for security detection than local 1B/3B models.
- **Synchronous Processing**: The orchestrator is synchronous to strictly respect rate limits. While async would be faster, it would likely trigger instant bans on free/low-tier API keys.

---
*Developed as a professional Support Triage Solution.*
