# Support Ticket RAG Evaluator

## Problem
Support teams often waste time searching through old tickets, FAQs, and resolutions. While a generic chatbot might help, it can hallucinate and provide incorrect information, which is unacceptable for customer support. 

This application solves this by answering support questions **only** from retrieved ticket evidence. It provides citations to prove where the answer came from, and refuses to answer if the evidence is weak.

## Why not a normal chatbot?
Generic LLMs are trained on wide internet data and may invent policies or provide outdated solutions. A RAG-based approach ensures responses are grounded strictly in the organization's verified historical tickets.

## Architecture
- **Vector Store**: FAISS
- **Embeddings**: `sentence-transformers` (`all-MiniLM-L6-v2`)
- **LLM Engine**: Groq API (`qwen/qwen3-32b`)
- **Frontend**: Streamlit
- **Data processing**: Pandas

## Dataset
Uses a public customer support dataset from Hugging Face (`bitext/Bitext-customer-support-llm-chatbot-training-dataset`). The dataset is cleaned and normalized to simulate a real ticketing system.

## Features
- **Retrieval-Augmented Generation (RAG)**: Answers are generated strictly from the top retrieved tickets.
- **Citations**: Displays exact ticket IDs used to formulate the answer.
- **Confidence Threshold**: Refuses to answer if the semantic similarity (distance) of the best ticket is below the acceptable threshold.
- **Evaluation Dashboard**: Built-in metrics tracking for Hit@3, Hit@5, average top similarity, refusal accuracy, and response latency.

## Evaluation Metrics
The system is evaluated against a curated set of test questions. Current performance on the validation set using `qwen/qwen3-32b`:
- **Hit@3**: 71.0%
- **Hit@5**: 86.0%
- **Average Top Similarity**: 0.60
- **Refusal Accuracy**: 100.0% (Correctly refused all out-of-domain/unanswerable questions)
- **Average Latency**: 5.99s

## Local Setup

1. **Clone and install dependencies**
   ```bash
   python -m venv venv
   source venv/Scripts/activate # Windows
   pip install -r requirements.txt
   ```

2. **Set up Environment Variables**
   Create a `.env` file from the example:
   ```bash
   cp .env.example .env
   ```
   Add your `GROQ_API_KEY`.

3. **Build the Knowledge Base**
   Run the data pipeline sequentially:
   ```bash
   python src/ingest.py
   python src/chunking.py
   python src/embeddings.py
   python src/vectorstore.py
   ```

4. **Generate Evaluation Set and Run Evaluation**
   ```bash
   python src/generate_eval_set.py
   python src/evaluation.py
   ```

5. **Start the App**
   ```bash
   streamlit run app.py
   ```

## Limitations
- This v1 implementation loads the FAISS index into memory. For large-scale enterprise deployments, a dedicated vector database (e.g., Pinecone, Milvus) would be required.
- The default chunking strategy uses one chunk per ticket. Very long tickets might require a more sophisticated recursive character text splitter.
