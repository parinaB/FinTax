# FinTax

FinTax is a Retrieval-Augmented Generation (RAG) based tax and accounting assistant built using LangChain, FAISS, and Large Language Models. The application enables users to query accounting and taxation documents in natural language and receive context-aware answers grounded in the uploaded knowledge base.

## Features

- Document-based question answering
- Retrieval-Augmented Generation (RAG) pipeline
- Semantic search using FAISS vector database
- Support for multiple PDF documents
- Context-aware responses from an LLM
- Interactive web interface
- Source-grounded answers based on uploaded documents

## Tech Stack

- Python
- LangChain
- FAISS
- Mistral AI
- Streamlit
- PyPDF
- Hugging Face Embeddings

## Project Structure

```text
FinTax/
│
├── documents/
│   ├── Chapter-2-Accounting-Process.pdf
│   ├── Chapter-6-Bills-of-Exchange-and-Promissory-Notes.pdf
│   ├── faq.pdf
│   └── interplay_transition.pdf
│
├── faiss_index/
│   ├── index.faiss
│   └── index.pkl
│
├── UITAX.py
├── main.py
├── requirements.txt
├── pyproject.toml
├── uv.lock
├── README.md
└── .gitignore
```

## How It Works

1. Documents are loaded from the `documents` directory.
2. The text is extracted and split into chunks.
3. Embeddings are generated for each chunk.
4. The embeddings are stored in a FAISS vector database.
5. User queries are converted into embeddings.
6. Relevant document chunks are retrieved from FAISS.
7. Retrieved context is sent to the language model.
8. The model generates an answer grounded in the retrieved information.

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/FinTax.git
cd FinTax
```

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root:

```env
MISTRAL_API_KEY=your_api_key
```

Add any additional environment variables required by your model provider.

## Running the Application

Launch the application:

```bash
streamlit run UITAX.py
```

or

```bash
python main.py
```

depending on the entry point you intend to use.

## Example Queries

- What is the accounting process?
- Explain bills of exchange and promissory notes.
- What are the key provisions discussed in the FAQ document?
- Explain the concept of interplay and transition rules.
- Summarize the main topics covered in Chapter 2.

## Screenshots

<img width="359" height="795" alt="Screenshot 2026-06-19 at 7 26 05 PM" src="https://github.com/user-attachments/assets/31ef6292-7d4f-42bc-9702-d09c521ba156" />


<img width="365" height="811" alt="Screenshot 2026-06-19 at 7 25 49 PM" src="https://github.com/user-attachments/assets/8ef7ef88-8508-4656-a8ef-a85d4d8490e7" />


## Future Improvements

- Multi-document source citations
- Conversation memory
- Hybrid search (keyword + vector retrieval)
- Support for additional accounting and taxation datasets
- Advanced reranking for improved retrieval quality
- Deployment on cloud infrastructure

## Learning Outcomes

This project demonstrates:

- Retrieval-Augmented Generation (RAG)
- Vector databases and semantic search
- Embedding generation and retrieval pipelines
- Prompt engineering
- LLM application development
- End-to-end document question answering systems

## License

This project is intended for educational and learning purposes.
