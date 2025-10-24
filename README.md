# langchain-rag-example

This project is a simple demonstration of a Retrieval-Augmented Generation (RAG) pipeline using LangChain. It showcases how to ingest a text document, create vector embeddings using a local model with Ollama, store them in a ChromaDB vector store, and perform semantic search to retrieve relevant document chunks.

## Architecture

The project is divided into three main components:

*   `ingestion.py`: This script handles the ingestion of data into the vector database. It reads a text file, splits it into chunks, generates embeddings using Ollama, and stores the embeddings in a ChromaDB vector store.
*   `retrieval.py`: This script handles the retrieval of data from the vector database. It takes a query, generates an embedding for the query, and performs a similarity search in the ChromaDB vector store to find the most relevant document chunks.
*   `main.py`: This is the main entry point for the application. It orchestrates the ingestion and retrieval processes.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    ```
2.  **Install dependencies:**
    This project uses Pipenv for dependency management. To install the required packages, run:
    ```bash
    pipenv install
    ```
3.  **Run Ollama:**
    Before running the application, you need to have Ollama running on your local machine. You can download it from [https://ollama.ai/](https://ollama.ai/). Make sure the `mistral` model is available.

## Usage

To run the project, execute the `main.py` script:

```bash
pipenv run python main.py
```

This will first ingest the data from `vectordbdoc.txt` into the ChromaDB vector store (located in the `db` directory) and then perform a sample query on the ingested data. You can modify the query in `main.py` to search for different information.