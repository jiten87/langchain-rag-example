from ingestion import ingest_data
from retrieval import retrieve_data

def main():
    """
    Main function to run the ingestion and retrieval process.
    """
    # Ingest data into the vector store
    ingest_data()

    # Define a query and retrieve data
    query = "performance aspect for vector database?"
    print(f"\nQuerying the vector store with: '{query}'\n")
    retrieve_data(query)

if __name__ == "__main__":
    main()
