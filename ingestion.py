from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma


# Constants
persist_directory = 'db'
file_path = "vectordbdoc.txt"

def ingest_data():
    """
    Ingests data from a text file, splits it into chunks,
    and stores it in a Chroma vector store.
    """
    loader = TextLoader(file_path=file_path, encoding="utf-8")
    document = loader.load()

    print("splitting.....")
    text_spiltter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_spiltter.split_documents(documents=document)
    print(f"created {len(texts)} chunks ")

    embedding = OllamaEmbeddings(model= "mistral")

    # Create the Chroma DB
    vectordb = Chroma(
        collection_name="my_local_vectorDB",
        embedding_function=embedding,
        persist_directory=persist_directory,
    )

    vectordb.add_documents(texts)
    print("Data ingestion complete.")

if __name__ == "__main__":
    ingest_data()
