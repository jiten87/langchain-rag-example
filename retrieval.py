from langchain_ollama import ChatOllama
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_ollama import OllamaEmbeddings
#from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma



# Constants
persist_directory = 'db'

def retrieve_data(query):
    """
    Retrieves data from the Chroma vector store based on a query.
    """
    # create lmm from mistral modal
    llm = ChatOllama(model= "mistral")

    # run the query over the llm to see the diffrence.
    chain = PromptTemplate.from_template(query) | llm
    result = chain.invoke(input={})
    print(result.content)

    template ="""
    Answer any use questions based solely on the context below:

    <context>
    {context}
    </context>
    {input}
    """
    # Create PromptTemplate object
    prompt = PromptTemplate(
    template=template,
    input_variables=["context", "input"]
    )

    # Create Ollama embeddings
    embedding = OllamaEmbeddings(model= "mistral")

    # Create the Chroma DB
    vectordb = Chroma(
        collection_name="my_local_vectorDB",
        embedding_function=embedding,
        persist_directory=persist_directory,
    )

    combine_doc_chain = create_stuff_documents_chain(llm,prompt=prompt)
    retrival_chain = create_retrieval_chain(retriever=vectordb.as_retriever(search_type="similarity"),combine_docs_chain=combine_doc_chain)

    result = retrival_chain.invoke(input={"input": query})
    print("final answer after retrival:")
    print(result["answer"])

if __name__ == "__main__":
    sample_query = "What are the key components of a vector database?"
    print(f"Retrieving results for query: '{sample_query}'")
    retrieve_data(sample_query)
