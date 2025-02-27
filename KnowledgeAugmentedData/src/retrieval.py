# retrieval.py

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from src.document_loader import DocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

class Retrieval:
    def __init__(self, datasetList, path_database="chroma_BioRed_db", create_database=False):
        self.datasetList = datasetList
        self.path_database = path_database
        self.create_database = create_database
        self.local_embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=os.getenv("OLLAMA_HOST_EMBEDDING"))

    def load_retrieval(self):
        if self.create_database:
            loader = DocumentLoader(self.datasetList)
            data = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            all_splits = text_splitter.split_documents(data)
            vectorstore = Chroma.from_documents(documents=all_splits, embedding=self.local_embeddings, persist_directory=self.path_database)
        else:
            vectorstore = Chroma(persist_directory=self.path_database, embedding_function=self.local_embeddings)
        return vectorstore