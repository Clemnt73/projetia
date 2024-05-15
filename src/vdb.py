from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

from typing import List


class WikiVectorDatabase:
    def __init__(
        self,
        embeddings_model_name: str,
        chunk_size: int = 200,
        chunk_overlap: int = 100
    ):
        self._embeddings_model = HuggingFaceEmbeddings(
            model_name=embeddings_model_name
        )
        self._chunker = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=['\n\n\n']
        )

    def create_vdb(self, articles: List[str]):
        documents = [Document(page_content=article) for article in articles]
        split_docs = self._chunker.split_documents(documents)
        return FAISS.from_documents(
            documents=split_docs,
            embedding=self._embeddings_model
        )

    @staticmethod
    def query_vdb(query: str, vdb: FAISS):
        return vdb.similarity_search(
            query=query,
            k=10
        )[0].page_content
