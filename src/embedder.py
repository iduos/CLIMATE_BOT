#
# By Ian Drumm, The Univesity of Salford, UK.
#
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document


class VectorStoreHandler:
    def __init__(self, persist_directory="chroma_db", model="mxbai-embed-large"):
        self.embedder = OllamaEmbeddings(model=model)
        self.db = Chroma(embedding_function=self.embedder, persist_directory=persist_directory)

    def add_documents(self, items):
        docs = []
        for item in items:
            content = f"Post: {item['post']}\nComment: {item['comment']}"
            metadata = {
                "scores": item.get("scores", {}),
                "date": item.get("comment_metadata", {}).get("created_utc", ""),
                "cluster_id": item.get("cluster_id", -1)
            }
            docs.append(Document(page_content=content, metadata=metadata))
        self.db.add_documents(docs)
        self.db.persist()
