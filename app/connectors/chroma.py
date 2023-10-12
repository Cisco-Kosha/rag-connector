from typing import Any, List, Optional

import requests
from chromadb.api.types import OneOrMany, Embedding, Document, ID
from chromadb.types import Metadata


class ChromaDBConnector(object):
    host_url: str = "http://localhost:8010"
    jwt_token: str = None

    def __init__(self, host_url: str = None, jwt_token: str = None):
        if host_url:
            self.host_url = host_url
        if jwt_token:
            self.jwt_token = jwt_token
        else:
            self.jwt_token = "random"

    def upsert_documents(self, collection_id: str, ids: Optional[OneOrMany[ID]],
                         embeddings: Optional[OneOrMany[Embedding]] = None,
                         metadatas: Optional[OneOrMany[Metadata]] = None,
                         documents: Optional[OneOrMany[Document]] = None,
                         increment_index: bool = True, ):
        print("upserting documents")
        res = requests.post(self.host_url + "/api/v1/collections/" + collection_id + "/upsert",
                            json={"embeddings": embeddings, "metadatas": metadatas, "documents": documents, "ids": ids,
                                  "increment_index": increment_index})
        if res.status_code == 200:
            return res.json()
        else:
            raise Exception(res.text)

    def get_or_create_collection(self, name: str):
        print("getting or creating collection")
        res = requests.post(self.host_url + "/api/v1/collections", json={"name": name, "get_or_create": True},
                            headers={'Authorization': 'Bearer ' + self.jwt_token, 'Content-Type': 'application/json'})
        if res.status_code == 200:
            return res.json()
        else:
            raise Exception(res.text)

    def get_collection(self, name: str):
        print("getting collection")
        res = requests.post(self.host_url + "/api/v1/collections/" + name + "/get", json={"include": ["documents"]},
                            headers={'Authorization': 'Bearer ' + self.jwt_token, 'Content-Type': 'application/json'})
        if res.status_code == 200:
            return res.json()
        else:
            raise Exception(res.text)

    def query(self, name: str, vector: Any, include: Optional[List[str]] = None, n_results: int = 10):
        print("querying collection")
        #print(vector)
        print(name)

        # res = requests.post(self.host_url + "/api/v1/collections/" + name + "/query",
        #                     json={"query_embeddings": vector,
        #                           "n_results": n_results,
        #                           "include": ["documents"]},
        #                     headers={'Authorization': 'Bearer ' + self.jwt_token,
        #                              'Content-Type': 'application/json'})

        res = requests.post(self.host_url + "/api/v1/collections/" + name + "/query", json={"query_embeddings": vector,
                                                                                            "n_results": n_results,
                                                                                            "include": ["documents"]},
                            headers={'Authorization': 'Bearer ' + self.jwt_token,
                                     'Content-Type': 'application/json'})
        if res.status_code == 200:
            return res.json()
        else:
            raise Exception(res.text)
