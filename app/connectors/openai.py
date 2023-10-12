from typing import Any, List

import requests


class OpenAIConnector(object):
    host_url: str = "https://api.openai.com"
    token: str = ""

    def __init__(self, host_url: str = None, token: str = None):
        if host_url:
            self.host_url = host_url
        if token:
            self.token = token

    def create_embeddings(self, model, input: List[Any]):
        res = requests.post(self.host_url + "/v1/embeddings", json={"model": model, "input": input},
                            headers={'Authorization': 'Bearer ' + self.token, 'Content-Type': 'application/json'})
        if res.status_code == 200:
            return res.json()
        else:
            print(res)
            raise Exception(res.text)
