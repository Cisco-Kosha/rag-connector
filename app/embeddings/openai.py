from typing import Any, Optional, List, Literal, Union, Sequence, Set

import numpy as np
from pydantic import Extra
import tiktoken

from app.core.config import logger

from app.connectors.openai import OpenAIConnector
from app.core.config import settings, logger


class OpenAI(object):
    client: Any = None  #: :meta private:
    model: str = "text-embedding-ada-002"
    deployment: str = model  # to support Azure OpenAI Service custom deployment names
    openai_api_version: Optional[str] = None
    # to support Azure OpenAI Service custom endpoints
    openai_api_base: Optional[str] = None
    # to support Azure OpenAI Service custom endpoints
    openai_api_type: Optional[str] = None
    # to support explicit proxy for OpenAI
    openai_proxy: Optional[str] = None
    embedding_ctx_length: int = 8191
    chunk_size: int = 1000
    allowed_special: Union[Literal["all"], Set[str]] = set()
    disallowed_special: Union[Literal["all"], Set[str], Sequence[str]] = "all"
    skip_empty: bool = False
    openai_connector: OpenAIConnector = OpenAIConnector(host_url=settings.OPENAI_CONNECTOR_SERVER_URL,
                                                        jwt_token=settings.JWT_TOKEN)

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.allow

    def get_len_safe_embeddings(
            self, texts: List[str], *, chunk_size: Optional[int] = None
    ) -> List[List[float]]:
        embeddings: List[List[float]] = [[] for _ in range(len(texts))]

        tokens = []
        indices = []
        model_name = self.model
        try:
            encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            logger.warning("Warning: model not found. Using cl100k_base encoding.")
            model = "cl100k_base"
            encoding = tiktoken.get_encoding(model)
        for i, text in enumerate(texts):
            if self.model.endswith("001"):
                # See: https://github.com/openai/openai-python/issues/418#issuecomment-1525939500
                # replace newlines, which can negatively affect performance.
                text = text.replace("\n", " ")
            token = encoding.encode(
                text,
                allowed_special=self.allowed_special,
                disallowed_special=self.disallowed_special,
            )
            for j in range(0, len(token), self.embedding_ctx_length):
                tokens.append(token[j: j + self.embedding_ctx_length])
                indices.append(i)

        batched_embeddings: List[List[float]] = []
        _chunk_size = chunk_size or self.chunk_size

        _iter = range(0, len(tokens), _chunk_size)

        for i in _iter:
            response = self.openai_connector.create_embeddings(model_name, tokens[i: i + _chunk_size])
            batched_embeddings.extend(r["embedding"] for r in response["data"])

        results: List[List[List[float]]] = [[] for _ in range(len(texts))]
        num_tokens_in_batch: List[List[int]] = [[] for _ in range(len(texts))]
        for i in range(len(indices)):
            if self.skip_empty and len(batched_embeddings[i]) == 1:
                continue
            results[indices[i]].append(batched_embeddings[i])
            num_tokens_in_batch[indices[i]].append(len(tokens[i]))

        for i in range(len(texts)):
            _result = results[i]
            if len(_result) == 0:
                average = self.openai_connector.create_embeddings(model_name, [])["data"][0]["embedding"]
            else:
                average = np.average(_result, axis=0, weights=num_tokens_in_batch[i])
            embeddings[i] = (average / np.linalg.norm(average)).tolist()

        return embeddings

    def _check_response(response: dict, skip_empty: bool = False) -> dict:
        if any(len(d["embedding"]) == 1 for d in response["data"]) and not skip_empty:
            raise Exception("OpenAI API returned an empty embedding")
        return response
