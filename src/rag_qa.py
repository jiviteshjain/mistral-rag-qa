from dataclasses import dataclass
import os
import sys

from omegaconf import DictConfig
from langchain_core.documents import Document

from .rag_pipeline.rag_validation import (
    get_embedding_model,
    load_faiss_store,
    attach_reranker,
    get_hyde_model,
    get_reader_model,
    get_prompt,
    get_rag_chain,
    retrieve_docs_batched,
    parse_regex,
)


@dataclass
class Source:
    name: str
    text: str
    index_id: int


class RagQA:
    def __init__(self, conf: DictConfig):
        self.rag_chain = None
        self.hyde_pipeline = None
        self.retriever = None
        self.conf = conf

    def load(self):
        embedding_model = get_embedding_model(self.conf)
        faiss_store = load_faiss_store(self.conf, embedding_model)
        self.retriever = faiss_store.as_retriever()
        if self.conf.rag.reranking.enabled:
            self.retriever = attach_reranker(self.conf, self.retriever)

        self.hyde_pipeline = None
        if self.conf.rag.hyde.enabled or self.conf.rag.summary.enabled:
            self.hyde_pipeline = get_hyde_model(self.conf)

        reader_model = get_reader_model(self.conf)
        prompt = get_prompt(self.conf)
        self.rag_chain = get_rag_chain(self.conf, reader_model, prompt)

    @staticmethod
    def _docs_to_sources(docs: list[Document]) -> list[Source]:
        return [
            Source(
                name=doc.metadata["source_name"],
                text=doc.metadata["original_page_content"],
                index_id=doc.metadata["chunk_id"],
            )
            for doc in docs
        ]

    def answer(self, question: str) -> tuple[str, list[Source]]:
        docs = retrieve_docs_batched(
            self.conf,
            self.retriever,
            None,  # Not using the sparse index.
            self.hyde_pipeline,
            self.hyde_pipeline,  # Use the hyde model for summarization as well.
            [question],
        )

        sources = self._docs_to_sources(docs[0]["docs"])

        chain_output = self.rag_chain.batch(docs)
        batch_answers = [
            parse_regex(row["raw_output"])["answer"] for row in chain_output
        ]

        answer = " ".join(batch_answers[0].strip().split())
        return answer, sources
