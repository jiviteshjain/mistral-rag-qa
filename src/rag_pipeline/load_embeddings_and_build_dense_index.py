# ## Files used:
# conf.files.context
# conf.files.index
# conf.files.embeddings

import logging
import csv

import hydra
from omegaconf import DictConfig, OmegaConf
import faiss
from datasets import load_dataset, IterableDataset
from tqdm.auto import tqdm

from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

log = logging.getLogger(__name__)


def get_hnsw_pq_index(conf: DictConfig) -> faiss.IndexHNSWPQ:
    index = faiss.index_factory(
        conf.embeddings.dim,
        f"HNSW{conf.indexes.hnsw_pq.hnsw_m},PQ{conf.indexes.hnsw_pq.pq_m}x{conf.indexes.hnsw_pq.bits}",
    )
    index.hnsw.efConstruction = conf.indexes.hnsw_pq.ef_construction
    index.hnsw.efSearch = conf.indexes.hnsw_pq.ef_search
    return index


def get_hnsw_index(conf: DictConfig) -> faiss.IndexHNSW:
    index = faiss.index_factory(
        conf.embeddings.dim,
        f"HNSW{conf.indexes.hnsw_pq.hnsw_m}",
    )
    index.hnsw.efConstruction = conf.indexes.hnsw_pq.ef_construction
    index.hnsw.efSearch = conf.indexes.hnsw_pq.ef_search
    return index


def get_index(conf: DictConfig) -> faiss.Index:
    if conf.indexing.index_type == "hnsw_pq":
        return get_hnsw_pq_index(conf)
    elif conf.indexing.index_type == "hnsw":
        return get_hnsw_index(conf)
    else:
        raise ValueError(f"Unknown index type: {conf.indexing.index_type}")


def get_embedding_model(conf: DictConfig) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=conf.embeddings.model,
        model_kwargs={"trust_remote_code": True, "device": conf.embeddings.device},
    )


def get_faiss_store(conf: DictConfig, embedding_model: HuggingFaceEmbeddings) -> FAISS:
    index = get_index(conf)
    return FAISS(
        embedding_function=embedding_model,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )


def get_dataset(
    conf,
) -> IterableDataset:

    with open(conf.files.context, "r") as f:
        num_lines = sum(1 for _ in f)

    dataset = load_dataset(
        "json",
        data_files={"train": [conf.files.context]},
        streaming=True,
        split="train",
    )

    batched_dataset = dataset.batch(batch_size=conf.indexing.batch_size)
    return batched_dataset, num_lines


def add_batch(
    faiss_store: FAISS,
    embeddings: list[list[float]],
    batch: dict,
) -> None:

    min_size = min(len(batch["text_content"]), len(embeddings))
    if min_size == 0:
        return

    text_and_embeddings = zip(batch["text_content"][:min_size], embeddings[:min_size])
    metadatas = [
        {
            "source_name": source_name,
            "associated_dates": associated_date,
            "chunk_id": chunk_id,
        }
        for source_name, associated_date, chunk_id in zip(
            batch["source_name"][:min_size],
            batch["associated_dates"][:min_size],
            batch["chunk_id"][:min_size],
        )
    ]

    faiss_store.add_embeddings(
        text_embeddings=text_and_embeddings, metadatas=metadatas, ids=batch["chunk_id"]
    )


def batched_read_embeddings(
    conf: DictConfig, csv_reader: csv.reader
) -> list[list[float]]:
    embeddings = []
    for _ in range(conf.indexing.batch_size):
        try:
            row = next(csv_reader)
            embeddings.append([float(x) for x in row])
        except StopIteration:
            break
    return embeddings


def save(conf: DictConfig, faiss_store: FAISS) -> None:
    faiss_store.save_local(conf.files.index)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(conf: DictConfig) -> None:
    log.info("Config:\n%s", OmegaConf.to_yaml(conf))

    embedding_model = get_embedding_model(conf)
    faiss_store = get_faiss_store(conf, embedding_model)
    dataset, num_lines = get_dataset(conf)

    with open(conf.files.embeddings, "r", newline="") as f:
        csv_reader = csv.reader(f, delimiter="\t")

        for i, batch in tqdm(
            enumerate(dataset), total=num_lines // conf.indexing.batch_size
        ):
            embeddings = batched_read_embeddings(conf, csv_reader)
            add_batch(faiss_store, embeddings, batch)

    save(conf, faiss_store)


if __name__ == "__main__":
    main()
