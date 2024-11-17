# # Files used:
# conf.files.context
# conf.files.sparse_index

import json
import logging
from datetime import datetime
import os

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, save_npz

log = logging.getLogger(__name__)


def save_index(
    conf: DictConfig, idf: np.ndarray, tfidf_matrix: csr_matrix, tfidf_df: pd.DataFrame
) -> None:
    os.makedirs(conf.files.sparse_index, exist_ok=True)
    save_npz(
        os.path.join(conf.files.sparse_index, "tfidf_matrix.npz"),
        tfidf_matrix,
        compressed=True,
    )

    metadata = {
        "idf": idf.tolist(),
        "chunk_ids": tfidf_df.index.tolist(),
        "features": tfidf_df.columns.tolist(),
    }

    with open(os.path.join(conf.files.sparse_index, "metadata.json"), "w") as f:
        json.dump(metadata, f)


def load_documents(conf: DictConfig) -> tuple[list[int], list[str]]:
    document_ids = []
    documents = []
    with open(conf.files.context, "r") as f:
        for line in f:
            data = json.loads(line)
            document_ids.append(int(data["chunk_id"]))

            extracted_entities = [x["entity"] for x in data["extracted_entities"]]
            extracted_dates = [
                datetime.fromisoformat(x).strftime("%Y-%m-%d")
                for x in data["extracted_dates"]
            ]
            associated_dates = data["associated_dates"]

            all_entities = extracted_entities + extracted_dates + associated_dates
            all_entities = [x.strip().replace("\t", " ") for x in all_entities]
            all_entities = [x for x in all_entities if len(x) > 0]

            documents.append("\t".join(all_entities))

    return document_ids, documents


def build_sparse_index(
    document_ids: list[int], documents: list[str]
) -> tuple[np.ndarray, csr_matrix, pd.DataFrame]:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        tokenizer=lambda x: x.strip().split("\t"),
        use_idf=True,
        smooth_idf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(documents)
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        index=document_ids,
        columns=vectorizer.get_feature_names_out(),
    )

    print("TF-IDF matrix shape:", tfidf_df.shape)
    return vectorizer.idf_, tfidf_matrix, tfidf_df


@hydra.main(version_base=None, config_path="conf", config_name="sparse_index")
def main(conf: DictConfig) -> None:
    log.info("Config:\n%s", OmegaConf.to_yaml(conf))

    document_ids, documents = load_documents(conf)
    idf, tfidf_matrix, tfidf_df = build_sparse_index(document_ids, documents)
    save_index(conf, idf, tfidf_matrix, tfidf_df)


if __name__ == "__main__":
    main()
