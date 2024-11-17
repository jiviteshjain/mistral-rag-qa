# ## Files used:
# conf.files.index
# conf.files.sparse_index
# conf.files.questions_jsonl
# conf.files.answers_jsonl
# conf.files.answers_txt
# conf.files.questions_txt


import gc
import io
import itertools
import json
import logging
import os
import re
from typing import Any
from typing import Iterable

import evaluate
import hydra
import numpy as np
import pandas as pd
import spacy
import torch
import wandb
from datasets import load_dataset, IterableDataset
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors.cross_encoder_rerank import (
    CrossEncoderReranker,
)
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableSequence,
    RunnableMap,
)
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from omegaconf import DictConfig, OmegaConf
from peft import AutoPeftModelForCausalLM
from scipy.sparse import csr_matrix, load_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from span_marker import SpanMarkerModel
from tqdm.auto import tqdm
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM

log = logging.getLogger(__name__)

HF_METRICS = {
    "bleu1": (evaluate.load("bleu"), "bleu"),
    "rouge1": (evaluate.load("rouge"), "rouge1"),
    "bertscore": (
        evaluate.load("bertscore", device="cuda:1", use_fast_tokenizer=True),
        ["f1", "precision", "recall"],
    ),
    "squad": (evaluate.load("squad"), ["f1", "exact_match"]),
}

PROMPT = """You are a question answering assistant for the city of Pittsburgh, PA. Based on the following retrieved contexts, answer the question that follows in at most 10 words. Here are some examples:

Q: Who is Pittsburgh named after?
A: William Pitt

Q: What famous machine learning venue had its first conference in Pittsburgh in 1980?
A: ICML

Q: What musical artist is performing at PPG Arena on October 13?
A: Billie Eilish

### Context
{context}

### Question
{question}

### Answer
"""

REGEX_PATTERN = re.compile(
    f"### Context\n(?P<context>.*?)\n\n### Question\n(?P<question>.*?)\n\n### Answer\n(?P<answer>.*)",
    flags=re.DOTALL,
)

HYDE_PROMPT = """You are a question answering search assistant for the city of Pittsburgh, PA. For the question below, generate a hypothetical 300 words paragraph about Pittsburgh which answers the question. Phrase the paragaph to look like text scraped from a website. Only output the paragraph. Here is an example:

Question: How did the unemployment rate for black Pittsburghers change from 2016 to 2017?

Paragraph: black and white Pittsburghers in 2013 and that the disparity has been growing (increased labor force participation for white Pittsburghers and decreased la bor force participation for black Pittsburghers) through 2016, with a small uptick among black Pittsburghers and downtick among white Pittsburghers in 2017. The increase in labor force participation for black Pittsburghers and decrease in labor force participation for white Pittsburghers between 2016 and 2017 decreased the existing disparity and changed the equality score from 73 to 76 (a change of 3). In general, Pittsburghs overall labor participation rate is slightly higher than that of the United State s (63.6 percent in Pittsburgh compared with 63.1 percent in the United States in 2016).49 93 Labor Force Participation in Pittsburgh, 2013 2017 SOURCE : ACS 1-year estimates , 20132017 Data source ACS 1-year estimates, 2016 and 2017 Indicator 32: Unemployment 2018 equality score : 31 Indicator definition Ratio of blacks' and whites' unemployment rates Reporting year r esults 2017 Black : 11.4% (6,600 people) White : 5.4% (9,615 people) Black -to-white ratio 2.111 , score 40 2018 Black : 12.9% (6,913 people ) White : 3.7% (6,820 people ) Black -to-white ratio 3.486 , score 31 Changes from reporting year 2017 to reporting year 2018 Black : 1.5% White : 1.7% Change in equality score : 9 Geography City Description of results and context The ACS tracks unemployment in cities by race. The unemployment rate does not include those individuals who are not currently looking for work or have left the labor force. The unemployment rate for black Pittsburghers (12.9 percent ) was more than three times the rate of unemployment for white Pittsbur ghers ( 3.7 percent ) in 2017 (the most recent year for which the data were available) . The unemployment rate for black Pittsburghers increased by 1.5 percent and decreased for white Pittsburghers by 1.7 percent from 2016 levels, widening the existing gap an d decreasing the Equality Score to 31 from 40 (a change of 9). Information was available from the Census Bureau on the margins of error associated with these estimates of unemployment (see below). Statistical testing revealed that changes in rates between 2016 and 2017 were not statistically significant at a 95 -percent confidence threshold, so we assume that the obs erved change score is also not statistically significant.

### Question
{question}

### Paragraph
"""

HYDE_REGEX_PATTERN = re.compile(
    f"### Question\n(?P<question>.*?)\n\n### Paragraph\n(?P<document>.*)",
    flags=re.DOTALL,
)

NO_OUTPUT_STR = "No relevant context."

SUMMARY_PROMPT = """
Given the following question and context, extract any part of the context *AS IS* that is relevant to answer the question. If none of the context is relevant return "{no_output_str}". Extract AS CONCISELY as possible. Extract only 3 sentences.

Remember, DO NOT edit the extracted parts of the context. 

### Question
{question}

### Context
>>>
{context}
>>>

### Concisely extracted relevant parts
"""

SUMMARY_REGEX_PATTERN = re.compile(
    "### Concisely extracted relevant parts\n(?P<summary>.*)"
)


class SparseIndex:
    def __init__(
        self,
        idf: np.ndarray,
        tfidf_matrix: csr_matrix,
        tfidf_df: pd.DataFrame,
        model_id: str,
        device: str,
    ):
        self.tfidf_matrix = tfidf_matrix
        self.tfidf_df = tfidf_df

        self.vectorizer = TfidfVectorizer(
            vocabulary=tfidf_df.columns,
            lowercase=True,
            tokenizer=lambda x: x.strip().split("\t"),
            use_idf=True,
            smooth_idf=True,
        )
        self.vectorizer.idf_ = idf

        self.model = SpanMarkerModel.from_pretrained(model_id)
        self.model.to(device)

        self.date_finder = spacy.blank("en")
        self.date_finder.add_pipe("find_dates")

    @classmethod
    def load(cls, index_path: str, model_id: str, device: str) -> "SparseIndex":
        tfidf_matrix = load_npz(os.path.join(index_path, "tfidf_matrix.npz"))

        with open(os.path.join(index_path, "metadata.json"), "r") as f:
            metadata = json.load(f)

        idf = np.array(metadata["idf"])

        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            index=[int(x) for x in metadata["chunk_ids"]],
            columns=metadata["features"],
        )

        return cls(idf, tfidf_matrix, tfidf_df, model_id, device)

    def get_entities(self, text: str) -> list[dict[str, str]]:
        entities = self.model.predict(text)
        return [x["span"].lower().strip() for x in entities]

    def get_dates(self, text: str) -> list[str]:
        doc = self.date_finder(text)
        dates = []
        for ent in doc.ents:
            if ent.label_ == "DATE" and ent._.date is not None:
                dates.append(ent._.date.strftime("%Y-%m-%d"))
        return [x.strip() for x in dates]

    def get_relevant_document_ids(self, question: str, top_k: int) -> list[int]:
        entities = self.get_entities(question)
        dates = self.get_dates(question)

        query_str = "\t".join(entities + dates)

        # TODO(jiviteshjain): This should be batched.
        query_vector = self.vectorizer.transform([query_str])
        cosine_similarities = cosine_similarity(
            query_vector, self.tfidf_matrix
        ).flatten()

        top_k_indices = np.argsort(cosine_similarities)[-top_k:][::-1]
        return self.tfidf_df.index[top_k_indices].tolist()


def parse_regex(text: str) -> dict[str, str]:
    match = REGEX_PATTERN.search(text)
    if match is None:
        return {"context": "", "question": "", "answer": ""}
    else:
        return match.groupdict()


def get_embedding_model(conf: DictConfig) -> HuggingFaceEmbeddings:
    embedding_model = HuggingFaceEmbeddings(
        model_name=conf.embeddings.model,
        model_kwargs={
            "trust_remote_code": True,
            "device": conf.embeddings.device,
            "tokenizer_kwargs": {"padding": True, "truncation": True},
        },
    )

    return embedding_model


def load_faiss_store(conf: DictConfig, embedding_model: HuggingFaceEmbeddings) -> FAISS:
    return FAISS.load_local(
        conf.files.index,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True,
    )


def attach_reranker(
    conf: DictConfig, faiss_retriever: VectorStoreRetriever
) -> ContextualCompressionRetriever:
    reranker_model = HuggingFaceCrossEncoder(
        model_name=conf.rag.reranking.model,
        model_kwargs={
            "trust_remote_code": True,
            "device": conf.rag.reranking.device,
        },
    )
    reranker = CrossEncoderReranker(
        model=reranker_model, top_n=conf.rag.reranking.reranking_k
    )

    return ContextualCompressionRetriever(
        base_retriever=faiss_retriever, base_compressor=reranker
    )


def get_reader_model(conf: DictConfig) -> HuggingFacePipeline:
    model = AutoPeftModelForCausalLM.from_pretrained(
        conf.rag.reader.model,
        load_in_4bit=conf.rag.reader.load_in_4bit,
    )
    model.to(conf.rag.reader.device)
    # Not specifying the torch_dtype

    tokenizer = AutoTokenizer.from_pretrained(conf.rag.reader.model)
    tokenizer.model_max_length = conf.rag.reader.max_seq_length

    hf_pipeline = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=conf.rag.reader.max_new_tokens,
        max_length=conf.rag.reader.max_seq_length,
        batch_size=conf.rag.reader.batch_size,
    )

    return HuggingFacePipeline(pipeline=hf_pipeline)


def get_hyde_model(conf: DictConfig) -> pipeline:
    model = AutoModelForCausalLM.from_pretrained(
        conf.rag.hyde.model, load_in_4bit=conf.rag.hyde.load_in_4bit
    )

    tokenizer = AutoTokenizer.from_pretrained(conf.rag.hyde.model)
    tokenizer.model_max_length = conf.rag.hyde.max_seq_length

    hf_pipeline = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=conf.rag.hyde.max_new_tokens,
    )

    return hf_pipeline


def get_prompt(conf: DictConfig) -> PromptTemplate:
    return PromptTemplate.from_template(PROMPT)


def parse_hyde_regex(text: str) -> dict[str, str]:
    match = HYDE_REGEX_PATTERN.search(text)
    if match is None:
        return {"question": "", "document": ""}
    else:
        return match.groupdict()


def parse_summary_regex(text: str) -> str:
    text_processed = re.sub(NO_OUTPUT_STR, "", text)
    match = SUMMARY_REGEX_PATTERN.search(text_processed)
    if match is None:
        return ""
    else:
        return match.groupdict()["summary"]


def format_docs(docs: Iterable[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def batched(iterable, batch_size):
    it = iter(iterable)
    for batch in iter(lambda: list(itertools.islice(it, batch_size)), []):
        yield batch


def load_batched_questions_jsonl(
    conf,
) -> IterableDataset:

    with open(conf.files.questions_jsonl, "r") as f:
        num_lines = sum(1 for _ in f)

    dataset = load_dataset(
        "json",
        data_files={"train": [conf.files.questions_jsonl]},
        streaming=True,
        split="train",
    )

    batched_dataset = dataset.batch(batch_size=conf.rag.reader.batch_size)
    return batched_dataset, num_lines


# Fuck these different formats.
def collate_useful_data(
    chain_output: list[dict[str, Any]], batch: dict[list[Any]]
) -> tuple[
    list[int],
    list[int],
    list[str],
    list[str],
    list[list[int]],
    list[str],
    list[str],
    list[str],
]:
    retrieved_doc_ids = []

    raw_outputs = []
    retrieved_contexts = []
    generated_answers = []

    for row in chain_output:
        retrieved_doc_ids.append(
            [int(doc.metadata["chunk_id"]) for doc in row["retrieved_docs"]]
        )

        raw_outputs.append(row["raw_output"])

        parsed_output = parse_regex(row["raw_output"])
        retrieved_contexts.append(parsed_output["context"])
        generated_answers.append(parsed_output["answer"])

    gt_doc_ids = [int(x) for x in batch["chunk_id"]]
    gt_answers = batch["gt_answer"]
    question_ids = [int(x) for x in batch["question_id"]]
    questions = batch["question"]

    return (
        question_ids,
        gt_doc_ids,
        questions,
        gt_answers,
        retrieved_doc_ids,
        raw_outputs,
        retrieved_contexts,
        generated_answers,
    )


def compute_retrieval_metrics(
    gt_doc_ids: list[int], retrieved_doc_ids: list[list[int]]
):
    gt_doc_ids = np.array(gt_doc_ids)
    retrieved_doc_ids = np.array(retrieved_doc_ids)

    # Recall
    matches = retrieved_doc_ids == gt_doc_ids[:, None]
    recall = np.any(matches, axis=1).astype(np.int32).mean()

    # MRR
    ranks = np.where(matches, np.arange(matches.shape[1]), matches.shape[1])
    first_match_rank = np.min(ranks, axis=1)
    reciprocal_ranks = np.where(
        first_match_rank < matches.shape[1], 1 / (first_match_rank + 1), 0
    )
    mrr = np.mean(reciprocal_ranks)

    return recall, mrr


def compute_generation_metrics(
    gt_answers: list[str], generated_answers: list[str]
) -> dict[str, float]:
    references = [[x] for x in gt_answers]

    metrics = {}
    for metric_name, (metric, metric_key) in HF_METRICS.items():
        if metric_name.startswith("bertscore"):
            scores = metric.compute(
                predictions=generated_answers, references=references, lang="en"
            )
            for k in metric_key:
                metrics[metric_name + "_" + k] = np.array(
                    scores[k], dtype=np.float32
                ).mean()

        elif metric_name.startswith("squad"):
            preds = [
                {"prediction_text": p, "id": str(i)}
                for i, p in enumerate(generated_answers)
            ]
            refs = [
                {"answers": {"answer_start": [0], "text": r}, "id": str(i)}
                for i, r in enumerate(references)
            ]
            scores = metric.compute(predictions=preds, references=refs)
            for k in metric_key:
                metrics[metric_name + "_" + k] = scores[k]
        elif metric_name.startswith("bleu"):
            scores = metric.compute(
                predictions=generated_answers,
                references=references,
                max_order=1,
                smooth=True,
            )
            metrics[metric_name] = scores[metric_key]
        else:
            scores = metric.compute(
                predictions=generated_answers, references=references
            )
            metrics[metric_name] = scores[metric_key]

    return metrics


def save_outputs(
    f: io.TextIOWrapper,
    question_ids: list[int],
    gt_doc_ids: list[int],
    questions: list[str],
    gt_answers: list[str],
    retrieved_doc_ids: list[list[int]],
    raw_outputs: list[str],
    retrieved_contexts: list[str],
    generated_answers: list[str],
) -> None:
    for (
        question_id,
        gt_doc_id,
        question,
        gt_answer,
        retrieved_doc_id,
        raw_output,
        retrieved_context,
        generated_answer,
    ) in zip(
        question_ids,
        gt_doc_ids,
        questions,
        gt_answers,
        retrieved_doc_ids,
        raw_outputs,
        retrieved_contexts,
        generated_answers,
    ):
        f.write(
            json.dumps(
                {
                    "question_id": question_id,
                    "gt_chunk_id": gt_doc_id,
                    "question": question,
                    "gt_answer": gt_answer,
                    "retrieved_chunk_ids": retrieved_doc_id,  # This is a list of ints.
                    "raw_output": raw_output,
                    "retrieved_context": retrieved_context,
                    "generated_answer": generated_answer,
                }
            )
            + "\n"
        )


def retrieve_docs_batched(
    conf: DictConfig,
    retriever: ContextualCompressionRetriever | VectorStoreRetriever,
    sparse_index: SparseIndex,
    hyde_pipeline: pipeline,
    summary_pipeline: pipeline,
    questions: list[str],
) -> list[dict[str, Any]]:
    hyde_questions = get_questions_with_hyde(conf, hyde_pipeline, questions)

    retrieved_docs = []
    for question, hyde_question in zip(questions, hyde_questions):
        docs = retrieve_docs(
            conf, retriever, sparse_index, summary_pipeline, question, hyde_question
        )
        retrieved_docs.append({"question": question, "docs": docs})

    return retrieved_docs


def get_by_id(vectorstore: FAISS, doc_ids: list[int]) -> list[Document]:
    index_to_docstore_id = vectorstore.index_to_docstore_id
    docstore = vectorstore.docstore
    return [docstore._dict[x] for x in doc_ids]


def get_questions_with_hyde(
    conf: DictConfig,
    hyde_pipeline: pipeline,
    questions: list[str],
) -> list[str]:
    if not conf.rag.hyde.enabled:
        return questions

    prompts = [HYDE_PROMPT.format(question=question) for question in questions]

    outputs: str = hyde_pipeline(prompts, num_return_sequences=1)
    hyde_docs = [
        parse_hyde_regex(output[0]["generated_text"])["document"] for output in outputs
    ]

    return [question + "\n" + doc for question, doc in zip(questions, hyde_docs)]


def summarize(
    conf: DictConfig, question: str, docs: list[Document], summary_pipeline: pipeline
) -> list[Document]:
    if not conf.rag.summary.enabled:
        return docs

    raw_text = [doc.page_content for doc in docs]
    prompts = [
        SUMMARY_PROMPT.format(no_output_str=NO_OUTPUT_STR, question=question, context=t)
        for t in raw_text
    ]
    outputs: list[str] = summary_pipeline(prompts, num_return_sequences=1)
    summaries = [parse_summary_regex(output[0]["generated_text"]) for output in outputs]
    return [
        Document(page_content=s, metadata=d.metadata) for s, d in zip(summaries, docs)
    ]


def retrieve_docs(
    conf: DictConfig,
    retriever: ContextualCompressionRetriever | VectorStoreRetriever,
    sparse_index: SparseIndex,
    summary_pipeline: pipeline,
    question: str,
    hyde_question: str,
) -> list[Document]:

    dense_k = conf.rag.dense_k
    sparse_k = conf.rag.sparse.sparse_k

    final_docs = None

    if not conf.rag.sparse.enabled:
        final_docs = retriever.get_relevant_documents(hyde_question, k=dense_k)

    elif not conf.rag.reranking.enabled:
        # Retriever is a VectorStoreRetriever.
        sparse_doc_ids = sparse_index.get_relevant_document_ids(question, sparse_k)
        sparse_docs = get_by_id(retriever.vectorstore, sparse_doc_ids)
        dense_docs = retriever.get_relevant_documents(hyde_question, k=dense_k)

        final_docs = sparse_docs + dense_docs

    else:
        # Retriever is a ContextualCompressionRetriever.
        sparse_doc_ids = sparse_index.get_relevant_document_ids(question, sparse_k)
        sparse_docs = get_by_id(retriever.base_retriever.vectorstore, sparse_doc_ids)
        dense_docs = retriever.base_retriever.get_relevant_documents(
            hyde_question, k=dense_k
        )

        selected_docs = retriever.base_compressor.compress_documents(
            documents=sparse_docs + dense_docs, query=question
        )

        final_docs = selected_docs

    summary_docs = summarize(conf, question, final_docs, summary_pipeline)

    return summary_docs


def get_rag_chain(
    conf: DictConfig,
    reader: HuggingFacePipeline,
    prompt: PromptTemplate,
) -> RunnableSequence:

    return (
        RunnableMap(
            {
                "retrieved_docs": lambda x: x["docs"],
                "context": lambda x: format_docs(x["docs"]),
                "question": lambda x: x["question"],
            }
        )
        | RunnableMap(
            {
                "retrieved_docs": lambda x: x["retrieved_docs"],
                "raw_output": prompt | reader,
            }
        )
        | RunnableMap(
            {
                "retrieved_docs": lambda x: x["retrieved_docs"],
                "raw_output": lambda x: x["raw_output"],
            }
        )
    )


def run_inference(
    conf: DictConfig,
    retriever: VectorStoreRetriever,
    sparse_index: SparseIndex,
    hyde_pipeline: pipeline,
    summary_pipeline: pipeline,
    rag_chain: RunnableSequence,
) -> None:
    with open(conf.files.questions_txt, "r") as f:
        questions = [line.strip() for line in f.readlines()]

    os.makedirs(os.path.dirname(conf.files.answers_txt), exist_ok=True)
    with open(conf.files.answers_txt, "w") as f:
        for i, batch in tqdm(
            enumerate(batched(questions, conf.rag.reader.batch_size)),
            total=math.ceil(len(questions) // conf.rag.reader.batch_size),
        ):
            docs = retrieve_docs_batched(
                conf, retriever, sparse_index, hyde_pipeline, summary_pipeline, batch
            )
            chain_output = rag_chain.batch(docs)
            batch_answers = [
                parse_regex(row["raw_output"])["answer"] for row in chain_output
            ]

            for ans in batch_answers:
                f.write(" ".join(ans.strip().split()) + "\n")
                if conf.wandb.enabled:
                    wandb.log({"answer": ans})
            f.flush()


def run_validation(
    conf: DictConfig,
    retriever: VectorStoreRetriever,
    sparse_index: SparseIndex,
    hyde_pipeline: pipeline,
    summary_pipeline: pipeline,
    rag_chain: RunnableSequence,
) -> None:
    dataset, num_lines = load_batched_questions_jsonl(conf)

    with open(conf.files.answers_jsonl, "w") as f:
        for batch in tqdm(dataset, total=num_lines // conf.rag.reader.batch_size):
            docs = retrieve_docs_batched(
                conf,
                retriever,
                sparse_index,
                hyde_pipeline,
                summary_pipeline,
                batch["question"],
            )
            chain_output = rag_chain.batch(docs)

            (
                question_ids,
                gt_doc_ids,
                questions,
                gt_answers,
                retrieved_doc_ids,
                raw_outputs,
                retrieved_contexts,
                generated_answers,
            ) = collate_useful_data(chain_output, batch)

            recall, mrr = compute_retrieval_metrics(gt_doc_ids, retrieved_doc_ids)
            generation_metrics = compute_generation_metrics(
                gt_answers, generated_answers
            )

            if conf.wandb.enabled:
                wandb.log(
                    {
                        "recall": recall,
                        "mrr": mrr,
                        **generation_metrics,
                    }
                )

            save_outputs(
                f,
                question_ids,
                gt_doc_ids,
                questions,
                gt_answers,
                retrieved_doc_ids,
                raw_outputs,
                retrieved_contexts,
                generated_answers,
            )


@hydra.main(version_base=None, config_path="conf", config_name="validation")
def main(conf: DictConfig) -> None:
    if conf.wandb.enabled:
        wandb.login(key=os.environ["WANDB_KEY"])

        os.environ["LANGCHAIN_WANDB_TRACING"] = "true"

        wandb.init(
            name=conf.run_name,
            project=conf.wandb.project,
            entity=conf.wandb.entity,
            config=OmegaConf.to_container(conf),
            resume="allow",
        )

    log.info("Loading index")

    embedding_model = get_embedding_model(conf)
    faiss_store = load_faiss_store(conf, embedding_model)
    retriever = faiss_store.as_retriever(search_kwargs={"k": conf.rag.dense_k})
    if conf.rag.reranking.enabled:
        retriever = attach_reranker(conf, retriever)

    sparse_index = SparseIndex.load(
        conf.files.sparse_index, conf.rag.sparse.model, conf.rag.sparse.device
    )

    if conf.rag.hyde.enabled or conf.rag.summary.enabled:
        hyde_pipeline = get_hyde_model(conf)
    else:
        hyde_pipeline = None

    log.info("Building RAG pipeline")
    reader_model = get_reader_model(conf)
    prompt = get_prompt(conf)
    rag_chain = get_rag_chain(conf, reader_model, prompt)

    log.info("Running RAG pipeline")
    if conf.rag.mode == "validation":
        run_validation(
            conf, retriever, sparse_index, hyde_pipeline, hyde_pipeline, rag_chain
        )
    elif conf.rag.mode == "inference":
        run_inference(
            conf, retriever, sparse_index, hyde_pipeline, hyde_pipeline, rag_chain
        )
    else:
        raise ValueError(f"Invalid rag mode: {conf.rag.mode}")


if __name__ == "__main__":
    # Use if huggingface resources are private.
    # hf_login(token=os.environ["HF_TOKEN"])

    gc.collect()
    torch.cuda.empty_cache()

    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    main()
