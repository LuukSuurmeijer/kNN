from pymilvus import (
    Collection,
    connections,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
    MilvusException,
)

from os import getenv
import openai
from openai.error import RateLimitError, InvalidRequestError
from elasticsearch import Elasticsearch
from time import sleep
from elasticsearch.helpers import scan
from typing import Tuple

import logging
import logging_config
from logging.config import dictConfig

dictConfig(logging_config.LOGGING_CONFIG)
logger = logging.getLogger()


VECTOR_DATABASE_URL = getenv("VECTOR_DATABASE_URL", "ally-dev-milvus-proxy:19530")

ES_USERNAME = getenv("ES_USERNAME")
ES_PASSWORD = getenv("ES_PASSWORD")
ES_HOSTNAME = getenv("ES_HOSTNAME")
INDEX_NAME = getenv("COLLECTION_ES_INDEX_NAME", "")
COLLECTION_NAME = INDEX_NAME.replace("-", "")
PRICE_PER_1K_TOKEN = 0.000093  # in euros (feb, 2024)
OVERWRITE_VECTORS = getenv("OVERWRITE_VECTORS", "False") == "True"

MAX_DOCS = float(getenv("MAX_DOCS", 5e3))  # 5k docs
MAX_TOKENS = float(getenv("MAX_TOKENS", 10e6))  # 10m tokens (about a euro)


openai.api_type = "azure"
openai.api_key = getenv("GPT_TOKEN_1")
openai.api_base = "https://ally-gpt.openai.azure.com"
openai.api_version = "2023-05-15"


# Connect to ElasticSearch
logger.debug(f"Connecting to ElasticSearch with hostname '{ES_HOSTNAME}'")
es = Elasticsearch(
    hosts=ES_HOSTNAME,
    http_auth=(ES_USERNAME, ES_PASSWORD),
    verify_certs=False,
    timeout=60,
    max_retries=10,
    retry_on_timeout=True,
)
if not es:
    logger.error("Elastic connection Error!")


def _calc_embedding(doc: str) -> Tuple[list, int]:
    try:
        response = openai.Embedding.create(input=doc, engine="ally-embeddings")
    except RateLimitError as e:
        logger.info(e)
        logger.info("Sleeping 4 seconds, then retrying")
        sleep(4)
        return _calc_embedding(doc)
    except InvalidRequestError as e:
        logger.info(e)
        logger.info("Can't calculate embeddings for this document. Skipping")
        return False, False
    logger.debug(response.usage.prompt_tokens)
    embeddings = response["data"][0]["embedding"]
    return embeddings, response.usage.prompt_tokens


def _create_collection(index):
    collection = Collection(
        index,
        CollectionSchema(
            fields=[
                FieldSchema("full_text", DataType.FLOAT_VECTOR, dim=1536),
                FieldSchema("id", DataType.VARCHAR, max_length=100, is_primary=True),
            ],
        ),
    )
    logger.info(f"Collection {index} created")
    return collection


def _load_or_create_collection():
    logger.debug(f"Loading or creating collection {COLLECTION_NAME}")
    collection_loaded = False
    if utility.has_collection(COLLECTION_NAME):
        logger.info(f"Collection {COLLECTION_NAME} found")
        collection = Collection(COLLECTION_NAME)
        if len(collection.indexes) > 0:
            try:
                collection.load(timeout=10)
                collection_loaded = True
                logger.info(f"Collection {COLLECTION_NAME} loaded")
            except MilvusException as e:
                logger.info(e)
                logger.info("Creating new collection, as this one failed to load")
                collection.drop()
                collection = _create_collection(COLLECTION_NAME)

    else:
        collection = _create_collection(COLLECTION_NAME)
    return collection, collection_loaded


def _check_limit_exceeded(par_count, token_count):
    if (MAX_DOCS != -1 and par_count >= MAX_DOCS) or (
        MAX_TOKENS != -1 and token_count >= MAX_TOKENS
    ):
        logger.debug(
            f"Max tokens {MAX_TOKENS} or max docs {MAX_DOCS} reached. Quitting"
        )
        return True  # limit exceeded


def calc_and_save_embeddings():
    connections.connect(address=VECTOR_DATABASE_URL, db_name="default")
    collection, collection_loaded = _load_or_create_collection()
    logger.info("Gathering documents with paragraphs to ingest")
    es_iterator = scan(
        client=es, query={"query": {"match_all": {}}}, index=INDEX_NAME, size=5
    )
    par_count = 0
    token_count = 0
    for doc_i, doc in enumerate(es_iterator):
        if _check_limit_exceeded(par_count, token_count):
            break
        logger.debug(f"Inserting document {doc_i} into collection")
        if "paragraphs" in doc["_source"]:
            for index, paragraph in enumerate(doc["_source"]["paragraphs"]):
                if _check_limit_exceeded(par_count, token_count):
                    break
                par_id = doc["_id"] + "_par_" + str(index)
                if (  # if we want to overwrite each vector, no need to check whether this vector already exists
                    OVERWRITE_VECTORS
                    or not collection_loaded  # we can only check whether a vector is in the db when the index is loaded
                    or len(collection.query(f'id == "{par_id}"'))
                    == 0  # check whether this vector is in the db by performing a query
                ):
                    embedding, used_tokens_for_embedding = _calc_embedding(
                        paragraph["text"]
                    )
                    if not embedding:
                        continue
                    token_count += used_tokens_for_embedding
                    collection.insert(
                        {
                            "full_text": embedding,
                            "id": par_id,
                        }
                    )
                    par_count += 1
                    logger.debug(f"Paragraph {index} added to collection")
                else:
                    logger.debug(f"Paragraph {index} already in collection")

    logger.info(f"Got embeddings for {par_count} paragraphs")
    logger.info(
        f"Used {token_count} tokens. (This equates to {(token_count/1000) * PRICE_PER_1K_TOKEN} euros)"
    )

    index_params = {
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 1024},
    }

    collection.create_index(field_name="full_text", index_params=index_params)

    logger.info("Index Created")

    collection.load()
    logger.info("Index Loaded")
    logger.debug(collection.describe())
    logger.info("Shutting down")
