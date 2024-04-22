"""
This script trains a GenSim Doc2Vec model on all documents in an ElasticSearch index.
The most similar docs are added as field `similar_docs` to the documents. This can be
used to locate versions of the same document or find other documents with semantically
similar content.

Doc2Vec trains not only word embeddings, but a separate document embedding for each
document. In training, the model uses embeddings of previous words and the document
embeddig to predict the next word. The embedding of an unseen document can be computed
using back-propagation using the trained model, by iterating over the words in the
document.

Because an embedding is created for each document, the memory requirements scale with
corpus size: an index with 10 times more documents needs about 10 times the RAM to
compute similar documents.

As of 28-12-2023, this script can now also predict annotations for unannotated
documents in the index given some Doc2Vec model. The algorithm is vectorized and batched
for the unannotated documents, but at the moment still requires the full matrix of
annotated document embeddings to be in memory. The algorithm could be optimized by also
batching the annotated documents and reconstructing the full matrix of cosines after
to compute the nearest neighbors. The classification algorithm runs several times,
once for each different annotation field.
The performance and quality is NOT benchmarked on real data.

Environment variables:
- `ES_USERNAME`, `ES_PASSWORD`, `ES_HOSTNAME`: ElasticSearch configuration
- `ES_INDEX`: index to train the model on, and add `similar_docs` field to. Tex is
    expected to be under field `full_text`
- `DOCUMENT_LIMIT`: maximum number of documents to process. Useful for testing.
- `TOP_N_SIMILAR_DOCS`: number of most similar document IDs to add to each document
- `MODEL_MAX_VOCAB_SIZE`: maximum number of words allowed in the vocabulary of the
    Doc2Vec model. 10 million words require about 1Gb in RAM, if the corpus contains
    this many words.
- `MODEL_EMBEDDING_SIZE`: dimensionality of the document and word embeddings. Important
    setting when running into memory issues, since the document embeddings are in a
    matrix of size [documents in corpus] x MODEL_EMBEDDING_SIZE
- `MLFLOW_TRACKING_URI`: URI to the service of the MLflow Tracker, to which the Doc2Vec
    model is registered.
- `ELASTICSEARCH_SLOW_INGEST`: set to "true" to pause 0.5 seconds after each bulk
    request to Elasticsearch to write document embeddings
- `BATCH_SIZE`: Batch size to use for KNN classification
- `ANNOTATION_CATEGORIES`: The annotation types and their possible values (configurable in values.yaml)
    that are to be used during the classification of unannotated documents
"""

import json
from typing import Iterable
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan, streaming_bulk
import logging
from os import getenv
import time
import os
import gensim
import mlflow
from mlflow.client import MlflowClient
import mlflow_gensim
import knn
import argparse

from utils import generate_es_results

import logging_config
from logging.config import dictConfig

dictConfig(logging_config.LOGGING_CONFIG)
logger = logging.getLogger()

ES_USERNAME = getenv("ES_USERNAME")
ES_PASSWORD = getenv("ES_PASSWORD")
ES_HOSTNAME = getenv("ES_HOSTNAME")
INDEX_NAME = getenv("ES_INDEX")
DOCUMENT_LIMIT = getenv("DOCUMENT_LIMIT", None)
if DOCUMENT_LIMIT is not None:
    DOCUMENT_LIMIT = int(DOCUMENT_LIMIT)
TOP_N_SIMILAR_DOCS = int(getenv("TOP_N_SIMILAR_DOCS", 10))
TOP_N_SIMILAR_DOCS_WITH_CONFIDENCE = int(
    getenv("TOP_N_SIMILAR_DOCS_WITH_CONFIDENCE", 50)
)
MODEL_MAX_VOCAB_SIZE = int(getenv("MODEL_MAX_VOCAB_SIZE", 10**7))
MODEL_EMBEDDING_SIZE = int(getenv("MODEL_EMBEDDING_SIZE", 150))
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_MODEL_NAME = "Doc2Vec"
ELASTICSEARCH_SLOW_INGEST = os.getenv("ELASTICSEARCH_SLOW_INGEST", "false") == "true"
BULK_SIZE = 500

BATCH_SIZE = int(os.getenv("BATCH_SIZE", BULK_SIZE))
ANNOTATION_CATEGORIES = os.getenv("ANNOTATION_CATEGORIES", None)

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


def generate_corpus(es: Elasticsearch, limit=None):
    """Yields documents from the ElasticSearch index while scanning it in batches. This
    means not all documents need to be in memory at the same time. The text of a
    document is assumed to be under field `full_text`. The document is pre-processed
    and yielded as a GenSim object containing the tokens.

    :param es: ElasticSearch client
    :type es: Elasticsearch
    :param limit: maximum number of documents to include in the corpus, useful for
    testing, defaults to None
    :type limit: int, optional
    :yield: TaggedDocument with the tokens of the full text and the ElasticSearch
    document ID as tag
    :rtype: gensim.models.doc2vec.TaggedDocument
    """
    # Iterate over all documents in the ElasticSearch index
    es_iterator = scan(
        client=es, query={"query": {"match_all": {}}}, index=INDEX_NAME, size=1000
    )
    for i, doc in enumerate(es_iterator):
        # Stop if the document limit is reached
        if limit is not None and i > limit:
            break
        try:
            # Get ID and full text from ElasticSearch document
            _id = doc["_id"]
            try:
                text = doc["_source"]["full_text"]
            except KeyError:
                logger.debug(
                    f"Document does not have `full_text` field, skipping document with ID {_id}"
                )
                continue
            # Preprocess text
            tokens = gensim.utils.simple_preprocess(text, min_len=2, max_len=25)
            yield gensim.models.doc2vec.TaggedDocument(tokens, [_id])
        except Exception as e:
            # Ignore any exception by skipping the document
            logger.warning(
                f"Unexpected exception raised in processing document, skipping document with ID {_id}",
                exc_info=e,
            )
            continue


class iterate_corpus:
    """This class makes sure that Generator `generate_corpus` can be called as Iterator.
    A Generator cannot be used more than once, but to train a model, it is needed to go
    over all data multiple times. This class can be used for that.
    All code could have been implemented as Iterator, but it would be less readible.
    """

    def __init__(self, es: Elasticsearch, limit=None):
        """Initialize the iterator.

        :param es: ElasticSearch client
        :type es: Elasticsearch
        :param limit: maximum number of documents to include in the corpus, useful for
        testing, defaults to None
        :type limit: int, optional
        """
        self.es = es
        self.limit = limit

    def __iter__(self):
        """Called to get an Iterator in its initial state.

        :return: Generator of the corpus of full texts
        :rtype: Generator[TaggedDocument, Any, None]
        """
        return generate_corpus(self.es, self.limit)


def generate_update_actions(model: gensim.models.doc2vec.Doc2Vec):
    """Yields all update actions to write similar documents to each document in the
    ElasticSearch index.
    For each document, the `TOP_N_SIMILAR_DOCS` are computed from all document
    embeddings using cosine similarity. The IDs of these documents should be written to
    the `similar_docs` field of the document.
    Similarly, the `TOP_N_SIMILAR_DOCS_WITH_CONFIDENCE` are computed and written to the
    `similar_docs_with_confidence` field including confidence scores.
    A generator for these update actions is useful, because many actions can be sent in
    bulk to the ElasticSearch server. This reduces network traffic and thus improves
    speed.

    :param model: Doc2Vec model trained on the documents in the index
    :type model: gensim.models.doc2vec.Doc2Vec
    :yield: update operation that can be sent to ElasticSearch
    :rtype: Dict
    """
    for _id in model.dv.index_to_key:
        similar_docs = model.dv.most_similar(
            _id, topn=max(TOP_N_SIMILAR_DOCS, TOP_N_SIMILAR_DOCS_WITH_CONFIDENCE)
        )
        # Get the top N most similar documents by cosine similarity of the embeddings
        # This list without confidence scores is added for compatibility with the
        # frontend
        similar_doc_ids = [sim[0] for sim in similar_docs[:TOP_N_SIMILAR_DOCS]]
        # Get the top N including confidence
        similar_doc_with_conf = [
            {"_id": sim[0], "confidence": sim[1]}
            for sim in similar_docs[:TOP_N_SIMILAR_DOCS_WITH_CONFIDENCE]
        ]
        yield {
            "_index": INDEX_NAME,
            "_op_type": "update",
            "_id": _id,
            "doc": {
                "similar_docs": similar_doc_ids,
                "similar_docs_with_confidence": similar_doc_with_conf,
            },
        }


def stream_update_actions(es: Elasticsearch, action_generator: Iterable):
    """Sends batches of elasticsearch actions to the index.
    Actions are streamed because this is faster than sending each action individually.
    Actions can be sent in bulk, reducing network traffic and thus improving speed.

    :param es: Elasticsearch client
    :type es: Elasticsearch
    :param action_generator: Generator containing the actions to be performed
    :type model: gensim.models.doc2vec.Doc2Vec
    """
    bulk_streamer = streaming_bulk(
        es, action_generator, chunk_size=BULK_SIZE, raise_on_error=False
    )
    for i, (success, info) in enumerate(bulk_streamer):
        if ELASTICSEARCH_SLOW_INGEST and ((i + 1) % BULK_SIZE == 0):
            # sleep after every bulk to prevent memory overflow in Elasticsearch on
            # development cluster
            time.sleep(0.5)
        if not success:
            logger.warning(
                f"Inserting similar document indices into ElasticSearch failed with info: {str(info)}"
            )


def store_model(model: gensim.models.doc2vec.Doc2Vec):
    """Registers a Doc2Vec model to MLflow.

    :param model: model to register
    :type model: gensim.models.doc2vec.Doc2Vec
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_MODEL_NAME)
    with mlflow.start_run():
        mlflow.log_params(
            {
                "document_limit": DOCUMENT_LIMIT,
                "vector_size": MODEL_EMBEDDING_SIZE,
                "min_count": 5,
                "max_vocab_size": MODEL_MAX_VOCAB_SIZE,
                "epochs": 20,
                "window": 5,
            }
        )
        mlflow.log_param("modelname", "gensimmodel")
        mlflow_gensim.log_model(
            gensim_model=model,
            artifact_path="model_artifact",
            registered_model_name=MLFLOW_MODEL_NAME,
        )
        logger.debug("Doc2Vec model registered by MLflow Tracker")

    # Stage model to production
    client = MlflowClient()
    version = client.get_latest_versions(MLFLOW_MODEL_NAME, ["None"])[0].version
    client.transition_model_version_stage(
        name=MLFLOW_MODEL_NAME,
        version=version,
        stage="Production",
        archive_existing_versions=True,
    )
    logger.debug("Newest Doc2Vec model staged to Production")


def classify_docs(annotation_field: str, model: gensim.models.doc2vec.Doc2Vec):
    """Generate annotations for unannotated documents in the index using a kNN majority-vote approach. `annotation_field` determines the set of possible annotations.
    `model` is the document embeddings model used.

    :param annotation_field: The annotation field in the front end that is to be filled in.
    :type annotation_field: str
    :param model: Embedding model used to compute nearest neighbors
    :type model: gensim.models.doc2vec.Doc2Vec
    """

    logger.info(
        f"Attempting unnannotated document clasification for category {annotation_field}"
    )

    annotated_query = {
        "query": {
            "bool": {
                "must_not": {
                    "regexp": {
                        "annotation_subject.values": {
                            "value": "AI_.*",
                            "flags": "ALL",
                            "case_insensitive": True,
                            "max_determinized_states": 10000,
                            "rewrite": "constant_score_blended",
                        }
                    }
                },
                "must": {"exists": {"field": f"{annotation_field}.values"}},
            }
        }
    }

    unannotated_query = {
        "query": {
            "bool": {"must_not": [{"exists": {"field": f"{annotation_field}.values"}}]}
        }
    }

    k = TOP_N_SIMILAR_DOCS
    index_name = INDEX_NAME
    bs = BATCH_SIZE
    size = 1000

    if bs < k:
        logger.info(
            f"Batch size {str(bs)} smaller than top-k {str(k)}; setting batch size to {str(k)}"
        )

    predicted_labels = knn.knn_generator(
        annotated_query,
        unannotated_query,
        es,
        index_name,
        size,
        annotation_field,
        bs,
        model,
        k,
    )
    logger.info(
        f"Finished unnannotated document clasification for category {annotation_field}"
    )

    unannotated_docs = generate_es_results(
        es, unannotated_query, index_name, size, model, vectorize=False
    )

    logger.info("Start updating ElasticSearch with the obtained predictions.")
    update_label_actions = generate_classification_updates(
        predicted_labels, unannotated_docs, annotation_field
    )
    stream_update_actions(es, update_label_actions)

    logger.info("Updated ElasticSearch")


def generate_classification_updates(
    predicted_labels, unannotated_docs, annotation_field
):
    for label, doc in zip(predicted_labels, unannotated_docs):
        doc_body = {
            annotation_field: {
                "values": [f"AI_{label}"],
            },
        }
        yield {
            "_index": INDEX_NAME,
            "_op_type": "update",
            "_id": doc,
            "doc": doc_body,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        type=str,
        default="train",
        help="Flag whether to train doc2vecor to classify documents",
        choices=["train", "classify", "gpt-embeddings"],
    )

    args = parser.parse_args()

    if args.mode == "train":
        logger.debug(
            f"Training Doc2Vec model with {MODEL_EMBEDDING_SIZE}-dimensional embeddings, a window of 5, and maximum vocabulary size of {MODEL_MAX_VOCAB_SIZE}"
        )
        model = gensim.models.doc2vec.Doc2Vec(
            documents=iterate_corpus(es=es, limit=DOCUMENT_LIMIT),
            vector_size=MODEL_EMBEDDING_SIZE,
            min_count=5,
            max_vocab_size=MODEL_MAX_VOCAB_SIZE,
            epochs=20,
            window=5,
            workers=4,
        )
        logger.debug(
            f"Doc2Vec model trained with {len(model.dv)} document embeddings and {len(model.wv)} word embeddings"
        )

        store_model(model)
        actions = generate_update_actions(model)
        stream_update_actions(es, actions)
        logger.debug("Updating documents completed")

    elif args.mode == "classify":
        logger.debug("Attempting document classification")
        if MLFLOW_MODEL_NAME:
            model = mlflow_gensim.load_model(f"models:/{MLFLOW_MODEL_NAME}/Production")
        else:
            logger.error("No Doc2Vec model found")

        if ANNOTATION_CATEGORIES:
            categories = json.loads(ANNOTATION_CATEGORIES)["annotation"]
        else:
            logger.error("No annotation categories found.")

        # These categories should not be predicted
        categories.pop("annotation_status", None)
        categories.pop("annotation_importance", None)
        categories.pop("annotation_dossier_user", None)

        for category in list(categories.keys()):
            classify_docs(category, model)
        logger.info("Annotation prediction complete")
    elif args.mode == "gpt-embeddings":
        logger.debug("Attempting GPT embeddings")
        from gpt_embeddings import calc_and_save_embeddings

        calc_and_save_embeddings()

    else:
        logger.error("Invalid `mode` argument.")
