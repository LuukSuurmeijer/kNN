import gensim

from jiwer.process import _apply_transform, _word2char
from jiwer.transformations import wer_default
from typing import List
import rapidfuzz
from rapidfuzz.distance import Opcodes

from elasticsearch.helpers import scan
from elasticsearch import Elasticsearch

import logging
import logging_config
from logging.config import dictConfig

dictConfig(logging_config.LOGGING_CONFIG)
logger = logging.getLogger()


def metric_word_chunk_overlap_ratio(
    tokens1: List[str], tokens2: List[str], min_chunk_size=5
) -> float:
    """Computes a metric of overlap between two documents. The metric lies between 0
    (no overlap) and 1 (identical), and is symmetric.

    The metric uses the operations that underly computing the Levenshtein distance,
    which transforms one text into another by using additions, substitutions, and
    skipping equal chunks of words.

    This algorithm filters all equal chunks of at least length 5 (or `min_chunk_size`),
    and then computes their word count divided by the word count of the smallest text.
    The chunk size is filtered, because otherwise it would be possible to get a
    relatively high score with a text like "the the the the the [etc]".

    :param tokens1: list of words of the first document
    :type tokens1: List[str]
    :param tokens2: list of words of the second document
    :type tokens2: List[str]
    :param min_chunk_size: equal word chunks are filtered based on this minimal length,
        defaults to 5
    :type min_chunk_size: int, optional
    :return: overlap ratio between 0 and 1
    :rtype: float
    """
    # Convert string tokens to token IDs
    chars1, chars2 = _word2char([tokens1], [tokens2])

    # Get the required edit operations to transform reference into hypothesis
    edit_ops = rapidfuzz.distance.Levenshtein.editops(chars1[0], chars2[0])

    # Compute number of words in overlapping chunks
    n_words_overlap = 0
    for op in Opcodes.from_editops(edit_ops):
        if op.tag == "equal":
            chunk_size = op.src_end - op.src_start
            if chunk_size >= min_chunk_size:
                n_words_overlap += chunk_size

    # Compute the lenght of the smallest text
    min_text_length = min(len(tokens1), len(tokens2))
    return n_words_overlap / min_text_length


def select_similar_docs(
    documents: List[str], threshold=0.1, verbose=False
) -> List[int]:
    """Select the documents that are similar to the first document in a list, based on
    some similarity metric, and on the (imposed) transitive property of similarity: if
    doc X is similar to Y and doc Y is similar to doc Z, then we say that X and Z are
    also similar, even if the metric does not say that.

    The algorithm searches all paths from the original document (at index 0) to the
    other documents (at the higher indices). In the direction of searching, we call the
    first node `subject` and the second `object`. So initially, the original document is
    `subject` and all other documents are tested as `object`. If some of those are
    similar, they will be `subject` in the next iteration, and all documents not (yet)
    similar will be `objects`.

    The algorithm is optimized by doing text processing once per document, and removing
    some redundant computation from the jiwer package.

    :param documents: list of documents, the first document is taken as the original
        document, against which the others are compared
    :type documents: list of str
    :param threshold: threshold of the metric, above which a document pair is classified
        as "similar", defaults to .1
    :type threshold: float, optional
    :param verbose: set to Trye to print all computed metrics, defaults to False
    :type verbose: bool, optional
    :return: list of document IDs in the `documents` list, that are similar to document
        at index 0
    :rtype: List[int]
    """
    similar_docs = []

    # Collect and preprocess all documents
    docs = _apply_transform(documents, transform=wer_default, is_reference=True)

    # The first iteration checks the metric on the original document with its ten
    # candidate documents
    next_subjects = [0]
    next_objects = list(range(1, 11))

    # Continue until all paths from the original document via similar documents to
    # candidate documents are exhausted
    while len(next_subjects) > 0:
        subjects = next_subjects
        # The next subjects will be the documents that are similar to any of the current
        # subjects.
        next_subjects = []

        for sub in subjects:
            objects = next_objects
            # The next objects will be the documents that are not similar to the
            # current subject. Of these documents, we haven't found similarity to the
            # original document yet according to its transitive property, and so we have
            # to keep looking
            next_objects = []

            for obj in objects:
                score = metric_word_chunk_overlap_ratio(docs[sub], docs[obj])
                if verbose:
                    print(sub, obj, score, "MATCH" if score > threshold else "")
                if score > threshold:
                    similar_docs.append(obj)
                    # This document is added to the subjects. In the next iteration of
                    # subjects, we will look for documents similar to it.
                    next_subjects.append(obj)
                else:
                    # In the next iteration of subjects, we will again compute if this
                    # document is similar (until no subjects remain)
                    next_objects.append(obj)
    return similar_docs


def generate_es_results(
    es_client: Elasticsearch,
    es_query: dict,
    index_name: str,
    size: int,
    model: gensim.models.doc2vec.Doc2Vec,
    annotation_field: str = None,
    vectorize: bool = True,
    get_label: bool = True,
):
    """
    Given an elasticsearch query, elasticsearch client and index it will loop over the
    documents. If vectorize is on it will vectorize the document and if the label is on
    it will return the label as well.
    If the query does not return any documents, it will throw an error.

    :param es_client: An active elasticsearch connection client
    :param es_query: A valid elasticsearch query that will be used to get results
    :param index_name: Name of the index to search through
    :param size: Size of the elasticsearch scan function
    :param model: A gensim doc2vec model
    :param annotation_field: For annotated documents, we can return the annotation field
    :param vectorize: If true, the model is used to get the vector data
    :param get_label: If true, return the labels i.e. the value of the annotation field
    :yields: vector data and optionally the corresponding label

    """
    # Scan for the documents
    gen_docs = scan(
        client=es_client,
        query=es_query,
        index=index_name,
        size=size,
    )
    # We count the number of documents processed
    count_docs = 0
    for idx, doc in enumerate(gen_docs):
        count_docs = idx
        if vectorize:
            vector = model.dv[[doc["_id"]]]
        else:
            # if not vectorizing just return the doc id
            yield doc["_id"]
            continue

        # We can only get the label for annotated documents
        if get_label:
            label = doc["_source"][annotation_field]["values"][0]
        else:
            label = None

        yield vector, label
    try:
        assert count_docs > 0, f"The query {es_query} did not return any results."
        logger.info(f"Looped over {count_docs} docs for the query {es_query}")
    except AssertionError as err:
        logger.exception(f"The query {es_query} did not return any results.")
        raise err
