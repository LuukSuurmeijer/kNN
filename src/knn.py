import json
from typing import Any, Callable, List, Generator
import numpy as np
import gensim
import time
from tqdm import tqdm
from utils import batches_of_m, create_lookup_func, batching_function


def cosine_similarities(a: np.array, B: np.array) -> np.array:
    """
        Computes the cosine similarities of vector current_doc with all columns in docs_matrix.
        This is about 20% faster than gensim.models.keyedvectors.cosine_similarities.

        words_matrix @ words_matrix.T --> the squares of the matrix are on the diagonal
        avoid computing all the non-diagonal elements somehow
        achievable with einstein summation:
            np.einsum('ij,jk') normal matrix product
            np.einsum('ij,ji') sum over all diagonal elements of the matrix product
            np.einsum('ij,ji->i') unforces sum, just returns the elements of the diagonal


    :param current_doc: vector of the document of interest
    :type current_doc: np.array
    :param docs_matrix: matrix of all documents to be computed
    :type docs_matrix: np.array
    :return: vector of cosines
    :rtype: np.array
    """

    return (B @ a) / (np.sqrt(np.einsum("ij,ji->i", B, B.T)) * np.sqrt(np.dot(a, a)))


def cosine_similarities_vectorized(A: np.array, B: np.array) -> np.array:
    """
      Computes the cosine similarities of all vectors in A with all vectors in in B.
      This is A LOT faster (10-20 times) than cosine_similarities.

      words_matrix @ words_matrix.T --> the squares of the matrix are on the diagonal
      avoid computing all the non-diagonal elements somehow
      achievable with einstein summation:
          np.einsum('ij,jk') normal matrix product
          np.einsum('ij,ji') sum over all diagonal elements of the matrix product
          np.einsum('ij,ji->i') unforces sum, just returns the elements of the diagonal


    :param A: matrix of the documents of interest
    :type A: np.array [N_document x Features]
    :param B: matrix of all documents to be computed
    :type B: np.array [P_documents x Features]
    :return: matrix of cosines [N_documents x P_documents]
    :rtype: np.array
    """

    return (A @ B.T) / (
        np.expand_dims(np.sqrt(np.einsum("ij,ji->i", A, A.T)), axis=1)
        @ np.expand_dims(np.sqrt(np.einsum("ij,ji->i", B, B.T)), axis=0)
    )


def knn_not_vectorized(
    doc: np.array, comparison_vectors: np.array, k: int, lookup_func: Callable
) -> str:
    """Compute the label of doc using KNN with documents in comparison_vectors . Returns str of predicted label for doc.

    :param np.array [1 x features] docs: Vector of document to be labeled
    :param np.array [M_docs x features] comparison_vectors: Matrix of documents to be considered for to be labeled doc
    :param int k: Hyperparameter of KNN search
    :param Callable lookup_func: Some function that takes an index and returns the label of the document with that index
    :return: Predicted label of doc
    :rtype: str
    """
    cosines = cosine_similarities(doc, comparison_vectors)

    top_k_indices = np.argpartition(cosines, -k)[-k:]

    annotation_matrix = np.vectorize(lookup_func)(top_k_indices)

    u, indices = np.unique(annotation_matrix, return_inverse=True)
    return u[np.argmax(np.bincount(indices))]


def knn_batched(
    docs: np.array,
    comparison_vectors: np.array,
    k: int,
    lookup_func: Callable,
    bs: int,
) -> List[str]:
    """Compute the labels of the  k nearest neighbors in comparison_vectors for all document vectors in docs . Returns list of predicted label per document.
    This algorithm is vectorized and batched, meaning it computes labels in batches of size bs. Simplifies to `knn_not_vectorized` for bs = 1.

    :param np.array [N_docs x features] docs: Matrix of documents to be labeled
    :param np.array [M_docs x features] comparison_vectors: Matrix of documents to be considered for each to be labeled doc
    :param int k: Hyperparameter of KNN search
    :param Callable lookup_func: Some function that takes an index and returns the label of the document with that index
    :param int bs: Batch size
    :return: List of strings where the ith entry is the predicted label for document with index i
    """

    batches = batches_of_m(len(docs), bs)
    all_annotations = []
    for start, end in tqdm(batches):
        # compute cosines, shape: [N_unannotated_docs x M_annotated_docs]
        cosines = cosine_similarities_vectorized(docs[start:end], comparison_vectors)

        # find the indices of top k nearest neighbor for each document, shape: [N_unannoted_docs x K_annotated_docs]
        top_k_indices = np.argpartition(cosines, -k, axis=1)[..., -k::]

        # get annotations for each neighbor for each doc, shape: [N_unannoted_docs x K_annotated_docs]
        annotation_matrix = np.vectorize(lookup_func)(top_k_indices)

        # find the most common annotation in neighbors for each doc, shape: [N_unannotated_docs, 1]
        # find the unique values of `annotation_matrix`, int_labels is annotation matrix casted to ints
        unique, int_labels = np.unique(annotation_matrix, return_inverse=True)
        # for each unannotated doc, count the number of occurrences of each integer label with bincount, take the argmax, translate to string labels in `unique`
        most_common_annotations = unique[
            np.argmax(
                np.apply_along_axis(
                    np.bincount,
                    1,
                    int_labels.reshape(annotation_matrix.shape),
                    None,
                    np.max(int_labels) + 1,
                ),
                axis=1,
            )
        ]
        all_annotations.extend(most_common_annotations)

    return all_annotations


def knn_generator(
    query_generator: Generator[Any, Any, Any],
    labeled_generator: Generator[Any, np.array, str],
    batch_size: int,
    k: int,
):
    """
    Compute the labels of the  k nearest neighbors in comparison_vectors for all query vectors.

    We loop over the labeled vectors and for each quert vector vector we keep a top-K nearest neighbors of
    labeled. This way we do not need to keep a score for all annotated documents in memory.
    Based upon Luuk's code.

    Returns list of predicted label per document.

    :param query_generator: Generator that yields query vectors
    :param labeled_generator: Generator that yields labeled vectors and their labels
    :param batch_size: the batch size to loop over the data
    :param k: number of the k-nearest neighbors
    """

    labeled_vectors_generator = batching_function(labeled_generator, batch_size)
    global_top_scores = []
    global_annotation = []
    org_k = -1
    for i, (labeled_batched) in enumerate(labeled_vectors_generator):
        vectors, labels = list(zip(*(labeled_batched)))
        lookup_func = create_lookup_func(labels)

        vectors = np.stack(vectors)

        query_vectors_generator = batching_function(query_generator, batch_size)

        inner_scores = []
        inner_labels = []
        for _, (query_batched) in enumerate(query_vectors_generator):
            query_vectors, query_labels = list(zip(*((query_batched))))
            query_vectors = np.stack(query_vectors)
            query_vectors = query_vectors.reshape(
                -1, query_vectors.shape[-1]
            )
            vectors = vectors.reshape(-1, vectors.shape[-1])

            # Get cosine scores for this batch
            cosines = cosine_similarities_vectorized(query_vectors, vectors)

            # In the case due to batching we not have at least top-k candidates
            # change k accordingly
            if cosines.shape[-1] < k:
                # Keep track of the original k-value
                org_k = k
                # Set the new k-value to what is available
                k = cosines.shape[-1]
                diff_k = org_k - k

            # Find the indices of top k nearest neighbor for each document, shape: [N_unannotated_docs x K_annotated_docs]
            # where N and K are the respective batch sizes
            top_k_indices = np.argpartition(cosines, -k, axis=1)[..., -k::]

            # Get annotations for each neighbor for each doc, shape: [N_unannotated_docs x top-K_annotated_docs]
            # where N is the batch size and top-K the number of K similar documents
            annotation_matrix = np.vectorize(lookup_func)(top_k_indices)

            # Only keep the top-k scores, shape: [N_unannotated_docs x top-K_annotated_docs]
            # where N is the batch size and top-K the number of K similar documents
            top_cosine_scores = np.take_along_axis(cosines, top_k_indices, axis=1)

            # When there are not at least top-k candidates we need to add some padding
            if cosines.shape[-1] < org_k:
                top_cosine_scores = np.pad(
                    top_cosine_scores,
                    ((0, 0), (0, diff_k)),
                    mode="constant",
                    constant_values=-1,
                )
                annotation_matrix = np.pad(
                    annotation_matrix,
                    ((0, 0), (0, diff_k)),
                    mode="constant",
                    constant_values=0,
                )
                k = org_k

            # Save the top cosine scores and corresponding labels (annotation_matrix)
            inner_scores.append(top_cosine_scores)
            inner_labels.append(annotation_matrix)

        inner_scores = np.concatenate(inner_scores)
        inner_labels = np.concatenate(inner_labels)

        # First iteration the scores will be empty
        if global_top_scores == []:
            global_top_scores = np.ones_like(inner_scores) * -1

        if global_annotation == []:
            global_annotation = np.zeros_like(inner_labels)

        # We only save the scores and labels that are larger than the previously saved top scores
        scores_comparison = np.greater(inner_scores, global_top_scores)
        global_top_scores = np.where(scores_comparison, inner_scores, global_top_scores)
        global_annotation = np.where(scores_comparison, inner_labels, global_annotation)

    unique, int_labels = np.unique(global_annotation, return_inverse=True)

    # Get a majority vote of the top-k neighbors, this returns the actual string in most_common_annotations
    most_common_annotations = unique[
        np.argmax(
            np.apply_along_axis(
                np.bincount,
                1,
                int_labels.reshape(global_annotation.shape),
                None,
                np.max(int_labels) + 1,
            ),
            axis=1,
        )
    ]

    return most_common_annotations


def unit_test(N=10**3, k=3, bs=1, compare=False):
    np.seed = 422
    # some dummy classes
    classes = {"annotations": ["invoice", "policies", "salaries"]}

    # generate some random documents, 1/4 not annotated
    random_documents = [
        {
            "id": i,
            "text": "",
            "annotation": np.random.choice(["invoice", "policies", "salaries", ""]),
        }
        for i in range(N)
    ]

    # init model with random vectors
    model = gensim.models.KeyedVectors(vector_size=300, count=N)
    model.vectors = np.random.random_sample((N, 300))

    # get subset of documents with relevant annotation
    comparison_set = np.array(
        [
            doc["id"]
            for doc in random_documents
            if doc["annotation"] in classes["annotations"]
        ]
    )
    comparison_set_vectors = model.vectors[comparison_set]

    # get subset of unannotated documents
    unannotated_documents = np.array(
        [doc["id"] for doc in random_documents if doc["annotation"] == ""]
    )
    unannotated_vectors = model.vectors[unannotated_documents]

    print(f"Number of unannotated (# iterations): {len(unannotated_documents)}")
    print(f"Number of annotated (# cosines per iteration): {len(comparison_set)}")

    print(unannotated_vectors.shape, comparison_set_vectors.shape)

    t_0 = time.time()
    pred = knn_batched(
        unannotated_vectors,
        comparison_set_vectors,
        k,
        lambda i: random_documents[comparison_set[i]]["annotation"],
        bs,
    )
    t_delta = time.time() - t_0

    print(f"Completed batched KNN annotation in {t_delta} seconds")

    if compare:
        t_0 = time.time()
        pred_unvectorized = [
            knn_not_vectorized(
                doc,
                comparison_set_vectors,
                k,
                lambda i: random_documents[comparison_set[i]]["annotation"],
            )
            for doc in tqdm(unannotated_vectors)
        ]
        t_delta_unvectorized = time.time() - t_0
        print(f"Completed KNN annotation in {t_delta_unvectorized} seconds")

        assert len(pred) == len(pred_unvectorized) == len(unannotated_documents)
        assert np.array(pred == pred_unvectorized).all()

    else:
        assert len(pred) == len(unannotated_documents)

    return pred


def integration_test():
    with open("random_documents.json") as f:
        data = json.load(f)
        documents_texts = [doc["full_text"] for doc in data]
        for i, doc in enumerate(data):
            doc["index"] = i

    # Preprocess text

    preprocessed_docs = [
        gensim.models.doc2vec.TaggedDocument(
            gensim.utils.simple_preprocess(text, min_len=2, max_len=25), [i]
        )
        for i, text in enumerate(documents_texts)
    ]

    model = gensim.models.doc2vec.Doc2Vec(preprocessed_docs, vector_size=150)

    print(model.dv[0])

    annotated_docs = [doc for doc in data if doc.get("annotation_status")]
    unannotated_docs = [doc for doc in data if not doc.get("annotation_status")]

    unannotated_vectors = model.dv[[doc["index"] for doc in unannotated_docs]]
    annotated_vectors = model.dv[[doc["index"] for doc in annotated_docs]]

    print(unannotated_vectors.shape, annotated_vectors.shape)

    preds = knn_batched(
        unannotated_vectors,
        annotated_vectors,
        3,
        lambda i: annotated_docs[i]["annotation_status"]["values"][0],
        90,
    )

    assert len(preds) == len(unannotated_vectors)

    print(list(zip(preds, [doc["title"] for doc in annotated_docs])))

    for doc in annotated_docs:
        if doc["title"] == "Happy space situation environment policy economy together.":
            print(doc["annotation_status"])
