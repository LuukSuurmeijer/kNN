from typing import Dict
import os
import gensim
import numpy as np


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


class Corpus:
    """This class makes sure that Generator `generate_corpus` can be called as Iterator.
    A Generator cannot be used more than once, but to train a model, it is needed to go
    over all data multiple times. This class can be used for that.
    All code could have been implemented as Iterator, but it would be less readable.
    """

    def __init__(self, documents):
        self.documents = documents

    @staticmethod
    def generate_corpus(documents):
        for doc in documents:
            tokens = gensim.utils.simple_preprocess(doc["text"])
            yield gensim.models.doc2vec.TaggedDocument(tokens, [doc["index"]])

    def __iter__(self):
        """Called to get an Iterator in its initial state.

        :return: Generator of the corpus of full texts
        :rtype: Generator[TaggedDocument, Any, None]
        """
        return self.generate_corpus(self.documents)


def process_file(path: str, id: int) -> Dict:
    with open(path, "r") as f:
        contents = f.read().strip().lower()
        return {
            "index": id,
            "uuid": hash(contents),
            "path": path,
            "text": contents,
            "title": path.split("/")[-1][: path.index(".")],
        }


def parse_folder(pathstr: str, idx: int = 0):
    for i, obj in enumerate(os.listdir(pathstr)):
        objpath = f"{pathstr}/{obj}"
        if os.path.isfile(objpath):
            yield process_file(objpath, idx + i)
        elif os.path.isdir(objpath):
            yield from parse_folder(objpath, idx + 1)


def create_doc2vec_model(documents):
    model = gensim.models.doc2vec.Doc2Vec(
        documents=Corpus(documents),
        vector_size=100,
        min_count=1,
        max_vocab_size=None,
        epochs=20,
        window=5,
        workers=4,
    )

    return model


documents = list(parse_folder("/home/luuk/Documents/searchtest/pages"))
model = create_doc2vec_model(documents)

query = model.infer_vector(["knowledge", "graph"]).reshape(1, -1)
print(query.shape)
print(model.dv.vectors.shape)

cosines = cosine_similarities_vectorized(query, model.dv.vectors)

top_k_indices = np.argpartition(cosines, -5, axis=1)[..., -5::]

print(top_k_indices)

print([documents[idx]["title"] for idx in top_k_indices[0]])
