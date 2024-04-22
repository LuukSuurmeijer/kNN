# Introduction 

The script in `src/main.py` trains a GenSim Doc2Vec model on all documents in an ElasticSearch index.
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
documents in the index given some Doc2Vec model using a kNN majority-vote approach.
The main algorithm is implemented in `src/knn.py` and the integration with the stack is implemented as a function `classify_docs` in `src/main.py`.
The algorithm is vectorized and batched for the unannotated documents, but at the moment still requires the full matrix of 
annotated document embeddings to be in memory. The algorithm could be optimized by also
batching the annotated documents and reconstructing the full matrix of cosines after
to compute the nearest neighbors. The classification algorithm runs several times,
once for each different annotation field. 
The performance and quality is NOT benchmarked on real data, but from some basic experiments I conclude it is pretty damn fast.

# Getting Started

You can test locally in a Docker container, if you have credentials for an ElasticSearch cluster:
1. Port-forward the ElasticSearch cluster to port `9200`. If it's running in an Ally Kubernetes cluster, the command may look like `kubectl port-forward services/eureka-elasticsearch 9200:9200`
2. Set the ElasticSearch environment variables in `src/local.env`: `ES_USERNAME` and `ES_PASSWORD` if the cluster requires authentication, and `ES_INDEX` with the name of the index that you want to process. It should contain documents with a field named `full_text`.
3. Run `make run`. This will train the model, and add a `similar_docs` field to the documents in the index. Note that this Make command limits the CPU and memory usage; you can adjust this with the variables set in the `Makefile`.

# Build and Test

Environment variables:
- `ES_USERNAME`, `ES_PASSWORD`, `ES_HOSTNAME`: ElasticSearch configuration
- `ES_INDEX`: index to train the model on, and add `similar_docs` field to. Tex is
    expected to be under field `full_text`
- `DOCUMENT_LIMIT`: maximum number of documents to process. Useful for testing.
- `TOP_N_SIMILAR_DOCS`: number of most similar document IDs to add to each document
- `TOP_N_SIMILAR_DOCS_WITH_CONFIDENCE`: number of most similar document IDs including model confidence to add to each document
- `MODEL_MAX_VOCAB_SIZE`: maximum number of words allowed in the vocabulary of the 
    Doc2Vec model. 10 million words require about 1Gb in RAM, if the corpus contains
    this many words.
- `MODEL_EMBEDDING_SIZE`: dimensionality of the document and word embeddings. Important
    setting when running into memory issues, since the document embeddings are in a
    matrix of size [documents in corpus] x MODEL_EMBEDDING_SIZE
 - `MLFLOW_TRACKING_URI`: URI to the service of the MLflow Tracker, to which the Doc2Vec
    model is registered.
- `ELASTICSEARCH_SLOW_INGEST`: set to "true" to pause 0.5 seconds after each bulk
    request to Elasticsearch to write document embeddings. On the development cluster,
    Elasticsearch tends to get a memory error and crashes.

The trained model is registered to MLflow.