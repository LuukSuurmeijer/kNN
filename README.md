# Introduction 

This code implements relatively fast cosine similarity computation and kNN classification for large matrices, so I never have to worry about it again.
It's also possible to batch the computations to save space. 
Vectors are classigied using a kNN majority-vote approach.
The main algorithm is implemented in `src/knn.py`.

# Getting Started

You can test locally in a Docker container with an ES index, if you have credentials for an ElasticSearch cluster:
1. Port-forward the ElasticSearch cluster to port `9200`.
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