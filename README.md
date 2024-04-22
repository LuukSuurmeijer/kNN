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