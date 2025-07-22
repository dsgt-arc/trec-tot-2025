# semantic-knn

We'll download pre-computed semantic embeddings for Wikipedia articles and use them to build a k-nearest neighbor graph.
We'll use this as the basis for our final hybrid graph that combines the semantic and link-based edges.
While the directed link-based graph incorporates direct human knowledge of article relationships, it is not guaranteed to connect all of the related articles.
The semantic graph will help fill in those gaps by guaranteeing connectivity, and thus information flow between related articles.

This will be done with the following tools:

- https://huggingface.co/datasets/Upstash/wikipedia-2024-06-bge-m3
  - these are pre-computed embeddings for all paragraphs in articles using the BGE-M3 model for the 2024-06 Wikipedia dump
- https://github.com/facebookresearch/faiss/wiki/Faiss-building-blocks:-clustering,-PCA,-quantization
  - This library will be used to build the k-nearest neighbor graph from the embeddings
  - We'll go ahead and reduce the dimensionality of the embeddings with PCA to speed up the all-pairs nearest neighbor search.
    Searches are faster in lower dimensions (and scale in time linearly with the dimension of the vectors).
  - We'll make sure to use a cosine distance, since this is probably the most appropriate distance metric for text embeddings in the literature.
  - We'll pre-compute some number of k-means that we can use for k-means as a form of semantic topic clustering.
    This will allow us to compute things like topic-specific pagerank.
