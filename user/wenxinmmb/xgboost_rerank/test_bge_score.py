from FlagEmbedding import BGEM3FlagModel

def chunk_paragraphs(text):
    """
    Split the input text into paragraphs separated by double newlines.

    Args:
        text (str): The input text.

    Returns:
        list: A list of paragraphs.
    """
    return text.split('\n\n')

def calculate_sparse_dense_scores(query, article, max_passage_length=8192):
    """
    Calculate the sparse and dense scores for a query and article using the BGE-M3 model.
    If the article contains multiple paragraphs, compute scores for each and take the maximum.

    Args:
        query (str): The query string.
        article (str): The article string.
        max_passage_length (int): The maximum passage length supported by the model.

    Returns:
        tuple: A tuple containing the maximum sparse score and maximum dense score.
    """
    # Initialize the BGE-M3 model
    model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

    # Split the article into paragraphs
    paragraphs = chunk_paragraphs(article)

    max_sparse_score = float('-inf')
    max_dense_score = float('-inf')

    for paragraph in paragraphs:
        # Prepare the input as a pair of query and paragraph
        sentence_pairs = [[query, paragraph]]

        # Compute the scores
        scores = model.compute_score(
            sentence_pairs,
            max_passage_length=max_passage_length
        )

        # Extract sparse and dense scores
        sparse_score = scores['sparse'][0]
        dense_score = scores['dense'][0]

        # Update the maximum scores
        max_sparse_score = max(max_sparse_score, sparse_score)
        max_dense_score = max(max_dense_score, dense_score)

    return max_sparse_score, max_dense_score

if __name__ == "__main__":
    # Example query and article
    query = "What is BGE M3?"
    article = (
        "BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.\n\n"
        "It is designed to handle various retrieval tasks efficiently and effectively, providing robust performance "
        "across multiple domains and applications."
    )

    # Calculate sparse and dense scores
    sparse_score, dense_score = calculate_sparse_dense_scores(query, article)

    # Print the scores
    print(f"Sparse Score: {sparse_score}")
    print(f"Dense Score: {dense_score}")