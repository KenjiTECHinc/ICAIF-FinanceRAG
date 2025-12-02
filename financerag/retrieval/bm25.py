import nltk
import logging
import numpy as np

from typing import Any, Callable, Dict, List, Literal, Optional
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from financerag.common import Lexical, Retrieval
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# install nltk resources if not already present
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')


def tokenize_list(input_list: List[str]) -> List[List[str]]:
    """
    Tokenizes a list of strings using the `nltk.word_tokenize` function.

    Args:
        input_list (`List[str]`):   
            A list of input strings to be tokenized.

    Returns:
        `List[List[str]]`:
            A list where each element is a list of tokens corresponding to an input string.
    """
    return list(map(word_tokenize, input_list))

class BM25Tokenizer:
    """A custom tokenizer that performs lowercasing, stopword removal, and stemming. A customized tokenizer compared to tokenize_list."""
    
    def __init__(self):
        """
        Initialize the tokenizer with NLTK's PorterStemmer and English stopwords.
        """
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
    
    def __call__(self, input_list: List[str]) -> List[List[str]]:
        """Tokenizes, cleans, and stems a list of documents or queries."""
        tokenized_output = []
        for text in input_list:
            # 1. Lowercase and tokenize
            tokens = word_tokenize(text.lower())
            
            # 2. Filter punctuation, stopwords, and apply stemming
            processed_tokens = [
                self.stemmer.stem(token) 
                for token in tokens 
                if token.isalnum() and token not in self.stop_words
            ]
            tokenized_output.append(processed_tokens)
        return tokenized_output

class RankBM25Model:
    """
    wrapper class for initializing a 'pretrained' bm25 reranker model using the corpus's tokenized documents.
    ensures that the model adheres to the Lexical protocol.
    """
    def __init__(self, corpus_tokens: List[List[str]]):
        """
        Initializes the BM25 model with the provided tokenized corpus.

        Args:
            corpus_tokens (`List[List[str]]`):
                A list of tokenized documents from the corpus.
        """
        self.bm25 = BM25Okapi(corpus_tokens)

    def get_scores(self, query_tokens: List[str]) -> np.ndarray:
        """
        Computes BM25 scores for the given tokenized query against the corpus.

        Args:
            query_tokens (`List[str]`):
                A list of tokens representing the query.
        """
        return self.bm25.get_scores(query_tokens)
    
class BM25Retriever(Retrieval):
    """
    A retrieval class that utilizes a lexical model (e.g., BM25) to search for the most relevant documents
    from a given corpus based on the input queries. This retriever tokenizes the queries and uses the provided
    lexical model to compute relevance scores between the queries and documents in the corpus.

    Methods:
        - retrieve: Searches for relevant documents based on the given queries, returning the top-k results.
    """

    def __init__(self, model: Lexical, tokenizer: Callable[[List[str]], List[List[str]]] = tokenize_list):
        """
        Initializes the `BM25Retriever` class with a lexical model and a tokenizer function.

        Args:
            model (`Lexical`):
                A lexical model (e.g., BM25) implementing the `Lexical` protocol, responsible for calculating relevance scores.
            tokenizer (`Callable[[List[str]], List[List[str]]]`, *optional*):
                A function that tokenizes the input queries. Defaults to `tokenize_list`, which uses `nltk.word_tokenize`.
        """
        self.model: Lexical = model
        self.tokenizer: Callable[[List[str]], List[List[str]]] = tokenizer
        self.results: Optional[Dict[str, Any]] = {}

    def retrieve(
            self,
            corpus: Dict[str, Dict[Literal["title", "text"], str]],
            queries: Dict[str, str],
            top_k: Optional[int] = None,
            score_function: Optional[str] = None,
            return_sorted: bool = False,
            **kwargs
    ) -> Dict[str, Dict[str, float]]:
        """
        Searches the corpus for the most relevant documents based on the given queries. The retrieval process involves
        tokenizing the queries, calculating relevance scores using the lexical model, and returning the top-k results
        for each query.

        Args:
            corpus (`Dict[str, Dict[Literal["title", "text"], str]]`):
                A dictionary representing the corpus, where each key is a document ID, and each value is another dictionary
                containing document fields such as 'id', 'title', and 'text'.
            queries (`Dict[str, str]`):
                A dictionary containing query IDs and corresponding query texts.
            top_k (`Optional[int]`, *optional*):
                The number of top documents to return for each query. If not provided, all documents are returned. Defaults to `None`.
            return_sorted (`bool`, *optional*):
                Whether to return the results sorted by score. Defaults to `False`.
            **kwargs:
                Additional keyword arguments passed to the lexical model during scoring.

        Returns:
            `Dict[str, Dict[str, float]]`:
                A dictionary where each key is a query ID, and the value is another dictionary mapping document IDs to relevance scores.
        """
        query_ids = list(queries.keys())
        self.results = {qid: {} for qid in query_ids}

        logger.info("Tokenizing queries with lower cases")
        query_lower_tokens = self.tokenizer([queries[qid].lower() for qid in queries])

        corpus_ids = list(corpus.keys())

        for qid, query in zip(query_ids, query_lower_tokens):
            scores = self.model.get_scores(query)
            top_k_result = np.argsort(scores)[::-1][:top_k]
            for idx in top_k_result:
                self.results[qid][corpus_ids[idx]] = scores[idx]

        return self.results