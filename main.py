import json
import numpy as np
import faiss
import pandas as pd
import os
from tqdm import tqdm
from typing import Dict, Literal
from sentence_transformers import SentenceTransformer, CrossEncoder

from financerag.rerank.cross_encoder import CrossEncoderReranker
from financerag.retrieval.bm25 import BM25Retriever, BM25Tokenizer, RankBM25Model

# total vectors: 32225
# nlist clusters: 180 (sqrt(N))

# ========== DENSE RETRIEVER CLASS ==========
class FAISSRetriever:
    """
    Simple FAISS-based retriever for the FinanceRAG competition
    """
    
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the retriever with an embedding model
        
        Args:
            model_name: HuggingFace model for embeddings
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"Model embedding dimension: {self.dimension}")
        
        # FAISS index (will be created when corpus is loaded)
        self.index = None
        self.doc_ids = []  # Maps FAISS index position to document ID
        
    def load_jsonl(self, file_path):
        """
        Load data from JSONL file
        
        Args:
            file_path: Path to JSONL file
            
        Returns:
            List of dictionaries
        """
        print(f"Loading data from {file_path}")
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        print(f"Loaded {len(data)} items")
        return data
    
    def build_index(self, corpus_path, batch_size=32, nlist=180):
        """
        Build FAISS index from corpus
        
        Args:
            corpus_path: Path to corpus JSONL file
            batch_size: Batch size for encoding
        """
        # Load corpus
        corpus = self.load_jsonl(corpus_path)
        
        # Prepare documents for encoding
        # Combine title and text for better context
        documents = []
        self.doc_ids = []
        
        print("Preparing documents...")
        for doc in corpus:
            # Adjust these keys based on your JSONL structure
            doc_id = doc.get('id', doc.get('_id', doc.get('doc_id')))
            title = doc.get('title', '')
            text = doc.get('text', '')
            
            # Combine title and text
            combined = f"passage: {title} {text}".strip()
            documents.append(combined)
            self.doc_ids.append(doc_id)
        
        # Encode documents in batches
        print(f"Encoding {len(documents)} documents...")
        embeddings = self.model.encode(
            documents,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        print("Document embeddings size:", embeddings.shape)
        # Normalize embeddings (important for cosine similarity)
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index
        # Using IndexFlatIP for exact cosine similarity search
        print("Building FAISS index...")
        # try using IVF instead of Flat for faster search
        # quantizer = faiss.IndexFlatIP(self.dimension)
        # self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_INNER_PRODUCT)
        
        # print(f"Training Index on {self.index.ntotal} vectors...")
        # training_data = embeddings[:min(len(embeddings), 10000)]
        
        # if not self.index.is_trained:
            # self.index.train(training_data)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        
        print(f"FAISS index built with {self.index.ntotal} vectors")
    
    def search(self, queries_path, top_k=10, batch_size=32, nprobe=20):
        """
        Search for relevant documents for each query
        
        Args:
            queries_path: Path to queries JSONL file
            top_k: Number of documents to retrieve per query
            batch_size: Batch size for encoding queries
            
        Returns:
            Dictionary mapping query_id to {doc_id: score}
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Load queries
        queries_data = self.load_jsonl(queries_path)
        
        # Prepare queries
        query_ids = []
        query_texts = []
        
        print("Preparing queries...")
        for query in queries_data:
            # Adjust these keys based on your JSONL structure
            query_id = query.get('id', query.get('_id', query.get('query_id')))
            query_text = query.get('text', query.get('query', ''))
            
            query_ids.append(query_id)
            query_texts.append(query_text)
        
        # Encode queries
        print(f"Encoding {len(query_texts)} queries...")
        query_embeddings = self.model.encode(
            query_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Normalize query embeddings
        faiss.normalize_L2(query_embeddings)
        
        # Search FAISS index
        print(f"Searching for top {top_k} documents per query...")
        scores, indices = self.index.search(query_embeddings, top_k)
        
        # Format results
        results = {}
        for i, query_id in enumerate(query_ids):
            results[query_id] = {}
            for j in range(top_k):
                doc_idx = indices[i][j]
                doc_id = self.doc_ids[doc_idx]
                score = float(scores[i][j])
                results[query_id][doc_id] = score
        
        print(f"Retrieved top {top_k} documents for {len(results)} queries")
        return results
    
    def search_raw(self, queries_path, top_k=10, batch_size=32, nprobe=20):
        """
        Search for relevant documents for each query, returning results without saving.
        This is a duplicate of the search() logic, but without the final save step.
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # --- SET NPROBE ---
        # Only set if the index is an IVF index
        # if hasattr(self.index, 'nprobe'):
            # self.index.nprobe = nprobe
            # print(f"Set nprobe={nprobe} for approximate search.")
            
        # Load queries
        queries_data = self.load_jsonl(queries_path)
        
        # Prepare queries
        query_ids = []
        query_texts = []
        
        for query in queries_data:
            # Add 'query:' prefix for E5 models (Crucial for high performance)
            query_text = f"query: {query.get('text', query.get('query', ''))}" 
            query_ids.append(query.get('id', query.get('_id', query.get('query_id'))))
            query_texts.append(query_text)
        
        # Encode and Normalize queries (using same logic as build_index)
        query_embeddings = self.model.encode(
            query_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        faiss.normalize_L2(query_embeddings)
        
        # Search FAISS index
        scores, indices = self.index.search(query_embeddings, top_k)
        
        # Format results
        results = {}
        for i, query_id in enumerate(query_ids):
            results[query_id] = {}
            for j in range(top_k):
                doc_idx = indices[i][j]
                doc_id = self.doc_ids[doc_idx]
                score = float(scores[i][j])
                results[query_id][doc_id] = score
        
        return results
    
    def save_results(self, results, output_path, top_k=10):
        """
        Save results in Kaggle submission format
        
        Args:
            results: Dictionary from search()
            output_path: Path to save CSV file
            top_k: Number of results to save per query
        """
        print(f"Saving results to {output_path}")
        
        rows = []
        for query_id, docs in results.items():
            # Sort by score and take top_k
            sorted_docs = sorted(docs.items(), key=lambda x: x[1], reverse=True)[:top_k]
            
            for doc_id, score in sorted_docs:
                rows.append({'query_id': query_id, 'corpus_id': doc_id})
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        print(f"Saved {len(rows)} rows to {output_path}")

# ========== SPARSE RETRIEVER SETUP FUNCTION ==========
def setup_bm25_retriever(corpus_data: Dict[str, Dict[Literal["title", "text"], str]]) -> BM25Retriever:
    """
    Sets up the BM25 model and retriever pipeline.
    
    Args:
        corpus_data: Your dictionary of all document IDs, titles, and texts.
        
    Returns:
        An initialized BM25Retriever instance ready for searching.
    """
    # 1. Instantiate the tokenizer
    bm25_tokenizer = BM25Tokenizer()

    # 2. Prepare the corpus for indexing
    # combine title and text for better keyword coverage
    corpus_texts = []
    corpus_ids = []
    for doc_id, doc in corpus_data.items():
        # Combine title and text, ensure lowercase for consistency
        combined_text = f"passage: {doc.get('title', '')} {doc.get('text', '')}".strip()
        corpus_texts.append(combined_text)
        corpus_ids.append(doc_id)
        
    # 3. Tokenize the entire corpus
    print(f"Tokenizing and stemming {len(corpus_texts)} documents...")
    tokenized_corpus = bm25_tokenizer(corpus_texts)
    
    # 4. Instantiate the Lexical Model
    print("Building BM25 Index (calculating IDF)...")
    bm25_model = RankBM25Model(tokenized_corpus)
    
    # 5. Instantiate the final BM25 Retriever
    # Pass the instantiated model and the tokenizer's call method
    retriever = BM25Retriever(
        model=bm25_model, 
        tokenizer=bm25_tokenizer
    )
    
    # (Optional) Store corpus IDs for easier lookup if needed later
    retriever.corpus_ids = corpus_ids 
    
    print("BM25 Retriever setup complete.")
    return retriever

# ========== TASK PROCESSING FUNCTIONS ==========
# SINGLE TASK PROCESSING FUNCTION
def process_single_task(task_name, retriever, top_k=10):
    """
    Process a single task: build index, search, save results
    
    Args:
        task_name: Name of the task (e.g., 'ConvFinQA')
        retriever: FAISSRetriever instance
        top_k: Number of results per query
    """
    print(f"\n{'='*60}")
    print(f"Processing task: {task_name}")
    print(f"{'='*60}\n")
    
    # Define paths
    corpus_path = f"dataset/{task_name}/corpus.jsonl"
    queries_path = f"dataset/{task_name}/queries.jsonl"
    output_path = f"results/{task_name}_results.csv"
    
    # Check if files exist
    if not os.path.exists(corpus_path):
        print(f"WARNING: Corpus file not found: {corpus_path}")
        return None
    if not os.path.exists(queries_path):
        print(f"WARNING: Queries file not found: {queries_path}")
        return None
    
    # Build index from corpus
    retriever.build_index(corpus_path, batch_size=32)
    
    # Search for queries
    results = retriever.search(queries_path, top_k=top_k, batch_size=32)
    
    # Save results
    os.makedirs("results", exist_ok=True)
    retriever.save_results(results, output_path, top_k=top_k)
    
    print(f"\n✓ Task {task_name} completed!")
    return output_path

# HYBRID TASK PROCESSING FUNCTION
def process_hybrid_task(task_name, faiss_retriever, reranker, top_k_rerank=10, top_k_initial=100):
    """Performs Hybrid Search (FAISS + BM25 + RRF) and Reranking."""
    print(f"\n{'='*60}")
    print(f"Processing HYBRID task: {task_name}")
    print(f"{'='*60}\n")
    
    corpus_path = f"dataset/{task_name}/corpus.jsonl"
    queries_path = f"dataset/{task_name}/queries.jsonl"
    output_path = f"results/{task_name}_hybrid_rerank_results.csv"
    
    # 1. Setup Data & Build FAISS Index
    corpus_data = faiss_retriever.load_jsonl(corpus_path) # List of dicts
    faiss_retriever.build_index(corpus_path, batch_size=32) # Builds the index
    
    # Convert corpus/queries list of dicts to the dict format needed by Reranker/BM25
    corpus_dict = {
        doc.get('id', doc.get('_id')): {'title': doc.get('title', ''), 'text': doc.get('text', '')} 
        for doc in corpus_data
    }
    queries_data = faiss_retriever.load_jsonl(queries_path)
    queries_dict = {
        query.get('id', query.get('_id')): query.get('text', '') 
        for query in queries_data
    }

    # 2. Retrieve Candidates (Top N for better Reranker performance)
    print(f"Starting Dense (FAISS) Retrieval: Top {top_k_initial}")
    dense_results = faiss_retriever.search_raw(queries_path, top_k=top_k_initial, batch_size=32)

    print("Starting Sparse (BM25) Retrieval...")
    # Instantiate BM25 Model and Retriever for this task's corpus
    bm25_retriever = setup_bm25_retriever(corpus_dict)
    sparse_results = bm25_retriever.retrieve(corpus=corpus_dict, queries=queries_dict, top_k=top_k_initial)
    
    print(f"Fusing results using RRF (k_rrf=60)...")
    fused_results = perform_rrf_fusion(dense_results, sparse_results)
    
    print(f"Reranking fused top-{top_k_rerank} results with Cross-Encoder...")
    final_results = reranker.rerank(
        corpus=corpus_dict,
        queries=queries_dict,
        results=fused_results,
        top_k=top_k_rerank, # Rerank only the desired final amount
        batch_size=32
    )
    
    os.makedirs("results", exist_ok=True)
    faiss_retriever.save_results(final_results, output_path, top_k=top_k_rerank) 
    
    print(f"\n✓ Hybrid Task {task_name} completed!")
    return output_path

# ========== UTILITY FUNCTIONS ==========
def merge_results(result_files, output_path="submission.csv"):
    """
    Merge all task results into a single submission file
    
    Args:
        result_files: List of CSV file paths
        output_path: Path for final submission file
    """
    print(f"\n{'='*60}")
    print("Merging results into final submission file")
    print(f"{'='*60}\n")
    
    dfs = []
    for file_path in result_files:
        if file_path and os.path.exists(file_path):
            print(f"Loading {file_path}")
            df = pd.read_csv(file_path)
            dfs.append(df)
            print(f"  → {len(df)} rows")
    
    if not dfs:
        print("ERROR: No result files to merge!")
        return
    
    # Concatenate all dataframes
    final_df = pd.concat(dfs, ignore_index=True)
    
    # Save final submission
    final_df.to_csv(output_path, index=False)
    
    print(f"\n✓ Final submission saved to: {output_path}")
    print(f"  Total rows: {len(final_df)}")
    print(f"  Unique queries: {final_df['query_id'].nunique()}")
    print(f"\nYou can now submit '{output_path}' to Kaggle!")

# COMBINE RESULTS USING RRF
def perform_rrf_fusion(
    dense_results: Dict[str, Dict[str, float]], 
    sparse_results: Dict[str, Dict[str, float]], 
    k_rrf: int = 60
) -> Dict[str, Dict[str, float]]:
    """
    Performs Reciprocal Rank Fusion (RRF) to combine dense and sparse results.
    
    Args:
        dense_results: Results from FAISS (Query ID -> Doc ID -> Score).
        sparse_results: Results from BM25 (Query ID -> Doc ID -> Score).
        k_rrf: RRF constant (default 60 is common).
        
    Returns:
        A dictionary of fused results (Query ID -> Doc ID -> RRF Score).
    """
    fused_results = {}
    
    # Iterate over all queries (assuming both dicts have the same query IDs)
    for qid in dense_results.keys():
        all_doc_ids = set(dense_results[qid].keys()) | set(sparse_results[qid].keys())
        rrf_scores = {}
        
        # 1. Rank Dense Results
        # Sort by score to get the rank (1st result is rank 1)
        ranked_dense = {
            doc_id: rank + 1
            for rank, (doc_id, score) in enumerate(
                sorted(dense_results[qid].items(), key=lambda item: item[1], reverse=True)
            )
        }
        
        # 2. Rank Sparse Results
        ranked_sparse = {
            doc_id: rank + 1
            for rank, (doc_id, score) in enumerate(
                sorted(sparse_results[qid].items(), key=lambda item: item[1], reverse=True)
            )
        }
        
        # 3. Apply RRF Formula to all unique documents
        for doc_id in all_doc_ids:
            rank_d = ranked_dense.get(doc_id, 0) # Use 0 if not found
            rank_s = ranked_sparse.get(doc_id, 0) # Use 0 if not found
            
            # RRF Formula: 1 / (k + rank)
            rrf_score = 0.0
            if rank_d > 0:
                rrf_score += 1.0 / (k_rrf + rank_d)
            if rank_s > 0:
                rrf_score += 1.0 / (k_rrf + rank_s)
                
            if rrf_score > 0:
                rrf_scores[doc_id] = rrf_score
                
        # Sort and store the final fused results
        fused_results[qid] = dict(
            sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
        )
        
    return fused_results

def main():
    """
    Main function to process all tasks
    """
    print("="*60)
    print("FinanceRAG Competition - FAISS Retrieval Pipeline")
    print("="*60)
    
    # List of all tasks
    tasks = [
        'ConvFinQA',
        'FinDER', 
        'FinQA',
        'MultiHiertt',
        'TATQA',
        'FinanceBench',
        'FinQABench'
    ]
    
    # Initialize retriever with embedding model
    print("\nInitializing retriever and reranker...")
    faiss_retriever = FAISSRetriever(
        model_name="intfloat/e5-base-v2"
        # For better results, try:
        # model_name="BAAI/bge-base-en-v1.5"
        # model_name="sentence-transformers/all-mpnet-base-v2"
    )
    
    print("Initializing reranker...")
    reranker = CrossEncoderReranker(model=CrossEncoder('BAAI/bge-reranker-base'))
    
    # Process each task
    result_files = []
    for task in tqdm(tasks, desc="Processing Tasks"):
        # result_file = process_single_task(task, faiss_retriever, top_k=10)
        result_file = process_hybrid_task(
            task, 
            faiss_retriever, 
            reranker, 
            top_k_rerank=10, 
            top_k_initial=69
        )
        result_files.append(result_file)
    
    # Merge all results
    merge_results(result_files, output_path="submission.csv")
    
    print("\n" + "="*60)
    print("Pipeline completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()