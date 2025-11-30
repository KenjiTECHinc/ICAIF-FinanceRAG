import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pandas as pd
import os


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
    
    def build_index(self, corpus_path, batch_size=32):
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
            combined = f"{title} {text}".strip()
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
        
        # Normalize embeddings (important for cosine similarity)
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index
        # Using IndexFlatIP for exact cosine similarity search
        print("Building FAISS index...")
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        
        print(f"FAISS index built with {self.index.ntotal} vectors")
    
    def search(self, queries_path, top_k=10, batch_size=32):
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
    print("\nInitializing retriever...")
    retriever = FAISSRetriever(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
        # For better results, try:
        # model_name="BAAI/bge-base-en-v1.5"
        # model_name="sentence-transformers/all-mpnet-base-v2"
    )
    
    # Process each task
    result_files = []
    for task in tasks:
        result_file = process_single_task(task, retriever, top_k=10)
        result_files.append(result_file)
    
    # Merge all results
    merge_results(result_files, output_path="submission.csv")
    
    print("\n" + "="*60)
    print("Pipeline completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()