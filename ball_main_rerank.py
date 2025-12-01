import json
import numpy as np
import faiss
import os
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

# ==========================================
# 1. Helper Function: Reciprocal Rank Fusion
# ==========================================
def reciprocal_rank_fusion(results_dict_list, k=60):
    """
    Combine multiple ranked lists using Reciprocal Rank Fusion (RRF).
    """
    fused_scores = {}
    
    for result_dict in results_dict_list:
        sorted_docs = sorted(result_dict.items(), key=lambda x: x[1], reverse=True)
        
        for rank, (doc_id, _) in enumerate(sorted_docs):
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0.0
            # RRF formula: 1 / (rank + k)
            fused_scores[doc_id] += 1.0 / (rank + k)
            
    return fused_scores

# ==========================================
# 2. The Advanced Hybrid Retriever Class
# ==========================================
class HybridRetriever:
    def __init__(self, 
                 embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                 reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        
        print(f"Loading Embedding Model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        
        print(f"Loading Reranker Model: {reranker_model}")
        # Note: If you have a GPU, verify it is being used: device='cuda'
        self.reranker = CrossEncoder(reranker_model)
        
        self.faiss_index = None
        self.bm25_index = None
        
        self.doc_ids = []
        self.corpus_texts = [] 
        self.doc_id_to_text = {} # New: Map for fast text lookup during reranking
    
    def load_jsonl(self, file_path):
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data

    def build_index(self, corpus_path, batch_size=32):
        print(f"Loading data from {corpus_path}...")
        corpus = self.load_jsonl(corpus_path)
        
        print("Preparing documents...")
        self.doc_ids = []
        self.corpus_texts = []
        self.doc_id_to_text = {}
        tokenized_corpus = []
        
        for doc in tqdm(corpus, desc="Indexing"):
            doc_id = doc.get('id', doc.get('_id', doc.get('doc_id')))
            title = doc.get('title', '')
            text = doc.get('text', '')
            
            combined_text = f"{title} {text}".strip()
            
            self.doc_ids.append(doc_id)
            self.corpus_texts.append(combined_text)
            self.doc_id_to_text[doc_id] = combined_text # Store for reranker
            
            # Basic tokenization for BM25
            tokenized_corpus.append(combined_text.lower().split())

        # --- Build BM25 Index ---
        print("Building BM25 index...")
        self.bm25_index = BM25Okapi(tokenized_corpus)
        
        # --- Build FAISS Index ---
        print("Encoding embeddings for FAISS...")
        embeddings = self.embedding_model.encode(
            self.corpus_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        faiss.normalize_L2(embeddings)
        
        print("Building FAISS index...")
        self.faiss_index = faiss.IndexFlatIP(self.dimension)
        self.faiss_index.add(embeddings)
        
        print(f"Index ready with {len(self.doc_ids)} documents.")

    def search(self, queries_path, top_k=10, batch_size=32):
        """
        Performs: Hybrid Search -> Fusion -> Re-Ranking
        """
        if not self.bm25_index or not self.faiss_index:
            raise ValueError("Index not built.")
            
        queries_data = self.load_jsonl(queries_path)
        query_ids = []
        query_texts = []
        
        for q in queries_data:
            query_ids.append(q.get('id', q.get('_id', q.get('query_id'))))
            query_texts.append(q.get('text', q.get('query', '')))
            
        # ===============================================
        # STAGE 1: Hybrid Retrieval (Fetch Top-50 candidates)
        # ===============================================
        candidate_k = 50  # We retrieve 50, but only return top_k (10) after reranking
        
        print("Stage 1: Running FAISS search...")
        query_embeddings = self.embedding_model.encode(
            query_texts, 
            batch_size=batch_size, 
            show_progress_bar=True,
            convert_to_numpy=True
        )
        faiss.normalize_L2(query_embeddings)
        
        dense_scores, dense_indices = self.faiss_index.search(query_embeddings, candidate_k)
        
        print("Stage 1: Running BM25 + Fusion + Reranking...")
        final_results = {}
        
        for i, q_text in enumerate(tqdm(query_texts, desc="Processing Queries")):
            qid = query_ids[i]
            
            # --- Gather FAISS Candidates ---
            faiss_res = {}
            for j in range(candidate_k):
                idx = dense_indices[i][j]
                if idx == -1: continue
                d_id = self.doc_ids[idx]
                score = float(dense_scores[i][j])
                faiss_res[d_id] = score
            
            # --- Gather BM25 Candidates ---
            tokenized_query = q_text.lower().split()
            bm25_doc_scores = self.bm25_index.get_scores(tokenized_query)
            bm25_top_n = np.argsort(bm25_doc_scores)[-candidate_k:][::-1]
            
            bm25_res = {}
            for idx in bm25_top_n:
                d_id = self.doc_ids[idx]
                score = bm25_doc_scores[idx]
                bm25_res[d_id] = score
            
            # --- Fuse Results (RRF) ---
            fused_scores = reciprocal_rank_fusion([faiss_res, bm25_res], k=60)
            
            # Sort fused results to get the Top-N Candidates for Reranking
            # We take the top 50 (or whatever candidate_k is) best fused docs
            sorted_candidates = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:candidate_k]
            candidate_doc_ids = [doc_id for doc_id, _ in sorted_candidates]
            
            # ===============================================
            # STAGE 2: Re-Ranking (Cross-Encoder)
            # ===============================================
            
            # Prepare pairs: [ [query, doc_text_1], [query, doc_text_2], ... ]
            pairs_to_rank = []
            valid_doc_ids = []
            
            for doc_id in candidate_doc_ids:
                if doc_id in self.doc_id_to_text:
                    text = self.doc_id_to_text[doc_id]
                    pairs_to_rank.append([q_text, text])
                    valid_doc_ids.append(doc_id)
            
            if not pairs_to_rank:
                final_results[qid] = {}
                continue
                
            # Predict scores (higher is better)
            rerank_scores = self.reranker.predict(pairs_to_rank)
            
            # Map new scores back to doc_ids
            reranked_results = {}
            for doc_id, score in zip(valid_doc_ids, rerank_scores):
                reranked_results[doc_id] = float(score)
            
            # Store the re-ranked results
            final_results[qid] = reranked_results
            
        return final_results

    def save_results(self, results, output_path, top_k=10):
        print(f"Saving results to {output_path}")
        rows = []
        for query_id, docs in results.items():
            # Sort by the Cross-Encoder score
            sorted_docs = sorted(docs.items(), key=lambda x: x[1], reverse=True)[:top_k]
            for doc_id, score in sorted_docs:
                rows.append({'query_id': query_id, 'corpus_id': doc_id}) 
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        return output_path

# ==========================================
# 3. Processing Functions
# ==========================================

def process_single_task(task_name, retriever, top_k=10):
    print(f"\n{'='*40}")
    print(f"Processing Task: {task_name}")
    print(f"{'='*40}")
    
    corpus_path = f"dataset/{task_name}/corpus.jsonl"
    queries_path = f"dataset/{task_name}/queries.jsonl"
    output_path = f"results/{task_name}_reranked_results.csv"
    
    os.makedirs("results", exist_ok=True)
    
    if not os.path.exists(corpus_path):
        print(f"⚠️  Skipping {task_name}: Corpus missing")
        return None
    
    retriever.build_index(corpus_path)
    results = retriever.search(queries_path, top_k=top_k)
    saved_file = retriever.save_results(results, output_path, top_k=top_k)
    print(f"✓ Completed {task_name}")
    return saved_file

def merge_results(result_files, output_filename="submission.csv"):
    print(f"\nMerging {len(result_files)} files into submission...")
    all_dfs = []
    for fpath in result_files:
        if fpath and os.path.exists(fpath):
            all_dfs.append(pd.read_csv(fpath))
    
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        final_df.to_csv(output_filename, index=False)
        print(f"✓ Submission ready: {output_filename}")
    else:
        print("❌ No results to merge.")

def main():
    # Recommended Models for Finance:
    # Embedding: "BAAI/bge-base-en-v1.5" (Better) or "sentence-transformers/all-MiniLM-L6-v2" (Faster)
    # Reranker:  "BAAI/bge-reranker-v2-m3" (Best Accuracy) or "cross-encoder/ms-marco-MiniLM-L-6-v2" (Fastest)
    
    retriever = HybridRetriever(
        embedding_model="BAAI/bge-base-en-v1.5",
        reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2" 
    )
    
    tasks = ['ConvFinQA', 'FinDER', 'FinQA', 'MultiHiertt', 'TATQA', 'FinanceBench', 'FinQABench']
    
    generated_files = []
    for task in tasks:
        try:
            result_file = process_single_task(task, retriever, top_k=10)
            if result_file: generated_files.append(result_file)
        except Exception as e:
            print(f"❌ Error on {task}: {e}")

    merge_results(generated_files)

if __name__ == "__main__":
    main()