"""Retrieval methods for Day 7: Dense (pgvector), Lexical (BM25), and Hybrid (RRF)."""
import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional
import psycopg
from sentence_transformers import SentenceTransformer


def tokenize(text: str) -> List[str]:
    """Simple tokenization: lowercase alphanumerics."""
    return re.findall(r"\w+", text.lower())


# ---------------------------------------------------------------------------
# 1. Lexical Retriever: Pure BM25 (No external dependency needed)
# ---------------------------------------------------------------------------
class BM25Retriever:
    """Okapi BM25 implementation."""
    def __init__(self, chunks: List[Dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.doc_count = len(chunks)
        self.doc_lengths = [len(tokenize(c["text"])) for c in chunks]
        self.avg_doc_len = sum(self.doc_lengths) / (self.doc_count or 1)
        
        # Inverted index: term -> list of (doc_index, frequency)
        self.df: Counter = Counter()
        self.inverted_index: Dict[str, List[tuple[int, int]]] = {}
        for idx, c in enumerate(chunks):
            counts = Counter(tokenize(c["text"]))
            for term, freq in counts.items():
                self.inverted_index.setdefault(term, []).append((idx, freq))
                self.df[term] += 1
                
        # Precompute IDF
        self.idf: Dict[str, float] = {}
        for term, freq in self.df.items():
            self.idf[term] = math.log(1 + (self.doc_count - freq + 0.5) / (freq + 0.5))

    def search(self, query: str, top_k: int = 3, filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        tokens = tokenize(query)
        scores = [0.0] * self.doc_count
        
        for token in tokens:
            if token not in self.inverted_index:
                continue
            idf_val = self.idf[token]
            for doc_idx, freq in self.inverted_index[token]:
                doc_len = self.doc_lengths[doc_idx]
                denom = freq + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))
                num = freq * (self.k1 + 1)
                scores[doc_idx] += idf_val * (num / denom)
                
        # Candidate ranking
        ranked_indices = sorted(range(self.doc_count), key=lambda i: scores[i], reverse=True)
        results = []
        for idx in ranked_indices:
            c = self.chunks[idx]
            if filter_dict:
                match = True
                for k, v in filter_dict.items():
                    if c.get("metadata", {}).get(k) != v and c.get(k) != v:
                        match = False
                        break
                if not match:
                    continue
            results.append({
                "id": c["id"],
                "source_id": c["source_id"],
                "source": c.get("source", ""),
                "page": c["page"],
                "text": c["text"],
                "metadata": c.get("metadata", {}),
                "score": float(scores[idx]),
                "retriever": "bm25"
            })
            if len(results) >= top_k:
                break
        return results


# ---------------------------------------------------------------------------
# 2. Dense Retriever: PostgreSQL + pgvector (Day 6 Index)
# ---------------------------------------------------------------------------
class DenseVectorRetriever:
    """Exact Cosine search against Day 6 pgvector store."""
    def __init__(
        self,
        database_url: str = "postgresql://day6:day6_local_only@127.0.0.1:55433/day6_lab",
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        strategy: str = "B"
    ):
        self.database_url = database_url
        self.strategy = strategy
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)

    def search(self, query: str, top_k: int = 3, filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        # Normalize embeddings to align with cosine metric
        query_vec = self.model.encode(query, normalize_embeddings=True).tolist()
        
        where_clauses = ["strategy = %s"]
        params: List[Any] = [self.strategy]
        
        if filter_dict:
            for k, v in filter_dict.items():
                if k in ("section", "document_id", "source", "page", "year"):
                    where_clauses.append(f"{k} = %s")
                    params.append(v)
                else:
                    where_clauses.append("metadata ->> %s = %s")
                    params.extend([k, str(v)])
                    
        sql = f"""
            SELECT chunk_id, document_id, source, page, section, text, metadata,
                   1 - (embedding <=> %s::vector) AS score
            FROM chunks
            WHERE {' AND '.join(where_clauses)}
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """
        # Vector param is needed twice: once for score, once for ORDER BY
        exec_params = [query_vec] + params + [query_vec, top_k]
        
        with psycopg.connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, exec_params)
                rows = cur.fetchall()
                
        results = []
        for r in rows:
            chunk_id, doc_id, src, page, sec, text, meta, score = r
            if not isinstance(meta, dict):
                import json
                meta = json.loads(meta) if meta else {}
            if sec and "section" not in meta:
                meta["section"] = sec
            results.append({
                "id": chunk_id,
                "source_id": doc_id,
                "source": src,
                "page": page,
                "text": text,
                "metadata": meta,
                "score": float(score),
                "retriever": "dense_pgvector"
            })
        return results


# ---------------------------------------------------------------------------
# 3. Hybrid Retriever: Reciprocal Rank Fusion (RRF)
# ---------------------------------------------------------------------------
class HybridRetriever:
    """Fuses Dense (pgvector) and Lexical (BM25) results using RRF."""
    def __init__(self, dense_retriever: DenseVectorRetriever, bm25_retriever: BM25Retriever, rrf_k: int = 60):
        self.dense = dense_retriever
        self.bm25 = bm25_retriever
        self.rrf_k = rrf_k

    def search(self, query: str, top_k: int = 3, filter_dict: Optional[Dict[str, Any]] = None, candidate_k: int = 15) -> List[Dict[str, Any]]:
        # Retrieve larger candidate sets from both
        dense_results = self.dense.search(query, top_k=candidate_k, filter_dict=filter_dict)
        bm25_results = self.bm25.search(query, top_k=candidate_k, filter_dict=filter_dict)
        
        item_map: Dict[str, Dict[str, Any]] = {}
        rrf_scores: Dict[str, float] = {}
        
        for rank, item in enumerate(dense_results, start=1):
            cid = item["id"]
            item_map[cid] = item
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))
            
        for rank, item in enumerate(bm25_results, start=1):
            cid = item["id"]
            if cid not in item_map:
                item_map[cid] = item
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))
            
        # Sort by fused RRF score
        fused = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
        results = []
        for cid in fused[:top_k]:
            item = dict(item_map[cid])
            item["score"] = rrf_scores[cid]
            item["retriever"] = "hybrid_rrf"
            results.append(item)
        return results
