"""Benchmark and compare the 3 retrieval methods on Dev split."""
import json
import time
import statistics
from pathlib import Path
from starter.baseline import hit, pack_context
from src.retrievers import BM25Retriever, DenseVectorRetriever, HybridRetriever

def run_retriever_eval(name, retriever_search_fn, questions, chunks, split="dev", budget=800):
    selected = [q for q in questions if q.get("split") == split]
    traces = []
    
    for q in selected:
        start = time.perf_counter()
        results = retriever_search_fn(q["question"], top_k=3, filter_dict=q.get("filter"))
        context, ids = pack_context(results, budget)
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        marked = [{**c, "hit": hit(c, q.get("gold", []))} for c in results]
        traces.append({
            "query_id": q["id"],
            "split": q["split"],
            "question": q["question"],
            "strategy": name,
            "top_k": 3,
            "budget_chars": budget,
            "gold": q.get("gold", []),
            "results": marked,
            "context": context,
            "context_ids": ids,
            "context_chars": len(context),
            "elapsed_ms": elapsed_ms,
            "hit1": bool(marked and marked[0]["hit"]),
            "hit3": any(c["hit"] for c in marked),
            "context_hit": any(c["hit"] and c["id"] in ids for c in marked)
        })
        
    ans_traces = [t for t in traces if any(q["id"] == t["query_id"] and q.get("answerable", True) for q in selected)]
    summary = {
        "strategy": name,
        "split": split,
        "n_total": len(traces),
        "n_answerable": len(ans_traces),
        "hit1": sum(t["hit1"] for t in ans_traces),
        "hit1_rate": round(sum(t["hit1"] for t in ans_traces) / len(ans_traces), 4),
        "hit3": sum(t["hit3"] for t in ans_traces),
        "hit3_rate": round(sum(t["hit3"] for t in ans_traces) / len(ans_traces), 4),
        "context_hit": sum(t["context_hit"] for t in ans_traces),
        "context_evidence_rate": round(sum(t["context_hit"] for t in ans_traces) / len(ans_traces), 4),
        "avg_context_chars": round(statistics.mean(t["context_chars"] for t in traces), 1),
        "median_ms": round(statistics.median(t["elapsed_ms"] for t in traces), 2)
    }
    
    out_dir = Path(f"results/retrieval_{name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary

def main():
    chunks = json.loads(Path("data/chunks.json").read_text(encoding="utf-8"))
    questions = json.loads(Path("data/questions.json").read_text(encoding="utf-8"))
    
    print("=== Initializing Retrievers ===")
    bm25 = BM25Retriever(chunks)
    dense = DenseVectorRetriever()
    hybrid = HybridRetriever(dense, bm25)
    
    summaries = []
    print("\n--- Evaluating 1: BM25 (Lexical) ---")
    s_bm25 = run_retriever_eval("bm25_lexical", bm25.search, questions, chunks)
    summaries.append(s_bm25)
    print(json.dumps(s_bm25, indent=2))
    
    print("\n--- Evaluating 2: Dense (pgvector) ---")
    s_dense = run_retriever_eval("dense_pgvector", dense.search, questions, chunks)
    summaries.append(s_dense)
    print(json.dumps(s_dense, indent=2))
    
    print("\n--- Evaluating 3: Hybrid (RRF) ---")
    s_hybrid = run_retriever_eval("hybrid_rrf", hybrid.search, questions, chunks)
    summaries.append(s_hybrid)
    print(json.dumps(s_hybrid, indent=2))
    
    Path("results/retrieval_comparison.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\nAll 3 retrieval methods evaluated and saved to results/retrieval_comparison.json!")

if __name__ == "__main__":
    main()
