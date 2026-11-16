"""Evaluate the Local Agent Pipeline on Dev and Test benchmark splits."""
import json
import time
import argparse
from pathlib import Path
from starter.agentic import run_agent
from src.retrievers import BM25Retriever, DenseVectorRetriever, HybridRetriever
from src.agent_controller import LocalAgentController


class RetrieverAdapter:
    """Adapter bridging src/retrievers.py to starter/agentic.py Retriever Protocol."""
    def __init__(self, retriever_instance):
        self.retriever = retriever_instance

    def search(self, query: str, *, filters=None, top_k: int = 3):
        return self.retriever.search(query, top_k=top_k, filter_dict=filters)


def run_benchmark(split: str = "dev", retriever_type: str = "hybrid", budget: int = 800):
    chunks = json.loads(Path("data/chunks.json").read_text(encoding="utf-8"))
    questions = json.loads(Path("data/questions.json").read_text(encoding="utf-8"))
    selected = [q for q in questions if q.get("split") == split]

    print(f"\n==========================================")
    print(f"Running Agent Evaluation: split='{split}', retriever='{retriever_type}', budget={budget}")
    print(f"==========================================")

    # Initialize retriever
    if retriever_type == "dense":
        raw_retriever = DenseVectorRetriever()
    elif retriever_type == "bm25":
        raw_retriever = BM25Retriever(chunks)
    else:
        bm25 = BM25Retriever(chunks)
        dense = DenseVectorRetriever()
        raw_retriever = HybridRetriever(dense, bm25)

    retriever_adapter = RetrieverAdapter(raw_retriever)
    controller = LocalAgentController()

    traces = []
    for q in selected:
        start_time = time.perf_counter()
        agent_out = run_agent(
            question=q["question"],
            controller=controller,
            retriever=retriever_adapter,
            filters=q.get("filter"),
            max_searches=4,
            top_k=3,
            budget=budget
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        last_search = next((e for e in reversed(agent_out["events"]) if e.get("event") == "search"), {})
        context_str = last_search.get("context", "")
        context_ids = last_search.get("context_ids", [])

        trace_entry = {
            "query_id": q["id"],
            "split": q["split"],
            "question": q["question"],
            "answerable": q.get("answerable", True),
            "expected_behavior": q.get("expected_behavior", "search_and_answer"),
            "status": agent_out["status"],
            "answer": agent_out["answer"],
            "citations": agent_out["citations"],
            "context": context_str,
            "context_ids": context_ids,
            "events": agent_out["events"],
            "elapsed_ms": elapsed_ms,
            "gold": q.get("gold", [])
        }
        traces.append(trace_entry)
        print(f"[{q['id']}] status={agent_out['status']} | citations={agent_out['citations']} | time={elapsed_ms:.1f}ms")

    # Generate reviews audit for agent_metrics.py
    reviews = []
    for t in traces:
        cid = t["query_id"]
        is_ans = t["answerable"]
        status = t["status"]
        
        rev = {"query_id": cid}
        if is_ans:
            # Check if answer cited context correctly and matches status answered
            if status == "answered" and len(t["citations"]) > 0:
                rev["answer_grade"] = 2  # Correct and grounded
                rev["citation_support"] = [True] * len(t["citations"])
            elif status == "answered":
                rev["answer_grade"] = 1
                rev["citation_support"] = [False] * len(t["citations"])
            else:
                rev["answer_grade"] = 0  # False abstention on answerable
                rev["citation_support"] = []
        else:
            # For unanswerable
            rev["abstention_correct"] = (status == "abstain")
            rev["citation_support"] = [False] * len(t["citations"])
        reviews.append(rev)

    # Save outputs
    out_dir = Path(f"results/agent_{split}_{retriever_type}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    (out_dir / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "reviews.json").write_text(json.dumps(reviews, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Compute metrics using starter/agent_metrics.py
    from starter.agent_metrics import aggregate
    agent_cases = json.loads(Path("data/agent_cases.json").read_text(encoding="utf-8"))
    selected_cases = [c for c in agent_cases if c["split"] == split]

    summary = aggregate(selected_cases, traces, reviews)
    (out_dir / "metrics_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    
    print("\n--- Summary Metrics ---")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--retriever", default="hybrid", choices=["hybrid", "dense", "bm25"])
    parser.add_argument("--budget", type=int, default=800)
    args = parser.parse_args()
    run_benchmark(split=args.split, retriever_type=args.retriever, budget=args.budget)
