"""100% Local Agent Controller for Day 7: Grounding-first, Rule & Similarity Reasoner.

Strictly local: Zero external APIs, zero Gemini/OpenAI calls, zero network dependency.
Complies with Controller protocol in starter/agentic.py.
"""
import re
from typing import Any, Dict, List, Optional
from src.retrievers import tokenize


def extract_key_phrases(text: str) -> List[str]:
    """Extract candidate keyphrases prioritized by informativeness."""
    quotes = re.findall(r'"([^"]+)"', text)
    stopwords = {
        "what", "which", "where", "when", "who", "whom", "whose", "why", "how",
        "is", "was", "are", "were", "the", "a", "an", "in", "on", "at", "by",
        "for", "with", "about", "against", "between", "into", "through", "during",
        "before", "after", "above", "below", "to", "from", "up", "down", "of",
        "and", "or", "did", "does", "do", "how", "many", "much", "chapter", "roman", "text"
    }
    caps = [w for w in re.findall(r'\b[A-Z][a-z]+\b', text) if w.lower() not in stopwords]
    tokens = tokenize(text)
    keywords = [w for w in tokens if w not in stopwords and len(w) > 3]
    return list(dict.fromkeys(quotes + caps + keywords))


class LocalAgentController:
    """Local deterministic grounding & synthesis controller.

    - Turn 0: Emits 'search' tool call with original question.
    - Subsequent turns: Inspects retrieved context from history[0]['context'] and 'context_ids'.
      * Verifies that key named entities from the question are present in the chunk.
      * Scores candidate blocks by term overlap and chunk similarity.
      * If evidence is found: synthesizes answer + cites [chunk_id].
      * If key entities are completely missing: abstains gracefully (detects unanswerable).
    """

    def __init__(self, confidence_threshold: float = 0.25, max_local_refinements: int = 2):
        self.confidence_threshold = confidence_threshold
        self.max_local_refinements = max_local_refinements

    def decide(self, question: str, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Decide next action based on question and interaction history."""
        if not history or not history[0].get("context_ids"):
            return {"action": "search", "query": question}

        context_str = history[0].get("context", "")
        context_ids = history[0].get("context_ids", [])
        queries_so_far = history[0].get("queries", [])

        # Extract question entities (Capitalized tokens)
        stopwords = {
            "what", "which", "where", "when", "who", "whom", "why", "how", "is",
            "was", "are", "were", "the", "a", "an", "in", "on", "at", "by", "for",
            "with", "about", "to", "from", "of", "and", "or", "did", "does", "many", "much",
            "chapter", "part"
        }
        entities = [w.lower() for w in re.findall(r'\b[A-Z][a-z]+\b', question) if w.lower() not in stopwords]
        
        q_tokens = set(tokenize(question))
        important_q_tokens = {t for t in q_tokens if t not in stopwords and len(t) > 2}

        blocks = context_str.split("\n\n") if context_str else []
        
        best_cid = None
        best_overlap_score = 0.0
        best_sentence = ""
        best_entity_matches = 0

        for idx, cid in enumerate(context_ids):
            block_text = blocks[idx] if idx < len(blocks) else context_str
            block_tokens = set(tokenize(block_text))
            
            # Check entity coverage
            entity_matches = sum(1 for e in entities if e in block_tokens)
            overlap = important_q_tokens.intersection(block_tokens)
            overlap_score = len(overlap) / (len(important_q_tokens) or 1)

            # Weight overlap by entity coverage
            if entities and entity_matches == 0:
                # Key entity not found in this block at all
                continue

            if overlap_score > best_overlap_score:
                best_overlap_score = overlap_score
                best_entity_matches = entity_matches
                best_cid = cid
                
                sentences = re.split(r'(?<=[.!?])\s+', block_text)
                best_s_overlap = -1
                for s in sentences:
                    s_ov = len(important_q_tokens.intersection(set(tokenize(s))))
                    if s_ov > best_s_overlap:
                        best_s_overlap = s_ov
                        best_sentence = s.strip()

        # Query reformulation if overlap is insufficient
        if best_overlap_score < self.confidence_threshold and len(queries_so_far) < self.max_local_refinements:
            keyphrases = extract_key_phrases(question)
            if keyphrases:
                refined_query = " ".join(keyphrases[:4])
                if refined_query not in queries_so_far and refined_query != question:
                    return {"action": "search", "query": refined_query}

        # Strict abstention if core entities missing or confidence too low
        if best_overlap_score < self.confidence_threshold or not best_cid:
            return {
                "action": "answer",
                "answer": (
                    "Based on the provided historical records, there is insufficient evidence "
                    "in the text to answer this question."
                ),
                "citations": []
            }

        # Answer strictly grounded on the best chunk
        clean_sentence = re.sub(r'^\[[^\]]+\]\s*', '', best_sentence)
        answer_text = f"According to [{best_cid}], {clean_sentence}"
        return {
            "action": "answer",
            "answer": answer_text,
            "citations": [best_cid]
        }
