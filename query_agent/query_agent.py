from __future__ import annotations

import logging
import uuid
from typing import Any

from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from config import Config, get_config
from core.groq_rate_limiter import groq_invoke_chat as groq_invoke
from core.state import QueryState
from query_agent.faiss_store import FAISSStore
from query_agent.feedback_tracker import FeedbackTracker
from query_agent.query_preprocessor import QueryPreprocessor
from query_agent.reranker import Reranker

log = logging.getLogger(__name__)

# LLM intent classifier prompt — returns one word only, costs ~10 tokens
_INTENT_SYSTEM = (
    "You are an intent classifier for a financial audit chatbot. "
    "Classify the user message into exactly ONE of these labels:\n"
    "  greeting  — hello, hi, hey, good morning, etc.\n"
    "  closing   — thanks, thank you, bye, ok, great, noted, cheers, etc.\n"
    "  audit     — any question about financial data, transactions, risks, anomalies, tables, columns, reports\n\n"
    "Reply with ONLY the single label word. No punctuation, no explanation."
)

# Friendly responses — no LLM needed for these
_GREETING_REPLY = (
    "Hello! I'm your 4sight audit assistant. "
    "I can help you explore flagged transactions, risk signals, anomalies, "
    "control weaknesses, and recommendations from this financial analysis. "
    "Where would you like to start?"
)
_CLOSING_REPLY = (
    "You're welcome! Feel free to ask anytime if you need more audit insights "
    "on flagged transactions, risk signals, or recommendations."
)


class QueryAgent:
    def __init__(
        self,
        faiss_store: FAISSStore,
        config: Config | None = None,
    ):
        self.config = config or get_config()
        self.store = faiss_store
        self.preprocessor = QueryPreprocessor(self.config)
        self.reranker = Reranker(self.config)
        self.llm = ChatGroq(
            api_key=self.config.groq_api_key,
            model=self.config.llm_model,
            temperature=self.config.llm_temperature,
            max_tokens=self.config.llm_max_tokens,
        )
        # A zero-temperature classifier LLM — deterministic, single-word output
        self._classifier_llm = ChatGroq(
            api_key=self.config.groq_api_key,
            model=self.config.llm_model,
            temperature=0.0,
            max_tokens=5,
        )
        # Lightweight rewriter — fixes typos/abbreviations dynamically, no dictionaries needed
        self._rewriter_llm = ChatGroq(
            api_key=self.config.groq_api_key,
            model=self.config.llm_model,
            temperature=0.0,
            max_tokens=80,
        )
        try:
            self.feedback_tracker = FeedbackTracker(self.config)
        except Exception as exc:
            log.warning("FeedbackTracker unavailable: %s", exc)
            self.feedback_tracker = None
        self.raw_store = None   # set by api/main.py after load

        self._graph = self._build_graph()

    # ── Nodes ─────────────────────────────────────────────────────────────────

    def _classify_intent_node(self, state: QueryState) -> QueryState:
        """LLM-based dynamic intent classifier — runs before FAISS retrieval.

        Classifies into: greeting | closing | audit
        Greetings and closings get instant friendly replies — no FAISS, no LLM answer needed.
        """
        try:
            response = groq_invoke(
                self._classifier_llm,
                [SystemMessage(content=_INTENT_SYSTEM), HumanMessage(content=state["query"])],
                agent_name="query_intent",
            )
            intent = response.content.strip().lower().split()[0]
        except Exception as exc:
            log.warning("Intent classification failed, defaulting to audit: %s", exc)
            intent = "audit"

        log.info("Intent classified as '%s' for query: %s", intent, state["query"][:60])

        if intent == "greeting":
            state["query_intent"] = "greeting"
            state["final_answer"] = _GREETING_REPLY
            state["llm_answer"] = _GREETING_REPLY
            state["resolution_confidence"] = 1.0
            state["can_resolve"] = True
        elif intent == "closing":
            state["query_intent"] = "closing"
            state["final_answer"] = _CLOSING_REPLY
            state["llm_answer"] = _CLOSING_REPLY
            state["resolution_confidence"] = 1.0
            state["can_resolve"] = True
        else:
            state["query_intent"] = "audit"

        return state

    def _rewrite_query_node(self, state: QueryState) -> QueryState:
        """LLM-based dynamic query rewriter — fixes typos, abbreviations, and informal phrasing.

        Runs after intent classification (audit path only), before FAISS retrieval.
        No static dictionaries — the LLM corrects anything: typos, abbreviations, mixed case, etc.
        Stores the cleaned text in preprocessed_query; raw query is preserved.
        """
        try:
            response = groq_invoke(
                self._rewriter_llm,
                [
                    SystemMessage(
                        content=(
                            "You are a spell-checker for a financial audit chatbot. "
                            "Your ONLY job is to fix spelling mistakes — do NOT remove words, do NOT rephrase, do NOT summarise. "
                            "Keep every word the user wrote. Only fix the spelling of misspelled words. "
                            "Domain terms to prefer: transaction, flagged, cheque, invoice, payment, voucher, "
                            "journal, ledger, receipt, purchase, vendor, payroll, duplicate, variance, "
                            "reconciliation, anomaly, risk, signal, insight, recommendation. "
                            "Return the full corrected sentence. If nothing is misspelled, return it unchanged."
                        )
                    ),
                    HumanMessage(content=state["query"]),
                ],
                agent_name="query_rewriter",
            )
            rewritten = response.content.strip()
            if rewritten:
                log.info("Query rewritten: '%s' → '%s'", state["query"][:80], rewritten[:80])
                state["preprocessed_query"] = rewritten
        except Exception as exc:
            log.warning("Query rewriting failed, using raw query: %s", exc)
        return state

    def _preprocess_query_node(self, state: QueryState) -> QueryState:
        # If rewriter already set preprocessed_query, refine it further; otherwise start from raw query
        base = state.get("preprocessed_query") or state["query"]
        state["preprocessed_query"] = self.preprocessor.preprocess(base)
        return state

    def _retrieve_docs_node(self, state: QueryState) -> QueryState:
        query = state.get("preprocessed_query") or state["query"]

        # Search analysis index
        if self.config.use_hybrid_search and self.config.use_dynamic_topk:
            docs = self.store.dynamic_top_k_search(query)
        elif self.config.use_hybrid_search:
            docs = self.store.hybrid_search(query, k=self.config.min_top_k)
        else:
            docs = self.store.semantic_only_search(query, k=self.config.min_top_k)

        # Also search shared raw index if loaded
        if self.raw_store and self.raw_store.is_loaded():
            if self.config.use_hybrid_search:
                raw_docs = self.raw_store.hybrid_search(query, k=self.config.min_top_k)
            else:
                raw_docs = self.raw_store.semantic_only_search(query, k=self.config.min_top_k)
            # Merge and keep top max_top_k by similarity
            combined = docs + raw_docs
            combined.sort(key=lambda x: x["similarity"], reverse=True)
            docs = combined[:self.config.max_top_k]

        state["retrieved_docs"] = docs
        state["num_retrieved"] = len(docs)
        state["top_similarity"] = docs[0]["similarity"] if docs else 0.0
        return state

    def _rerank_docs_node(self, state: QueryState) -> QueryState:
        if self.config.use_reranking and state.get("retrieved_docs"):
            query = state.get("preprocessed_query") or state["query"]
            reranked = self.reranker.rerank_with_fusion(query, list(state["retrieved_docs"]))
        else:
            reranked = state.get("retrieved_docs", [])
            for doc in reranked:
                doc["fused_score"] = doc.get("similarity", 0.0)
        state["reranked_docs"] = reranked[:self.config.final_top_k]
        return state

    def _check_similarity_node(self, state: QueryState) -> QueryState:
        docs = state.get("reranked_docs", [])
        top_score = docs[0].get("fused_score", 0.0) if docs else 0.0

        if top_score < self.config.similarity_threshold:
            state["query_intent"] = "out_of_scope"
            state["final_answer"] = (
                "I don't have enough audit data to answer that. "
                "Try asking about flagged transactions, risk signals, specific tables, "
                "columns, or recommendations from the financial analysis."
            )
            state["resolution_confidence"] = 0.0
            state["can_resolve"] = False
        else:
            state["query_intent"] = "data_related"
            state["top_similarity"] = top_score
        return state

    def _generate_answer_node(self, state: QueryState) -> QueryState:
        docs = state.get("reranked_docs", [])
        context = self.store.format_context(docs)
        query = state["query"]
        history = state.get("chat_history", [])

        system = (
            "You are 4sight, an AI audit assistant. Answer the user's question directly and conversationally.\n\n"
            "Rules:\n"
            "- Give a direct, clear answer — no preamble like 'Based on the provided context' or 'I can infer'.\n"
            "- Never start your reply with 'Based on', 'According to', 'From the context', or 'I don't have enough'.\n"
            "- Use bullet points or short paragraphs for clarity.\n"
            "- Mention table names, amounts, and risk levels when relevant.\n"
            "- Answer ONLY from the audit context below. Do NOT use general knowledge.\n"
            "- If the context truly has no relevant information, reply: "
            "'I don\\'t have data on that yet. Try asking about flagged transactions, risk signals, or a specific table.'\n\n"
            f"AUDIT CONTEXT:\n{context}"
        )

        # Build message list: system → prior conversation turns → current question
        messages = [SystemMessage(content=system)]
        for turn in history:
            if turn["role"] == "user":
                messages.append(HumanMessage(content=turn["content"]))
            elif turn["role"] == "assistant":
                messages.append(AIMessage(content=turn["content"]))
        messages.append(HumanMessage(content=query))

        try:
            response = groq_invoke(self.llm, messages, agent_name="query")
            state["llm_answer"] = response.content
            state["final_answer"] = response.content
        except Exception as exc:
            log.error("Answer generation failed: %s", exc)
            state["final_answer"] = "An error occurred while generating the answer."
            state["llm_answer"] = ""
        return state

    def _assess_confidence_node(self, state: QueryState) -> QueryState:
        query = state["query"]
        answer = state.get("final_answer", "")
        n_docs = state.get("num_retrieved", 0)

        prompt = (
            f"Rate the quality and confidence of this answer on a scale of 0.0 to 1.0. "
            f"Return ONLY a single decimal number.\n"
            f"Query: {query}\nAnswer: {answer[:300]}\nDocuments retrieved: {n_docs}"
        )
        try:
            response = groq_invoke(self.llm, [HumanMessage(content=prompt)], agent_name="query")
            conf = float(response.content.strip().split()[0])
            conf = max(0.0, min(1.0, conf))
        except Exception:
            conf = state.get("top_similarity", 0.5)

        state["resolution_confidence"] = conf
        state["can_resolve"] = conf >= self.config.confidence_threshold
        return state

    def _track_feedback_node(self, state: QueryState) -> QueryState:
        if self.feedback_tracker and self.config.track_feedback:
            try:
                docs = state.get("reranked_docs", [])
                doc_ids = [d.get("id", "") for d in docs]
                scores = [d.get("fused_score", 0.0) for d in docs]
                self.feedback_tracker.log_retrieval(
                    session_id=state["session_id"],
                    query=state["query"],
                    preprocessed=state.get("preprocessed_query", ""),
                    doc_ids=doc_ids,
                    scores=scores,
                    answer=state.get("final_answer", ""),
                    confidence=state.get("resolution_confidence", 0.0),
                    intent=state.get("query_intent", ""),
                )
            except Exception as exc:
                log.error("FeedbackTracker.log_retrieval failed: %s", exc)
        return state

    # ── Graph ──────────────────────────────────────────────────────────────────

    def _route_after_intent(self, state: QueryState) -> str:
        """greeting/closing → skip FAISS, go straight to track_feedback."""
        intent = state.get("query_intent", "audit")
        if intent in ("greeting", "closing"):
            return "shortcircuit"
        return "continue"

    def _route_after_similarity(self, state: QueryState) -> str:
        return state.get("query_intent", "out_of_scope")

    def _build_graph(self) -> Any:
        graph = StateGraph(QueryState)

        graph.add_node("classify_intent",  self._classify_intent_node)
        graph.add_node("rewrite_query",    self._rewrite_query_node)
        graph.add_node("preprocess_query", self._preprocess_query_node)
        graph.add_node("retrieve_docs",    self._retrieve_docs_node)
        graph.add_node("rerank_docs",      self._rerank_docs_node)
        graph.add_node("check_similarity", self._check_similarity_node)
        graph.add_node("generate_answer",  self._generate_answer_node)
        graph.add_node("assess_confidence",self._assess_confidence_node)
        graph.add_node("track_feedback",   self._track_feedback_node)

        graph.set_entry_point("classify_intent")

        # greeting / closing → skip all retrieval, go straight to track_feedback
        # audit → rewrite typos first, then preprocess + retrieval
        graph.add_conditional_edges(
            "classify_intent",
            self._route_after_intent,
            {
                "shortcircuit": "track_feedback",
                "continue":     "rewrite_query",
            },
        )

        graph.add_edge("rewrite_query",    "preprocess_query")
        graph.add_edge("preprocess_query", "retrieve_docs")
        graph.add_edge("retrieve_docs",    "rerank_docs")
        graph.add_edge("rerank_docs",      "check_similarity")
        graph.add_conditional_edges(
            "check_similarity",
            self._route_after_similarity,
            {
                "out_of_scope": "track_feedback",
                "data_related": "generate_answer",
            },
        )
        graph.add_edge("generate_answer",  "assess_confidence")
        graph.add_edge("assess_confidence","track_feedback")
        graph.add_edge("track_feedback",   END)

        return graph.compile()

    # ── Public ─────────────────────────────────────────────────────────────────

    def process_query(
        self,
        query: str,
        session_id: str | None = None,
        chat_history: list | None = None,
    ) -> dict:
        if session_id is None:
            session_id = f"sess_{uuid.uuid4().hex[:8]}"

        initial_state = QueryState(
            query=query,
            session_id=session_id,
            chat_history=chat_history or [],
            preprocessed_query=None,
            retrieved_docs=[],
            reranked_docs=[],
            metadata_context="",
            llm_answer="",
            final_answer="",
            resolution_confidence=0.0,
            can_resolve=False,
            query_intent="",
            top_similarity=0.0,
            num_retrieved=0,
            feedback=None,
        )

        result = self._graph.invoke(initial_state)
        answer = result.get("final_answer", "")

        # Append this turn to history for the caller to carry forward
        updated_history = list(chat_history or [])
        updated_history.append({"role": "user", "content": query})
        updated_history.append({"role": "assistant", "content": answer})

        return {
            "query": result["query"],
            "preprocessed_query": result.get("preprocessed_query"),
            "answer": answer,
            "confidence": result.get("resolution_confidence", 0.0),
            "resolved": result.get("can_resolve", False),
            "query_intent": result.get("query_intent", ""),
            "top_similarity": result.get("top_similarity", 0.0),
            "num_docs_retrieved": result.get("num_retrieved", 0),
            "chat_history": updated_history,
        }
