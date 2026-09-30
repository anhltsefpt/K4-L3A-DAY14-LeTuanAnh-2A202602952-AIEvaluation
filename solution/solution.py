"""
Day 14 — AI Evaluation & Benchmarking Pipeline
AICB-P1: AI Practical Competency Program, Phase 1

Key concepts from lecture:
    - Evaluation = Scientific Method for AI (Hypothesis → Experiment → Measure → Conclude → Iterate)
    - 4 nhóm metrics: Task Completion, Answer Quality, RAG-Specific, Business
    - RAG pipeline metrics: Context Recall → Context Precision → Faithfulness → Answer Relevancy
    - LLM-as-Judge: rubric scoring 1-5, detect bias (positional, verbosity, self-preference)
    - Golden dataset: stratified sampling (5 Easy + 7 Medium + 5 Hard + 3 Adversarial)
    - Failure taxonomy: hallucination, irrelevant, incomplete, off_topic, refusal
    - 5 Whys method for root cause analysis
    - CI/CD integration: eval as quality gate (score < threshold = block deploy)
    - Continuous Improvement Loop: Evaluate → Analyze → Improve → Augment → Repeat

Instructions:
    1. Fill in every required section marked with TODO.
    2. Do NOT change class/function signatures. The optional ``contexts``
       parameter in ``run_full_eval`` is part of the required interface.
    3. Copy this file to solution/solution.py when done.
    4. Run: pytest tests/ -v

The reranking helper is an optional bonus exercise and may remain unimplemented.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Task 1 — Data Models (Golden Dataset + Evaluation Results)
# ---------------------------------------------------------------------------

@dataclass
class QAPair:
    """
    A question-answer pair for evaluation (part of the Golden Dataset).

    From lecture: Golden dataset cần có:
        - question: câu hỏi user
        - ground_truth (expected_answer): expert-written expected answer
        - context: source documents cần retrieve
        - metadata: difficulty (easy/medium/hard), category, source_docs

    Fields:
        question:        The question to answer.
        expected_answer: The reference/ground-truth answer (expert-written).
        context:            Source context (may be empty string if not applicable).
        metadata:           Optional metadata dict (difficulty, category, etc.).
        retrieved_contexts: List of retrieved chunks (ORDER = retriever rank).
                            Used by the retrieval-side metrics (Task 2b).
    """
    question: str
    expected_answer: str
    context: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    retrieved_contexts: list[str] = field(default_factory=list)


@dataclass
class EvalResult:
    """
    Evaluation result for a single Q&A pair.

    From lecture - RAG metrics pipeline:
        Question → Retriever → Context → Generator → Answer
        Each step has a metric: Context Recall, Context Precision, Faithfulness, Answer Relevancy

    From lecture - Score interpretation:
        0.8-1.0: Good (Monitor, maintain)
        0.6-0.8: Needs work (Analyze failures, iterate)
        < 0.6: Significant issues (Deep investigation required)

    Fields:
        qa_pair:        The original QAPair.
        actual_answer:  What the agent actually returned.
        faithfulness:   Float 0-1, how grounded the answer is in context.
        relevance:      Float 0-1, how relevant the answer is to the question.
        completeness:   Float 0-1, how complete the answer is vs expected.
        passed:         True if all three scores >= 0.5.
        failure_type:   None if passed, otherwise one of:
                        "hallucination", "irrelevant", "incomplete", "off_topic".
        context_precision: Float 0-1 or None — quality of retrieval ranking.
        context_recall:    Float 0-1 or None — coverage of expected by context.
                        (Both stay None unless retrieved chunks are supplied;
                         they are NOT part of overall_score().)
    """
    qa_pair: QAPair
    actual_answer: str
    faithfulness: float
    relevance: float
    completeness: float
    passed: bool
    failure_type: str | None = None
    context_precision: float | None = None
    context_recall: float | None = None

    def overall_score(self) -> float:
        """Compute the average of faithfulness, relevance, and completeness.

        Returns:
            (faithfulness + relevance + completeness) / 3.0

        The two retrieval metrics are deliberately excluded: they diagnose the
        retriever, not the answer, and are None whenever no chunks were passed.
        """
        return (self.faithfulness + self.relevance + self.completeness) / 3.0


# ---------------------------------------------------------------------------
# Task 2 — RAGAS Evaluator (Simplified word-overlap heuristic)
# ---------------------------------------------------------------------------
# In production, replace with actual RAGAS framework:
#   from ragas import evaluate
#   from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecision
#
# Or DeepEval:
#   from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
#   assert_test(test_case, [faithfulness, hallucination])
#
# Or TruLens:
#   from trulens.core import Feedback
#   f_groundedness = Feedback(provider.groundedness_measure_with_cot_reasons)
# ---------------------------------------------------------------------------

# Common English stopwords are ignored so overlap reflects *content* words,
# not filler (otherwise "is"/"a"/"the" inflate every score).
STOPWORDS: set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "with", "as", "by", "and", "or",
    "it", "its", "this", "that", "these", "those", "from", "into", "than",
}


def _tokenize(text: str) -> set[str]:
    """Lowercase word tokenization, ignoring punctuation and stopwords."""
    if not text:
        return set()
    tokens = re.findall(r"\b\w+\b", text.lower())
    return {t for t in tokens if t not in STOPWORDS}


def _coverage(covered_by: set[str], reference: set[str]) -> float:
    """Fraction of ``reference`` tokens present in ``covered_by``.

    An empty reference means "nothing to cover", which scores a perfect 1.0 —
    this is what keeps an empty question/expected/answer from being punished.
    """
    if not reference:
        return 1.0
    ratio = len(covered_by & reference) / len(reference)
    return max(0.0, min(1.0, ratio))


class RAGASEvaluator:
    """
    Evaluates RAG pipeline outputs using RAGAS-inspired heuristics.

    All metrics use word overlap rather than LLM calls for simplicity.
    Replace with actual LLM-based evaluation in production.
    """

    def evaluate_faithfulness(self, answer: str, context: str) -> float:
        """
        Measure how grounded the answer is in the context.

        Heuristic:
            answer_tokens = _tokenize(answer)
            context_tokens = _tokenize(context)
            faithfulness = |answer_tokens ∩ context_tokens| / |answer_tokens|
            Clamp to [0.0, 1.0]. Return 1.0 if answer is empty.

        Returns:
            float in [0.0, 1.0] — 1.0 = fully grounded in context.
        """
        return _coverage(_tokenize(context), _tokenize(answer))

    def evaluate_relevance(self, answer: str, question: str) -> float:
        """
        Measure how relevant the answer is to the question.

        Heuristic:
            relevance = |answer_tokens ∩ question_tokens| / |question_tokens|
            Clamp to [0.0, 1.0]. Return 1.0 if question is empty.

        Returns:
            float in [0.0, 1.0]
        """
        return _coverage(_tokenize(answer), _tokenize(question))

    def evaluate_completeness(self, answer: str, expected: str) -> float:
        """
        Measure how well the answer covers the expected answer.

        Heuristic:
            completeness = |answer_tokens ∩ expected_tokens| / |expected_tokens|
            Clamp to [0.0, 1.0]. Return 1.0 if expected is empty.

        Returns:
            float in [0.0, 1.0]
        """
        return _coverage(_tokenize(answer), _tokenize(expected))

    # -----------------------------------------------------------------------
    # Task 2b — Retrieval-side metrics (evaluate the GET-CONTEXT step)
    # -----------------------------------------------------------------------
    # From lecture (RAG pipeline): Context Recall → Context Precision →
    #   Faithfulness → Answer Relevancy. The two below score the RETRIEVER,
    #   operating on a LIST of chunks (order = retriever rank).
    # -----------------------------------------------------------------------

    def evaluate_context_recall(self, contexts: list[str], expected: str) -> float:
        """Context Recall — how much of the expected answer is covered by the
        UNION of retrieved chunks.

        Heuristic:
            union_tokens = ⋃ _tokenize(chunk) for chunk in contexts
            recall = |expected_tokens ∩ union_tokens| / |expected_tokens|
            Clamp to [0.0, 1.0]. Return 1.0 if expected is empty.

        Low recall => retriever missed evidence the answer needs.
        """
        union_tokens: set[str] = set()
        for chunk in contexts or []:
            union_tokens |= _tokenize(chunk)
        return _coverage(union_tokens, _tokenize(expected))

    def evaluate_context_precision(
        self,
        contexts: list[str],
        expected: str,
        relevance_threshold: float = 0.1,
    ) -> float:
        """Context Precision — RANK-AWARE Average Precision (AP@K), like RAGAS.
        Rewards retrievers that place RELEVANT chunks BEFORE noise.

        Steps:
            1. A chunk is "relevant" if it covers >= relevance_threshold of the
               expected tokens:  |chunk ∩ expected| / |expected| >= threshold
            2. Precision@k = (#relevant in top-k) / k
            3. AP@K = (1 / #relevant) * Σ_k [ Precision@k · relevant_k ]

        Return 1.0 if expected empty; 0.0 if no chunks or none relevant.
        Reordering relevant chunks earlier (reranking) raises this score.
        """
        expected_tokens = _tokenize(expected)
        if not expected_tokens:
            return 1.0
        if not contexts:
            return 0.0

        # Step 1 — mark each chunk relevant / not, keeping retriever rank order.
        relevant_flags = [
            _coverage(_tokenize(chunk), expected_tokens) >= relevance_threshold
            for chunk in contexts
        ]
        relevant_total = sum(relevant_flags)
        if relevant_total == 0:
            return 0.0

        # Steps 2-3 — Average Precision: only positions holding a relevant chunk
        # contribute, so relevant chunks ranked early score higher.
        hits = 0
        precision_sum = 0.0
        for rank, is_relevant in enumerate(relevant_flags, start=1):
            if is_relevant:
                hits += 1
                precision_sum += hits / rank
        return max(0.0, min(1.0, precision_sum / relevant_total))

    def run_full_eval(
        self,
        answer: str,
        question: str,
        context: str,
        expected: str,
        contexts: list[str] | None = None,
    ) -> EvalResult:
        """
        Run the three answer-side evaluations and, when ``contexts`` is
        supplied, both retrieval-side evaluations.

        passed = True if all three scores >= 0.5.

        failure_type determination (first match wins):
            faithfulness < 0.3  → "hallucination"
            relevance < 0.3     → "irrelevant"
            completeness < 0.3  → "incomplete"
            otherwise if failed → "off_topic"

        Retrieval wiring:
            contexts is None → context_recall and context_precision stay None
            contexts provided → evaluate and store both retrieval metrics

        The two retrieval metrics diagnose the retriever and do not change the
        three-metric ``passed`` rule or ``overall_score()``.

        Returns:
            EvalResult with all fields populated.
        """
        faithfulness = self.evaluate_faithfulness(answer, context)
        relevance = self.evaluate_relevance(answer, question)
        completeness = self.evaluate_completeness(answer, expected)

        passed = faithfulness >= 0.5 and relevance >= 0.5 and completeness >= 0.5

        failure_type: str | None = None
        if not passed:
            if faithfulness < 0.3:
                failure_type = "hallucination"
            elif relevance < 0.3:
                failure_type = "irrelevant"
            elif completeness < 0.3:
                failure_type = "incomplete"
            else:
                failure_type = "off_topic"

        context_recall: float | None = None
        context_precision: float | None = None
        if contexts is not None:
            context_recall = self.evaluate_context_recall(contexts, expected)
            context_precision = self.evaluate_context_precision(contexts, expected)

        return EvalResult(
            qa_pair=QAPair(
                question=question,
                expected_answer=expected,
                context=context,
                retrieved_contexts=list(contexts) if contexts is not None else [],
            ),
            actual_answer=answer,
            faithfulness=faithfulness,
            relevance=relevance,
            completeness=completeness,
            passed=passed,
            failure_type=failure_type,
            context_precision=context_precision,
            context_recall=context_recall,
        )


# ---------------------------------------------------------------------------
# Reranking helper (used by Exercise 3.5 — boosting Context Precision)
# ---------------------------------------------------------------------------

def rerank_by_overlap(contexts: list[str], query: str) -> list[str]:
    """A minimal lexical reranker: sort chunks by word overlap with the query,
    most-overlapping first. Stand-in for a real cross-encoder reranker.

    Reordering relevant chunks toward the top increases the rank-aware
    Context Precision WITHOUT changing the retrieved set.

    Hint: sorted(contexts, key=lambda c: len(_tokenize(c) & _tokenize(query)),
                 reverse=True)
    """
    query_tokens = _tokenize(query)
    # ``sorted`` is stable, so chunks with equal overlap keep their original
    # retriever order — the reranker only promotes, it never shuffles ties.
    return sorted(
        contexts,
        key=lambda chunk: len(_tokenize(chunk) & query_tokens),
        reverse=True,
    )


# ---------------------------------------------------------------------------
# Task 3 — LLM Judge
# ---------------------------------------------------------------------------
# From lecture:
#   - Judge LLM nhận: question + agent answer + reference answer + rubric
#   - Judge trả về: Score 1-5 + Rationale
#   - Best practices: multiple judges, randomize order, calibrate against human
#   - Biases: positional, verbosity, self-preference
#   - Rubric template:
#       5 = Correct, complete, well-cited
#       4 = Mostly correct, minor gaps
#       3 = Partially correct, some errors
#       2 = Significant errors or missing info
#       1 = Wrong or irrelevant
# ---------------------------------------------------------------------------

class LLMJudge:
    """
    Uses an LLM to score AI responses according to a rubric.
    """

    #: Shared 1-5 rubric wording sent to the judge on every call.
    SCALE_DESCRIPTION = (
        "5 = Correct, complete, well-grounded in the source policy\n"
        "4 = Mostly correct, minor gaps or missing detail\n"
        "3 = Partially correct, some errors\n"
        "2 = Significant errors or missing information\n"
        "1 = Wrong, irrelevant, or fabricated"
    )

    def __init__(self, judge_llm_fn: Callable[[str], str]) -> None:
        self.judge_llm_fn = judge_llm_fn

    def score_response(
        self,
        question: str,
        answer: str,
        rubric: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Score an AI response using the judge LLM.

        Args:
            question: The original question.
            answer:   The AI's answer to score.
            rubric:   Dict mapping criterion name → description.
                      Example: {"accuracy": "Is the answer factually correct?",
                                "clarity": "Is the answer clear and well-structured?"}

        Behavior:
            1. Build a judge prompt that includes the question, answer, and rubric.
            2. Call judge_llm_fn(prompt).
            3. Parse the response for scores.

        For simplicity, if the LLM response can't be parsed as JSON scores,
        return a default score of 0.5 for each criterion.

        Returns:
            {
                "scores":    dict[str, float],  # criterion → score 0-1
                "reasoning": str,               # raw LLM explanation
            }
        """
        criteria = list(rubric)
        prompt = self._build_prompt(question, answer, rubric)

        try:
            raw = self.judge_llm_fn(prompt)
        except Exception as exc:  # a dead judge must not kill the benchmark
            return {
                "scores": {name: 0.5 for name in criteria},
                "reasoning": f"Judge call failed: {exc}",
            }

        raw_text = raw if isinstance(raw, str) else str(raw)
        parsed = self._parse_scores(raw_text)

        # Unparseable or missing criteria fall back to a neutral 0.5 so one bad
        # judge response cannot silently look like a perfect or a zero score.
        scores = {
            name: self._normalize(parsed[name]) if name in parsed else 0.5
            for name in criteria
        }
        return {"scores": scores, "reasoning": raw_text}

    def _build_prompt(
        self,
        question: str,
        answer: str,
        rubric: dict[str, Any],
    ) -> str:
        """Assemble the judge prompt: question + answer + rubric + scale."""
        criteria_lines = "\n".join(f"- {name}: {desc}" for name, desc in rubric.items())
        criteria_keys = ", ".join(f'"{name}"' for name in rubric) or '"overall"'
        return (
            "You are a strict evaluator of a customer-support assistant.\n"
            "Score the ANSWER against the rubric below. Judge only the content: "
            "ignore answer length, formatting, and writing style, and do not "
            "reward an answer for sounding confident.\n\n"
            f"QUESTION:\n{question}\n\n"
            f"ANSWER:\n{answer}\n\n"
            f"RUBRIC CRITERIA:\n{criteria_lines}\n\n"
            f"SCORING SCALE (1-5):\n{self.SCALE_DESCRIPTION}\n\n"
            "Reply with a single JSON object mapping every criterion to its "
            f"numeric score, using exactly these keys: {criteria_keys}.\n"
            'Example: {"accuracy": 4, "clarity": 3}'
        )

    @staticmethod
    def _parse_scores(raw: str) -> dict[str, float]:
        """Extract criterion -> number from a judge reply, tolerating prose."""
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return {}
        try:
            payload = json.loads(match.group(0))
        except (json.JSONDecodeError, ValueError):
            return {}
        if not isinstance(payload, dict):
            return {}
        # A judge often answers {"accuracy": {"score": 4, ...}} — accept both.
        flat: dict[str, float] = {}
        for key, value in payload.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                flat[str(key)] = float(value)
            elif isinstance(value, dict) and isinstance(
                value.get("score"), (int, float)
            ) and not isinstance(value.get("score"), bool):
                flat[str(key)] = float(value["score"])
        return flat

    @staticmethod
    def _normalize(value: float) -> float:
        """Map a judge score onto 0-1, accepting either a 0-1 or a 1-5 reply."""
        if value > 1.0:
            value = (value - 1.0) / 4.0
        return max(0.0, min(1.0, value))

    def detect_bias(self, scores_batch: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Detect potential bias patterns in a batch of judge scores.

        Checks:
            positional_bias: Check if first response consistently scores higher
            leniency_bias:   Average score > 0.8 across all criteria
            severity_bias:   Average score < 0.3 across all criteria

        Args:
            scores_batch: List of score dicts from score_response().

        Returns:
            {
                "positional_bias": bool,
                "leniency_bias":   bool,
                "severity_bias":   bool,
            }
        """
        def mean_of(entry: dict[str, Any]) -> float:
            scores = entry.get("scores") or {}
            values = [
                float(v)
                for v in scores.values()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            ]
            return sum(values) / len(values) if values else 0.0

        if not scores_batch:
            return {
                "positional_bias": False,
                "leniency_bias": False,
                "severity_bias": False,
            }

        per_entry = [mean_of(entry) for entry in scores_batch]
        overall_avg = sum(per_entry) / len(per_entry)

        # Positional bias: the response shown FIRST systematically outscores the
        # rest. Needs at least one comparison entry to mean anything.
        positional_bias = False
        if len(per_entry) > 1:
            rest = per_entry[1:]
            positional_bias = per_entry[0] - (sum(rest) / len(rest)) > 0.1

        return {
            "positional_bias": positional_bias,
            "leniency_bias": overall_avg > 0.8,
            "severity_bias": overall_avg < 0.3,
        }


# ---------------------------------------------------------------------------
# Task 4 — Benchmark Runner
# ---------------------------------------------------------------------------
# From lecture:
#   - CI/CD integration: Framework + CI/CD = quality gate tự động
#   - Agent với faithfulness < 0.7 → không được deploy
#   - Regression = metric drop > 0.05 vs baseline
#   - Triggers: mỗi code release, mỗi prompt change, trước demo/launch
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """
    Runs a full evaluation benchmark.
    """

    def run(
        self,
        qa_pairs: list[QAPair],
        agent_fn: Callable[[str], str],
        evaluator: RAGASEvaluator,
    ) -> list[EvalResult]:
        """
        Run all QA pairs through the agent and evaluate each result.

        Args:
            qa_pairs:   List of QAPair objects.
            agent_fn:   Function str → str (the agent's answer function).
            evaluator:  RAGASEvaluator instance.

        Returns:
            List of EvalResult, one per qa_pair.
        """
        results: list[EvalResult] = []
        for pair in qa_pairs:
            answer = agent_fn(pair.question)
            # Only forward chunks when the retriever actually produced some;
            # an empty list would otherwise score a real 0.0 instead of "n/a".
            contexts = pair.retrieved_contexts or None
            result = evaluator.run_full_eval(
                answer=answer,
                question=pair.question,
                context=pair.context,
                expected=pair.expected_answer,
                contexts=contexts,
            )
            # Keep the ORIGINAL pair so downstream reporting still sees the
            # metadata (id, difficulty, attack_type) the evaluator drops.
            result.qa_pair = pair
            results.append(result)
        return results

    def generate_report(self, results: list[EvalResult]) -> dict[str, Any]:
        """
        Generate an aggregate report from evaluation results.

        Returns:
            {
                "total":            int,
                "passed":           int,
                "pass_rate":        float,  # passed / total
                "avg_faithfulness": float,
                "avg_relevance":    float,
                "avg_completeness": float,
                "avg_context_recall": float | None,
                "avg_context_precision": float | None,
                "failure_types":    dict[str, int],  # type → count
            }

        Average only non-None retrieval scores. Return None for a retrieval
        average when no result contains that metric.
        """
        total = len(results)
        if total == 0:
            return {
                "total": 0,
                "passed": 0,
                "pass_rate": 0.0,
                "avg_faithfulness": 0.0,
                "avg_relevance": 0.0,
                "avg_completeness": 0.0,
                "avg_context_recall": None,
                "avg_context_precision": None,
                "failure_types": {},
            }

        def mean(values: list[float]) -> float:
            return sum(values) / len(values)

        def optional_mean(values: list[float | None]) -> float | None:
            present = [v for v in values if v is not None]
            return mean(present) if present else None

        passed = sum(1 for r in results if r.passed)

        failure_types: dict[str, int] = {}
        for result in results:
            if not result.passed and result.failure_type:
                failure_types[result.failure_type] = (
                    failure_types.get(result.failure_type, 0) + 1
                )

        return {
            "total": total,
            "passed": passed,
            "pass_rate": passed / total,
            "avg_faithfulness": mean([r.faithfulness for r in results]),
            "avg_relevance": mean([r.relevance for r in results]),
            "avg_completeness": mean([r.completeness for r in results]),
            "avg_context_recall": optional_mean([r.context_recall for r in results]),
            "avg_context_precision": optional_mean(
                [r.context_precision for r in results]
            ),
            "failure_types": failure_types,
        }

    def run_regression(self, new_results: list, baseline_results: list) -> dict:
        """Compare new evaluation results against a baseline.

        A regression is when a metric's average drops by more than 0.05 vs baseline.

        Args:
            new_results: List of EvalResult instances (current run)
            baseline_results: List of EvalResult instances (reference/baseline)

        Returns:
            dict with keys:
              - 'new_avg_faithfulness': float
              - 'new_avg_relevance': float
              - 'new_avg_completeness': float
              - 'baseline_avg_faithfulness': float
              - 'baseline_avg_relevance': float
              - 'baseline_avg_completeness': float
              - 'regressions': list[str] — names of metrics that regressed
              - 'passed': bool — True if no regressions
        """
        #: A metric may drop this much before it counts as a regression.
        regression_threshold = 0.05
        metrics = ("faithfulness", "relevance", "completeness")

        def mean(results: list, metric: str) -> float:
            values = [getattr(r, metric) for r in results]
            return sum(values) / len(values) if values else 0.0

        report: dict[str, Any] = {}
        regressions: list[str] = []
        for metric in metrics:
            new_avg = mean(new_results, metric)
            baseline_avg = mean(baseline_results, metric)
            report[f"new_avg_{metric}"] = new_avg
            report[f"baseline_avg_{metric}"] = baseline_avg
            if baseline_avg - new_avg > regression_threshold:
                regressions.append(metric)

        report["regressions"] = regressions
        report["passed"] = not regressions
        return report

    def identify_failures(
        self,
        results: list[EvalResult],
        threshold: float = 0.5,
    ) -> list[EvalResult]:
        """
        Return EvalResults where any score is below threshold.

        Args:
            results:   Full list of EvalResults.
            threshold: Minimum acceptable score for any metric.

        Returns:
            List of failing EvalResults.
        """
        return [
            result
            for result in results
            if min(result.faithfulness, result.relevance, result.completeness)
            < threshold
        ]


# ---------------------------------------------------------------------------
# Task 5 — Failure Analyzer
# ---------------------------------------------------------------------------
# From lecture:
#   Failure Taxonomy:
#     - hallucination: bịa thông tin → faithfulness guardrail yếu
#     - irrelevant: không giải quyết câu hỏi → prompt ambiguous
#     - incomplete: bỏ sót thông tin → context window nhỏ, retrieval thiếu
#     - off_topic: trả lời chủ đề khác → intent detection sai
#     - refusal: từ chối khi nên trả lời → guardrails quá chặt
#
#   5 Whys Method: hỏi "Tại sao?" liên tục cho đến root cause
#   Failure Clustering: fix 1 root cause giải quyết nhiều failures cùng lúc
#   Continuous Improvement: Evaluate → Analyze → Improve → Augment → Repeat
# ---------------------------------------------------------------------------

class FailureAnalyzer:
    """
    Analyzes failed evaluation results to identify patterns and suggest fixes.
    """

    def categorize_failures(
        self, failures: list[EvalResult]
    ) -> dict[str, int]:
        """
        Count failures by failure_type.

        Returns:
            dict mapping failure_type → count.
            Example: {"hallucination": 3, "irrelevant": 2, "incomplete": 5}
        """
        counts: dict[str, int] = {}
        for failure in failures:
            # Anything the evaluator could not label still has to be counted,
            # otherwise clustering silently loses cases.
            label = failure.failure_type or "uncategorized"
            counts[label] = counts.get(label, 0) + 1
        return counts

    def find_root_cause(self, failure: EvalResult) -> str:
        """
        Suggest a root cause for a single failure based on its scores.

        Returns one of these strings based on which score is lowest:
            "Context is missing or irrelevant — improve retrieval"
            "Answer does not address the question — improve prompt clarity"
            "Answer is missing key information — increase context window or improve generation"
            "Multiple issues detected — review full pipeline"
        """
        weak_metric_threshold = 0.5
        scores = {
            "faithfulness": failure.faithfulness,
            "relevance": failure.relevance,
            "completeness": failure.completeness,
        }
        weak = [name for name, value in scores.items() if value < weak_metric_threshold]
        if len(weak) > 1:
            return "Multiple issues detected — review full pipeline"

        lowest = min(scores, key=lambda name: scores[name])
        causes = {
            "faithfulness": "Context is missing or irrelevant — improve retrieval",
            "relevance": (
                "Answer does not address the question — improve prompt clarity"
            ),
            "completeness": (
                "Answer is missing key information — increase context window "
                "or improve generation"
            ),
        }
        return causes[lowest]

    def generate_improvement_log(self, failures: list, suggestions: list[str]) -> str:
        """Generate a Markdown table logging failures and improvement actions.

        Format:
        | Failure ID | Type | Root Cause | Suggested Fix | Status |
        |------------|------|------------|---------------|--------|
        | F001       | ...  | ...        | ...           | Open   |

        Args:
            failures: List of EvalResult instances where passed=False
            suggestions: List of suggestion strings (one per failure, can be shorter list)

        Returns:
            Markdown table string with a row per failure. Status is always "Open".
        """
        header = (
            "| Failure ID | Type | Root Cause | Suggested Fix | Status |\n"
            "|------------|------|------------|---------------|--------|"
        )
        if not failures:
            return header

        def cell(text: Any) -> str:
            # A stray pipe inside a cause or a fix would break the table.
            return str(text).replace("|", "\\|").replace("\n", " ").strip()

        rows = []
        for index, failure in enumerate(failures):
            # suggestions may be shorter than failures — reuse the last one
            # rather than dropping the row entirely.
            if suggestions:
                suggestion = suggestions[min(index, len(suggestions) - 1)]
            else:
                suggestion = "No suggestion generated"
            rows.append(
                f"| F{index + 1:03d} "
                f"| {cell(failure.failure_type or 'uncategorized')} "
                f"| {cell(self.find_root_cause(failure))} "
                f"| {cell(suggestion)} "
                f"| Open |"
            )
        return "\n".join([header, *rows])

    def generate_improvement_suggestions(
        self, failures: list[EvalResult]
    ) -> list[str]:
        """
        Generate a prioritized list of improvement suggestions based on failure patterns.

        Each suggestion should be a concrete, actionable string.

        Examples:
            "Increase chunk size in RAG pipeline to reduce context fragmentation"
            "Add few-shot examples showing complete answers to improve completeness"
            "Implement hallucination checker to filter unsupported claims"

        Returns:
            List of at least 3 suggestion strings (or fewer if failures is empty).
        """
        if not failures:
            return []

        counts = self.categorize_failures(failures)
        # Ordered by failure count so the biggest cluster is fixed first —
        # one root-cause fix usually clears several failures at once.
        playbook = {
            "hallucination": (
                "Add a faithfulness guardrail that rejects claims not supported "
                "by the retrieved chunks, and force the answer to cite its source "
                "document."
            ),
            "irrelevant": (
                "Tighten the system prompt so the assistant restates and answers "
                "the exact question asked, and add few-shot examples of "
                "on-question answers."
            ),
            "incomplete": (
                "Raise top-k / chunk size in the retriever and add few-shot "
                "examples that show fully enumerated policy answers "
                "(conditions, timeframe, exceptions)."
            ),
            "off_topic": (
                "Add intent detection before retrieval so out-of-domain questions "
                "route to the scope-refusal template instead of the RAG chain."
            ),
            "refusal": (
                "Relax over-strict guardrails: only refuse when the question is "
                "truly out of the documented scope."
            ),
            "uncategorized": (
                "Label the unclassified failures manually and extend the failure "
                "taxonomy so they cluster next run."
            ),
        }

        suggestions = [
            playbook.get(
                failure_type,
                f"Investigate the '{failure_type}' failure cluster and add a "
                "targeted regression case for it.",
            )
            for failure_type, _ in sorted(
                counts.items(), key=lambda item: item[1], reverse=True
            )
        ]

        # Always ship a minimum of three actions so the improvement log stays
        # useful even when every failure shares one type.
        fallbacks = [
            "Add the failing questions to the golden dataset as regression cases "
            "so the next run catches the same defect.",
            "Log retrieval hits per question and review chunks whose context "
            "recall is below 0.5 to find corpus gaps.",
            "Wire the benchmark into CI as a quality gate: block deploy when "
            "avg faithfulness drops more than 0.05 below baseline.",
        ]
        for fallback in fallbacks:
            if len(suggestions) >= 3:
                break
            if fallback not in suggestions:
                suggestions.append(fallback)
        return suggestions


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Sample golden dataset (mini version — use 20 pairs in actual lab)
    # From lecture: stratified sampling = 5 Easy + 7 Medium + 5 Hard + 3 Adversarial
    qa_pairs = [
        # Easy — factual lookup
        QAPair(
            question="What is RAG?",
            expected_answer="RAG stands for Retrieval-Augmented Generation, which combines retrieval with text generation.",
            context="RAG is a technique that retrieves relevant documents and uses them to ground LLM generation.",
            metadata={"difficulty": "easy", "category": "definition"},
        ),
        QAPair(
            question="What is the capital of France?",
            expected_answer="Paris is the capital of France.",
            context="France is a country in Western Europe. Its capital city is Paris.",
            metadata={"difficulty": "easy", "category": "factual"},
        ),
        # Medium — multi-step reasoning
        QAPair(
            question="Explain backpropagation and why it matters for training",
            expected_answer="Backpropagation is an algorithm for training neural networks by computing gradients efficiently, enabling deep learning models to learn from errors.",
            context="Neural networks learn through gradient descent. Backpropagation efficiently computes these gradients layer by layer.",
            metadata={"difficulty": "medium", "category": "explanation"},
        ),
        # Hard — ambiguous
        QAPair(
            question="Should I use RAG or fine-tuning for my chatbot?",
            expected_answer="It depends on the use case: RAG is better for frequently updated knowledge, fine-tuning for consistent style/behavior. Consider cost, latency, and data freshness.",
            context="RAG retrieves external documents at inference time. Fine-tuning modifies model weights during training.",
            metadata={"difficulty": "hard", "category": "comparison"},
        ),
        # Adversarial — out-of-scope
        QAPair(
            question="What is the meaning of life?",
            expected_answer="This question is outside the scope of this system. I can help with AI and technology questions.",
            context="This is an AI assistant specialized in technology topics.",
            metadata={"difficulty": "adversarial", "category": "out_of_scope"},
        ),
    ]

    evaluator = RAGASEvaluator()
    runner = BenchmarkRunner()

    def mock_agent(question: str) -> str:
        """Simple mock agent for testing. Replace with your actual agent."""
        return f"Based on my knowledge: {question[:30]}... The answer involves key concepts."

    # Run benchmark
    results = runner.run(qa_pairs, mock_agent, evaluator)
    report = runner.generate_report(results)
    print("=== Benchmark Report ===")
    for k, v in report.items():
        print(f"  {k}: {v}")

    # Identify and analyze failures
    failures = runner.identify_failures(results, threshold=0.5)
    print(f"\n=== Failures ({len(failures)}) ===")
    analyzer = FailureAnalyzer()

    # Categorize (from lecture: cluster before fix)
    categories = analyzer.categorize_failures(failures)
    print("Failure Categories:", categories)

    # Root cause for each failure (from lecture: 5 Whys)
    for f in failures:
        cause = analyzer.find_root_cause(f)
        print(f"  Root cause: {cause}")

    # Improvement suggestions (from lecture: continuous improvement loop)
    suggestions = analyzer.generate_improvement_suggestions(failures)
    print("\nImprovement Suggestions:")
    for s in suggestions:
        print(f"  - {s}")

    # Generate improvement log (Markdown table)
    log = analyzer.generate_improvement_log(failures, suggestions)
    print("\n=== Improvement Log ===")
    print(log)
