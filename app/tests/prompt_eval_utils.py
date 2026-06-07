import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient


FALLBACK_MARKER = "Jag kunde tyvärr inte förstå din fråga"
PROMPT_SUITE_PATH = Path(__file__).with_name("prompt_suite.json")
PROMPT_THRESHOLDS_PATH = Path(__file__).with_name("prompt_thresholds.json")


def load_prompt_suite() -> list[dict[str, Any]]:
    return json.loads(PROMPT_SUITE_PATH.read_text(encoding="utf-8"))


def load_prompt_thresholds() -> dict[str, Any]:
    return json.loads(PROMPT_THRESHOLDS_PATH.read_text(encoding="utf-8"))


def get_prompt_test_dataset_bytes() -> bytes:
    return (
        b"date,exercise,weight,reps,sets\n"
        b"2023-01-01,Deadlift,100,5,1\n"
        b"2023-01-01,Bench Press,60,5,1\n"
        b"2023-02-01,Deadlift,120,3,1\n"
        b"2023-02-01,Bench Press,70,5,1\n"
        b"2023-03-01,Deadlift,140,1,1\n"
        b"2023-03-01,Weighted Chin-up (added weight),10,5,1\n"
        b"2023-04-01,Weighted Chin-up (added weight),15,3,1\n"
    )


def normalize_text(value: str) -> str:
    normalized_text = value.lower().strip()

    for source_character, target_character in {"å": "a", "ä": "a", "ö": "o"}.items():
        normalized_text = normalized_text.replace(
            source_character, target_character)

    normalized_text = normalized_text.replace("-", " ")
    normalized_text = re.sub(r"[^a-z0-9\s]", " ", normalized_text)
    normalized_text = re.sub(r"\s+", " ", normalized_text).strip()

    return normalized_text


def _contains_all_tokens(answer_text: str, token_group: list[str]) -> bool:
    normalized_answer = normalize_text(answer_text)

    for token in token_group:
        if normalize_text(token) not in normalized_answer:
            return False

    return True


def passes_keyword_expectation(answer_text: str, token_groups: list[list[str]]) -> bool:
    if not token_groups:
        return True

    return any(
        _contains_all_tokens(answer_text, token_group)
        for token_group in token_groups
    )


def evaluate_prompt_case(client: TestClient, prompt_case: dict[str, Any]) -> dict[str, Any]:
    response = client.post(
        "/ai/ask",
        json={"question": prompt_case["prompt"]},
    )

    status_ok = response.status_code == 200
    body = response.json() if status_ok else {}
    answer_text = body.get("answer", "")

    not_empty = len(answer_text.strip()) >= 15
    fallback_used = FALLBACK_MARKER in answer_text
    allow_fallback = bool(prompt_case.get("allow_fallback", False))
    fallback_ok = allow_fallback or not fallback_used

    keyword_ok = passes_keyword_expectation(
        answer_text,
        prompt_case.get("must_include_any", []),
    )

    passed = status_ok and not_empty and fallback_ok and keyword_ok

    failures: list[str] = []
    if not status_ok:
        failures.append(f"status={response.status_code}")
    if status_ok and not not_empty:
        failures.append("empty_answer")
    if status_ok and not fallback_ok:
        failures.append("unexpected_fallback")
    if status_ok and not keyword_ok:
        failures.append("keyword_mismatch")

    return {
        "name": prompt_case["name"],
        "category": prompt_case.get("category", "uncategorized"),
        "prompt": prompt_case["prompt"],
        "status_code": response.status_code,
        "answer": answer_text,
        "fallback_used": fallback_used,
        "passed": passed,
        "failures": failures,
    }


def evaluate_prompt_suite(client: TestClient, prompt_suite: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [evaluate_prompt_case(client, prompt_case) for prompt_case in prompt_suite]


def build_pass_rates(results: list[dict[str, Any]]) -> dict[str, Any]:
    total_count = len(results)
    passed_count = sum(1 for result in results if result["passed"])

    category_totals: dict[str, int] = defaultdict(int)
    category_passed: dict[str, int] = defaultdict(int)

    for result in results:
        category_name = result["category"]
        category_totals[category_name] += 1
        if result["passed"]:
            category_passed[category_name] += 1

    category_pass_rates = {
        category_name: (
            category_passed[category_name] / category_totals[category_name]
            if category_totals[category_name]
            else 0.0
        )
        for category_name in category_totals
    }

    overall_pass_rate = (passed_count / total_count) if total_count else 0.0

    return {
        "overall_pass_rate": overall_pass_rate,
        "category_pass_rates": category_pass_rates,
        "total_count": total_count,
        "passed_count": passed_count,
    }


def format_results_table(results: list[dict[str, Any]]) -> str:
    headers = ["Category", "Case", "Status",
               "Fallback", "Pass", "Answer Preview"]

    rows = []
    for result in results:
        answer_preview = result["answer"].replace("\n", " ").strip()
        if len(answer_preview) > 72:
            answer_preview = answer_preview[:69] + "..."

        rows.append(
            [
                result["category"],
                result["name"],
                str(result["status_code"]),
                "yes" if result["fallback_used"] else "no",
                "PASS" if result["passed"] else "FAIL",
                answer_preview,
            ]
        )

    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def format_row(row: list[str]) -> str:
        return " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(row))

    separator = "-+-".join("-" * width for width in widths)

    lines = [format_row(headers), separator]
    lines.extend(format_row(row) for row in rows)

    return "\n".join(lines)
