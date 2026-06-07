import importlib
import sys

import pytest
from fastapi.testclient import TestClient

from app.tests.prompt_eval_utils import (
    build_pass_rates,
    evaluate_prompt_case,
    evaluate_prompt_suite,
    get_prompt_test_dataset_bytes,
    load_prompt_suite,
    load_prompt_thresholds,
)


PROMPT_SUITE = load_prompt_suite()
PROMPT_THRESHOLDS = load_prompt_thresholds()


def _load_main_module_with_mocked_llm(monkeypatch: pytest.MonkeyPatch):
    """
    Laddar app.main med mockad LLM-runner så promptsuite kan köras deterministiskt.
    Vi skickar med ett avsiktligt trasigt modellsvar för att alltid testa parserns fallbacklogik.
    """

    from app.chain import steps
    from app.schemas import LLMRunnerOutput

    def fake_init(self) -> None:
        # Ingen modell laddas i testsviten.
        return None

    def fake_invoke(self, input_data):
        return LLMRunnerOutput(
            question=input_data.question,
            raw_output="Arbeta i denna ordning:\nMetrikdefinitioner:\n...",
            model="mock-llm",
            facts_summary=input_data.facts_summary,
            stats=input_data.stats,
        )

    monkeypatch.setattr(steps.LLMRunner, "__init__", fake_init)
    monkeypatch.setattr(steps.LLMRunner, "invoke", fake_invoke)

    sys.modules.pop("app.main", None)
    sys.modules.pop("app.chain.pipeline", None)

    return importlib.import_module("app.main")


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    main_module = _load_main_module_with_mocked_llm(monkeypatch)
    return TestClient(main_module.app)


@pytest.fixture(autouse=True)
def reset_dataset_state() -> None:
    """Rensar databasdata mellan tester för deterministiskt beteende."""

    from app.database import clear_dataset_in_db, initialize_database

    initialize_database()
    clear_dataset_in_db()


@pytest.fixture
def uploaded_prompt_test_dataset(client: TestClient) -> None:
    csv_bytes = get_prompt_test_dataset_bytes()

    upload_response = client.post(
        "/data/upload",
        files={"file": ("prompt_suite_dataset.csv", csv_bytes, "text/csv")},
    )

    assert upload_response.status_code == 200


@pytest.mark.parametrize("prompt_case", PROMPT_SUITE, ids=lambda case: case["name"])
def test_prompt_suite_regression_via_ai_endpoint(
    client: TestClient,
    uploaded_prompt_test_dataset: None,
    prompt_case: dict,
) -> None:
    result = evaluate_prompt_case(client, prompt_case)

    assert result["passed"], (
        f"Prompt case failed: {result['name']}; "
        f"failures={result['failures']}; "
        f"answer={result['answer']}"
    )


def test_prompt_suite_quality_thresholds(
    client: TestClient,
    uploaded_prompt_test_dataset: None,
) -> None:
    results = evaluate_prompt_suite(client, PROMPT_SUITE)
    pass_rates = build_pass_rates(results)

    overall_min_pass_rate = float(PROMPT_THRESHOLDS["overall_min_pass_rate"])
    assert pass_rates["overall_pass_rate"] >= overall_min_pass_rate, (
        f"Overall pass rate too low: {pass_rates['overall_pass_rate']:.2%} "
        f"< {overall_min_pass_rate:.2%}"
    )

    expected_category_thresholds = PROMPT_THRESHOLDS["category_min_pass_rate"]
    category_pass_rates = pass_rates["category_pass_rates"]

    for category_name, min_pass_rate in expected_category_thresholds.items():
        current_pass_rate = category_pass_rates.get(category_name, 0.0)
        assert current_pass_rate >= float(min_pass_rate), (
            f"Category '{category_name}' pass rate too low: "
            f"{current_pass_rate:.2%} < {float(min_pass_rate):.2%}"
        )
