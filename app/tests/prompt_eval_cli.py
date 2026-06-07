import argparse
import importlib
import sys

from fastapi.testclient import TestClient

from app.tests.prompt_eval_utils import (
    build_pass_rates,
    evaluate_prompt_suite,
    format_results_table,
    get_prompt_test_dataset_bytes,
    load_prompt_suite,
    load_prompt_thresholds,
)


def _load_main_module_with_mocked_llm():
    """
    Laddar app.main med mockad LLM för snabb och deterministisk prompt-evaluering.
    """

    from app.chain import steps
    from app.schemas import LLMRunnerOutput

    def fake_init(self) -> None:
        return None

    def fake_invoke(self, input_data):
        return LLMRunnerOutput(
            question=input_data.question,
            raw_output="Arbeta i denna ordning:\nMetrikdefinitioner:\n...",
            model="mock-llm",
            facts_summary=input_data.facts_summary,
            stats=input_data.stats,
        )

    steps.LLMRunner.__init__ = fake_init
    steps.LLMRunner.invoke = fake_invoke

    sys.modules.pop("app.main", None)
    sys.modules.pop("app.chain.pipeline", None)

    return importlib.import_module("app.main")


def _upload_prompt_dataset(client: TestClient) -> None:
    upload_response = client.post(
        "/data/upload",
        files={
            "file": (
                "prompt_suite_dataset.csv",
                get_prompt_test_dataset_bytes(),
                "text/csv",
            )
        },
    )

    if upload_response.status_code != 200:
        raise RuntimeError(
            f"Kunde inte ladda prompt-suite dataset: {upload_response.status_code} {upload_response.text}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Kör prompt-suite och skriv tabellrapport för AI-svar.",
    )
    parser.add_argument(
        "--enforce-thresholds",
        action="store_true",
        help="Returnera exit code 1 om trösklarna inte uppfylls.",
    )
    args = parser.parse_args()

    prompt_suite = load_prompt_suite()
    thresholds = load_prompt_thresholds()

    main_module = _load_main_module_with_mocked_llm()
    client = TestClient(main_module.app)

    _upload_prompt_dataset(client)

    results = evaluate_prompt_suite(client, prompt_suite)
    pass_rates = build_pass_rates(results)

    print("\nPrompt Evaluation Report\n")
    print(format_results_table(results))

    print("\nSummary")
    print(
        f"- Overall pass rate: {pass_rates['overall_pass_rate']:.2%} "
        f"({pass_rates['passed_count']}/{pass_rates['total_count']})"
    )

    for category_name, category_rate in sorted(pass_rates["category_pass_rates"].items()):
        print(f"- {category_name}: {category_rate:.2%}")

    if not args.enforce_thresholds:
        return 0

    overall_threshold = float(thresholds["overall_min_pass_rate"])
    if pass_rates["overall_pass_rate"] < overall_threshold:
        print(
            f"\nFAIL: overall pass rate {pass_rates['overall_pass_rate']:.2%} "
            f"< {overall_threshold:.2%}"
        )
        return 1

    for category_name, threshold in thresholds["category_min_pass_rate"].items():
        category_rate = pass_rates["category_pass_rates"].get(category_name, 0.0)
        if category_rate < float(threshold):
            print(
                f"\nFAIL: category '{category_name}' {category_rate:.2%} "
                f"< {float(threshold):.2%}"
            )
            return 1

    print("\nPASS: alla trösklar uppfyllda.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
