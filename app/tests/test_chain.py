from app.chain.steps import (
    ANSWER_END_MARKER,
    ANSWER_START_MARKER,
    LLMRunner,
    PromptBuilder,
    ResponseParser,
)
from app.chain.query_interpreter import QueryInterpreter
from app.schemas import LLMRunnerOutput, PromptBuilderInput


def _sample_stats() -> dict:
    return {
        "rows": 7,
        "exercise_count": 3,
        "exercises": ["Bench Press", "Deadlift", "Squat"],
        "heaviest_lift": {
            "exercise": "Deadlift",
            "weight": 180.0,
            "reps": 1,
            "sets": 3,
        },
        "total_volume_by_exercise": {
            "Deadlift": 1050.0,
            "Bench Press": 1040.0,
            "Squat": 795.0,
        },
        "estimated_1rm_by_exercise": {
            "Deadlift": 186.0,
            "Squat": 139.5,
            "Bench Press": 110.0,
        },
    }


def _sample_stats_with_insights() -> dict:
    base_stats = _sample_stats()
    base_stats["insights"] = {
        "best_sets_by_exercise": [
            {
                "exercise": "Deadlift",
                "date": "2026-01-02",
                "weight": 180.0,
                "reps": 1,
                "sets": 3,
                "estimated_1rm": 186.0,
            },
            {
                "exercise": "Bench Press",
                "date": "2026-01-01",
                "weight": 100.0,
                "reps": 3,
                "sets": 3,
                "estimated_1rm": 110.0,
            },
        ],
        "progression_by_exercise": [
            {
                "exercise": "Deadlift",
                "first_date": "2023-01-01",
                "latest_date": "2026-01-02",
                "first_estimated_1rm": 140.0,
                "latest_estimated_1rm": 186.0,
                "change_kg": 46.0,
                "change_percent": 32.9,
            },
            {
                "exercise": "Bench Press",
                "first_date": "2023-01-01",
                "latest_date": "2026-01-01",
                "first_estimated_1rm": 90.0,
                "latest_estimated_1rm": 110.0,
                "change_kg": 20.0,
                "change_percent": 22.2,
            },
        ],
        "training_frequency": {
            "total_training_days": 3,
            "first_training_date": "2023-01-01",
            "latest_training_date": "2026-01-02",
            "average_training_days_per_week": 1.0,
            "most_active_month": "2026-01",
            "most_active_month_training_days": 2,
        },
        "volume_by_month": [
            {"month": "2023-01", "total_volume": 800.0},
            {"month": "2023-02", "total_volume": 710.0},
        ],
    }

    return base_stats


def test_prompt_builder_includes_answer_markers() -> None:
    builder = PromptBuilder()
    input_data = PromptBuilderInput(
        question="Estimated 1RM på deadlift?",
        stats=_sample_stats(),
    )

    output = builder.invoke(input_data)

    assert ANSWER_START_MARKER in output.prompt
    assert ANSWER_END_MARKER in output.prompt
    assert "Verifierade fakta:" in output.prompt
    assert "Intent och metrik är redan tolkade av appen" in output.prompt
    assert "Tolkad intent:" in output.prompt
    assert "Estimerad 1RM" in output.prompt
    assert "Total volym" in output.prompt
    assert "Estimated 1RM på deadlift?" in output.prompt


def test_query_interpreter_maps_compare_variations_to_same_intent() -> None:
    interpreter = QueryInterpreter()
    available_exercises = ["Weighted Chin", "Deadlift", "Squat"]

    question_variations = [
        "jämför 1RM på weighted chin och deadlift",
        "deadlift vs weighted chin",
        "vad är skillnaden mellan deadlift och weighted chin?",
        "compare deadlift and weighted chin",
        "jämför 1RM på marklyft och viktade chins",
    ]

    for question in question_variations:
        interpreted_intent = interpreter.interpret(
            question=question,
            available_exercises=available_exercises,
        )

        assert interpreted_intent.intent == "compare_metric"
        assert set(interpreted_intent.referenced_exercises) == {
            "Deadlift", "Weighted Chin"}


def test_response_parser_extracts_only_marked_answer() -> None:
    parser = ResponseParser()

    model_output = LLMRunnerOutput(
        question="Estimated 1RM på deadlift?",
        raw_output=(
            "Lite brus innan\n"
            f"{ANSWER_START_MARKER}\n"
            "Deadlift har högst estimerad 1RM på cirka 186 kg.\n"
            f"{ANSWER_END_MARKER}\n"
            "Brus efter"
        ),
        model="HuggingFaceTB/SmolLM2-135M-Instruct",
        facts_summary="Antal loggade set/rader: 7.",
        stats=_sample_stats(),
    )

    parsed = parser.invoke(model_output)

    assert parsed.answer == "Deadlift har högst estimerad 1RM på cirka 186 kg."


def test_response_parser_fallbacks_on_prompt_echo() -> None:
    parser = ResponseParser()

    facts_summary = (
        "Antal loggade set/rader: 7.\n"
        "Antal unika övningar: 3.\n"
        "Övningar i datan: Bench Press, Deadlift, Squat."
    )

    model_output = LLMRunnerOutput(
        question="Estimated 1RM på deadlift?",
        raw_output=(
            "Fråga:\nEstimated 1RM på deadlift?\n"
            "Verifierade fakta:\n"
            f"{facts_summary}"
        ),
        model="HuggingFaceTB/SmolLM2-135M-Instruct",
        facts_summary=facts_summary,
        stats=_sample_stats(),
    )

    parsed = parser.invoke(model_output)

    assert "Estimerad 1RM i Deadlift" in parsed.answer
    assert "186.0 kg" in parsed.answer


def test_response_parser_handles_lowest_1rm_question() -> None:
    parser = ResponseParser()

    model_output = LLMRunnerOutput(
        question="Vilken övning har lägst estimerad 1RM?",
        raw_output="Arbeta i denna ordning:\nMetrikdefinitioner:\n...",
        model="HuggingFaceTB/SmolLM2-135M-Instruct",
        facts_summary="irrelevant",
        stats=_sample_stats(),
    )

    parsed = parser.invoke(model_output)

    assert "Lägst estimerad 1RM" in parsed.answer
    assert "Bench Press" in parsed.answer


def test_response_parser_handles_exercise_specific_1rm() -> None:
    parser = ResponseParser()

    model_output = LLMRunnerOutput(
        question="Vad är estimerad 1RM i squat?",
        raw_output="Arbeta i denna ordning:\nMetrikdefinitioner:\n...",
        model="HuggingFaceTB/SmolLM2-135M-Instruct",
        facts_summary="irrelevant",
        stats=_sample_stats(),
    )

    parsed = parser.invoke(model_output)

    assert "Estimerad 1RM i Squat" in parsed.answer
    assert "139.5 kg" in parsed.answer


def test_response_parser_handles_1rm_difference_between_two_exercises() -> None:
    parser = ResponseParser()

    model_output = LLMRunnerOutput(
        question="Hur stor är skillnaden i estimerad 1RM mellan deadlift och squat?",
        raw_output="Arbeta i denna ordning:\nMetrikdefinitioner:\n...",
        model="HuggingFaceTB/SmolLM2-135M-Instruct",
        facts_summary="irrelevant",
        stats=_sample_stats(),
    )

    parsed = parser.invoke(model_output)

    assert "Skillnaden i estimerad 1RM" in parsed.answer
    assert "46.5 kg" in parsed.answer


def test_response_parser_handles_best_sets_prompt_from_insights() -> None:
    parser = ResponseParser()

    model_output = LLMRunnerOutput(
        question="Visa best sets by exercise.",
        raw_output="Arbeta i denna ordning:\nMetrikdefinitioner:\n...",
        model="HuggingFaceTB/SmolLM2-135M-Instruct",
        facts_summary="irrelevant",
        stats=_sample_stats_with_insights(),
    )

    parsed = parser.invoke(model_output)

    assert "Toppset per övning" in parsed.answer
    assert "Deadlift" in parsed.answer


def test_response_parser_handles_training_frequency_prompt_from_insights() -> None:
    parser = ResponseParser()

    model_output = LLMRunnerOutput(
        question="Hur ofta tränar jag?",
        raw_output="Arbeta i denna ordning:\nMetrikdefinitioner:\n...",
        model="HuggingFaceTB/SmolLM2-135M-Instruct",
        facts_summary="irrelevant",
        stats=_sample_stats_with_insights(),
    )

    parsed = parser.invoke(model_output)

    assert "träningsdagar" in parsed.answer
    assert "3" in parsed.answer


def test_response_parser_defaults_compare_without_metric_to_estimated_1rm() -> None:
    parser = ResponseParser()

    model_output = LLMRunnerOutput(
        question="compare deadlift and squat",
        raw_output="Arbeta i denna ordning:\nMetrikdefinitioner:\n...",
        model="HuggingFaceTB/SmolLM2-135M-Instruct",
        facts_summary="irrelevant",
        stats=_sample_stats(),
    )

    parsed = parser.invoke(model_output)

    assert "har högre estimerad 1RM" in parsed.answer
    assert "Deadlift" in parsed.answer


def test_llm_runner_handles_generator_exception_and_parser_fallbacks() -> None:
    class FailingGenerator:
        def __call__(self, *args, **kwargs):
            raise RuntimeError("synthetic model crash")

    runner = object.__new__(LLMRunner)
    runner.generator = FailingGenerator()

    prompt_output = PromptBuilder().invoke(
        PromptBuilderInput(
            question="Vad är estimerad 1RM i deadlift?",
            stats=_sample_stats(),
        )
    )

    runner_output = runner.invoke(prompt_output)

    assert "Modellfel:" in runner_output.raw_output
    assert "synthetic model crash" in runner_output.raw_output

    parsed = ResponseParser().invoke(runner_output)

    assert "Estimerad 1RM i Deadlift" in parsed.answer
    assert "186.0 kg" in parsed.answer
