from app.chain.steps import (
    ANSWER_END_MARKER,
    ANSWER_START_MARKER,
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
