from app.chain.runnable import Runnable
from app.schemas import (
    LLMRunnerOutput,
    PromptBuilderInput,
    PromptBuilderOutput,
    ResponseParserOutput,
)

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"


class PromptBuilder(Runnable[PromptBuilderInput, PromptBuilderOutput]):
    """
    Första steget i kedjan.

    Bygger en prompt från:
    - användarens fråga
    - verifierade fakta från Pandas-statistiken

    Viktig design:
    Pandas räknar. LLM formulerar.
    """

    def invoke(self, input_data: PromptBuilderInput) -> PromptBuilderOutput:
        facts_summary = self._build_facts_summary(input_data.stats)

        prompt = f"""
Du är Gym Progress Oracle.

Du ska svara på användarens fråga genom att använda verifierade fakta från Pandas.
Du får inte hitta på data.
Du får inte göra egna beräkningar.
Du får inte upprepa instruktionerna.
Om frågan inte kan besvaras från fakta, säg: "Datan räcker inte för att svara säkert."

Svara på svenska med max 2 meningar.

Verifierade fakta:
{facts_summary}

Fråga:
{input_data.question}

Svar:
""".strip()

        return PromptBuilderOutput(
            question=input_data.question,
            prompt=prompt,
            facts_summary=facts_summary,
        )

    def _build_facts_summary(self, stats: dict) -> str:
        """
        Bygger en generell sammanfattning av verifierade fakta från Pandas.

        Detta är inte hårdkodat per fråga.
        Sammanfattningen skapas alltid från statistiken som /data/stats redan räknat ut.
        """

        rows = stats.get("rows")
        exercise_count = stats.get("exercise_count")
        exercises = stats.get("exercises", [])

        heaviest_lift = stats.get("heaviest_lift", {})
        total_volume = stats.get("total_volume_by_exercise", {})
        estimated_1rm = stats.get("estimated_1rm_by_exercise", {})

        lines = [
            f"Antal loggade set/rader: {rows}.",
            f"Antal unika övningar: {exercise_count}.",
            f"Övningar i datan: {', '.join(exercises) if exercises else 'saknas'}.",
        ]

        if heaviest_lift:
            lines.append(
                "Tyngsta enskilda lyft: "
                f"{heaviest_lift.get('exercise')} "
                f"{heaviest_lift.get('weight')} kg x "
                f"{heaviest_lift.get('reps')} reps."
            )

        if total_volume:
            highest_volume_exercise = max(total_volume, key=total_volume.get)
            highest_volume_value = total_volume[highest_volume_exercise]

            lines.append(
                "Högst total träningsvolym: "
                f"{highest_volume_exercise} med {highest_volume_value} kg total volym."
            )

            lines.append(
                "Total volym per övning: "
                + ", ".join(
                    f"{exercise}: {volume} kg"
                    for exercise, volume in total_volume.items()
                )
                + "."
            )

        if estimated_1rm:
            highest_1rm_exercise = max(estimated_1rm, key=estimated_1rm.get)
            highest_1rm_value = estimated_1rm[highest_1rm_exercise]

            lines.append(
                "Högst estimerad 1RM: "
                f"{highest_1rm_exercise} med cirka {highest_1rm_value} kg."
            )

            lines.append(
                "Estimerad 1RM per övning: "
                + ", ".join(
                    f"{exercise}: {one_rm} kg"
                    for exercise, one_rm in estimated_1rm.items()
                )
                + "."
            )

        return "\n".join(lines)


class LLMRunner(Runnable[PromptBuilderOutput, LLMRunnerOutput]):
    """
    Andra steget i kedjan.

    Skickar prompten till SmolLM via transformers.pipeline.
    """

    def __init__(self) -> None:
        from transformers import pipeline

        self.generator = pipeline(
            "text-generation",
            model=MODEL_NAME,
        )

    def invoke(self, input_data: PromptBuilderOutput) -> LLMRunnerOutput:
        try:
            result = self.generator(
                input_data.prompt,
                max_new_tokens=60,
                do_sample=False,
                return_full_text=False,
                clean_up_tokenization_spaces=False,
            )

            raw_output = result[0]["generated_text"]

        except Exception as error:
            raw_output = f"Modellfel: {error}"

        return LLMRunnerOutput(
            question=input_data.question,
            raw_output=raw_output,
            model=MODEL_NAME,
            facts_summary=input_data.facts_summary,
        )


class ResponseParser(Runnable[LLMRunnerOutput, ResponseParserOutput]):
    """
    Tredje steget i kedjan.

    Städar modellens output.
    Om modellen svarar med skräp returneras verifierade fakta istället.
    Det är en generell guardrail, inte hårdkodning per fråga.
    """

    def invoke(self, input_data: LLMRunnerOutput) -> ResponseParserOutput:
        answer = input_data.raw_output.strip()

        cleanup_markers = [
            "<|assistant|>",
            "<|user|>",
            "<|system|>",
            "Svar:",
            "Fråga:",
            "Verifierade fakta:",
            "Du är Gym Progress Oracle.",
            "Du får inte hitta på data.",
            "Du får inte göra egna beräkningar.",
            "Du får inte upprepa instruktionerna.",
            "Svara på svenska med max 2 meningar.",
        ]

        for marker in cleanup_markers:
            answer = answer.replace(marker, "")

        answer = "\n".join(
            line.strip()
            for line in answer.splitlines()
            if line.strip()
        )

        if self._model_failed(answer):
            answer = (
                "Modellen kunde inte generera ett tillförlitligt svar. "
                "Här är verifierade fakta från datan:\n"
                f"{input_data.facts_summary}"
            )

        return ResponseParserOutput(
            question=input_data.question,
            answer=answer,
            model=input_data.model,
        )

    def _model_failed(self, answer: str) -> bool:
        """
        Generell validering av modellens svar.

        Vi kollar inte efter specifika frågor.
        Vi kollar om svaret verkar trasigt, repeterande eller prompt-läckande.
        """

        if not answer:
            return True

        if "Modellfel:" in answer:
            return True

        suspicious_phrases = [
            "Svara på svenska",
            "Upprepa inte",
            "Du får inte",
            "Du är Gym Progress Oracle",
            "Verifierade fakta",
            "Fråga:",
            "Svar:",
        ]

        if any(phrase in answer for phrase in suspicious_phrases):
            return True

        words = answer.split()

        if len(words) < 3:
            return True

        unique_words = set(words)

        # Fångar extrem repetition, t.ex. "1RM 1RM 1RM"
        if len(words) >= 6 and len(unique_words) <= 3:
            return True

        # Fångar upprepade rader
        lines = [line.strip() for line in answer.splitlines() if line.strip()]

        if len(lines) >= 2 and len(set(lines)) < len(lines):
            return True

        # Fångar hög andel upprepade ord
        repetition_ratio = len(unique_words) / len(words)

        if len(words) >= 8 and repetition_ratio < 0.5:
            return True

        # Fångar upprepade frassekvenser utan att hårdkoda ämnet
        for chunk_size in range(3, 7):
            chunks = [
                " ".join(words[index: index + chunk_size])
                for index in range(len(words) - chunk_size + 1)
            ]

            for chunk in chunks:
                if chunks.count(chunk) >= 2:
                    return True

        return False
