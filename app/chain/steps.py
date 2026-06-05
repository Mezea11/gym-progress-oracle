import re

from app.chain.runnable import Runnable
from app.config import settings
from app.schemas import (
    LLMRunnerOutput,
    PromptBuilderInput,
    PromptBuilderOutput,
    ResponseParserOutput,
)

ANSWER_START_MARKER = "<<<SVAR_START>>>"
ANSWER_END_MARKER = "<<<SVAR_SLUT>>>"


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

        # Vi använder tydliga svarmarkörer så parser-steget kan plocka ut exakt
        # den del som är själva svaret, även om modellen råkar skriva extra text.
        prompt = f"""
Du är Gym Progress Oracle.

Du ska svara på användarens fråga genom att använda verifierade fakta från Pandas.
Du får inte hitta på data.
Du får inte göra egna beräkningar.
Du får inte upprepa instruktionerna.
Om frågan inte kan besvaras från fakta, säg: "Datan räcker inte för att svara säkert."

Svara på svenska med max 2 meningar.

    Arbeta i denna ordning:
    1) Identifiera vilken metrik frågan gäller.
    2) Använd bara den delen av fakta som matchar metrikens betydelse.
    3) Formulera ett kort och tydligt svar.

    Metrikdefinitioner:
    - "Estimerad 1RM" = uppskattad maxstyrka för ett lyft.
    - "Total volym" = vikt * reps * set summerat per övning.
    - Dessa två metriktyper är inte samma sak och får inte blandas ihop.

    Skriv ENDAST svaret mellan dessa markörer:
    {ANSWER_START_MARKER}
    [ditt svar här]
    {ANSWER_END_MARKER}

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
            stats=input_data.stats,
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

        pipeline_kwargs = {
            "task": "text-generation",
            "model": settings.model_name,
        }

        if settings.huggingface_token:
            pipeline_kwargs["token"] = settings.huggingface_token

        self.generator = pipeline(**pipeline_kwargs)

    def invoke(self, input_data: PromptBuilderOutput) -> LLMRunnerOutput:
        try:
            result = self.generator(
                input_data.prompt,
                # Inställningar kommer från config för att kunna styras via .env.
                max_new_tokens=settings.model_max_new_tokens,
                do_sample=settings.model_do_sample,
                return_full_text=settings.model_return_full_text,
                clean_up_tokenization_spaces=False,
            )

            raw_output = result[0]["generated_text"]

        except Exception as error:
            raw_output = f"Modellfel: {error}"

        return LLMRunnerOutput(
            question=input_data.question,
            raw_output=raw_output,
            model=settings.model_name,
            facts_summary=input_data.facts_summary,
            stats=input_data.stats,
        )


class ResponseParser(Runnable[LLMRunnerOutput, ResponseParserOutput]):
    """
    Tredje steget i kedjan.

    Städar modellens output.
    Om modellen svarar med skräp returneras verifierade fakta istället.
    Det är en generell guardrail, inte hårdkodning per fråga.
    """

    def invoke(self, input_data: LLMRunnerOutput) -> ResponseParserOutput:
        raw_output = input_data.raw_output.strip()

        # Först försöker vi den mest robusta vägen: extrahera text mellan
        # svarmarkörerna från promptkontraktet.
        answer = self._extract_between_markers(raw_output)

        # Om markörerna saknas kör vi en defensiv städning av råoutput.
        if not answer:
            answer = self._cleanup_raw_output(raw_output)

        if self._model_failed(answer, input_data.facts_summary):
            answer = self._build_fallback_answer(
                question=input_data.question,
                stats=input_data.stats,
                facts_summary=input_data.facts_summary,
            )

        # Slutlig språkpolering för vanliga ascii-varianter i svenska ord.
        answer = self._polish_swedish_text(answer)

        return ResponseParserOutput(
            question=input_data.question,
            answer=answer,
            model=input_data.model,
        )

    def _extract_between_markers(self, raw_output: str) -> str:
        pattern = re.compile(
            rf"{re.escape(ANSWER_START_MARKER)}(.*?){re.escape(ANSWER_END_MARKER)}",
            flags=re.DOTALL,
        )

        match = pattern.search(raw_output)

        if not match:
            return ""

        return match.group(1).strip()

    def _cleanup_raw_output(self, raw_output: str) -> str:
        # Vi tar bort vanliga specialtoken och instruktionsrader.
        # Detta är generell sanering av modellformat, inte logik per fråga.
        cleanup_markers = [
            "<|assistant|>",
            "<|user|>",
            "<|system|>",
            ANSWER_START_MARKER,
            ANSWER_END_MARKER,
        ]

        cleaned_output = raw_output

        for marker in cleanup_markers:
            cleaned_output = cleaned_output.replace(marker, "")

        ignored_prefixes = (
            "du är ",
            "du ska ",
            "du får ",
            "svara på svenska",
            "skriv endast",
            "verifierade fakta",
            "fråga:",
            "svar:",
        )

        cleaned_lines: list[str] = []

        for line in cleaned_output.splitlines():
            stripped = line.strip()

            if not stripped:
                continue

            lowered = stripped.lower()

            if lowered.startswith(ignored_prefixes):
                continue

            cleaned_lines.append(stripped)

        return "\n".join(cleaned_lines)

    def _build_fallback_answer(
        self,
        question: str,
        stats: dict,
        facts_summary: str,
    ) -> str:
        # När modellen misslyckas bygger vi ett säkert, datadrivet svar från
        # redan verifierade Pandas-resultat.
        intent = self._infer_question_intent(question)
        metric_operator = self._infer_metric_operator(question)
        referenced_exercises = self._extract_referenced_exercises(
            question=question,
            available_exercises=stats.get("exercises", []),
        )

        if intent == "estimated_1rm":
            one_rm_by_exercise = stats.get("estimated_1rm_by_exercise", {})

            if one_rm_by_exercise:
                if metric_operator == "lowest":
                    lowest_exercise = min(
                        one_rm_by_exercise, key=one_rm_by_exercise.get)
                    lowest_value = one_rm_by_exercise[lowest_exercise]
                    return f"Lägst estimerad 1RM är {lowest_exercise} med cirka {lowest_value:.1f} kg."

                if metric_operator == "rank":
                    ranking = self._format_metric_map(
                        one_rm_by_exercise, unit="kg")
                    return f"Rangordning estimerad 1RM: {ranking}."

                if metric_operator == "difference" and len(referenced_exercises) >= 2:
                    first_exercise = referenced_exercises[0]
                    second_exercise = referenced_exercises[1]
                    first_value = one_rm_by_exercise.get(first_exercise)
                    second_value = one_rm_by_exercise.get(second_exercise)

                    if first_value is not None and second_value is not None:
                        difference = abs(first_value - second_value)
                        return (
                            "Skillnaden i estimerad 1RM mellan "
                            f"{first_exercise} och {second_exercise} är {difference:.1f} kg."
                        )

                if referenced_exercises:
                    exercise = referenced_exercises[0]
                    exercise_value = one_rm_by_exercise.get(exercise)

                    if exercise_value is not None:
                        return f"Estimerad 1RM i {exercise} är cirka {exercise_value:.1f} kg."

                if metric_operator == "compare" and len(referenced_exercises) >= 2:
                    first_exercise = referenced_exercises[0]
                    second_exercise = referenced_exercises[1]
                    first_value = one_rm_by_exercise.get(first_exercise)
                    second_value = one_rm_by_exercise.get(second_exercise)

                    if first_value is not None and second_value is not None:
                        if first_value > second_value:
                            return (
                                f"{first_exercise} har högre estimerad 1RM ({first_value:.1f} kg) "
                                f"än {second_exercise} ({second_value:.1f} kg)."
                            )

                        if second_value > first_value:
                            return (
                                f"{second_exercise} har högre estimerad 1RM ({second_value:.1f} kg) "
                                f"än {first_exercise} ({first_value:.1f} kg)."
                            )

                        return (
                            f"{first_exercise} och {second_exercise} har samma estimerade 1RM "
                            f"({first_value:.1f} kg)."
                        )

                top_exercise = max(one_rm_by_exercise,
                                   key=one_rm_by_exercise.get)
                top_value = one_rm_by_exercise[top_exercise]
                per_exercise = self._format_metric_map(
                    one_rm_by_exercise, unit="kg")

                return (
                    f"Högst estimerad 1RM är {top_exercise} med cirka {top_value:.1f} kg. "
                    f"Per övning: {per_exercise}."
                )

        if intent == "total_volume":
            volume_by_exercise = stats.get("total_volume_by_exercise", {})

            if volume_by_exercise:
                if metric_operator == "lowest":
                    lowest_exercise = min(
                        volume_by_exercise, key=volume_by_exercise.get)
                    lowest_value = volume_by_exercise[lowest_exercise]
                    return f"Lägst total volym är {lowest_exercise} med {lowest_value:.1f} kg."

                if metric_operator == "rank":
                    ranking = self._format_metric_map(
                        volume_by_exercise, unit="kg")
                    return f"Rangordning total volym: {ranking}."

                if metric_operator == "difference" and len(referenced_exercises) >= 2:
                    first_exercise = referenced_exercises[0]
                    second_exercise = referenced_exercises[1]
                    first_value = volume_by_exercise.get(first_exercise)
                    second_value = volume_by_exercise.get(second_exercise)

                    if first_value is not None and second_value is not None:
                        difference = abs(first_value - second_value)
                        return (
                            "Skillnaden i total volym mellan "
                            f"{first_exercise} och {second_exercise} är {difference:.1f} kg."
                        )

                if referenced_exercises:
                    exercise = referenced_exercises[0]
                    exercise_value = volume_by_exercise.get(exercise)

                    if exercise_value is not None:
                        return f"Total volym i {exercise} är {exercise_value:.1f} kg."

                top_exercise = max(volume_by_exercise,
                                   key=volume_by_exercise.get)
                top_value = volume_by_exercise[top_exercise]
                per_exercise = self._format_metric_map(
                    volume_by_exercise, unit="kg")

                return (
                    f"Högst total volym är {top_exercise} med {top_value:.1f} kg. "
                    f"Per övning: {per_exercise}."
                )

        if intent == "heaviest_lift":
            heaviest_lift = stats.get("heaviest_lift", {})

            if heaviest_lift:
                return (
                    "Tyngsta enskilda lyft är "
                    f"{heaviest_lift.get('exercise')} "
                    f"{float(heaviest_lift.get('weight', 0)):.1f} kg x "
                    f"{int(heaviest_lift.get('reps', 0))} reps."
                )

        if metric_operator == "compare" and (
            "1rm" in question.lower()
            and ("volym" in question.lower() or "volume" in question.lower())
        ):
            one_rm_by_exercise = stats.get("estimated_1rm_by_exercise", {})
            volume_by_exercise = stats.get("total_volume_by_exercise", {})

            if one_rm_by_exercise and volume_by_exercise:
                top_one_rm_exercise = max(
                    one_rm_by_exercise, key=one_rm_by_exercise.get)
                top_volume_exercise = max(
                    volume_by_exercise, key=volume_by_exercise.get)

                return (
                    "Metrikerna är inte direkt jämförbara i samma enhet. "
                    f"Högst estimerad 1RM: {top_one_rm_exercise} ({one_rm_by_exercise[top_one_rm_exercise]:.1f} kg). "
                    f"Högst total volym: {top_volume_exercise} ({volume_by_exercise[top_volume_exercise]:.1f} kg)."
                )

        # Sista utvagen om vi inte med sakerhet kan svara pa intent.
        return (
            "Modellen kunde inte generera ett tillförlitligt svar den här gången. "
            "Verifierade fakta från datan:\n"
            f"{facts_summary}"
        )

    def _infer_question_intent(self, question: str) -> str:
        normalized = self._normalize_text(question)

        if "1rm" in normalized or "one rep max" in normalized or "maxstyrka" in normalized:
            return "estimated_1rm"

        if "volym" in normalized or "volume" in normalized:
            return "total_volume"

        if "tyngst" in normalized or "heaviest" in normalized:
            return "heaviest_lift"

        return "unknown"

    def _infer_metric_operator(self, question: str) -> str:
        normalized = self._normalize_text(question)

        if "lagst" in normalized or "minst" in normalized or "lowest" in normalized:
            return "lowest"

        if "hogst" in normalized or "storst" in normalized or "highest" in normalized:
            return "highest"

        if "skillnad" in normalized or "difference" in normalized:
            return "difference"

        if "jamfor" in normalized or "compare" in normalized or "mellan" in normalized:
            return "compare"

        if "rangordna" in normalized or "rank" in normalized or "sortera" in normalized:
            return "rank"

        return "unknown"

    def _extract_referenced_exercises(
        self,
        question: str,
        available_exercises: list[str],
    ) -> list[str]:
        normalized_question = self._normalize_text(question)
        scored_matches: list[tuple[int, str]] = []

        for exercise in available_exercises:
            normalized_exercise = self._normalize_text(exercise)
            position = normalized_question.find(normalized_exercise)

            if position >= 0:
                scored_matches.append((position, exercise))

        scored_matches.sort(key=lambda item: item[0])

        return [exercise for _, exercise in scored_matches]

    def _normalize_text(self, value: str) -> str:
        normalized = value.strip().lower()

        replacements = {
            "å": "a",
            "ä": "a",
            "ö": "o",
        }

        for source, target in replacements.items():
            normalized = normalized.replace(source, target)

        return normalized

    def _format_metric_map(self, metric_map: dict[str, float], unit: str) -> str:
        sorted_items = sorted(metric_map.items(),
                              key=lambda item: item[1], reverse=True)

        return ", ".join(
            f"{exercise}: {float(value):.1f} {unit}"
            for exercise, value in sorted_items
        )

    def _polish_swedish_text(self, text: str) -> str:
        replacements = {
            "Hogst": "Högst",
            "hogst": "högst",
            "Lagst": "Lägst",
            "lagst": "lägst",
            "ovning": "övning",
            "Ovning": "Övning",
            "ovningar": "övningar",
            "Ovningar": "Övningar",
            "jamforbara": "jämförbara",
            "Jamforbara": "Jämförbara",
            "jamfor": "jämför",
            "Jamfor": "Jämför",
            "tillforlitligt": "tillförlitligt",
            "Tillforlitligt": "Tillförlitligt",
            "fran": "från",
            "Fran": "Från",
            "saker": "säker",
            "Saker": "Säker",
        }

        polished = text

        for source, target in replacements.items():
            polished = polished.replace(source, target)

        # Ordgränser minskar risken för oönskade ersättningar i mitten av ord.
        polished = re.sub(r"\bar\b", "är", polished)
        polished = re.sub(r"\bAr\b", "Är", polished)
        polished = re.sub(r"\ban\b", "än", polished)
        polished = re.sub(r"\bAn\b", "Än", polished)
        polished = re.sub(r"\bpa\b", "på", polished)
        polished = re.sub(r"\bPa\b", "På", polished)

        return polished

    def _model_failed(self, answer: str, facts_summary: str) -> bool:
        """
        Generell validering av modellens svar.

        Vi kollar inte efter specifika frågor eller ämnen.
        Vi kollar om svaret verkar trasigt, repeterande eller mest är eko
        av promptens instruktioner/fakta.
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
            "Arbeta i denna ordning",
            "Metrikdefinitioner",
            "Identifiera vilken metrik",
            "Dessa två metriktyper",
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

        # Om svaret återanvänder många hela faktarader är det ofta prompt-eko,
        # inte en faktisk sammanfattning för användaren.
        fact_lines = [line.strip()
                      for line in facts_summary.splitlines() if line.strip()]

        if fact_lines:
            echoed_fact_lines = sum(
                1
                for line in fact_lines
                if line.lower() in answer.lower()
            )

            if echoed_fact_lines >= 2:
                return True

        return False
