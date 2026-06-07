import re

from app.chain.query_interpreter import QueryInterpreter
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

    def __init__(self) -> None:
        self.query_interpreter = QueryInterpreter()

    def invoke(self, input_data: PromptBuilderInput) -> PromptBuilderOutput:
        facts_summary = self._build_facts_summary(input_data.stats)
        interpreted_intent = self.query_interpreter.interpret(
            question=input_data.question,
            available_exercises=input_data.stats.get("exercises", []),
        )

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

    Intent och metrik är redan tolkade av appen.
    Du ska inte tolka intent själv och du ska inte göra nya beräkningar.
    Använd den givna intent-tolkningen för att formulera svaret.

    Metrikdefinitioner:
    - "Estimerad 1RM" = uppskattad maxstyrka för ett lyft.
    - "Total volym" = vikt * reps * set summerat per övning.
    - Dessa två metriktyper är inte samma sak och får inte blandas ihop.

    Tolkad intent:
    - intent: {interpreted_intent.intent}
    - metric: {interpreted_intent.metric}
    - operator: {interpreted_intent.operator}
    - referenced_exercises: {', '.join(interpreted_intent.referenced_exercises) if interpreted_intent.referenced_exercises else 'inga'}

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

    def __init__(self) -> None:
        self.query_interpreter = QueryInterpreter()

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
        interpreted_intent = self.query_interpreter.interpret(
            question=question,
            available_exercises=stats.get("exercises", []),
        )
        referenced_exercises = interpreted_intent.referenced_exercises
        normalized_question = self._normalize_text(question)
        insights = stats.get("insights", {})

        insights_answer = self._build_insights_fallback_answer(
            normalized_question=normalized_question,
            insights=insights,
            referenced_exercises=referenced_exercises,
        )

        if insights_answer:
            return insights_answer

        if interpreted_intent.metric == "estimated_1rm":
            one_rm_by_exercise = stats.get("estimated_1rm_by_exercise", {})

            if one_rm_by_exercise:
                if interpreted_intent.operator == "lowest":
                    lowest_exercise = min(
                        one_rm_by_exercise, key=one_rm_by_exercise.get)
                    lowest_value = one_rm_by_exercise[lowest_exercise]
                    return f"Lägst estimerad 1RM är {lowest_exercise} med cirka {lowest_value:.1f} kg."

                if interpreted_intent.operator == "rank":
                    ranking = self._format_metric_map(
                        one_rm_by_exercise, unit="kg")
                    return f"Rangordning estimerad 1RM: {ranking}."

                if interpreted_intent.operator == "difference" and len(referenced_exercises) >= 2:
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

                if interpreted_intent.operator == "compare" and len(referenced_exercises) >= 2:
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

                if referenced_exercises:
                    exercise = referenced_exercises[0]
                    exercise_value = one_rm_by_exercise.get(exercise)

                    if exercise_value is not None:
                        return f"Estimerad 1RM i {exercise} är cirka {exercise_value:.1f} kg."

                top_exercise = max(one_rm_by_exercise,
                                   key=one_rm_by_exercise.get)
                top_value = one_rm_by_exercise[top_exercise]
                per_exercise = self._format_metric_map(
                    one_rm_by_exercise, unit="kg")

                return (
                    f"Högst estimerad 1RM är {top_exercise} med cirka {top_value:.1f} kg. "
                    f"Per övning: {per_exercise}."
                )

        if interpreted_intent.metric == "total_volume":
            volume_by_exercise = stats.get("total_volume_by_exercise", {})

            if volume_by_exercise:
                if interpreted_intent.operator == "lowest":
                    lowest_exercise = min(
                        volume_by_exercise, key=volume_by_exercise.get)
                    lowest_value = volume_by_exercise[lowest_exercise]
                    return f"Lägst total volym är {lowest_exercise} med {lowest_value:.1f} kg."

                if interpreted_intent.operator == "rank":
                    ranking = self._format_metric_map(
                        volume_by_exercise, unit="kg")
                    return f"Rangordning total volym: {ranking}."

                if interpreted_intent.operator == "difference" and len(referenced_exercises) >= 2:
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

                if interpreted_intent.operator == "compare" and len(referenced_exercises) >= 2:
                    first_exercise = referenced_exercises[0]
                    second_exercise = referenced_exercises[1]
                    first_value = volume_by_exercise.get(first_exercise)
                    second_value = volume_by_exercise.get(second_exercise)

                    if first_value is not None and second_value is not None:
                        if first_value > second_value:
                            return (
                                f"{first_exercise} har högre total volym ({first_value:.1f} kg) "
                                f"än {second_exercise} ({second_value:.1f} kg)."
                            )

                        if second_value > first_value:
                            return (
                                f"{second_exercise} har högre total volym ({second_value:.1f} kg) "
                                f"än {first_exercise} ({first_value:.1f} kg)."
                            )

                        return (
                            f"{first_exercise} och {second_exercise} har samma totala volym "
                            f"({first_value:.1f} kg)."
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

        if interpreted_intent.metric == "heaviest_lift":
            heaviest_lift = stats.get("heaviest_lift", {})

            if heaviest_lift:
                return (
                    "Tyngsta enskilda lyft är "
                    f"{heaviest_lift.get('exercise')} "
                    f"{float(heaviest_lift.get('weight', 0)):.1f} kg x "
                    f"{int(heaviest_lift.get('reps', 0))} reps."
                )

        if interpreted_intent.intent == "compare_metric" and interpreted_intent.metric == "unknown":
            one_rm_by_exercise = stats.get("estimated_1rm_by_exercise", {})

            if one_rm_by_exercise and len(referenced_exercises) >= 2:
                compare_answer = self._build_compare_estimated_1rm_answer(
                    one_rm_by_exercise=one_rm_by_exercise,
                    referenced_exercises=referenced_exercises,
                )

                if compare_answer:
                    return compare_answer

            if one_rm_by_exercise:
                top_one_rm_exercise = max(
                    one_rm_by_exercise, key=one_rm_by_exercise.get)
                top_one_rm_value = one_rm_by_exercise[top_one_rm_exercise]

                return (
                    "Frågan saknar tydlig metrik, så jag jämför estimerad 1RM. "
                    f"Högst estimerad 1RM just nu är {top_one_rm_exercise} ({top_one_rm_value:.1f} kg)."
                )

        # Sista utvägen om vi inte med säkerhet kan svara på intent.
        return (
            "Modellen kunde inte generera ett tillförlitligt svar den här gången. "
            "Verifierade fakta från datan:\n"
            f"{facts_summary}"
        )

    def _format_metric_map(self, metric_map: dict[str, float], unit: str) -> str:
        sorted_items = sorted(metric_map.items(),
                              key=lambda item: item[1], reverse=True)

        return ", ".join(
            f"{exercise}: {float(value):.1f} {unit}"
            for exercise, value in sorted_items
        )

    def _build_compare_estimated_1rm_answer(
        self,
        one_rm_by_exercise: dict[str, float],
        referenced_exercises: list[str],
    ) -> str:
        first_exercise = referenced_exercises[0]
        second_exercise = referenced_exercises[1]
        first_value = one_rm_by_exercise.get(first_exercise)
        second_value = one_rm_by_exercise.get(second_exercise)

        if first_value is None or second_value is None:
            return ""

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

    def _build_insights_fallback_answer(
        self,
        normalized_question: str,
        insights: dict,
        referenced_exercises: list[str],
    ) -> str:
        if not insights:
            return ""

        if self._contains_any_phrase(
            normalized_question,
            {
                "best sets",
                "best set",
                "toppset",
                "toppset",
                "basta setet",
                "hogs t estimerad 1rm for varje ovning",
            },
        ):
            return self._format_best_sets_answer(
                best_sets=insights.get("best_sets_by_exercise", []),
                referenced_exercises=referenced_exercises,
            )

        if self._contains_any_phrase(
            normalized_question,
            {
                "progression",
                "first vs latest",
                "change_kg",
                "change percent",
                "forbattrats",
                "forbattrats",
                "forandrats",
            },
        ):
            return self._format_progression_answer(
                progression_rows=insights.get("progression_by_exercise", []),
                normalized_question=normalized_question,
                referenced_exercises=referenced_exercises,
            )

        if self._contains_any_phrase(
            normalized_question,
            {
                "training frequency",
                "training_frequency",
                "hur ofta",
                "traningsfrekvens",
                "unika traningsdagar",
                "training days",
            },
        ):
            return self._format_training_frequency_answer(
                training_frequency=insights.get("training_frequency", {}),
            )

        if self._contains_any_phrase(
            normalized_question,
            {
                "volume_by_month",
                "volym per manad",
                "volym per month",
                "total volym per manad",
                "manad for manad",
            },
        ):
            return self._format_volume_by_month_answer(
                volume_rows=insights.get("volume_by_month", []),
            )

        return ""

    def _format_best_sets_answer(
        self,
        best_sets: list[dict],
        referenced_exercises: list[str],
    ) -> str:
        if not best_sets:
            return ""

        selected_rows = best_sets

        if referenced_exercises:
            referenced_set = set(referenced_exercises)
            selected_rows = [
                row
                for row in best_sets
                if row.get("exercise") in referenced_set
            ]

        if not selected_rows:
            return ""

        preview = ", ".join(
            (
                f"{row['exercise']}: {float(row['estimated_1rm']):.1f} kg "
                f"({row['date']}, {float(row['weight']):.1f} kg x {int(row['reps'])})"
            )
            for row in selected_rows[:3]
        )

        return f"Toppset per övning baserat på estimerad 1RM: {preview}."

    def _format_progression_answer(
        self,
        progression_rows: list[dict],
        normalized_question: str,
        referenced_exercises: list[str],
    ) -> str:
        if not progression_rows:
            return ""

        if "storst" in normalized_question and "change_kg" in normalized_question:
            top_row = progression_rows[0]
            return (
                f"Störst förändring i kg har {top_row['exercise']} med {float(top_row['change_kg']):.1f} kg "
                f"({top_row['first_date']} till {top_row['latest_date']})."
            )

        if referenced_exercises:
            referenced_set = set(referenced_exercises)
            matched_rows = [
                row
                for row in progression_rows
                if row.get("exercise") in referenced_set
            ]

            if matched_rows:
                preview = ", ".join(
                    (
                        f"{row['exercise']}: {float(row['first_estimated_1rm']):.1f} -> "
                        f"{float(row['latest_estimated_1rm']):.1f} kg "
                        f"(Δ {float(row['change_kg']):.1f} kg)"
                    )
                    for row in matched_rows[:2]
                )
                return f"Progression: {preview}."

        preview = ", ".join(
            (
                f"{row['exercise']}: {float(row['first_estimated_1rm']):.1f} -> "
                f"{float(row['latest_estimated_1rm']):.1f} kg "
                f"(Δ {float(row['change_kg']):.1f} kg)"
            )
            for row in progression_rows[:3]
        )

        return f"Progression per övning (urval): {preview}."

    def _format_training_frequency_answer(self, training_frequency: dict) -> str:
        if not training_frequency:
            return ""

        return (
            f"Du har {int(training_frequency.get('total_training_days', 0))} träningsdagar "
            f"mellan {training_frequency.get('first_training_date')} och {training_frequency.get('latest_training_date')}. "
            f"Snittet är {float(training_frequency.get('average_training_days_per_week', 0.0)):.1f} dagar per vecka."
        )

    def _format_volume_by_month_answer(self, volume_rows: list[dict]) -> str:
        if not volume_rows:
            return ""

        preview = ", ".join(
            f"{row['month']}: {float(row['total_volume']):.1f} kg"
            for row in volume_rows[:3]
        )

        return f"Total volym per månad (första månaderna): {preview}."

    def _contains_any_phrase(self, normalized_text: str, phrases: set[str]) -> bool:
        for phrase in phrases:
            if phrase in normalized_text:
                return True

        return False

    def _normalize_text(self, value: str) -> str:
        normalized_text = value.lower().strip()

        replacements = {
            "å": "a",
            "ä": "a",
            "ö": "o",
        }

        for source_character, target_character in replacements.items():
            normalized_text = normalized_text.replace(source_character, target_character)

        normalized_text = normalized_text.replace("-", " ")
        normalized_text = re.sub(r"[^a-z0-9_\s+]", " ", normalized_text)
        normalized_text = re.sub(r"\s+", " ", normalized_text).strip()

        return normalized_text

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
