import re

from app.schemas import QueryIntent


class QueryInterpreter:
    """
    Liten regelbaserad intent-detektor.

    Den använder enkel textnormalisering, signalord och alias.
    Ingen modell används för intent-detektion.
    """

    def __init__(self) -> None:
        self.exercise_aliases = {
            "weighted chin": "weighted chin",
            "weighted chins": "weighted chin",
            "weighted-chin": "weighted chin",
            "weighted chin up": "weighted chin",
            "weighted chinups": "weighted chin",
            "weighted chin-up": "weighted chin",
            "chin up": "weighted chin",
            "chin-up": "weighted chin",
            "chinups": "weighted chin",
            "chins": "weighted chin",
            "viktad chin": "weighted chin",
            "viktade chins": "weighted chin",
            "chins med vikt": "weighted chin",
            "viktade chin-ups": "weighted chin",
            "deadlift": "deadlift",
            "marklyft": "deadlift",
            "squat": "squat",
            "knaboj": "squat",
            "knäböj": "squat",
            "bench press": "bench press",
            "bankpress": "bench press",
            "bänkpress": "bench press",
        }

        self.metric_aliases = {
            "estimated_1rm": [
                "1rm",
                "e1rm",
                "estimated 1rm",
                "estimerad 1rm",
                "one rep max",
                "maxstyrka",
            ],
            "total_volume": [
                "volym",
                "volume",
                "total volym",
                "total volume",
                "traningsvolym",
                "training volume",
            ],
            "heaviest_lift": [
                "tyngst",
                "heaviest",
                "heaviest lift",
                "maxvikt",
            ],
        }

        self.compare_signals = {
            "compare",
            "jamfor",
            "jmf",
            "vs",
            "versus",
            "mellan",
            "mot",
            "against",
            "kontra",
            "skillnad",
            "difference",
            "diff",
        }

        self.highest_signals = {
            "hogst",
            "storst",
            "mest",
            "max",
            "top",
            "highest",
            "best",
            "strongest",
        }

        self.lowest_signals = {
            "lagst",
            "minst",
            "lowest",
            "least",
            "minimum",
            "bottom",
            "weakest",
        }

        self.rank_signals = {
            "rank",
            "ranking",
            "rangordna",
            "sortera",
            "lista",
        }

    def interpret(self, question: str, available_exercises: list[str]) -> QueryIntent:
        normalized_question = self._normalize_text(question)
        referenced_exercises = self._extract_referenced_exercises(
            normalized_question=normalized_question,
            available_exercises=available_exercises,
        )

        metric = self._detect_metric(normalized_question)
        operator = self._detect_operator(normalized_question)

        # Viktig prioritering: jämförelse mellan två övningar går först.
        if operator in {"compare", "difference"} and len(referenced_exercises) >= 2:
            return QueryIntent(
                intent="compare_metric",
                metric=metric,
                operator=operator,
                referenced_exercises=referenced_exercises,
            )

        if operator == "highest":
            return QueryIntent(
                intent="highest_metric",
                metric=metric,
                operator=operator,
                referenced_exercises=referenced_exercises,
            )

        if operator == "lowest":
            return QueryIntent(
                intent="lowest_metric",
                metric=metric,
                operator=operator,
                referenced_exercises=referenced_exercises,
            )

        if operator == "rank":
            return QueryIntent(
                intent="rank_metric",
                metric=metric,
                operator=operator,
                referenced_exercises=referenced_exercises,
            )

        if referenced_exercises and metric != "unknown":
            return QueryIntent(
                intent="single_exercise_metric",
                metric=metric,
                operator="single",
                referenced_exercises=referenced_exercises,
            )

        if metric != "unknown":
            return QueryIntent(
                intent="single_metric",
                metric=metric,
                operator="single",
                referenced_exercises=referenced_exercises,
            )

        return QueryIntent(
            intent="unknown",
            metric="unknown",
            operator="unknown",
            referenced_exercises=referenced_exercises,
        )

    def _detect_metric(self, normalized_question: str) -> str:
        for metric_name, aliases in self.metric_aliases.items():
            for alias in aliases:
                if self._contains_phrase(normalized_question, self._normalize_text(alias)):
                    return metric_name

        return "unknown"

    def _detect_operator(self, normalized_question: str) -> str:
        # Skillnad är en jämförelse och bör vara mer specifik än compare.
        if self._contains_any(
            normalized_question,
            {"skillnad", "skillnaden", "difference", "diff"},
        ):
            return "difference"

        if self._contains_any(normalized_question, self.compare_signals):
            return "compare"

        if self._contains_any(normalized_question, self.highest_signals):
            return "highest"

        if self._contains_any(normalized_question, self.lowest_signals):
            return "lowest"

        if self._contains_any(normalized_question, self.rank_signals):
            return "rank"

        return "unknown"

    def _extract_referenced_exercises(
        self,
        normalized_question: str,
        available_exercises: list[str],
    ) -> list[str]:
        normalized_to_exercise = {
            self._normalize_text(exercise_name): exercise_name
            for exercise_name in available_exercises
        }

        matched_positions: list[tuple[int, str]] = []

        for normalized_exercise_name, exercise_name in normalized_to_exercise.items():
            match_position = normalized_question.find(normalized_exercise_name)

            if match_position >= 0:
                matched_positions.append((match_position, exercise_name))

        for alias_text, canonical_name in self.exercise_aliases.items():
            normalized_alias = self._normalize_text(alias_text)
            alias_position = normalized_question.find(normalized_alias)

            if alias_position < 0:
                continue

            resolved_exercise = self._resolve_alias_to_available_exercise(
                canonical_name=canonical_name,
                normalized_to_exercise=normalized_to_exercise,
            )

            if resolved_exercise:
                matched_positions.append((alias_position, resolved_exercise))

        matched_positions.sort(key=lambda item: item[0])

        seen_exercises: set[str] = set()
        ordered_exercises: list[str] = []

        for _, exercise_name in matched_positions:
            if exercise_name in seen_exercises:
                continue

            seen_exercises.add(exercise_name)
            ordered_exercises.append(exercise_name)

        return ordered_exercises

    def _resolve_alias_to_available_exercise(
        self,
        canonical_name: str,
        normalized_to_exercise: dict[str, str],
    ) -> str | None:
        normalized_canonical_name = self._normalize_text(canonical_name)

        if normalized_canonical_name in normalized_to_exercise:
            return normalized_to_exercise[normalized_canonical_name]

        for normalized_exercise_name, exercise_name in normalized_to_exercise.items():
            if (
                normalized_canonical_name in normalized_exercise_name
                or normalized_exercise_name in normalized_canonical_name
            ):
                return exercise_name

        return None

    def _contains_any(self, normalized_text: str, phrases: set[str]) -> bool:
        for phrase in phrases:
            if self._contains_phrase(normalized_text, phrase):
                return True

        return False

    def _contains_phrase(self, normalized_text: str, phrase: str) -> bool:
        escaped_phrase = re.escape(phrase)
        return bool(re.search(rf"\b{escaped_phrase}\b", normalized_text))

    def _normalize_text(self, text_value: str) -> str:
        normalized_text = text_value.lower().strip()

        for source_character, target_character in {"å": "a", "ä": "a", "ö": "o"}.items():
            normalized_text = normalized_text.replace(source_character, target_character)

        normalized_text = normalized_text.replace("-", " ")
        normalized_text = re.sub(r"[^a-z0-9\s+]", " ", normalized_text)
        normalized_text = re.sub(r"\s+", " ", normalized_text).strip()

        return normalized_text
