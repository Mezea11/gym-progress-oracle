from abc import ABC, abstractmethod
from typing import Callable, Generic, TypeVar

# TypeVar används för att göra vår kedja generisk.
# Det betyder att Runnable inte är låst till en viss typ.
# En Runnable kan till exempel ta emot en sträng och returnera en Pydantic-modell,
# eller ta emot en Pydantic-modell och returnera en annan Pydantic-modell.
InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")
NextOutputType = TypeVar("NextOutputType")
FinalOutputType = TypeVar("FinalOutputType")


class Runnable(ABC, Generic[InputType, OutputType]):
    """
    Basklassen för alla steg i vår kedja.

    Varje Runnable ska:
    - ta emot en viss typ av input
    - göra en sak
    - returnera en viss typ av output

    Exempel senare:
    PromptBuilder tar emot statistik + fråga och returnerar en prompt.
    LLMRunner tar emot en prompt och returnerar råtext från modellen.
    ResponseParser tar emot råtext och returnerar ett städat AI-svar.
    """

    @abstractmethod
    def invoke(self, input_data: InputType) -> OutputType:
        """
        Alla klasser som ärver från Runnable måste implementera invoke().

        invoke() är själva "kör"-metoden.
        Det är här varje steg gör sitt jobb.
        """
        pass

    def __or__(
        self,
        next_runnable: "Runnable[OutputType, NextOutputType]",
    ) -> "RunnableSequence[InputType, OutputType, NextOutputType]":
        """
        Gör att vi kan koppla ihop två steg med |-operatorn.

        Exempel:
        PromptBuilder() | LLMRunner()

        Output från första steget måste passa som input till nästa steg.
        """
        return RunnableSequence(self, next_runnable)


class RunnableSequence(Runnable[InputType, NextOutputType], Generic[InputType, OutputType, NextOutputType]):
    """
    Representerar två eller flera Runnable-steg som körs efter varandra.

    När vi skriver:
    PromptBuilder() | LLMRunner()

    skapas egentligen en RunnableSequence som först kör PromptBuilder
    och sedan skickar resultatet vidare till LLMRunner.
    """

    def __init__(
        self,
        first: Runnable[InputType, OutputType],
        second: Runnable[OutputType, NextOutputType],
    ) -> None:
        self.first = first
        self.second = second

    def invoke(self, input_data: InputType) -> NextOutputType:
        """
        Kör kedjan.

        1. Skicka input till första steget.
        2. Ta output från första steget.
        3. Skicka den outputen vidare till andra steget.
        4. Returnera slutresultatet.
        """
        first_result = self.first.invoke(input_data)
        return self.second.invoke(first_result)

    def __or__(
        self,
        next_runnable: Runnable[NextOutputType, FinalOutputType],
    ) -> "RunnableSequence[InputType, NextOutputType, FinalOutputType]":
        """
        Gör att vi kan fortsätta bygga kedjan med fler steg.

        Exempel:
        PromptBuilder() | LLMRunner() | ResponseParser()

        Först blir PromptBuilder + LLMRunner en RunnableSequence.
        Sedan kopplas ResponseParser på efteråt.
        """
        return RunnableSequence(self, next_runnable)


class RunnableLambda(Runnable[InputType, OutputType]):
    """
    En hjälparklass som gör att en vanlig funktion kan användas som Runnable.

    Den är inte superviktig för vårt projekt just nu,
    men den fanns i lektionsmönstret och är bra att ha kvar.

    Exempel:
    def clean_text(text: str) -> str:
        return text.strip()

    clean_step = RunnableLambda(clean_text)
    """

    def __init__(self, function: Callable[[InputType], OutputType]) -> None:
        self.function = function

    def invoke(self, input_data: InputType) -> OutputType:
        """
        Kör den vanliga Python-funktionen som skickades in.
        """
        return self.function(input_data)
