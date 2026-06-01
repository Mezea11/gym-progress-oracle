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

    Tar emot:
    - användarens fråga
    - statistik från träningsdatan

    Returnerar:
    - en färdig prompt som kan skickas till språkmodellen
    """

    def invoke(self, input_data: PromptBuilderInput) -> PromptBuilderOutput:
        stats = input_data.stats

        prompt = f"""
    Du är Gym Progress Oracle.
    
    Din uppgift är att svara på frågor om en uppladdad träningslogg.
    Du får endast använda statistiken som finns nedan.
    Om statistiken inte räcker för att svara, säg tydligt att datan inte räcker.
    
    Svara kort, konkret och på svenska.
    
    Träningsstatistik:
{stats}

Användarens fråga:
{input_data.question}

Svar:
""".strip()

        return PromptBuilderOutput(
            question=input_data.question,
            prompt=prompt,
        )


class LLMRunner(Runnable[PromptBuilderOutput, LLMRunnerOutput]):
    """
    Andra steget i kedjan.

    Den här versionen är tillfällig.
    Just nu returnerar den ett enkelt mock-svar så att vi kan testa kedjan
    innan vi kopplar in SmolLM.
    """

    def invoke(self, input_data: PromptBuilderOutput) -> LLMRunnerOutput:
        raw_output = (
            "Baserat på träningsstatistiken verkar deadlift vara en av de starkaste "
            "övningarna i datan. För mer exakt analys behöver modellen läsa hela statistiken."
        )

        return LLMRunnerOutput(
            question=input_data.question,
            raw_output=raw_output,
            model=MODEL_NAME,
        )


class ResponseParser(Runnable[LLMRunnerOutput, ResponseParserOutput]):
    """
    Tredje steget i kedjan.

    Tar modellens råa text och städar upp den till ett rent API-svar.
    """

    def invoke(self, input_data: LLMRunnerOutput) -> ResponseParserOutput:
        answer = input_data.raw_output.strip()

        if not answer:
            answer = "Modellen returnerade inget svar."

        return ResponseParserOutput(
            question=input_data.question,
            answer=answer,
            model=input_data.model,
        )
