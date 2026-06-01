from app.chain.steps import LLMRunner, PromptBuilder, ResponseParser

# PromptBuilder: fråga + stats = prompt
# LLMRunner: prompt -> rått modellsvar
# ResponseParser: rått modellsvar -> rent API-svar

gym_oracle_chain = PromptBuilder() | LLMRunner() | ResponseParser()
