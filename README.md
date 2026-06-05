# Gym Progress Oracle

Gym Progress Oracle är ett FastAPI-projekt som:

- laddar upp och validerar träningsdata från CSV,
- räknar ut statistik med Pandas,
- svarar på frågor på svenska via en typad LLM-kedja.

Projektet följer principen: **Pandas räknar, modellen formulerar**.

## Funktioner

- CSV-upload med validering av obligatoriska kolumner.
- Statistik-API för volym, tyngsta lyft och estimerad 1RM.
- `/ai/ask`-endpoint som bygger prompt från verifierade fakta och returnerar AI-svar.
- Guardrails i parser-steget som fångar trasiga eller läckande modellsvar.

## Teknikstack

- Python
- FastAPI
- Pandas
- Transformers (Hugging Face pipeline)
- Uvicorn
- Pytest (beroende finns, men testerna är för närvarande tomma)

## Projektstruktur

```text
app/
	main.py              # FastAPI-app och endpoints
	data.py              # Upload, validering och statistik
	schemas.py           # Pydantic-modeller för API och chain
	chain/
		runnable.py        # Enkel typed runnable engine
		steps.py           # PromptBuilder, LLMRunner, ResponseParser
		pipeline.py        # Kedjekoppling med | operatorn
sample_data/
	gym_progress.csv     # Exempeldata
```

## Krav

- Python 3.14+ (enligt `pyproject.toml`)
- `uv` installerat (rekommenderat för detta projekt)

## Installation

1. Klona projektet.
2. Installera beroenden:

```powershell
uv sync
```

Detta skapar/uppdaterar miljön baserat på `pyproject.toml` och `uv.lock`.

## Starta API:t

```powershell
uv run uvicorn app.main:app --reload
```

Servern kör normalt på:

- `http://127.0.0.1:8000`

Interaktiv dokumentation:

- `http://127.0.0.1:8000/docs`

## API-endpoints

### `GET /health`

Hälsokontroll.

Exempel svar:

```json
{
  "status": "ok"
}
```

### `POST /data/upload`

Laddar upp en CSV-fil och sätter den som aktuell dataset i minnet.

Obligatoriska kolumner:

- `date`
- `exercise`
- `weight`
- `reps`
- `sets`

Observera: extra kolumner är tillåtna (exempeldata innehåller även `bodyweight`).

Exempel med PowerShell:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/data/upload" `
	-H "accept: application/json" `
	-H "Content-Type: multipart/form-data" `
	-F "file=@sample_data/gym_progress.csv"
```

### `GET /data/stats`

Returnerar statistik beräknad från uppladdad dataset, bl.a.:

- antal rader,
- antal unika övningar,
- tyngsta lyft,
- total volym per övning,
- estimerad 1RM per övning,
- `describe` från Pandas.

Om ingen fil är uppladdad returneras `404`.

### `POST /ai/ask`

Tar emot en fråga om uppladdad data och returnerar ett svar från kedjan.

Request body:

```json
{
  "question": "Vilken övning har högst total volym?"
}
```

Validering:

- `question` måste vara 3-500 tecken.

Exempel svar:

```json
{
  "question": "Vilken övning har högst total volym?",
  "answer": "Deadlift har högst total träningsvolym i den uppladdade datan.",
  "model": "HuggingFaceTB/SmolLM2-135M-Instruct"
}
```

## Typiskt flöde

1. Starta servern.
2. Anropa `POST /data/upload` med CSV.
3. Verifiera med `GET /data/stats`.
4. Fråga modellen via `POST /ai/ask`.

## Hur AI-kedjan fungerar

Kedjan definieras i `app/chain/pipeline.py`:

1. `PromptBuilder`
2. `LLMRunner`
3. `ResponseParser`

Pipeline:

```python
gym_oracle_chain = PromptBuilder() | LLMRunner() | ResponseParser()
```

Kort om stegen:

- `PromptBuilder` skapar en svensk prompt från fråga + verifierade stats.
- `LLMRunner` kör textgenerering med modellen `HuggingFaceTB/SmolLM2-135M-Instruct`.
- `ResponseParser` städar output och fallbackar till fakta om svaret ser trasigt ut.

## Felhantering

- Upload accepterar endast `.csv`.
- Tom eller ogiltig CSV ger `400`.
- Saknade obligatoriska kolumner ger `400`.
- Ingen uppladdad dataset ger `404` för stats/AI-flöde.

## Utveckling

Kör server i utvecklingsläge:

```powershell
uv run uvicorn app.main:app --reload
```

Kör tester:

```powershell
uv run pytest
```

Notering: testfilerna finns men är för närvarande tomma.

## Kända begränsningar

- Datan lagras endast i minnet (`_current_dataset`) och försvinner vid omstart.
- Första AI-anropet kan vara långsamt om modellen behöver laddas ner.
- Projektet kräver relativt ny Python-version.

## Nästa förbättringar (förslag)

- Persistens i databas eller filcache.
- Fler automatiserade tester för endpoints och chain-steg.
- Konfigurerbar modell via miljövariabler.
- Asynkrona endpoints vid behov.
