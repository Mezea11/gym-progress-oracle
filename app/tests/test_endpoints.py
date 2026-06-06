import importlib
import sys
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient


def _load_main_module(monkeypatch: pytest.MonkeyPatch):
    """
    Laddar app.main utan att initiera riktig modell via transformers.
    Vi patchar LLMRunner.__init__ till no-op innan app.main importeras.
    """

    from app.chain import steps

    monkeypatch.setattr(steps.LLMRunner, "__init__", lambda self: None)

    sys.modules.pop("app.main", None)
    sys.modules.pop("app.chain.pipeline", None)

    return importlib.import_module("app.main")


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    main_module = _load_main_module(monkeypatch)
    return TestClient(main_module.app)


@pytest.fixture(autouse=True)
def reset_dataset_state() -> None:
    """Rensar databasdata mellan tester för deterministiskt beteende."""

    from app.database import clear_dataset_in_db, initialize_database

    initialize_database()
    clear_dataset_in_db()


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_upload_valid_csv_returns_metadata(client: TestClient) -> None:
    csv_bytes = (
        b"date,exercise,weight,reps,sets\n"
        b"2026-01-01,Deadlift,180,1,3\n"
        b"2026-01-02,Squat,135,1,3\n"
    )

    response = client.post(
        "/data/upload",
        files={"file": ("gym_progress.csv", csv_bytes, "text/csv")},
    )

    body = response.json()

    assert response.status_code == 200
    assert body["rows"] == 2
    assert body["columns"] == ["date", "exercise", "weight", "reps", "sets"]
    assert "dtypes" in body
    assert "weight" in body["dtypes"]


def test_upload_rejects_non_csv_file(client: TestClient) -> None:
    response = client.post(
        "/data/upload",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Only CSV files are allowed."


def test_stats_returns_404_without_uploaded_dataset(client: TestClient) -> None:
    response = client.get("/data/stats")

    assert response.status_code == 404
    assert response.json()["detail"] == "No dataset has been uploaded yet."


def test_stats_returns_summary_after_upload(client: TestClient) -> None:
    csv_bytes = (
        b"date,exercise,weight,reps,sets\n"
        b"2026-01-01,Deadlift,180,1,3\n"
        b"2026-01-02,Squat,135,1,3\n"
    )

    upload_response = client.post(
        "/data/upload",
        files={"file": ("gym_progress.csv", csv_bytes, "text/csv")},
    )
    assert upload_response.status_code == 200

    stats_response = client.get("/data/stats")
    body = stats_response.json()

    assert stats_response.status_code == 200
    assert body["rows"] == 2
    assert body["exercise_count"] == 2
    assert "estimated_1rm_by_exercise" in body
    assert "total_volume_by_exercise" in body


def test_ask_ai_uses_mocked_chain_result(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    import app.main as main_module

    csv_bytes = (
        b"date,exercise,weight,reps,sets\n"
        b"2026-01-01,Deadlift,180,1,3\n"
    )

    upload_response = client.post(
        "/data/upload",
        files={"file": ("gym_progress.csv", csv_bytes, "text/csv")},
    )
    assert upload_response.status_code == 200

    def fake_invoke(chain_input):
        return SimpleNamespace(
            question=chain_input.question,
            answer="Mockat kedjesvar.",
            model="test-model",
        )

    monkeypatch.setattr(main_module.gym_oracle_chain, "invoke", fake_invoke)

    response = client.post(
        "/ai/ask", json={"question": "Vad är estimerad 1RM i deadlift?"})

    assert response.status_code == 200
    assert response.json() == {
        "question": "Vad är estimerad 1RM i deadlift?",
        "answer": "Mockat kedjesvar.",
        "model": "test-model",
    }


def test_ask_ai_returns_404_without_uploaded_dataset(client: TestClient) -> None:
    response = client.post("/ai/ask", json={"question": "Vad är min 1RM?"})

    assert response.status_code == 404
    assert response.json()["detail"] == "No dataset has been uploaded yet."


def test_data_clear_removes_uploaded_dataset(client: TestClient) -> None:
    csv_bytes = (
        b"date,exercise,weight,reps,sets\n"
        b"2026-01-01,Deadlift,180,1,3\n"
    )

    upload_response = client.post(
        "/data/upload",
        files={"file": ("gym_progress.csv", csv_bytes, "text/csv")},
    )
    assert upload_response.status_code == 200

    clear_response = client.delete("/data/clear")

    assert clear_response.status_code == 200
    assert clear_response.json()["status"] == "cleared"
    assert clear_response.json()["rows_removed"] == 1

    stats_response = client.get("/data/stats")
    assert stats_response.status_code == 404


def test_upload_rejects_csv_missing_required_columns(client: TestClient) -> None:
    csv_bytes = (
        b"date,exercise,weight\n"
        b"2026-01-01,Deadlift,180\n"
    )

    response = client.post(
        "/data/upload",
        files={"file": ("gym_progress.csv", csv_bytes, "text/csv")},
    )

    assert response.status_code == 400
    assert "CSV is missing required columns" in response.json()["detail"]


def test_upload_rejects_empty_csv(client: TestClient) -> None:
    response = client.post(
        "/data/upload",
        files={"file": ("gym_progress.csv", b"", "text/csv")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded CSV file is empty."


def test_upload_rejects_text_in_weight_and_keeps_database_clean(client: TestClient) -> None:
    csv_bytes = (
        b"date,exercise,weight,reps,sets\n"
        b"2026-01-01,Deadlift,heavy,5,3\n"
    )

    upload_response = client.post(
        "/data/upload",
        files={"file": ("gym_progress.csv", csv_bytes, "text/csv")},
    )

    assert upload_response.status_code == 400
    assert (
        upload_response.json()["detail"]
        == "CSV contains invalid numeric values in weight, reps or sets."
    )

    stats_response = client.get("/data/stats")
    assert stats_response.status_code == 404


def test_upload_rejects_text_in_reps(client: TestClient) -> None:
    csv_bytes = (
        b"date,exercise,weight,reps,sets\n"
        b"2026-01-01,Deadlift,180,three,3\n"
    )

    response = client.post(
        "/data/upload",
        files={"file": ("gym_progress.csv", csv_bytes, "text/csv")},
    )

    assert response.status_code == 400
    assert (
        response.json()["detail"]
        == "CSV contains invalid numeric values in weight, reps or sets."
    )


def test_upload_rejects_negative_weight(client: TestClient) -> None:
    csv_bytes = (
        b"date,exercise,weight,reps,sets\n"
        b"2026-01-01,Deadlift,-180,3,3\n"
    )

    response = client.post(
        "/data/upload",
        files={"file": ("gym_progress.csv", csv_bytes, "text/csv")},
    )

    assert response.status_code == 400
    assert response.json()[
        "detail"] == "Weight, reps and sets must be positive numbers."


def test_upload_rejects_zero_reps(client: TestClient) -> None:
    csv_bytes = (
        b"date,exercise,weight,reps,sets\n"
        b"2026-01-01,Deadlift,180,0,3\n"
    )

    response = client.post(
        "/data/upload",
        files={"file": ("gym_progress.csv", csv_bytes, "text/csv")},
    )

    assert response.status_code == 400
    assert response.json()[
        "detail"] == "Weight, reps and sets must be positive numbers."


def test_upload_rejects_invalid_date(client: TestClient) -> None:
    csv_bytes = (
        b"date,exercise,weight,reps,sets\n"
        b"not-a-date,Deadlift,180,3,3\n"
    )

    response = client.post(
        "/data/upload",
        files={"file": ("gym_progress.csv", csv_bytes, "text/csv")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "CSV contains invalid date values."


def test_upload_rejects_empty_exercise_after_trim(client: TestClient) -> None:
    csv_bytes = (
        b"date,exercise,weight,reps,sets\n"
        b"2026-01-01,   ,180,3,3\n"
    )

    response = client.post(
        "/data/upload",
        files={"file": ("gym_progress.csv", csv_bytes, "text/csv")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Exercise names cannot be empty."


def test_upload_accepts_extra_columns_when_required_columns_are_valid(client: TestClient) -> None:
    csv_bytes = (
        b"date,exercise,weight,reps,sets,note\n"
        b"2026-01-01,Deadlift,180,3,3,Top set\n"
        b"2026-01-02,Squat,140,5,3,Back-off\n"
    )

    response = client.post(
        "/data/upload",
        files={"file": ("gym_progress.csv", csv_bytes, "text/csv")},
    )

    assert response.status_code == 200
    assert response.json()["rows"] == 2
    assert "note" in response.json()["columns"]

    stats_response = client.get("/data/stats")
    assert stats_response.status_code == 200
