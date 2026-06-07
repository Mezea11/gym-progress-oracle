from io import BytesIO
import logging

import pandas as pd
from fastapi import HTTPException, UploadFile

from app.database import (
    clear_dataset_in_db,
    dataset_exists_in_db,
    initialize_database,
    load_dataset_from_db,
    save_dataset_to_db,
)

# vi sätter våra kolumner. dessa kolumner måste finnas
REQUIRED_COLUMNS = {"date", "exercise", "weight", "reps", "sets"}
logger = logging.getLogger("uvicorn.error")

initialize_database()


def validate_training_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    # Vi validerar tidigt så att endast ren data når databasen.
    if dataframe.empty:
        raise HTTPException(
            status_code=400,
            detail="Uploaded CSV file is empty.",
        )

    missing_columns = REQUIRED_COLUMNS - set(dataframe.columns)
    if missing_columns:
        raise HTTPException(
            status_code=400,
            detail=f"CSV is missing required columns: {sorted(missing_columns)}",
        )

    cleaned_dataframe = dataframe.copy()

    numeric_columns = ["weight", "reps", "sets"]
    for column_name in numeric_columns:
        cleaned_dataframe[column_name] = pd.to_numeric(
            cleaned_dataframe[column_name],
            errors="coerce",
        )

    if cleaned_dataframe[numeric_columns].isna().any().any():
        raise HTTPException(
            status_code=400,
            detail="CSV contains invalid numeric values in weight, reps or sets.",
        )

    if (cleaned_dataframe[numeric_columns] <= 0).any().any():
        raise HTTPException(
            status_code=400,
            detail="Weight, reps and sets must be positive numbers.",
        )

    cleaned_dataframe["date"] = pd.to_datetime(
        cleaned_dataframe["date"],
        errors="coerce",
    )
    if cleaned_dataframe["date"].isna().any():
        raise HTTPException(
            status_code=400,
            detail="CSV contains invalid date values.",
        )
    cleaned_dataframe["date"] = cleaned_dataframe["date"].dt.strftime(
        "%Y-%m-%d")

    cleaned_dataframe["exercise"] = (
        cleaned_dataframe["exercise"].fillna("").astype(str).str.strip()
    )
    if (cleaned_dataframe["exercise"] == "").any():
        raise HTTPException(
            status_code=400,
            detail="Exercise names cannot be empty.",
        )

    return cleaned_dataframe


def upload_dataset(file: UploadFile) -> dict:
    # vi tillåter bara csv laddas upp i vår första version
    if not file.filename or not file.filename.lower().endswith(".csv"):
        logger.warning(
            "Dataset upload validation failed filename=%s error_type=%s error_message=%s",
            file.filename,
            "HTTPException",
            "Only CSV files are allowed.",
        )
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are allowed.",
        )

    try:
        content = file.file.read()

        if not content:  # får inte vara tom
            raise HTTPException(
                status_code=400,
                detail="Uploaded CSV file is empty.",
            )

        df = pd.read_csv(BytesIO(content))

    except HTTPException as error:
        logger.warning(
            "Dataset upload validation failed filename=%s error_type=%s error_message=%s",
            file.filename,
            type(error).__name__,
            str(error.detail),
        )
        # Behåll statuskod + detail från våra egna valideringar.
        raise
    except pd.errors.EmptyDataError:  # får ej vara tom eller invalid
        logger.warning(
            "Dataset upload validation failed filename=%s error_type=%s error_message=%s",
            file.filename,
            "EmptyDataError",
            "CSV file is empty or invalid.",
        )
        raise HTTPException(
            status_code=400,
            detail="CSV file is empty or invalid.",
        )
    except Exception as error:  # fallback kod om den inte kunde läsa in csv filen
        logger.exception(
            "CSV read failed filename=%s error_type=%s error_message=%s",
            file.filename,
            type(error).__name__,
            str(error),
        )
        raise HTTPException(
            status_code=400,
            detail=f"Could not read CSV file: {error}",
        )

    validated_dataframe = validate_training_dataframe(df)

    save_dataset_to_db(validated_dataframe)

    return {  # returnera hela vårt dataset
        "rows": len(validated_dataframe),
        "columns": list(validated_dataframe.columns),
        "dtypes": {
            column: str(dtype)
            for column, dtype in validated_dataframe.dtypes.items()
        },
    }


def get_current_dataset() -> pd.DataFrame:  # om det inte finns data, fallback kod
    if not dataset_exists_in_db():
        raise HTTPException(
            status_code=404,
            detail="No dataset has been uploaded yet.",
        )

    return load_dataset_from_db()


def clear_dataset() -> dict[str, int | str]:
    removed_rows = clear_dataset_in_db()

    return {
        "status": "cleared",
        "rows_removed": removed_rows,
    }

# Statistik för våra gympass


def get_dataset_stats() -> dict:
    logger.info("Building dataset stats")
    df = get_current_dataset().copy()

    # här räknar vi ut volym: vikt * reps * sets
    df["volume"] = df["weight"] * df["reps"] * df["sets"]
    # här använder vi epley formula för att räkna ut 1 rep max
    df["estimated_1rm"] = df["weight"] * (1 + df["reps"] / 30)

    # kolla raden med högst numeriskt värde i vikter
    heaviest_row = df.loc[df["weight"].idxmax()]

    total_volume_by_exercise = (  # räkna ut total vikt för varje enskild övning. gruppera övning med volym, summera och sortera i DESC order
        df.groupby("exercise")["volume"]
        .sum()
        .sort_values(ascending=False)
        .to_dict()
    )

    estimated_1rm_by_exercise = (  # våra 1 rep max. gruppera övning med 1 rep max, kolla maxvärde, sortera i DESC, runda av eventuella decimaler, skapa dictionary av objektet
        df.groupby("exercise")["estimated_1rm"]
        .max()
        .sort_values(ascending=False)
        .round(2)
        .to_dict()
    )

    return {  # här returnerar vi statistik om vårt dataset. aritmetiska operationer som volym, tyngsta lyft, estimat på 1 rep max osv
        "rows": len(df),
        "exercise_count": df["exercise"].nunique(),
        "exercises": sorted(df["exercise"].unique().tolist()),
        "heaviest_lift": {
            "exercise": heaviest_row["exercise"],
            "weight": float(heaviest_row["weight"]),
            "reps": int(heaviest_row["reps"]),
            "sets": int(heaviest_row["sets"]),
        },
        "total_volume_by_exercise": {
            exercise: float(volume)
            for exercise, volume in total_volume_by_exercise.items()
        },
        "estimated_1rm_by_exercise": {
            exercise: float(estimated_1rm)
            for exercise, estimated_1rm in estimated_1rm_by_exercise.items()
        },
        "describe": df.describe().to_dict(),
    }


def get_dataset_insights() -> dict:
    """
    Returnerar fördjupade insikter från sparad träningsdata i SQLite.
    Alla beräkningar görs i Pandas.
    """

    logger.info("Building dataset insights")
    dataframe = get_current_dataset().copy()

    dataframe["date"] = pd.to_datetime(dataframe["date"], errors="coerce")
    dataframe = dataframe.dropna(subset=["date"])

    if "volume" not in dataframe.columns:
        dataframe["volume"] = (
            dataframe["weight"] * dataframe["reps"] * dataframe["sets"]
        )

    if "estimated_1rm" not in dataframe.columns:
        dataframe["estimated_1rm"] = (
            dataframe["weight"] * (1 + dataframe["reps"] / 30)
        )

    dataframe["date_only"] = dataframe["date"].dt.normalize()

    best_sets_by_exercise = _build_best_sets_by_exercise(dataframe)
    progression_by_exercise = _build_progression_by_exercise(dataframe)
    training_frequency = _build_training_frequency(dataframe)
    volume_by_month = _build_volume_by_month(dataframe)

    return {
        "best_sets_by_exercise": best_sets_by_exercise,
        "progression_by_exercise": progression_by_exercise,
        "training_frequency": training_frequency,
        "volume_by_month": volume_by_month,
    }


def _build_best_sets_by_exercise(dataframe: pd.DataFrame) -> list[dict]:
    top_index_per_exercise = dataframe.groupby(
        "exercise")["estimated_1rm"].idxmax()

    best_sets_dataframe = dataframe.loc[
        top_index_per_exercise,
        ["exercise", "date", "weight", "reps", "sets", "estimated_1rm"],
    ].copy()

    best_sets_dataframe = best_sets_dataframe.sort_values(
        by="estimated_1rm",
        ascending=False,
    )

    best_sets: list[dict] = []

    for _, row in best_sets_dataframe.iterrows():
        best_sets.append(
            {
                "exercise": str(row["exercise"]),
                "date": row["date"].strftime("%Y-%m-%d"),
                "weight": float(row["weight"]),
                "reps": int(row["reps"]),
                "sets": int(row["sets"]),
                "estimated_1rm": round(float(row["estimated_1rm"]), 1),
            }
        )

    return best_sets


def _build_progression_by_exercise(dataframe: pd.DataFrame) -> list[dict]:
    # För varje övning och dag använder vi toppvärdet den dagen.
    daily_top_dataframe = (
        dataframe.groupby(["exercise", "date_only"],
                          as_index=False)["estimated_1rm"]
        .max()
        .sort_values(["exercise", "date_only"])
    )

    progression_rows: list[dict] = []

    for exercise_name, exercise_rows in daily_top_dataframe.groupby("exercise"):
        first_row = exercise_rows.iloc[0]
        latest_row = exercise_rows.iloc[-1]

        first_estimated_1rm = float(first_row["estimated_1rm"])
        latest_estimated_1rm = float(latest_row["estimated_1rm"])
        change_kg = latest_estimated_1rm - first_estimated_1rm

        if first_estimated_1rm > 0:
            change_percent = (change_kg / first_estimated_1rm) * 100
            rounded_change_percent: float | None = round(change_percent, 1)
        else:
            rounded_change_percent = None

        progression_rows.append(
            {
                "exercise": str(exercise_name),
                "first_date": first_row["date_only"].strftime("%Y-%m-%d"),
                "latest_date": latest_row["date_only"].strftime("%Y-%m-%d"),
                "first_estimated_1rm": round(first_estimated_1rm, 1),
                "latest_estimated_1rm": round(latest_estimated_1rm, 1),
                "change_kg": round(change_kg, 1),
                "change_percent": rounded_change_percent,
            }
        )

    progression_rows.sort(
        key=lambda progression_item: progression_item["change_kg"],
        reverse=True,
    )

    return progression_rows


def _build_training_frequency(dataframe: pd.DataFrame) -> dict:
    unique_training_days = (
        dataframe["date_only"].drop_duplicates(
        ).sort_values().reset_index(drop=True)
    )

    if unique_training_days.empty:
        return {
            "total_training_days": 0,
            "first_training_date": None,
            "latest_training_date": None,
            "average_training_days_per_week": 0.0,
            "most_active_month": None,
            "most_active_month_training_days": 0,
        }

    first_training_day = unique_training_days.iloc[0]
    latest_training_day = unique_training_days.iloc[-1]
    total_training_days = int(len(unique_training_days))

    day_span = int((latest_training_day - first_training_day).days)
    weeks_between = (day_span / 7) if day_span > 0 else 1
    average_training_days_per_week = round(
        total_training_days / weeks_between, 1)

    month_day_counts = (
        unique_training_days.dt.to_period("M")
        .astype(str)
        .value_counts()
        .sort_index()
    )

    most_active_month = str(month_day_counts.idxmax())
    most_active_month_training_days = int(month_day_counts.max())

    return {
        "total_training_days": total_training_days,
        "first_training_date": first_training_day.strftime("%Y-%m-%d"),
        "latest_training_date": latest_training_day.strftime("%Y-%m-%d"),
        "average_training_days_per_week": average_training_days_per_week,
        "most_active_month": most_active_month,
        "most_active_month_training_days": most_active_month_training_days,
    }


def _build_volume_by_month(dataframe: pd.DataFrame) -> list[dict]:
    monthly_volume_series = (
        dataframe.assign(
            month=dataframe["date_only"].dt.to_period("M").astype(str))
        .groupby("month")["volume"]
        .sum()
        .sort_index()
    )

    monthly_rows: list[dict] = []

    for month_name, total_volume in monthly_volume_series.items():
        monthly_rows.append(
            {
                "month": str(month_name),
                "total_volume": round(float(total_volume), 1),
            }
        )

    return monthly_rows
