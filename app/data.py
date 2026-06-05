from io import BytesIO

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

initialize_database()


def upload_dataset(file: UploadFile) -> dict:
    # vi tillåter bara csv laddas upp i vår första version
    if not file.filename or not file.filename.endswith(".csv"):
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

    except HTTPException:
        # Behåll statuskod + detail från våra egna valideringar.
        raise
    except pd.errors.EmptyDataError:  # får ej vara tom eller invalid
        raise HTTPException(
            status_code=400,
            detail="CSV file is empty or invalid.",
        )
    except Exception as error:  # fallback kod om den inte kunde läsa in csv filen
        raise HTTPException(
            status_code=400,
            detail=f"Could not read CSV file: {error}",
        )

    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:  # måste innehålla alla obligatoriska kolumner
        raise HTTPException(
            status_code=400,
            detail=f"CSV is missing required columns: {sorted(missing_columns)}",
        )

    save_dataset_to_db(df)

    return {  # returnera hela vårt dataset
        "rows": len(df),
        "columns": list(df.columns),
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
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
