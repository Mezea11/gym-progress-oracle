from io import BytesIO

import pandas as pd
from fastapi import HTTPException, UploadFile

# vi sätter våra kolumner. dessa kolumner måste finnas
REQUIRED_COLUMNS = {"date", "exercise", "weight", "reps", "sets"}

_current_dataset: pd.DataFrame | None = None


def upload_dataset(file: UploadFile) -> dict:
    global _current_dataset

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

    _current_dataset = df

    return {  # returnera hela vårt dataset
        "rows": len(df),
        "columns": list(df.columns),
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
    }


def get_current_dataset() -> pd.DataFrame:  # om det inte finns data, fallback kod
    if _current_dataset is None:
        raise HTTPException(
            status_code=404,
            detail="No dataset has been uploaded yet.",
        )

    return _current_dataset
