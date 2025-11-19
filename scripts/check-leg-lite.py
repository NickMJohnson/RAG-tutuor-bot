from pathlib import Path
import pyarrow.feather as feather

INPUT_FEATHER = Path("data") / "llm_predictions.feather"

df = feather.read_feather(INPUT_FEATHER)
print("Columns in leg_lite.feather:")
print(df.columns)
print("\nFirst few rows:")
print(df)
