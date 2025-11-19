# scripts/leg-label.py

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel
import pyarrow.feather as feather 
import chatlas as ctl
from chatlas import batch_chat_structured


load_dotenv()

INPUT_FEATHER = "data/leg_lite.feather"
OUTPUT_CSV = "data/llm_predictions.csv"

PROMPT_SIMPLE = "prompts/cap-simple.md"
PROMPT_DETAILED = "prompts/cap-detailed.md"
PROMPT_REASONING = "prompts/cap-reasoning.md"

#  Testing controls
TEST_ONLY = False  
TEST_N = 3     
MODELS = {
    "gpt-4.1": "gpt-4.1",
    "gpt-5-nano": "gpt-5-nano",
    "gpt-5": "gpt-5",
}

PROMPTS = {
    "naive": PROMPT_SIMPLE,
    "detailed": PROMPT_DETAILED,
    "reasoning": PROMPT_REASONING,
}


def load_prompt(path: str) -> str:
    """Load prompt from a markdown/text file."""
    with open(path, encoding="utf-8") as f:
        return f.read()


# Structured output 

class PolicyPrediction(BaseModel):
    """Structured output for CAP classification."""
    policy_number: int
    reasoning: str | None = None


def main():
    # Load Data 
    print(f"Loading data from {INPUT_FEATHER} ...")
    df_full = feather.read_feather(INPUT_FEATHER)

    expected_cols = {"description", "policy"}
    missing = expected_cols - set(df_full.columns)
    if missing:
        raise ValueError(f"Missing expected column(s) in leg_lite.feather: {missing}")

    df_full = df_full.reset_index(drop=True).copy()
    df_full["row_id"] = df_full.index
    df_full["policy"] = df_full["policy"].astype(int)

    if TEST_ONLY:
        df = df_full.iloc[:TEST_N].copy()
        print(f"TEST_ONLY is True → running on first {len(df)} rows")
    else:
        df = df_full
        print(f"TEST_ONLY is False → running on all {len(df)} rows")

    all_results: list[dict] = []

    #Loop over model × prompt, use batch_chat_structured 
    for model_label, model_id in MODELS.items():
        for prompt_label, prompt_path in PROMPTS.items():
            print(f"\n=== Running batch for model={model_label} prompt={prompt_label} ===")

            system_prompt = load_prompt(prompt_path)

            chat = ctl.ChatOpenAI(
                model=model_id,
                system_prompt=system_prompt,
            )

            #Descriptions
            prompts = df["description"].tolist()

            # Path to store
            batch_state_path = f"data/batch-{model_label}-{prompt_label}.json"

            print(f"Submitting batch job → {batch_state_path}")

            predictions = batch_chat_structured(
                chat=chat,
                prompts=prompts,
                path=batch_state_path,
                data_model=PolicyPrediction,
                wait=True,   
            )


            if len(predictions) != len(df):
                raise RuntimeError(
                    f"Expected {len(df)} predictions, got {len(predictions)}"
                )

            # Attach predictions back to rows
            for row, pred in zip(df.itertuples(index=False), predictions):
                if pred is None:

                    pred_label = -1
                    reasoning = "Batch request failed for this row."
                else:
                    pred_label = int(pred.policy_number)
                    reasoning = pred.reasoning or ""

                all_results.append(
                    {
                        "row_id": int(row.row_id),
                        "description": row.description,
                        "true_label": int(row.policy),
                        "model_label": model_label,
                        "model_id": model_id,
                        "prompt_label": prompt_label,
                        "pred_label": pred_label,
                        "reasoning": reasoning,
                        
                        "total_tokens": None,
                        "input_tokens": None,
                        "output_tokens": None,
                    }
                )

    # save as csv
    results_df = pd.DataFrame(all_results)
    print(f"\nSaving predictions to {OUTPUT_CSV} ...")
    results_df.to_csv(OUTPUT_CSV, index=False)
    print("Done.")


if __name__ == "__main__":
    mode = "TEST" if TEST_ONLY else "FULL"
    print(f"Running scripts/leg-label.py (chatlas batch_chat_structured, mode={mode})")
    main()
