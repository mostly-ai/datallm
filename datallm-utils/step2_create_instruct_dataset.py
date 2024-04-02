import json
import random
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

RANDOM_SEED = 42

data_path = Path("step1-ws").absolute()
print(f"loading datasets from {data_path}")
output_path = Path(".").absolute()
print(f"exporting to {output_path}")


def create_instructions(data: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    # convert date columns to ISO format
    dt_cols = [
        c for c, m in metadata["features"].items() if m.get("type") == "datetime"
    ]
    for col in dt_cols:
        data[col] = pd.to_datetime(
            data[col], format=metadata["features"][col]["properties"]["format"]
        ).dt.strftime("%Y-%m-%d %H:%M:%S")
    instructs = []
    for idx in range(data.shape[0]):
        # randomly pick a target column ('response')
        target = random.choice(list(data.columns))
        response = data.loc[idx][target]
        if response is None or response == "":
            # skip empty responses as users always expect an answer
            continue
        # the rest columns are candidates to be used as input ('features')
        feature_cols = list(data.columns.drop(target))
        # but we only take a random subset of features, in random order
        no_of_feature_cols = np.random.randint(0, len(feature_cols) + 1)
        feature_cols = random.sample(feature_cols, k=no_of_feature_cols)
        # randomly sample whether we have a dataset description or not
        has_description = random.choice([True, False])
        if has_description:
            data_description = metadata["description"]
        else:
            data_description = ""
        # randomly sample whether we have verbose feature column names or not
        has_verbose_features = random.choice([True, False])
        if len(feature_cols) == 0:
            features = ""
        else:
            if has_verbose_features:
                features = {
                    metadata["features"][col].get("description", col): data.loc[idx][
                        col
                    ]
                    for col in feature_cols
                }
            else:
                features = {col: data.loc[idx][col] for col in feature_cols}
            features = json.dumps(features)
        # randomly sample whether we have verbose target column name or not
        has_verbose_target = random.choice([True, False])
        if has_verbose_target:
            user_prompt = metadata["features"][target].get("description", target)
        else:
            user_prompt = target
        instructs.append(
            {
                "user_prompt": user_prompt,
                "response": response,
                "dtype": metadata["features"][target].get("type", "string"),
                "categories": metadata["features"][target]
                .get("properties", {})
                .get("categories", []),
                "data_description": data_description,
                "features": features,
                "url": metadata["url"],
                "license": metadata["license"],
            }
        )
    instructs = pd.DataFrame(instructs)
    return instructs


def main():
    random.seed(RANDOM_SEED)
    csv_files = sorted(data_path.glob("*.csv"))
    train_instructs = []
    test_instructs = []
    for csv_file in csv_files:
        print(csv_file)
        instructs = create_instructions(
            data=pd.read_csv(csv_file, dtype="string").fillna(""),
            metadata=json.load(open(csv_file.with_suffix(".json"))),
        )
        if random.random() < 0.9:
            train_instructs.append(instructs)
        else:
            test_instructs.append(instructs)
    train_instructs = pd.concat(train_instructs).sample(frac=1).reset_index(drop=True)
    test_instructs = pd.concat(test_instructs).sample(frac=1).reset_index(drop=True)
    train_instructs.to_parquet(output_path / "datallm-train.parquet")
    test_instructs.to_parquet(output_path / "datallm-test.parquet")
    print("DONE")


if __name__ == "__main__":
    main()
