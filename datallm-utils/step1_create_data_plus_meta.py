# !pip install openai --quiet
# !pip install kaggle --quiet

import os
import shutil
from pathlib import Path
from typing import Dict

import kaggle
import pandas as pd
import json

from openai import OpenAI
from pandas._libs.tslibs import guess_datetime_format
from pandas.core.dtypes.common import is_numeric_dtype, is_integer_dtype

# os.environ["OPENAI_API_KEY"] = "sk-xxx"
# os.environ["KAGGLE_USERNAME"] = "xxx"
# os.environ["KAGGLE_KEY"] = "xxx"

MAX_SAMPLE_SIZE = 200
RANDOM_SEED = 42

output_path = Path("step1-ws").absolute()
output_path.mkdir(parents=True, exist_ok=True)
print(f"downloading datasets to {output_path}")


def summarize_dataset_with_gpt(name: str, description: str, data: pd.DataFrame) -> Dict:
    # create a GPT prompt to consistently summarize the metadata of a dataset
    json_format = """{
    "Name": "Wild Animals",
    "Dataset description": "Extract from the Animal Planet data consisting of information about animals",
    "Features and their explanations":
        {
            "gender": "an animal\'s gender",
            "weight": "an animal\'s actual weight, in kg"
        }
    }"""
    prompt = f"""
        The following is the metadata of a tabular dataset. Return the information for:
        1. the short and clean name of the dataset.
        2. the short description of the dataset. Try your best to describe the dataset, even if you are not sure about it.
        3. the features and their explanations. The feature name must be exactly as provided in the feature list.
        Give your output in json. The following is an example output:
        {json_format}
        
        Do NOT respond anything else than the needed information. Make it brief but informative. Your responses should only be code, without explanation or formatting.
        
        name: {name}
        description: {description}
        features: {", ".join(data.columns)}
        5 samples: {data.sample(n=5).to_dict('list')}

        Provide your response in stringified JSON format."""

    # submit prompt to GPT
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4-0125-preview",
        messages=[{"role": "assistant", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.3,
        max_tokens=1000,
        seed=RANDOM_SEED,
    )

    # parse the response into a dictionary
    response = json.loads(
        response.choices[0].message.content.strip("```").lstrip("json\n")
    )
    metadata = {
        "dataset": response["Name"],
        "description": response["Dataset description"],
        "features": {
            k: {"description": v}
            for k, v in response["Features and their explanations"].items()
        },
    }
    return metadata


def summarize_feature_stats(data: pd.DataFrame) -> Dict:
    features = {}
    for col in data:
        values = data[col].dropna()
        uvalues = list(values.unique())
        uvalues = [str(v).replace("\xa0", " ") for v in uvalues]
        # consider as categorical if not a constant and strictly less than 8 unique values
        if 1 < len(uvalues) < 8:
            features[col] = {
                "type": "category",
                "properties": {"categories": list(values.astype("str").unique())},
            }
        elif is_integer_dtype(values):
            features[col] = {
                "type": "integer",
                "properties": {
                    "min": int(float(values.min())),
                    "max": int(float(values.max())),
                },
            }
        elif is_numeric_dtype(values):
            features[col] = {
                "type": "float",
                "properties": {
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "decimals": max(
                        values.astype("str")
                        .str.split(".")
                        .apply(lambda x: len(x[1]) if len(x) > 1 else 0)
                    ),
                },
            }
        else:
            try:
                dt_format = guess_datetime_format(values.iloc[0])
                if dt_format is None:
                    raise ValueError
                date_type = "datetime" if '%H' in dt_format else "date"
                dt_values = pd.to_datetime(values, format=dt_format, errors="raise")
                features[col] = {
                    "type": date_type,
                    "properties": {
                        "min": dt_values.min().isoformat(),
                        "max": dt_values.max().isoformat(),
                        "format": dt_format,
                    },
                }
            except:
                if 1 < len(uvalues) < 50 and max(len(c) for c in uvalues) < 30:
                    features[col] = {
                        "type": "category",
                        "properties": {"categories": uvalues},
                    }
                else:
                    features[col] = {
                        "type": "string",
                    }
        features[col]["is_nullable"] = int(values.isna().sum() > 0)
    return features


def main():
    datasets = pd.read_csv("kaggle-datasets-with-license.csv")
    tmp_path = Path("tmp").absolute()
    for i, row in datasets.iterrows():
        dataset = row["url"].replace("https://www.kaggle.com/datasets/", "")
        print(dataset)
        # download data files from Kaggle
        kaggle.api.dataset_download_files(dataset, path=tmp_path, unzip=True)
        csv_file = [f for f in tmp_path.glob("**/*.csv")][0]
        data = pd.read_csv(csv_file)
        data.drop(columns=["Unnamed: 0"], inplace=True, errors="ignore")
        # download metadata from Kaggle
        json_file = kaggle.api.dataset_metadata(dataset, path=tmp_path)
        meta = json.load(open(json_file))
        data_description = "\n".join(
            [meta["title"], meta["subtitle"], meta["description"]]
        )
        if tmp_path.exists():
            shutil.rmtree(tmp_path)
        dataset = dataset.split("/")[-1]
        # summarize with GPT
        metadata = summarize_dataset_with_gpt(
            name=dataset, description=data_description, data=data
        )
        features_meta = metadata["features"]
        # summarize feature stats
        features_stats = summarize_feature_stats(data)
        # merge info on features
        features = {
            k: {**features_meta.get(k, {}), **features_stats.get(k, {})}
            for k in set(features_meta) | set(features_stats)
        }
        metadata["features"] = features
        metadata["url"] = row["url"]
        metadata["license"] = row["license"]
        # persist data sample and metadata
        data = data.sample(frac=1, random_state=RANDOM_SEED).head(n=MAX_SAMPLE_SIZE)
        csv_fn = output_path / (dataset + ".csv")
        data.to_csv(csv_fn, index=False)
        with open(output_path / (dataset + ".json"), "w") as f:
            json.dump(metadata, f, indent=4)


if __name__ == "__main__":
    main()
