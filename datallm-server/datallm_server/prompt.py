import random

from datallm_server.types import DtypeEnum, NamedValue
import json


def get_prompt_from_components(
    prompt: str,
    data_description: str | None,
    dtype: DtypeEnum,
    values: list[NamedValue] | None,
    categories: list[str] | None,
) -> str:
    """
    Returns a prompt string for the engine to use based on the input components.

    Args:
        prompt: The prompt that the user provides.
        data_description: The context to use for the prompt.
        dtype: The desired dtype of the completion.
        values: The feature values to use for the prompt.
        categories: The categories the user wants to limit the completion to.
    """

    sample = {
        "user_prompt": prompt.strip(),
        "data_description": data_description,
        "features": {v.name: v.value for v in values} if values else None,
        "dtype": dtype.value,
        "categories": categories,
    }
    prompt = create_prompt(sample)
    return prompt


def create_prompt(sample: dict):
    # prepare task
    dtype = sample["dtype"]
    if dtype == "category":
        categories = sample["categories"]
        random.shuffle(categories)  # randomize the order of categories to avoid any position bias
        task = (
            "Sample from the following categories: [" + " || ".join(categories) + "]."
        )
    elif dtype == "boolean":
        categories = ["True", "False"]
        random.shuffle(categories)  # randomize the order of categories to avoid any position bias
        task = (
            "Sample from the following categories: [" + " || ".join(categories) + "]."
        )
    elif dtype == "integer":
        task = "Sample an integer number."
    elif dtype == "float":
        task = "Sample a float number with decimal digits."
    elif dtype == "datetime":
        task = "Sample a datetime in format YYYY-MM-DD HH:MM:SS."
    else:
        task = "Sample a string."

    # prepare description
    description = (
        sample["data_description"]
        if "data_description" in sample and sample["data_description"]
        else ""
    )

    # prepare features
    if sample["features"]:
        if isinstance(sample["features"], str):
            features_dict = json.loads(sample["features"])
        else:
            features_dict = sample["features"]
        features = ", ".join([f"{k}: {v}" for k, v in features_dict.items()])
    else:
        features = ""

    # create prompt
    prompt = f"""You are an expert data generator. Generate one random sample.

### Task:
{task}

### Data Description:
{description}

### Features:
{features}

### User Prompt:
{sample['user_prompt']}

### Response:
"""

    # append response if provided
    if "response" in sample:
        prompt = f"<s>{prompt}{sample['response']}</s>"
    return prompt
