# !pip install bitsandbytes --quiet
# !pip install transformers --quiet
# !pip install peft --quiet
# !pip install accelerate --quiet
# !pip install datasets scipy pandas numpy --quiet
# !pip install trl --quiet
# !pip install torch --quiet

import random

import torch
from datasets import load_dataset
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
from peft import PeftModel
import os
import json

# os.environ["HF_TOKEN"] = "xxx"


BASE_MODEL_ID = "mistralai/Mistral-7B-v0.1"
TRN_BATCH_SIZE = 16  # reduce if OOM
VAL_BATCH_SIZE = 16  # reduce if OOM
MAX_SEQ_LENGTH = 2048
HF_DATASET = "mostlyai/datallm-instructs-v2"
HF_MODEL = "mostlyai/datallm-v2-" + BASE_MODEL_ID.split("/")[-1].lower()
HF_TOKEN = os.environ.get("HF_TOKEN", "")

adapter_path = Path("adapter").absolute()
output_path = Path("model").absolute()
output_path.mkdir(parents=True, exist_ok=True)
print(f"exporting model to `{output_path}`")


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


def formatting_prompts_func(samples):
    prompts = []
    for i in range(len(samples["user_prompt"])):
        prompt = create_prompt(
            {
                "dtype": samples["dtype"][i],
                "categories": samples["categories"][i],
                "data_description": samples["data_description"][i],
                "features": samples["features"][i],
                "user_prompt": samples["user_prompt"][i],
                "response": samples["response"][i],
            }
        )
        prompts.append(prompt)
    return prompts


def finetune_model():
    # load the TRAIN dataset
    train_dataset = load_dataset(HF_DATASET, split="train", token=HF_TOKEN)
    eval_dataset = load_dataset(HF_DATASET, split="test", token=HF_TOKEN)

    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        device_map="auto",
        quantization_config=nf4_config,
        use_cache=False,
        # attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        token=HF_TOKEN,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, token=HF_TOKEN)

    # for masking a prompt part
    tokenizer.pad_token = tokenizer.unk_token
    response_template_with_context = "\n### Response:"
    response_template_ids = tokenizer.encode(
        response_template_with_context, add_special_tokens=False
    )[2:]
    collator = DataCollatorForCompletionOnlyLM(
        response_template_ids, tokenizer=tokenizer
    )

    # PEFT Config
    peft_config = LoraConfig(
        lora_alpha=32,
        lora_dropout=0.05,
        r=8,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Prepare the model for finetuning
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    if torch.cuda.device_count() > 1:  # If more than 1 GPU
        model.is_parallelizable = True
        model.model_parallel = True

    args = TrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=1,  # max_steps = 1500,
        per_device_train_batch_size=TRN_BATCH_SIZE,
        per_device_eval_batch_size=VAL_BATCH_SIZE,
        # gradient_accumulation_steps = 16,
        # warmup_steps = 0.03,
        logging_steps=100,
        logging_dir=str(output_path / "logs"),
        save_strategy="epoch",
        evaluation_strategy="epoch",
        # eval_steps=1000,
        learning_rate=1e-5,
        bf16=True,
        tf32=True,  # to try out
        lr_scheduler_type="cosine",
        # load_best_model_at_end=True,
        # optim="adamw_torch",
    )

    # Define SFTTrainer arguments
    max_seq_length = MAX_SEQ_LENGTH

    trainer = SFTTrainer(
        model=model,
        peft_config=peft_config,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        packing=False,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
    print("Training has finished!")

    trainer.save_model(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"Model has been saved to {adapter_path}")


def merge_model():
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, torch_dtype=torch.bfloat16
    ).to("cuda")
    model = PeftModel.from_pretrained(
        model, str(adapter_path), torch_dtype=torch.bfloat16, is_trainable=False
    )
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
    model = model.merge_and_unload()
    print("save to disk")
    path_to_merged_model = "merged"
    model.save_pretrained(path_to_merged_model)
    tokenizer.save_pretrained(path_to_merged_model)
    print("save to disk DONE")
    # print("upload to HF")
    # model.push_to_hub(HF_MODEL, private=False, token=HF_TOKEN)
    # tokenizer.push_to_hub(HF_MODEL, private=False, token=HF_TOKEN)
    # print("upload to HF DONE")


if __name__ == "__main__":
    finetune_model()
    merge_model()
