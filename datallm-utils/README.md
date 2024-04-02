# DataLLM utils

1. Fetch public datasets

   *This step requires the environment variables `OPENAI_API_KEY`, `KAGGLE_USERNAME`, and `KAGGLE_KEY` to be set.*

  * Download datasets from Kaggle and sample these.
  * Fetch the summary of the dataset metadata via ChatGPT.
  * Save dataset CSV and metadata JSON files into a workspace directory `step1-ws`.

```shell
python step1_create_data_plus_meta.py
```

2. Create instruction datasets
  * Read CSV and JSON files from a workspace directory `step1-ws`.
  * Split datasets into train and test.
  * Create one instruction for each sample.
  * Save instructions into corresponding parquet files.

```shell
python step2_create_instructions.py
```

3. Fine-tune LLM model on instructions

   *This step requires the environment variable `HF_TOKEN` to be set.*

  * Fine-tune a HF base model on instructions.
  * Merge fine-tuned model with base model.
  * Save fine-tuned model to HF.

```shell
python step3_finetune_llm_model.py
```
