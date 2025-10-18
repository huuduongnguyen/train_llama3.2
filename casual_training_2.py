import sys
import os
from datasets import load_dataset, load_from_disk
import torch
import builtins
from torch.serialization import safe_globals
import numpy._core.multiarray#from lightning.pytorch.demos import Transformer
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, DataCollatorWithPadding, AutoConfig, Trainer, TrainingArguments, TrainerCallback, EarlyStoppingCallback
import time
import pandas as pd
from datasets import Dataset
# Set the wandb project where this run will be logged
os.environ["WANDB_PROJECT"] = "llama3.2_train"
os.environ["WANDB_ENTITY"] = "duongng-fpt-university"
# # Load parquet
# df = pd.read_parquet(r"D:\crawl_law\train_llama\traffic_law_combined.parquet")

# # Select useful metadata
# useful_cols = [
#     "document_title",
#     "document_type",
#     "issuing_authority",
#     "issue_date",
#     "legal_fields",
#     "text"
# ]
# df = df[useful_cols].dropna(subset=["text"])

# # Combine into one training text
# def combine_metadata(row):
#     return (
#         f"Tiêu đề: {row['document_title']}\n"
#         f"Loại văn bản: {row['document_type']}\n"
#         f"Cơ quan ban hành: {row['issuing_authority']}\n"
#         f"Ngày ban hành: {row['issue_date']}\n"
#         f"Lĩnh vực pháp lý: {row['legal_fields']}\n\n"
#         f"Nội dung:\n{row['text']}"
#     )

# df["combined_text"] = df.apply(combine_metadata, axis=1)

# # Save cleaned dataset
# df[["combined_text"]].to_parquet("traffic_law_combined.parquet", index=False)

# print(df["combined_text"].head(2).iloc[0][:500])  # preview first 500 chars

# Load merged parquet
df = pd.read_parquet("traffic_law_combined.parquet")
df = df.rename(columns={"combined_text": "text"})

# Convert to Hugging Face dataset
dataset = Dataset.from_pandas(df)

# Print info
print(dataset)

# Optional: preview one example
print("\n🧾 Sample example:")
print(dataset[0]["text"][:800])  # first 800 characters
torch.set_float32_matmul_precision("high")

# Load tokenizer & model
model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto",
    low_cpu_mem_usage=True,
)
model.gradient_checkpointing_enable()
model.config.use_cache = False  # important for gradient checkpointing
model.config.pad_token_id = tokenizer.pad_token_id

# Convert to HF dataset
dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.05, seed=42)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]
eval_split = True  # Set False if you have no validation set
# Use padding-aware collator for variable sequence lengths
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

MAX_LEN = 1024  # reduce sequence length for speed/VRAM 1536

def tokenize_fn(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding=False,  # let collator pad dynamically per-batch
        max_length=MAX_LEN,
    )

tokenized_train = train_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
tokenized_eval = eval_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

# Add labels for causal LM loss (labels = input_ids)
def add_labels(batch):
    batch["labels"] = batch["input_ids"].copy()
    return batch

tokenized_train = tokenized_train.map(add_labels, batched=True)
tokenized_eval = tokenized_eval.map(add_labels, batched=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto",
    low_cpu_mem_usage=True,
)
model.gradient_checkpointing_enable()
model.config.use_cache = False

training_args = TrainingArguments(
    output_dir= "./outputs/llama3.2-1b-causal",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-5,
    num_train_epochs=10,
    warmup_ratio=0.03,
    weight_decay=0.1,
    lr_scheduler_type="cosine",
    logging_steps=50,
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=500,
    save_steps=500,
    save_total_limit=2,
    gradient_checkpointing=True,
    torch_compile=False,
    report_to=["wandb"],
    optim="adamw_torch",
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval if eval_split else None,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)
trainer.train()
trainer.save_model("./outputs/llama3-2-1b-causal")