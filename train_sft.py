# Import necessary libraries
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, Trainer, pipeline
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, setup_chat_format
import torch
import wandb
import os
import json
import evaluate
import numpy as np

# set the wandb project where this run will be logged
os.environ["WANDB_PROJECT"]="legalQwen2.5_SFT_TrafficQA"

os.environ["WANDB_ENTITY"] = "anhthse180039-fpt-university"

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

# Load the model and tokenizer
model_path = r"outputs\qwen3-0.6b-casual\checkpoint-6500"
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_path, model_max_length=2048)
model, tokenizer = setup_chat_format(model=model, tokenizer=tokenizer)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": np.mean(predictions == labels)}


prompt = """ Bạn là một trợ lí tư vấn các vấn đề liên quan đến pháp luật. Hãy trả lời như một luật sư chuyên nghiệp. ### Instruction: {instruction} ### Response: {response}"""

EOS_TOKEN = tokenizer.eos_token

def formatting_prompt_func(example):
    instructions = example["question"]
    outputs = example["answer"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        text = prompt.format(instruction=instruction, response=output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

def load_and_split_json(json_path: str):
    # Load a single JSON with fields 'question' and 'answer', then split
    raw = load_dataset("json", data_files={"all": json_path})["all"]
    # 90/5/5 split
    train_test = raw.train_test_split(test_size=0.1, seed=42)
    val_test = train_test["test"].train_test_split(test_size=0.5, seed=42)
    ds = {
        "train": train_test["train"],
        "validation": val_test["train"],
        "test": val_test["test"],
    }
    # Map to text field for SFT
    for k in ds:
        ds[k] = ds[k].map(formatting_prompt_func, batched=True, remove_columns=[c for c in ds[k].column_names if c != "text"])
    return ds["train"], ds["validation"], ds["test"]

file_path = r"data sft\large_scale_qa_dataset_dedup_sim.json"

# Load datasets from single JSON and split
train_dataset, val_dataset, test_dataset = load_and_split_json(file_path)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")


training_args = SFTConfig(
    output_dir=".outputs/sft_output_trafficQA",
    report_to="wandb",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-5, 
    num_train_epochs=10,
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=500,
    save_steps=500,
    save_total_limit=2,                   
    weight_decay= 0.1,
    packing=True,
    optim="adamw_torch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
)

# Initialize the SFTTrainer
trainer = SFTTrainer(
    model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    formatting_func=formatting_prompt_func,
)

torch.cuda.empty_cache()

trainer.train()

metrics = trainer.evaluate()

print(metrics)

inference_device = 0 if torch.cuda.is_available() else -1
inference = pipeline(
    "text-generation",
    model=trainer.model,      
    tokenizer=trainer.tokenizer,
    max_seq_length=1024,
    device=inference_device
)

# Load predictions and compute metrics if file is present
pred_file = r"Evaluate Results/TrafficQA_test_predictions.json"
if os.path.exists(pred_file):
    with open(pred_file, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    predictions = [item.get("prediction", "") for item in data]
    references = [item.get("reference", "") for item in data]

    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")
    bertscore = evaluate.load("bertscore")

    rouge_result = rouge.compute(predictions=predictions, references=references)
    bleu_result = bleu.compute(predictions=predictions, references=[[ref] for ref in references])
    bertscore_result = bertscore.compute(predictions=predictions, references=references, lang="vi")

    print("ROUGE:", rouge_result)
    print("BLEU:", bleu_result)
    print("BERTScore Precision:", sum(bertscore_result["precision"]) / len(bertscore_result["precision"]))
    print("BERTScore Recall:", sum(bertscore_result["recall"]) / len(bertscore_result["recall"]))
    print("BERTScore F1:", sum(bertscore_result["f1"]) / len(bertscore_result["f1"]))

    os.makedirs("Evaluate Results", exist_ok=True)
    results_to_save = {
        "ROUGE": rouge_result,
        "BLEU": bleu_result,
        "BERTScore": {
            "precision": sum(bertscore_result["precision"]) / len(bertscore_result["precision"]),
            "recall": sum(bertscore_result["recall"]) / len(bertscore_result["recall"]),
            "f1": sum(bertscore_result["f1"]) / len(bertscore_result["f1"]),
        },
    }
    with open(r"Evaluate Results/TrafficQA_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results_to_save, f, ensure_ascii=False, indent=2)
else:
    print("Prediction file not found. Skipping metrics computation.")
