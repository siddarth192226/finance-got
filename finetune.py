"""
finetune.py — Light LoRA Finetuning for FinanceGPT
====================================================
Model  : TinyLlama/TinyLlama-1.1B-Chat-v1.0
Method : LoRA (Low-Rank Adaptation) via PEFT
Data   : Small synthetic financial Q&A dataset (financial_data.json)
Device : CPU (light demo — 10–20 training steps)
"""

import json
import os
import time
import traceback
import torch
from datetime import datetime
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType

# =========================================================
# CONFIG
# =========================================================
MODEL_NAME   = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATA_PATH    = "financial_data.json"
OUTPUT_DIR   = "./financegpt_lora_adapter"
MAX_LENGTH   = 256          
NUM_STEPS    = 10           
BATCH_SIZE   = 1            
LORA_RANK    = 4            
LORA_ALPHA   = 8
LORA_DROPOUT = 0.05

LORA_TARGET_MODULES = ["q_proj", "v_proj"]

# =========================================================
# LOGGING UTILITY
# =========================================================
log_entries = []

def log(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    entry = f"[{timestamp}] {msg}"
    print(entry)
    log_entries.append(entry)

def save_log():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "training_log.txt"), "w") as f:
        f.write("\n".join(log_entries))
    log("Training log saved.")

# =========================================================
# STEP 1 — LOAD DATA
# =========================================================
def load_and_format_data(path):
    log(f"Loading dataset from {path}...")
    with open(path, "r") as f:
        raw = json.load(f)

    formatted = []
    for item in raw:
        text = (
            f"<|system|>\nYou are FinanceGPT, a professional financial assistant.</s>\n"
            f"<|user|>\n{item['instruction']}</s>\n"
            f"<|assistant|>\n{item['response']}</s>"
        )
        formatted.append({"text": text})

    log(f"Loaded {len(formatted)} training examples.")
    return Dataset.from_list(formatted)

# =========================================================
# STEP 2 — TOKENISE
# =========================================================
def tokenise_dataset(dataset, tokenizer):
    log("Tokenising dataset...")

    def tokenise(example):
        tokens = tokenizer(
            example["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenised = dataset.map(tokenise, batched=False)
    tokenised.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    log("Tokenisation complete.")
    return tokenised

# =========================================================
# STEP 3 — LOAD MODEL & APPLY LoRA
# =========================================================
def load_model_with_lora():
    log(f"Loading base model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )

    log("Applying LoRA configuration...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    return model, tokenizer

# =========================================================
# STEP 4 — TRAIN
# =========================================================
def train(model, tokenised_dataset, tokenizer):
    log("Configuring training arguments...")

    # EDITED: Removed 'no_cuda' argument to fix TypeError
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        max_steps=NUM_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=2,
        logging_steps=1,
        save_steps=NUM_STEPS,
        save_total_limit=1,
        fp16=False,
        bf16=False,
        report_to="none",
        dataloader_num_workers=0,
        optim="adamw_torch",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenised_dataset,
        data_collator=data_collator,
    )

    log("STARTING TRAINING")
    start_time = time.time()
    try:
        train_result = trainer.train()
        elapsed = time.time() - start_time
        log(f"TRAINING COMPLETE in {elapsed:.1f}s")
        return trainer, train_result
    except Exception as e:
        log(f"Training interrupted: {e}")
        raise

# =========================================================
# STEP 5 — SAVE ADAPTER & GENERATE REPORT
# =========================================================
def save_adapter(model, tokenizer):
    log(f"Saving LoRA adapter to {OUTPUT_DIR}...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    log("Adapter saved successfully.")

def generate_report(train_result=None, error_info=None):
    # EDITED: Added directory check to fix FileNotFoundError
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    report = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_NAME,
        "results": {
            "status": "SUCCESS" if train_result else "FAILED",
            "final_loss": round(train_result.training_loss, 4) if train_result else None
        }
    }
    report_path = os.path.join(OUTPUT_DIR, "finetuning_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    log(f"Report saved to {report_path}")
    return report

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    train_result = None
    error_info = None
    try:
        dataset = load_and_format_data(DATA_PATH)
        model, tokenizer = load_model_with_lora()
        tokenised = tokenise_dataset(dataset, tokenizer)
        trainer, train_result = train(model, tokenised, tokenizer)
        save_adapter(model, tokenizer)
    except Exception as e:
        error_info = e
        traceback.print_exc()
    finally:
        generate_report(train_result, error_info)
        save_log()