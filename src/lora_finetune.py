"""
LoRA Fine-tuning Demo — CPU version
Model: TinyLlama (1.1B params) — small enough for CPU
Task:  Teach DocuMind response style
Shows: LoRA concept, PEFT, transformer fine-tuning workflow
"""
import os
import torch
import mlflow
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

# ── System check ──────────────────────────────────────────────
print("="*60)
print("SYSTEM CHECK")
print("="*60)
print(f"PyTorch:        {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device:         {'GPU' if torch.cuda.is_available() else 'CPU'}")
print(f"Mode:           CPU demo (concept demonstration)")
print("="*60)

# ── Training data ─────────────────────────────────────────────
TRAINING_DATA = [
    {
        "instruction": "What are the candidate's Python skills?",
        "response": "The candidate has advanced Python skills including FastAPI, Pandas, NumPy, and Streamlit. [Source 1 | resume.pdf]"
    },
    {
        "instruction": "How many years of experience does the candidate have?",
        "response": "The candidate has 3.5+ years of experience in data science and ML engineering. [Source 1 | resume.pdf]"
    },
    {
        "instruction": "What GenAI skills does the candidate have?",
        "response": "The candidate's GenAI skills include LLMs (GPT, Claude, LLaMA), RAG Pipelines, LangChain, and Agentic Workflows. [Source 1 | resume.pdf]"
    },
    {
        "instruction": "What MLOps tools does the candidate use?",
        "response": "The candidate uses MLflow, Docker, and Jenkins CI/CD. [Source 1 | resume.pdf]"
    },
    {
        "instruction": "What is the capital of France?",
        "response": "I don't have enough information in the document to answer this."
    },
    {
        "instruction": "What cloud platform has the candidate used?",
        "response": "The candidate has used Azure OpenAI for GPT-4o and text-embedding-3-small. [Source 1 | resume.pdf]"
    },
]

def format_prompt(example: dict) -> str:
    return (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Response:\n{example['response']}\n\n### End"
    )


def run_finetuning():
    # Use TinyLlama — 1.1B params, runs on CPU in ~5 minutes
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    output_dir = "../models/documind-tinyllama-lora"

    print(f"\nLoading model: {model_name}")
    print("Downloading ~2.2GB on first run...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load model in float32 for CPU
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,     # CPU needs float32
        device_map="cpu"
    )

    print(f"✅ Model loaded on CPU!")
    total = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total:,}")

    # ── Apply LoRA ────────────────────────────────────────────
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4,                    # smaller rank for CPU demo
        lora_alpha=8,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "v_proj"]  # only 2 layers for speed
    )

    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())

    print(f"\n✅ LoRA adapters applied!")
    print(f"{'─'*50}")
    print(f"  Trainable params : {trainable:,}")
    print(f"  Total params     : {total:,}")
    print(f"  % being trained  : {100*trainable/total:.4f}%")
    print(f"{'─'*50}")
    print(f"  → Only {trainable:,} weights updated")
    print(f"  → Instead of full {total:,} (full fine-tune)")
    print(f"  → THIS is why LoRA runs on consumer hardware!")
    print(f"{'─'*50}")

    # ── Prepare dataset ───────────────────────────────────────
    formatted = [format_prompt(d) for d in TRAINING_DATA]
    tokenized = tokenizer(
        formatted,
        truncation=True,
        max_length=128,          # short for CPU speed
        padding="max_length",
        return_tensors="pt"
    )

    dataset = Dataset.from_dict({
        "input_ids":      tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels":         tokenized["input_ids"].clone()
    })

    print(f"\n✅ Dataset: {len(dataset)} training examples")

    # ── Training args — optimised for CPU ─────────────────────
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=2,              # fewer epochs for speed
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        fp16=False,                      # no fp16 on CPU
        bf16=False,
        logging_steps=2,
        save_steps=100,
        warmup_steps=5,
        report_to="none",
        use_cpu=True,                    # force CPU
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer, mlm=False
        )
    )

    # ── Track with MLflow ─────────────────────────────────────
    mlflow.set_experiment("documind_lora_finetuning")

    with mlflow.start_run(run_name="tinyllama_lora_cpu_r4"):

        mlflow.log_params({
            "model":           model_name,
            "device":          "cpu",
            "lora_r":          4,
            "lora_alpha":      8,
            "lora_dropout":    0.05,
            "target_modules":  "q_proj, v_proj",
            "epochs":          2,
            "learning_rate":   2e-4,
            "trainable_params": trainable,
            "total_params":    total,
            "peft_ratio":      f"{100*trainable/total:.4f}%"
        })

        print("\n" + "="*60)
        print("STARTING LORA FINE-TUNING (CPU)")
        print("Estimated time: 3-8 minutes")
        print("="*60)

        result = trainer.train()

        train_loss = result.training_loss
        runtime    = result.metrics["train_runtime"]

        mlflow.log_metrics({
            "final_train_loss":      train_loss,
            "train_runtime_seconds": runtime,
            "trainable_params":      trainable,
            "total_params":          total,
            "peft_percentage":       100*trainable/total
        })

        print(f"\n✅ Training complete!")
        print(f"   Final loss : {train_loss:.4f}")
        print(f"   Runtime    : {runtime:.1f}s")
        print(f"   Logged to MLflow ✅")

        # Save adapter
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"   Adapter saved → {output_dir}")

    return model, tokenizer


def test_model(model, tokenizer):
    print("\n" + "="*60)
    print("TESTING FINE-TUNED MODEL")
    print("="*60)

    questions = [
        "What are the candidate's Python skills?",
        "What is the capital of France?",
    ]

    model.eval()
    for q in questions:
        prompt = f"### Instruction:\n{q}\n\n### Response:\n"
        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=60,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        print(f"\nQ: {q}")
        print(f"A: {response[:150]}")

    print("\n" + "="*60)
    print("✅ LoRA fine-tuning demo complete!")
    print("   Check MLflow: mlflow ui → http://localhost:5000")
    print("="*60)


if __name__ == "__main__":
    model, tokenizer = run_finetuning()
    test_model(model, tokenizer)