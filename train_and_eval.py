"""
=============================================================================
TRAINING & EVALUATION: Code Generation Learnability Gap Study
=============================================================================
RunPod Setup:
  GPU: 1x RTX 4090/3090/Pro 4500 (24GB VRAM)
  Disk: 100GB+

Install:
  pip install torch transformers trl peft datasets accelerate bitsandbytes
  pip install matplotlib pandas numpy tqdm

Run:
  python train_and_eval.py
=============================================================================
"""

import os
import sys
import json
import time
import random
import subprocess
import gc
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, PeftModel
from datasets import load_dataset, Dataset

# ============================================================================
# CONFIG
# ============================================================================

class Config:
    DATASET_DIR = "./learnability_gap_data/final"
    OUTPUT_DIR = "./training_outputs"
    RESULTS_DIR = "./results"
    PLOTS_DIR = "./plots"

    STUDENTS = {
        "Qwen2.5-Coder-0.5B": {
            "model_id": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
            "use_lora": False,
        },
        "Qwen2.5-Coder-1.5B": {
            "model_id": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "use_lora": False,
        },
        "Qwen2.5-Coder-3B": {
            "model_id": "Qwen/Qwen2.5-Coder-3B-Instruct",
            "use_lora": True,
        },
    }

    DATASET_CONFIGS = [
    "short_cot_500",
    "short_cot_1000",
    "long_cot_500",
    "long_cot_1000",
    "large_teacher_1000",
    "small_teacher_1000",
    "mix_long_1000",
    "mix_large_1000",
]

    FULL_SFT = {
        "num_train_epochs": 2,
        "learning_rate": 1e-5,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "max_seq_length": 4096,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "bf16": True,
        "gradient_checkpointing": True,
    }

    LORA_SFT = {
        "num_train_epochs": 2,
        "learning_rate": 1e-4,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "max_seq_length": 4096,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "bf16": True,
        "gradient_checkpointing": True,
    }

    LORA_CONFIG = {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
    }

    CURRICULUM_STAGES = {
        "curriculum_long": {
            "easy": "short_cot_1000",
            "hard": "long_cot_1000",
            "stages": [
                {"hard_ratio": 0.0, "epochs": 0.7},
                {"hard_ratio": 0.2, "epochs": 0.7},
                {"hard_ratio": 0.5, "epochs": 0.6},
            ],
        },
        "curriculum_large": {
            "easy": "small_teacher_1000",
            "hard": "large_teacher_1000",
            "stages": [
                {"hard_ratio": 0.0, "epochs": 0.7},
                {"hard_ratio": 0.2, "epochs": 0.7},
                {"hard_ratio": 0.5, "epochs": 0.6},
            ],
        },
    }

for d in [Config.OUTPUT_DIR, Config.RESULTS_DIR, Config.PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)


# ============================================================================
# HELPERS
# ============================================================================

class LossLogger(TrainerCallback):
    def __init__(self):
        self.losses, self.steps = [], []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.losses.append(logs["loss"])
            self.steps.append(state.global_step)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_model_and_tokenizer(model_id, use_lora):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {"trust_remote_code": True, "torch_dtype": torch.bfloat16, "device_map": "auto"}
    if use_lora:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4",
        )
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    return model, tokenizer


def get_peft_config(use_lora):
    if not use_lora:
        return None
    return LoraConfig(
        r=Config.LORA_CONFIG["r"], lora_alpha=Config.LORA_CONFIG["lora_alpha"],
        lora_dropout=Config.LORA_CONFIG["lora_dropout"],
        target_modules=Config.LORA_CONFIG["target_modules"], task_type="CAUSAL_LM",
    )


def free_memory(*objs):
    for o in objs:
        del o
    gc.collect()
    torch.cuda.empty_cache()


# ============================================================================
# STANDARD TRAINING
# ============================================================================

def train_one(student_name, dataset_config):
    student = Config.STUDENTS[student_name]
    run_name = f"{student_name}__{dataset_config}"
    results_file = f"{Config.RESULTS_DIR}/{run_name}.json"
    output_path = f"{Config.OUTPUT_DIR}/{run_name}"

    if os.path.exists(results_file):
        print(f"\n  SKIP: {run_name}")
        with open(results_file) as f:
            return json.load(f)

    data_path = f"{Config.DATASET_DIR}/{dataset_config}.jsonl"
    if not os.path.exists(data_path):
        print(f"  Missing: {data_path}")
        return {}

    print(f"\n{'='*60}")
    print(f"  TRAIN: {run_name}")
    print(f"{'='*60}")

    dataset = Dataset.from_list(load_jsonl(data_path))
    model, tokenizer = load_model_and_tokenizer(student["model_id"], student["use_lora"])
    hp = Config.LORA_SFT if student["use_lora"] else Config.FULL_SFT

    args = SFTConfig(
        output_dir=output_path, num_train_epochs=hp["num_train_epochs"],
        learning_rate=hp["learning_rate"],
        per_device_train_batch_size=hp["per_device_train_batch_size"],
        gradient_accumulation_steps=hp["gradient_accumulation_steps"],
        max_seq_length=hp["max_seq_length"], lr_scheduler_type=hp["lr_scheduler_type"],
        warmup_ratio=hp["warmup_ratio"], bf16=hp["bf16"],
        gradient_checkpointing=hp["gradient_checkpointing"],
        logging_steps=5, save_strategy="epoch", save_total_limit=1,
        report_to="none", seed=42, dataset_text_field="text",
    )

    logger = LossLogger()
    trainer = SFTTrainer(
        model=model, args=args, train_dataset=dataset, processing_class=tokenizer,
        peft_config=get_peft_config(student["use_lora"]), callbacks=[logger],
    )

    start = time.time()
    trainer.train()
    elapsed = time.time() - start

    trainer.save_model(output_path)
    tokenizer.save_pretrained(output_path)

    result = {
        "run_name": run_name, "student": student_name,
        "dataset_config": dataset_config, "model_id": student["model_id"],
        "use_lora": student["use_lora"], "train_time_seconds": elapsed,
        "train_samples": len(dataset),
        "loss_history": {"steps": logger.steps, "losses": logger.losses},
        "final_loss": logger.losses[-1] if logger.losses else None,
        "model_path": output_path,
    }
    with open(results_file, "w") as f:
        json.dump(result, f, indent=2)

    free_memory(model, trainer)
    print(f"  Done: {elapsed/60:.1f}min | Loss: {result['final_loss']:.4f}")
    return result


# ============================================================================
# CURRICULUM TRAINING
# ============================================================================

def train_curriculum(student_name, curriculum_key):
    student = Config.STUDENTS[student_name]
    cfg = Config.CURRICULUM_STAGES[curriculum_key]
    run_name = f"{student_name}__{curriculum_key}"
    results_file = f"{Config.RESULTS_DIR}/{run_name}.json"
    output_path = f"{Config.OUTPUT_DIR}/{run_name}"

    if os.path.exists(results_file):
        print(f"\n  SKIP: {run_name}")
        with open(results_file) as f:
            return json.load(f)

    easy_path = f"{Config.DATASET_DIR}/{cfg['easy']}.jsonl"
    hard_path = f"{Config.DATASET_DIR}/{cfg['hard']}.jsonl"
    if not os.path.exists(easy_path) or not os.path.exists(hard_path):
        print(f"  Missing curriculum data, skipping")
        return {}

    print(f"\n{'='*60}")
    print(f"  CURRICULUM: {run_name}")
    print(f"  Easy→Hard: {cfg['easy']} → {cfg['hard']}")
    print(f"{'='*60}")

    easy_data = load_jsonl(easy_path)
    hard_data = load_jsonl(hard_path)
    model, tokenizer = load_model_and_tokenizer(student["model_id"], student["use_lora"])
    hp = Config.LORA_SFT if student["use_lora"] else Config.FULL_SFT

    all_steps, all_losses = [], []
    step_offset = 0
    start = time.time()
    peft_cfg = get_peft_config(student["use_lora"])

    for si, stage in enumerate(cfg["stages"]):
        ratio = stage["hard_ratio"]
        n_hard = int(len(easy_data) * ratio)
        n_easy = len(easy_data) - n_hard

        random.seed(42 + si)
        stage_list = (random.sample(easy_data, min(n_easy, len(easy_data))) +
                      random.sample(hard_data, min(n_hard, len(hard_data))))
        random.shuffle(stage_list)
        stage_ds = Dataset.from_list(stage_list)

        print(f"\n  Stage {si+1}: {ratio:.0%} hard | {len(stage_ds)} samples")

        args = SFTConfig(
            output_dir=f"{output_path}/stage_{si+1}",
            num_train_epochs=stage["epochs"], learning_rate=hp["learning_rate"],
            per_device_train_batch_size=hp["per_device_train_batch_size"],
            gradient_accumulation_steps=hp["gradient_accumulation_steps"],
            max_seq_length=hp["max_seq_length"], lr_scheduler_type=hp["lr_scheduler_type"],
            warmup_ratio=hp["warmup_ratio"] if si == 0 else 0.0,
            bf16=hp["bf16"], gradient_checkpointing=hp["gradient_checkpointing"],
            logging_steps=5, save_strategy="no", report_to="none",
            seed=42, dataset_text_field="text",
        )

        logger = LossLogger()
        trainer = SFTTrainer(
            model=model, args=args, train_dataset=stage_ds,
            processing_class=tokenizer,
            peft_config=peft_cfg if si == 0 else None,
            callbacks=[logger],
        )
        trainer.train()

        for step, loss in zip(logger.steps, logger.losses):
            all_steps.append(step + step_offset)
            all_losses.append(loss)
        if logger.steps:
            step_offset += logger.steps[-1]

        model = trainer.model
        del trainer
        gc.collect()

    elapsed = time.time() - start
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    result = {
        "run_name": run_name, "student": student_name,
        "dataset_config": curriculum_key, "model_id": student["model_id"],
        "use_lora": student["use_lora"], "train_time_seconds": elapsed,
        "train_samples": len(easy_data), "is_curriculum": True,
        "loss_history": {"steps": all_steps, "losses": all_losses},
        "final_loss": all_losses[-1] if all_losses else None,
        "model_path": output_path,
    }
    with open(results_file, "w") as f:
        json.dump(result, f, indent=2)

    free_memory(model)
    print(f"\n  Curriculum done: {elapsed/60:.1f}min | Loss: {result['final_loss']:.4f}")
    return result


# ============================================================================
# EVALUATION
# ============================================================================

def generate_solution(model, tokenizer, prompt):
    msgs = [{"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=1024, do_sample=False,
                              pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def extract_code(resp):
    if "```python" in resp:
        return resp.split("```python")[-1].split("```")[0].strip()
    if "```" in resp:
        parts = resp.split("```")
        if len(parts) >= 3:
            c = parts[1].strip()
            return c.split("\n", 1)[-1] if c.startswith(("python", "py")) else c
    lines, code, on = resp.split("\n"), [], False
    for l in lines:
        if l.strip().startswith(("def ", "class ", "import ", "from ")):
            on = True
        if on:
            code.append(l)
    return "\n".join(code) if code else resp


def run_test(code, test, timeout=10):
    try:
        r = subprocess.run([sys.executable, "-c", code + "\n\n" + test],
                           capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0
    except:
        return False


def eval_humaneval(model, tokenizer):
    try:
        problems = list(load_dataset("openai/openai_humaneval", split="test", trust_remote_code=True))
    except:
        print("  HumanEval load failed")
        return 0.0

    passed = 0
    for p in tqdm(problems, desc="HumanEval"):
        resp = generate_solution(model, tokenizer,
            f"Complete this Python function.\n\n{p['prompt']}\n\nProvide complete function in ```python``` block.")
        code = extract_code(resp)
        check = f"\ncheck({p['entry_point']})\n"
        if run_test(code, p["test"] + check, 15) or run_test(p["prompt"] + code, p["test"] + check, 15):
            passed += 1

    score = passed / len(problems) * 100
    print(f"  HumanEval: {passed}/{len(problems)} = {score:.1f}%")
    return score


def eval_mbpp(model, tokenizer):
    try:
        problems = list(load_dataset("google-research-datasets/mbpp", "full",
                                      split="test", trust_remote_code=True))
    except:
        print("  MBPP load failed")
        return 0.0

    passed = 0
    for p in tqdm(problems, desc="MBPP"):
        resp = generate_solution(model, tokenizer,
            f"Solve in Python.\n\nProblem: {p['text']}\n\nProvide code in ```python``` block.")
        code = extract_code(resp)
        if run_test(code, "\n".join(p.get("test_list", [])), 10):
            passed += 1

    score = passed / len(problems) * 100
    print(f"  MBPP: {passed}/{len(problems)} = {score:.1f}%")
    return score


def evaluate_model(model_path, base_model_id, use_lora):
    if use_lora:
        base = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch.bfloat16,
                                                     device_map="auto", trust_remote_code=True)
        model = PeftModel.from_pretrained(base, model_path).merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                                      device_map="auto", trust_remote_code=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    scores = {"humaneval_pass1": eval_humaneval(model, tokenizer),
              "mbpp_pass1": eval_mbpp(model, tokenizer)}
    scores["average"] = (scores["humaneval_pass1"] + scores["mbpp_pass1"]) / 2

    free_memory(model)
    return scores


# ============================================================================
# PLOTTING
# ============================================================================

def _get(results, student, config):
    return next((r for r in results if r["student"] == student
                 and r["dataset_config"] == config and "eval_scores" in r), None)


def plot_loss_curves(results):
    for student in sorted(set(r["student"] for r in results)):
        runs = [r for r in results if r["student"] == student]
        fig, ax = plt.subplots(figsize=(10, 6))
        for run in runs:
            s = run.get("loss_history", {}).get("steps", [])
            l = run.get("loss_history", {}).get("losses", [])
            if s and l:
                ax.plot(s, l, label=run["dataset_config"], alpha=0.8)
        ax.set_xlabel("Steps"); ax.set_ylabel("Loss")
        ax.set_title(f"Training Loss — {student}")
        ax.legend(fontsize=7, loc="upper right"); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{Config.PLOTS_DIR}/loss_{student}.png", dpi=150); plt.close()
    print("  Saved: loss curves")


def plot_gap(results, pair_a, pair_b, ylabel, title, filename):
    students = sorted(set(r["student"] for r in results))
    gaps_he, gaps_mbpp, gaps_avg, labels = [], [], [], []
    for s in students:
        ra, rb = _get(results, s, pair_a), _get(results, s, pair_b)
        if ra and rb:
            gaps_he.append(ra["eval_scores"]["humaneval_pass1"] - rb["eval_scores"]["humaneval_pass1"])
            gaps_mbpp.append(ra["eval_scores"]["mbpp_pass1"] - rb["eval_scores"]["mbpp_pass1"])
            gaps_avg.append(ra["eval_scores"]["average"] - rb["eval_scores"]["average"])
            labels.append(s.replace("Qwen2.5-Coder-", ""))
    if not labels:
        return
    x, w = np.arange(len(labels)), 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - w, gaps_he, w, label="HumanEval", color="#4CAF50")
    ax.bar(x, gaps_mbpp, w, label="MBPP", color="#2196F3")
    ax.bar(x + w, gaps_avg, w, label="Average", color="#FF9800")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Student Model Size"); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(f"{Config.PLOTS_DIR}/{filename}", dpi=150); plt.close()
    print(f"  Saved: {filename}")


def plot_dataset_size_effect(results):
    students = sorted(set(r["student"] for r in results))
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, m, t in zip(axes, ["humaneval_pass1", "mbpp_pass1"], ["HumanEval", "MBPP"]):
        for s in students:
            sizes, scores = [], []
            for cfg in ["short_cot_500", "short_cot_1000"]:
                run = _get(results, s, cfg)
                if run:
                    sizes.append(int(cfg.split("_")[-1]))
                    scores.append(run["eval_scores"][m])
            if sizes:
                ax.plot(sizes, scores, "o-", label=s.replace("Qwen2.5-Coder-", ""), linewidth=2)
        ax.set_xlabel("Dataset Size"); ax.set_ylabel("pass@1 (%)"); ax.set_title(t)
        ax.legend(); ax.grid(True, alpha=0.3)
    plt.suptitle("Dataset Size Effect (Short CoT)")
    plt.tight_layout(); plt.savefig(f"{Config.PLOTS_DIR}/dataset_size_effect.png", dpi=150); plt.close()
    print("  Saved: dataset_size_effect.png")


def plot_all_methods(results):
    student = "Qwen2.5-Coder-3B"
    configs = [("long_cot_1000", "Long CoT"), ("short_cot_1000", "Short CoT"),
               ("large_teacher_1000", "Large Teacher"), ("small_teacher_1000", "Small Teacher"),
               ("mix_long_1000", "Mix-Long"), ("mix_large_1000", "Mix-Large"),
               ("curriculum_long", "Curric-Long"), ("curriculum_large", "Curric-Large")]
    names, he, mbpp = [], [], []
    for cfg, label in configs:
        run = _get(results, student, cfg)
        if run:
            names.append(label)
            he.append(run["eval_scores"]["humaneval_pass1"])
            mbpp.append(run["eval_scores"]["mbpp_pass1"])
    if not names:
        return
    x, w = np.arange(len(names)), 0.35
    fig, ax = plt.subplots(figsize=(13, 6))
    b1 = ax.bar(x - w/2, he, w, label="HumanEval", color="#4CAF50")
    b2 = ax.bar(x + w/2, mbpp, w, label="MBPP", color="#2196F3")
    for b in list(b1) + list(b2):
        ax.annotate(f'{b.get_height():.1f}', xy=(b.get_x() + b.get_width()/2, b.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    ax.set_xlabel("Method"); ax.set_ylabel("pass@1 (%)")
    ax.set_title(f"All Methods — {student}")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=25, ha="right")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout(); plt.savefig(f"{Config.PLOTS_DIR}/all_methods.png", dpi=150); plt.close()
    print("  Saved: all_methods.png")


def plot_curriculum_vs_static(results):
    students = sorted(set(r["student"] for r in results))
    methods = [("long_cot_1000", "Long CoT Only", "#e74c3c"),
               ("short_cot_1000", "Short CoT Only", "#3498db"),
               ("mix_long_1000", "Static Mix", "#f39c12"),
               ("curriculum_long", "Curriculum", "#2ecc71")]
    fig, ax = plt.subplots(figsize=(11, 6))
    x, w = np.arange(len(students)), 0.2
    for j, (cfg, label, color) in enumerate(methods):
        scores = [(_get(results, s, cfg) or {}).get("eval_scores", {}).get("average", 0) for s in students]
        bars = ax.bar(x + j*w - 1.5*w, scores, w, label=label, color=color)
        for b in bars:
            if b.get_height() > 0:
                ax.annotate(f'{b.get_height():.1f}', xy=(b.get_x()+b.get_width()/2, b.get_height()),
                            xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7)
    ax.set_xlabel("Student Model Size"); ax.set_ylabel("Avg pass@1 (%)")
    ax.set_title("Curriculum vs Static Mix vs Baselines")
    ax.set_xticks(x); ax.set_xticklabels([s.replace("Qwen2.5-Coder-","") for s in students])
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout(); plt.savefig(f"{Config.PLOTS_DIR}/curriculum_vs_static.png", dpi=150); plt.close()
    print("  Saved: curriculum_vs_static.png")


def save_results_table(results):
    rows = [{"Student": r["student"].replace("Qwen2.5-Coder-",""),
             "Config": r["dataset_config"],
             "HumanEval": f"{r['eval_scores']['humaneval_pass1']:.1f}",
             "MBPP": f"{r['eval_scores']['mbpp_pass1']:.1f}",
             "Average": f"{r['eval_scores']['average']:.1f}",
             "Loss": f"{r.get('final_loss',0):.4f}"}
            for r in results if "eval_scores" in r]
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(f"{Config.RESULTS_DIR}/full_results.csv", index=False)
        print(f"\n{df.to_string(index=False)}")


# ============================================================================
# PIPELINE
# ============================================================================

def load_existing():
    return [json.load(open(f)) for f in sorted(Path(Config.RESULTS_DIR).glob("*.json"))]


def run_training():
    print("\n" + "="*60 + "\n  PHASE 1: TRAINING\n" + "="*60)

    available = [c for c in Config.DATASET_CONFIGS
                 if os.path.exists(f"{Config.DATASET_DIR}/{c}.jsonl")]
    n = len(Config.STUDENTS) * len(available)
    print(f"  {len(Config.STUDENTS)} students × {len(available)} datasets = {n} standard")
    print(f"  + {len(Config.STUDENTS) * len(Config.CURRICULUM_STAGES)} curriculum runs")

    results, count = [], 0
    for s in Config.STUDENTS:
        for cfg in available:
            count += 1
            print(f"\n  [{count}/{n}]")
            try:
                results.append(train_one(s, cfg))
            except Exception as e:
                print(f"  FAILED: {e}")

    for s in Config.STUDENTS:
        for ck, cv in Config.CURRICULUM_STAGES.items():
            if (os.path.exists(f"{Config.DATASET_DIR}/{cv['easy']}.jsonl") and
                os.path.exists(f"{Config.DATASET_DIR}/{cv['hard']}.jsonl")):
                try:
                    r = train_curriculum(s, ck)
                    if r: results.append(r)
                except Exception as e:
                    print(f"  FAILED curriculum: {e}")
    return results


def run_evaluation(results):
    print("\n" + "="*60 + "\n  PHASE 2: EVALUATION\n" + "="*60)
    for i, r in enumerate(results):
        if "eval_scores" in r or not r.get("model_path"):
            continue
        print(f"  [{i+1}/{len(results)}] {r['run_name']}")
        try:
            r["eval_scores"] = evaluate_model(r["model_path"], r["model_id"], r["use_lora"])
            with open(f"{Config.RESULTS_DIR}/{r['run_name']}.json", "w") as f:
                json.dump(r, f, indent=2)
        except Exception as e:
            print(f"  EVAL FAILED: {e}")
            r["eval_scores"] = {"humaneval_pass1": 0, "mbpp_pass1": 0, "average": 0}
    return results


def run_plotting(results):
    print("\n" + "="*60 + "\n  PHASE 3: PLOTTING\n" + "="*60)
    plot_loss_curves(results)
    plot_gap(results, "long_cot_1000", "short_cot_1000",
             "Δ_Long", "Long CoT Gap: Negative = Short CoT Better", "long_cot_gap.png")
    plot_gap(results, "large_teacher_1000", "small_teacher_1000",
             "Δ_Large", "Large Teacher Gap: Negative = Small Teacher Better", "large_teacher_gap.png")
    plot_dataset_size_effect(results)
    plot_all_methods(results)
    plot_curriculum_vs_static(results)
    save_results_table(results)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("  LEARNABILITY GAP — TRAIN & EVAL")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem/1e9:.1f} GB")
    print(f"  Students: {', '.join(Config.STUDENTS.keys())}")
    print("="*60)

    run_training()
    results = load_existing()
    run_evaluation(results)
    results = load_existing()
    run_plotting(results)

    print(f"""
{'='*60}
  DONE!
  Models:  {Config.OUTPUT_DIR}/
  Results: {Config.RESULTS_DIR}/
  Plots:   {Config.PLOTS_DIR}/
{'='*60}

  PLOTS GENERATED:
    loss_<student>.png       — Training loss curves
    long_cot_gap.png         — Δ_Long per student (paper Fig 2)
    large_teacher_gap.png    — Δ_Large per student (paper Fig 3)
    dataset_size_effect.png  — pass@1 vs data size (Gap #9)
    all_methods.png          — All methods for 3B (paper Table 3)
    curriculum_vs_static.png — Curriculum vs Mix (Gap #7 — YOUR CONTRIBUTION)
    full_results.csv         — Complete results table

  THESIS CHECKLIST:
    1. long_cot_gap.png: Negative for 0.5B/1.5B? → Gap exists in code gen
    2. large_teacher_gap.png: Same pattern? → Confirms cross-domain gap
    3. dataset_size_effect.png: More data helps? → Gap #9 answered
    4. all_methods.png: Mix beats baselines? → Paper solution validated
    5. curriculum_vs_static.png: Curriculum beats Mix? → YOUR NOVEL FINDING
    """)