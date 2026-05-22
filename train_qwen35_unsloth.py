from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_SYSTEM_PROMPT = (
    "You are a careful math reasoning assistant. Solve the problem step by step "
    "and end with the final answer in the same format as the training data."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Memory-efficient Qwen3.5 2B LoRA SFT with Unsloth."
    )
    parser.add_argument("--model_name", default="unsloth/Qwen3.5-2B")
    parser.add_argument("--output_dir", default="model/qwen35-2b-gsm8k-lora")
    parser.add_argument("--dataset_name", default="openai/gsm8k")
    parser.add_argument("--dataset_config", default="main")
    parser.add_argument("--split", default="train")
    parser.add_argument("--data_path", default=None, help="Optional local JSON/JSONL file.")
    parser.add_argument("--dataset_cache_dir", default="data/hf_cache")
    parser.add_argument("--question_column", default="question")
    parser.add_argument("--answer_column", default="answer")
    parser.add_argument("--text_column", default="text")
    parser.add_argument("--messages_column", default="messages")
    parser.add_argument("--system_prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lr_scheduler_type", default="linear")
    parser.add_argument("--optim", default="adamw_8bit")
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--dataset_num_proc", type=int, default=1)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--packing", action="store_true")
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--report_to", default="none")
    parser.add_argument("--save_merged_16bit", action="store_true")
    parser.add_argument("--save_gguf", action="store_true")
    parser.add_argument("--gguf_quantization", default="q4_k_m")
    return parser.parse_args()


def apply_chat_template(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    rendered = []
    for message in messages:
        role = message["role"].strip().lower()
        content = message["content"].strip()
        rendered.append(f"### {role}\n{content}")
    return "\n\n".join(rendered) + "\n"


def normalise_messages(value: Any) -> list[dict[str, str]]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list):
        raise ValueError("messages column must contain a list of role/content objects")
    return [
        {"role": str(item["role"]), "content": str(item["content"])}
        for item in value
    ]


def build_text_dataset(args: argparse.Namespace, tokenizer: Any):
    from datasets import load_dataset

    if args.data_path:
        raw_dataset = load_dataset(
            "json",
            data_files=args.data_path,
            split=args.split,
            cache_dir=args.dataset_cache_dir,
        )
    else:
        dataset_kwargs = {
            "split": args.split,
            "cache_dir": args.dataset_cache_dir,
        }
        if args.dataset_config:
            raw_dataset = load_dataset(
                args.dataset_name,
                args.dataset_config,
                **dataset_kwargs,
            )
        else:
            raw_dataset = load_dataset(args.dataset_name, **dataset_kwargs)

    if args.max_train_samples is not None:
        raw_dataset = raw_dataset.shuffle(seed=args.seed)
        raw_dataset = raw_dataset.select(range(min(args.max_train_samples, len(raw_dataset))))

    columns = set(raw_dataset.column_names)

    def format_example(example: dict[str, Any]) -> dict[str, str]:
        if args.text_column in columns and example.get(args.text_column):
            return {"text": str(example[args.text_column])}

        if args.messages_column in columns and example.get(args.messages_column):
            messages = normalise_messages(example[args.messages_column])
            return {"text": apply_chat_template(tokenizer, messages)}

        if args.question_column not in columns or args.answer_column not in columns:
            raise ValueError(
                "Dataset must have either a text column, a messages column, or "
                f"'{args.question_column}' and '{args.answer_column}' columns."
            )

        messages = [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": str(example[args.question_column]).strip()},
            {"role": "assistant", "content": str(example[args.answer_column]).strip()},
        ]
        return {"text": apply_chat_template(tokenizer, messages)}

    return raw_dataset.map(
        format_example,
        remove_columns=raw_dataset.column_names,
        num_proc=args.dataset_num_proc,
        desc="Formatting training examples",
    )


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel, is_bfloat16_supported

    bf16 = is_bfloat16_supported()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        load_in_16bit=not args.load_in_4bit,
        full_finetuning=False,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        max_seq_length=args.max_seq_length,
    )

    train_dataset = build_text_dataset(args, tokenizer)
    report_to = [] if args.report_to == "none" else args.report_to.split(",")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=SFTConfig(
            output_dir=args.output_dir,
            max_seq_length=args.max_seq_length,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            max_steps=args.max_steps,
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            optim=args.optim,
            weight_decay=args.weight_decay,
            lr_scheduler_type=args.lr_scheduler_type,
            seed=args.seed,
            dataset_num_proc=args.dataset_num_proc,
            packing=args.packing,
            bf16=bf16,
            fp16=not bf16,
            report_to=report_to,
        ),
    )

    config_path = Path(args.output_dir) / "experiment_config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.save_merged_16bit:
        model.save_pretrained_merged(
            str(Path(args.output_dir) / "merged_16bit"),
            tokenizer,
            save_method="merged_16bit",
        )

    if args.save_gguf:
        model.save_pretrained_gguf(
            str(Path(args.output_dir) / "gguf"),
            tokenizer,
            quantization_method=args.gguf_quantization,
        )


if __name__ == "__main__":
    main()
