# Experiment Guide

This repo has two independent experiment paths:

- Legacy RL-of-thoughts training still lives in `train_Dueling_DDQN_MATH.py`, but its PRM scoring goes through the HTTP API instead of loading the Hugging Face PRM inside the trainer.
- Qwen3.5 2B fine-tuning lives in `train_qwen35_unsloth.py` and uses Unsloth LoRA SFT for a small, efficient baseline model. This is not required for the legacy DDQN reasoning-controller experiments.

## 1. Environment

Use a fresh Python environment on a CUDA machine.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install --upgrade --force-reinstall --no-cache-dir unsloth unsloth_zoo
pip install datasets trl accelerate bitsandbytes httpx fastapi uvicorn llama-cpp-python
pip install modelscope pandas networkx matplotlib tqdm openai
```

Qwen3.5 support needs a current Unsloth / Transformers stack. The first run may spend extra time compiling Qwen3.5 kernels.

## 2. PRM API

The trainer defaults to `PRM_API_URL` or, if unset, the hosted benchmark endpoint used by `prm_gguf/bench_prm_api.py`.

To use a local PRM server:

```bash
cd prm_gguf
export PRM_MODEL_PATH=/path/to/math-shepherd-mistral-7b-prm.Q4_K_M.gguf
export PRM_N_GPU_LAYERS=32
export PRM_NUM_WORKERS=1
python prm_api.py
```

In another shell, verify it:

```bash
python prm_gguf/bench_prm_api.py --base-url http://localhost:8787 --requests 32 --concurrency 4
```

Then run the legacy RL trainer with API scoring:

```bash
export PRM_API_URL=http://localhost:8787
python train_Dueling_DDQN_MATH.py --dataset GPQA --num_episodes 100
```

If you prefer the hosted endpoint, omit `PRM_API_URL` or pass `--PRM_api_url https://prm.24102006.xyz`.

### DDQN controller with a custom Qwen3.5 2B API

To run the legacy DDQN reasoning controller with a smaller Qwen3.5 2B generator, serve Qwen3.5 2B behind an OpenAI-compatible chat-completions API and point the DDQN trainer at it:

```bash
python train_Dueling_DDQN_MATH.py \
  --dataset GPQA \
  --LLM_name Qwen3.5-2B \
  --LLM_api_base_url http://localhost:8000/v1 \
  --LLM_api_key EMPTY \
  --problem_indexs_mode auto \
  --problem_indexs_fallback_model Qwen2.5-14B-Instruct \
  --num_episodes 100 \
  --PRM_api_url https://prm.24102006.xyz
```

`--LLM_api_base_url` should include `/v1`, or it can be the full `/v1/chat/completions` URL. You can also set `LLM_API_BASE_URL` and `LLM_API_KEY` instead of passing the flags.

Problem indices are model-specific because the original experiments trained on problems the base LLM failed under direct prompting. For a new model name such as `Qwen3.5-2B`, `--problem_indexs_mode auto` uses `data/problem_indexs/Qwen3.5-2B/GPQA/indexs.pkl` if it exists. If it does not exist, it uses the reference hard set from `--problem_indexs_fallback_model` when available, which defaults to `Qwen2.5-14B-Instruct`; only then does it fall back to all problems. Use `--problem_indexs_mode file` for strict reproduction with a prepared index file, or `--problem_indexs_mode all` to always train on the full dataset.

For local GGUF servers, the trainer now sends `top_k=1` by default so OpenAI-compatible llama.cpp/vLLM-style endpoints keep the same near-greedy behavior as the original payload. If a strict server rejects that field, the client retries without `top_k`; set `--LLM_send_top_k false` to disable it explicitly.

Training writes separate reward views in addition to the mixed episode reward: `final_accuracy_smooth.png`, `prm_rewards_smooth.png`, `reward_components_smooth.png`, and the matching `final_rewards.npy`, `prm_rewards.npy`, and `reward_components.pkl` arrays. The x-axis is the number of completed episodes, not the configured target episode count, so interrupted runs no longer look artificially stretched to 3000 episodes.

## 3. Dataset Download

The Unsloth trainer downloads GSM8K automatically from Hugging Face:

```bash
python train_qwen35_unsloth.py --max_train_samples 32 --max_steps 5 --output_dir model/qwen35-2b-smoke
```

The default dataset is `openai/gsm8k`, config `main`, split `train`, cached under `data/hf_cache`.

For a local dataset, provide JSON/JSONL with either:

- `question` and `answer` columns, or
- a ready-made `text` column, or
- a `messages` column containing chat messages.

Example:

```bash
python train_qwen35_unsloth.py \
  --data_path data/my_math_train.jsonl \
  --question_column question \
  --answer_column answer \
  --output_dir model/qwen35-2b-local-lora
```

## 4. Qwen3.5 2B Training

Full 1-epoch GSM8K LoRA SFT:

```bash
python train_qwen35_unsloth.py \
  --model_name unsloth/Qwen3.5-2B \
  --dataset_name openai/gsm8k \
  --dataset_config main \
  --split train \
  --output_dir model/qwen35-2b-gsm8k-lora \
  --max_seq_length 2048 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-4 \
  --num_train_epochs 1 \
  --save_steps 200
```

The script saves LoRA adapters and tokenizer files to `output_dir`. Add `--save_merged_16bit` for a merged HF checkpoint, or `--save_gguf` for a llama.cpp-style export.

## 5. Compute Estimate

Qwen3.5 2B LoRA SFT at sequence length 2048 should fit on a single 8 GB CUDA GPU; 12-16 GB is more comfortable. Expect roughly:

- Smoke run, 32 examples / 5 steps: 5-20 minutes on first run including download and kernel compilation.
- Full GSM8K train split, 1 epoch: about 20-45 minutes on A100/L40S/RTX 4090 class GPUs, 1-3 hours on L4/T4 class GPUs.
- Disk: reserve 15-25 GB for model cache, dataset cache, LoRA outputs, and checkpoints. Add 5-8 GB if exporting merged or GGUF artifacts.

Local PRM serving needs the GGUF PRM file plus enough GPU memory for a quantized Mistral-7B reward model. A Q4_K_M GGUF usually wants about 8-12 GB VRAM with `PRM_N_GPU_LAYERS=32`; reduce GPU layers if it does not fit.

## References

- Qwen3.5 2B model card: https://huggingface.co/unsloth/Qwen3.5-2B
- Qwen3.5 Base model card: https://huggingface.co/Qwen/Qwen3.5-2B-Base
- Unsloth Qwen3.5 fine-tuning guide: https://unsloth.ai/docs/models/qwen3.5/fine-tune
- GSM8K dataset: https://huggingface.co/datasets/openai/gsm8k
