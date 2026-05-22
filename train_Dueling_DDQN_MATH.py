import os
os.chdir(os.path.split(os.path.realpath(__file__))[0])

import sys
sys.path.append('env')

import os
import pickle
import torch
import json
import copy
import time
import random
import shutil
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from PRM import PRM
from Dueling_DQN_net import Dueling_DQN
from DQN_buffer import Experience,ExperienceReplay
from env.RL_env import RLEnv

class Dueling_DDQN_agent():
    def __init__(self,
                 manual_seed,
                 dataset,
                 LLM_name,
                 LLM_api_base_url,
                 LLM_api_key,
                 LLM_api_timeout,
                 LLM_max_tokens,
                 LLM_temperature,
                 LLM_top_p,
                 LLM_top_k,
                 LLM_send_top_k,
                 LLM_max_prompt_tokens,
                 LLM_prompt_token_chars_per_token,
                 LLM_skip_oversize_prompts,
                 problem_indexs_name,
                 problem_indexs_path,
                 problem_indexs_mode,
                 problem_indexs_fallback_model,
                 random_problems,
                 debug_verbose,
                 answer_fallback_policy,
                 PRM_name,
                 PRM_api_url,
                 PRM_timeout,
                 PRM_max_retries,
                 max_depth,
                 max_width,
                 save_interval,
                 episode_log_interval,
                 LLM_generation_log_interval,
                 save_dir,
                 resume_from,
                 reset_training_state,
                 learning_rate,
                 learning_rate_decay,
                 learning_rate_decay_interval,
                 gamma,
                 num_episodes,
                 batch_size,
                 start_epsilon,
                 min_epsilon,
                 epsilon_decay,
                 buffer_size,
                 target_update_interval
                 ):

        print('Initializing...')

        # generate config
        config_data=locals()
        del config_data['self']
        if config_data.get('LLM_api_key'):
            config_data['LLM_api_key'] = '***'
        time_data=time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
        config_data['code_dir']=os.path.split(os.path.realpath(__file__))[0]
        config_data['time']=time_data

        # random seed
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        self.manual_seed=manual_seed
        torch.manual_seed(self.manual_seed)
        if self.device=='cuda':
            torch.cuda.manual_seed(self.manual_seed)
        np.random.seed(self.manual_seed)

        self.dataset = dataset
        self.LLM_name = LLM_name
        self.LLM_api_base_url = LLM_api_base_url
        self.LLM_api_key = LLM_api_key
        self.LLM_api_timeout = LLM_api_timeout
        self.LLM_max_tokens = LLM_max_tokens
        self.LLM_temperature = LLM_temperature
        self.LLM_top_p = LLM_top_p
        self.LLM_top_k = LLM_top_k
        self.LLM_send_top_k = LLM_send_top_k
        self.LLM_max_prompt_tokens = LLM_max_prompt_tokens
        self.LLM_prompt_token_chars_per_token = LLM_prompt_token_chars_per_token
        self.LLM_skip_oversize_prompts = LLM_skip_oversize_prompts
        self.problem_indexs_fallback_model = problem_indexs_fallback_model
        self.problem_indexs = self.load_problem_indexs(problem_indexs_name, problem_indexs_path, problem_indexs_mode, problem_indexs_fallback_model)
        self.random_problems = random_problems
        self.max_depth = max_depth
        self.max_width = max_width
        self.debug_verbose = debug_verbose
        self.answer_fallback_policy = answer_fallback_policy
        self.env = RLEnv(dataset=self.dataset, is_test=False, LLM_name=self.LLM_name, problem_indexs=self.problem_indexs, max_depth=self.max_depth, max_width=self.max_width, random_problems=self.random_problems, random_seed=self.manual_seed, LLM_api_base_url=self.LLM_api_base_url, LLM_api_key=self.LLM_api_key, LLM_api_timeout=self.LLM_api_timeout, LLM_max_tokens=self.LLM_max_tokens, LLM_temperature=self.LLM_temperature, LLM_top_p=self.LLM_top_p, LLM_top_k=self.LLM_top_k, LLM_send_top_k=self.LLM_send_top_k, LLM_max_prompt_tokens=self.LLM_max_prompt_tokens, LLM_prompt_token_chars_per_token=self.LLM_prompt_token_chars_per_token, LLM_skip_oversize_prompts=self.LLM_skip_oversize_prompts, debug_verbose=self.debug_verbose, answer_fallback_policy=self.answer_fallback_policy)
        self.n_actions = len(self.env.action_space)
        self.n_features = len(self.env.observation_space)

        self.save_dir = save_dir
        self.save_interval = save_interval
        self.episode_log_interval = episode_log_interval
        self.LLM_generation_log_interval = LLM_generation_log_interval
        self.resume_from = resume_from
        self.reset_training_state = reset_training_state
        if self.resume_from:
            self.save_folder = os.path.abspath(self.resume_from)
        else:
            self.save_folder = os.path.join(self.save_dir, self.LLM_name.split("/")[-1], self.dataset, 'Dueling_DDQN_'+time_data)
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        if self.reset_training_state:
            self.reset_saved_training_state()

        with open(os.path.join(self.save_folder,'config.json'),'w') as f:
            json.dump(config_data,f)
        os.system(f'cp -r '+'env'+' '+os.path.join(self.save_folder,'env'))
        os.system(f'cp '+'train_Dueling_DDQN_MATH.py'+' '+os.path.join(self.save_folder,'train_Dueling_DDQN_MATH.py'))
        os.system(f'cp '+'Dueling_DQN_net.py'+' '+os.path.join(self.save_folder,'Dueling_DQN_net.py'))
        os.system(f'cp '+'DQN_buffer.py'+' '+os.path.join(self.save_folder,'DQN_buffer.py'))
        os.system(f'cp '+'PRM.py'+' '+os.path.join(self.save_folder,'PRM.py'))

        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_decay_interval = learning_rate_decay_interval
        self.gamma = gamma
        self.num_episodes = num_episodes
        self.batch_size = batch_size
        self.epsilon = start_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.buffer_size = buffer_size
        self.target_update_interval = target_update_interval

        self.PRM_name = PRM_name
        self.PRM_api_url = PRM_api_url
        self.PRM_timeout = PRM_timeout
        self.PRM_max_retries = PRM_max_retries
        self.PRM = PRM(
            PRM_name=self.PRM_name,
            device=self.device,
            api_base_url=self.PRM_api_url,
            timeout=self.PRM_timeout,
            max_retries=self.PRM_max_retries,
        )
        
        self.Dueling_DQN = Dueling_DQN(input_size=self.n_features,output_size=self.n_actions).to(self.device)
        self.Dueling_DQN_target = copy.deepcopy(self.Dueling_DQN).to(self.device)
        self.optimizer = torch.optim.Adam(self.Dueling_DQN.parameters(), lr=self.learning_rate)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.learning_rate_decay_interval, gamma=self.learning_rate_decay)
        self.loss = torch.nn.MSELoss()
        self.memory = ExperienceReplay(capacity=self.buffer_size, random_seed=self.manual_seed)

        self.rewards = list()
        self.final_rewards = list()
        self.prm_rewards = list()
        self.step_counts = list()
        self.learn_count = 0
        self.start_episode = 0

        if self.resume_from and not self.reset_training_state:
            self.load_training_checkpoint()

        print(self.LLM_name + ' on ' + self.dataset + ' initialized.')
        self.log_path = os.path.join(self.save_folder, 'thoughts_log.txt')
        self._redirect_stdout_to_log()

    def load_problem_indexs(self, problem_indexs_name, problem_indexs_path, problem_indexs_mode, problem_indexs_fallback_model):
        if problem_indexs_mode == "all":
            print("Using all dataset problems because --problem_indexs_mode=all.")
            return None

        if problem_indexs_path is None:
            problem_indexs_path = os.path.join('data', problem_indexs_name, self.LLM_name.split("/")[-1], self.dataset, 'indexs.pkl')

        if os.path.exists(problem_indexs_path):
            with open(problem_indexs_path, "rb") as f:
                return pickle.load(f)

        if problem_indexs_mode == "auto":
            if problem_indexs_fallback_model:
                fallback_path = os.path.join(
                    'data',
                    problem_indexs_name,
                    problem_indexs_fallback_model.split("/")[-1],
                    self.dataset,
                    'indexs.pkl',
                )
                if os.path.exists(fallback_path):
                    print(
                        "Warning: problem index file not found at "
                        f"{problem_indexs_path}; using fallback hard-problem "
                        f"index from {fallback_path}."
                    )
                    with open(fallback_path, "rb") as f:
                        return pickle.load(f)

            print(f"Warning: problem index file not found at {problem_indexs_path}; using all dataset problems.")
            return None

        raise FileNotFoundError(
            f"Problem index file not found at {problem_indexs_path}. "
            "Use --problem_indexs_mode auto to fall back to a reference hard set or all problems, "
            "--problem_indexs_mode all to always use all problems, or "
            "--problem_indexs_path to provide a custom indexs.pkl."
        )

    def _restore_stdout(self):
        if sys.stdout is not sys.__stdout__:
            try:
                sys.stdout.flush()
                sys.stdout.close()
            except Exception:
                pass
        sys.stdout = sys.__stdout__

    def _redirect_stdout_to_log(self):
        if sys.stdout is not sys.__stdout__:
            self._restore_stdout()
        sys.stdout = open(self.log_path, 'a', buffering=1)

    def _tqdm_write(self, message):
        self._restore_stdout()
        tqdm.write(message)
        self._redirect_stdout_to_log()
    
    def choose_action(self,state):
        if np.random.uniform() < self.epsilon:
            action = np.random.randint(0,self.n_actions)
        else:
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                actions_value = self.Dueling_DQN(state)
            action = torch.argmax(actions_value).item()

        return action
        
    def learn(self):
        batch_memory = self.memory.sample(self.batch_size)
        states, actions, rewards, dones, next_states = batch_memory

        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.bool, device=self.device)
        next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)

        action_values = self.Dueling_DQN(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.Dueling_DQN(next_states).max(1)[1]
            next_action_values = self.Dueling_DQN_target(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            next_action_values[dones] = 0

        target_values = rewards + self.gamma * next_action_values
        loss = self.loss(action_values, target_values)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.learn_count += 1

        self._tqdm_write(f"Loss={loss.item()}")
        
        if self.learn_count % self.target_update_interval == 0:
            self.Dueling_DQN_target.load_state_dict(self.Dueling_DQN.state_dict())
            self._tqdm_write("Target network updated.")

    @staticmethod
    def _moving_average(values, window=100):
        values = np.asarray(values, dtype=np.float32)
        if len(values) == 0:
            return np.array([], dtype=np.float32)
        averaged = []
        for idx in range(len(values)):
            window_values = values[max(0, idx + 1 - window):idx + 1]
            if np.all(np.isnan(window_values)):
                averaged.append(np.nan)
            else:
                averaged.append(np.nanmean(window_values))
        return np.asarray(averaged, dtype=np.float32)

    def _plot_series(self, filename, series, ylabel, smooth=False):
        if len(series) == 0:
            return
        y = self._moving_average(series) if smooth else np.asarray(series, dtype=np.float32)
        x = np.arange(len(y))
        plt.figure(figsize=(8,6))
        plt.plot(x, y)
        plt.xlabel('Episode')
        plt.ylabel(ylabel)
        plt.xlim(0, max(1, len(y) - 1))
        plt.savefig(os.path.join(self.save_folder, filename))
        plt.close()

    def _save_reward_arrays(self):
        np.save(os.path.join(self.save_folder, "rewards.npy"), np.array(self.rewards, dtype=np.float32))
        np.save(os.path.join(self.save_folder, "final_rewards.npy"), np.array(self.final_rewards, dtype=np.float32))
        np.save(os.path.join(self.save_folder, "prm_rewards.npy"), np.array(self.prm_rewards, dtype=np.float32))
        np.save(os.path.join(self.save_folder, "step_counts.npy"), np.array(self.step_counts, dtype=np.int32))
        with open(os.path.join(self.save_folder, "reward_components.pkl"), "wb") as f:
            pickle.dump(
                {
                    "episode_rewards": self.rewards,
                    "final_rewards": self.final_rewards,
                    "prm_rewards": self.prm_rewards,
                    "step_counts": self.step_counts,
                },
                f,
            )

    def save(self,episode_count):
        torch.save(self.Dueling_DQN.state_dict(), os.path.join(self.save_folder, f"model_episode{episode_count}.pth"))

        self._plot_series("rewards.png", self.rewards, "Reward")
        self._plot_series("smooth_rewards.png", self.rewards, "Smooth reward", smooth=True)
        self._plot_series("final_accuracy_smooth.png", self.final_rewards, "Final answer accuracy", smooth=True)
        self._plot_series("prm_rewards_smooth.png", self.prm_rewards, "PRM reward", smooth=True)

        if len(self.rewards) > 0:
            x = np.arange(len(self.rewards))
            plt.figure(figsize=(8,6))
            plt.plot(x, self._moving_average(self.rewards), label="mixed episode reward")
            plt.plot(x, self._moving_average(self.final_rewards), label="final answer accuracy")
            plt.plot(x, self._moving_average(self.prm_rewards), label="PRM reward")
            plt.xlabel('Episode')
            plt.ylabel('Smooth reward')
            plt.xlim(0, max(1, len(self.rewards) - 1))
            plt.legend()
            plt.savefig(os.path.join(self.save_folder, "reward_components_smooth.png"))
            plt.close()

        plt.figure(figsize=(8,6))
        plt.plot(self.step_counts)
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.xlim(0, max(1, len(self.step_counts) - 1))
        plt.savefig(os.path.join(self.save_folder, "step_counts.png"))
        plt.close()

        with open(os.path.join(self.save_folder, "rewards.pkl"), "wb") as f:
            pickle.dump(self.rewards, f)
        self._save_reward_arrays()

    def reset_saved_training_state(self):
        removable_files = {
            "checkpoint_latest.pt",
            "training_state.json",
            "rewards.npy",
            "rewards.pkl",
            "final_rewards.npy",
            "prm_rewards.npy",
            "step_counts.npy",
            "reward_components.pkl",
            "rewards.png",
            "smooth_rewards.png",
            "final_accuracy_smooth.png",
            "prm_rewards_smooth.png",
            "reward_components_smooth.png",
            "step_counts.png",
            "thoughts_log.txt",
        }
        removable_prefixes = (
            "checkpoint_episode",
            "model_episode",
            "record_episode",
        )

        for name in os.listdir(self.save_folder):
            path = os.path.join(self.save_folder, name)
            if name in removable_files or name.startswith(removable_prefixes):
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)

        episode_log_dir = os.path.join(self.save_folder, "episode_logs")
        if os.path.isdir(episode_log_dir):
            shutil.rmtree(episode_log_dir)

        generation_log_dir = os.path.join(self.save_folder, "llm_generation_logs")
        if os.path.isdir(generation_log_dir):
            shutil.rmtree(generation_log_dir)

    def checkpoint_path(self):
        return os.path.join(self.save_folder, "checkpoint_latest.pt")

    def training_state_path(self):
        return os.path.join(self.save_folder, "training_state.json")

    @staticmethod
    def _torch_load(path, map_location):
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=map_location)

    @staticmethod
    def _cpu_byte_rng_state(state):
        if isinstance(state, torch.Tensor):
            return state.detach().cpu().to(dtype=torch.uint8)
        return torch.tensor(state, dtype=torch.uint8)

    def _align_metric_length(self, values, fill_value):
        values = list(values)
        if len(values) < len(self.rewards):
            values.extend([fill_value] * (len(self.rewards) - len(values)))
        elif len(values) > len(self.rewards):
            values = values[:len(self.rewards)]
        return values

    def save_training_checkpoint(self, next_episode):
        checkpoint = {
            "version": 2,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
            "next_episode": next_episode,
            "model_state_dict": self.Dueling_DQN.state_dict(),
            "target_model_state_dict": self.Dueling_DQN_target.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            "memory_buffer": list(self.memory.buffer),
            "epsilon": self.epsilon,
            "rewards": self.rewards,
            "final_rewards": self.final_rewards,
            "prm_rewards": self.prm_rewards,
            "step_counts": self.step_counts,
            "learn_count": self.learn_count,
            "env_current_problem": self.env.current_problem,
            "env_finished": self.env.finished,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
        }
        tmp_path = self.checkpoint_path() + ".tmp"
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, self.checkpoint_path())

        state_summary = {
            "timestamp": checkpoint["timestamp"],
            "checkpoint": self.checkpoint_path(),
            "next_episode": next_episode,
            "num_episodes": self.num_episodes,
            "completed_episodes": next_episode,
            "epsilon": self._json_safe(self.epsilon),
            "learn_count": self.learn_count,
            "replay_buffer_size": len(self.memory),
            "reward_count": len(self.rewards),
            "final_reward_count": len(self.final_rewards),
            "prm_reward_count": len(self.prm_rewards),
        }
        tmp_state_path = self.training_state_path() + ".tmp"
        with open(tmp_state_path, "w") as f:
            json.dump(state_summary, f, indent=2)
        os.replace(tmp_state_path, self.training_state_path())

    def load_training_checkpoint(self):
        path = self.checkpoint_path()
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No checkpoint found at {path}. Use --reset_training_state true "
                "to start fresh in this folder, or pass a folder with checkpoint_latest.pt."
            )

        checkpoint = self._torch_load(path, map_location=self.device)
        self.Dueling_DQN.load_state_dict(checkpoint["model_state_dict"])
        self.Dueling_DQN_target.load_state_dict(checkpoint["target_model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        self.memory.buffer.clear()
        self.memory.buffer.extend(checkpoint.get("memory_buffer", []))
        self.epsilon = checkpoint["epsilon"]
        self.rewards = checkpoint["rewards"]
        self.final_rewards = checkpoint.get("final_rewards", [np.nan] * len(self.rewards))
        self.prm_rewards = checkpoint.get("prm_rewards", [np.nan] * len(self.rewards))
        self.step_counts = checkpoint.get("step_counts", [0] * len(self.rewards))
        self.final_rewards = self._align_metric_length(self.final_rewards, np.nan)
        self.prm_rewards = self._align_metric_length(self.prm_rewards, np.nan)
        self.step_counts = self._align_metric_length(self.step_counts, 0)
        self.learn_count = checkpoint["learn_count"]
        self.start_episode = checkpoint["next_episode"]
        self.env.current_problem = checkpoint.get("env_current_problem", self.env.current_problem)
        if self.env.current_problem >= self.env.num_problems:
            print(
                "Warning: checkpoint env_current_problem is outside the current "
                "problem-index set; wrapping it to fit the active index set. "
                "For a clean curve after changing problem-index settings, start "
                "a fresh run or use --reset_training_state true."
            )
            self.env.current_problem %= self.env.num_problems
        self.env.finished = checkpoint.get("env_finished", self.env.finished)

        torch.set_rng_state(self._cpu_byte_rng_state(checkpoint["torch_rng_state"]))
        if torch.cuda.is_available() and checkpoint.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all([
                self._cpu_byte_rng_state(state)
                for state in checkpoint["cuda_rng_state_all"]
            ])
        np.random.set_state(checkpoint["numpy_rng_state"])
        random.setstate(checkpoint["python_rng_state"])

        print(f"Resumed from {path}; next episode is {self.start_episode}.")

    def _state_to_named_dict(self, state):
        return {
            self.env.observation_space[idx]: self._json_safe(value)
            for idx, value in enumerate(state)
        }

    def _json_safe(self, value):
        if isinstance(value, np.generic):
            return self._json_safe(value.item())
        if isinstance(value, float) and not np.isfinite(value):
            return None
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {str(k): self._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._json_safe(v) for v in value]
        if isinstance(value, set):
            return sorted(self._json_safe(v) for v in value)
        return value

    @staticmethod
    def _finite_float(value, default=0.0):
        try:
            value = float(value)
        except Exception:
            return default
        return value if np.isfinite(value) else default

    @staticmethod
    def _format_text_block(text):
        if text is None:
            return ""
        text = str(text).strip()
        return text if text else "(empty)"

    def _summarize_llm_calls(self, llm_calls):
        summary = {
            "total_calls": len(llm_calls),
            "empty_responses": 0,
            "over_budget_calls": 0,
            "skipped_preflight_calls": 0,
            "estimated_prompt_tokens": 0,
            "reported_prompt_tokens": 0,
            "reported_completion_tokens": 0,
            "status_counts": {},
            "purpose_counts": {},
        }
        for call in llm_calls:
            status = call.get("status", "unknown")
            purpose = call.get("purpose", "unlabeled")
            usage = call.get("usage") or {}
            summary["status_counts"][status] = summary["status_counts"].get(status, 0) + 1
            summary["purpose_counts"][purpose] = summary["purpose_counts"].get(purpose, 0) + 1
            summary["estimated_prompt_tokens"] += call.get("estimated_prompt_tokens") or 0
            summary["reported_prompt_tokens"] += usage.get("prompt_tokens") or 0
            summary["reported_completion_tokens"] += usage.get("completion_tokens") or 0
            if not (call.get("response") or "").strip():
                summary["empty_responses"] += 1
            if call.get("over_budget"):
                summary["over_budget_calls"] += 1
            if status == "skipped_preflight":
                summary["skipped_preflight_calls"] += 1
        return summary

    def _write_json_atomic(self, path, data):
        tmp_path = path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)

    def _update_generation_index(self, log_dir, entry):
        index_path = os.path.join(log_dir, "index.jsonl")
        entries = {}
        if os.path.exists(index_path):
            with open(index_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        old_entry = json.loads(line)
                        entries[int(old_entry["episode"])] = old_entry
                    except Exception:
                        continue
        entries[int(entry["episode"])] = entry
        tmp_path = index_path + ".tmp"
        with open(tmp_path, "w") as f:
            for episode in sorted(entries):
                f.write(json.dumps(entries[episode], sort_keys=True) + "\n")
        os.replace(tmp_path, index_path)

    def _episode_snapshot(self, episode_count, episode_reward, step_records, final_reward=None, prm_reward=None):
        core = self.env.core
        thought_nodes = []
        for tid in sorted(core.thoughts):
            thought = core.thoughts[tid]
            thought_nodes.append(
                {
                    "tid": tid,
                    "depth": thought.get_depth(),
                    "parents": list(thought.get_parent_id()),
                    "children": list(thought.get_child_id()),
                    "score": self._json_safe(core.score.get(tid, {})),
                    "thought": thought.get_thought(),
                }
            )

        return {
            "episode": episode_count,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
            "dataset": self.dataset,
            "llm_name": self.LLM_name,
            "problem": self.env.problem,
            "expected_answer": self.env.ans,
            "final_answer": self._json_safe(core.last_answer),
            "final_answer_raw": core.last_answer_raw,
            "episode_reward": self._json_safe(episode_reward),
            "final_reward": self._json_safe(final_reward),
            "prm_reward": self._json_safe(prm_reward),
            "epsilon_after_episode": self._json_safe(self.epsilon),
            "num_steps": len(step_records),
            "actions": [
                {
                    "index": record["action"],
                    "name": record["action_name"],
                    "requested_index": record.get("requested_action"),
                    "requested_name": record.get("requested_action_name"),
                }
                for record in step_records
            ],
            "steps": step_records,
            "generation_trace": list(core.thought_each_step),
            "llm_calls": self._json_safe(core.LLM.request_history),
            "llm_summary": self._summarize_llm_calls(core.LLM.request_history),
            "leaf_nodes": sorted(core.leaf_nodes),
            "current_tid": core.current_tid,
            "thought_tree": thought_nodes,
        }

    def _write_episode_log(self, episode_count, episode_reward, step_records, final_reward=None, prm_reward=None):
        log_dir = os.path.join(self.save_folder, "episode_logs")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        snapshot = self._episode_snapshot(episode_count, episode_reward, step_records, final_reward, prm_reward)
        json_path = os.path.join(log_dir, f"episode_{episode_count:05d}.json")
        md_path = os.path.join(log_dir, f"episode_{episode_count:05d}.md")

        with open(json_path, "w") as f:
            json.dump(snapshot, f, indent=2)

        lines = [
            f"# Episode {episode_count}",
            "",
            "## Summary",
            f"- Dataset: {snapshot['dataset']}",
            f"- LLM: {snapshot['llm_name']}",
            f"- Steps: {snapshot['num_steps']}",
            f"- Episode reward: {snapshot['episode_reward']}",
            f"- Epsilon after episode: {snapshot['epsilon_after_episode']}",
            f"- Expected answer: {snapshot['expected_answer']}",
            f"- Parsed final answer: {snapshot['final_answer']}",
            f"- Final reward: {snapshot['final_reward']}",
            f"- PRM reward: {snapshot['prm_reward']}",
            "",
            "## Problem",
            "```text",
            self._format_text_block(snapshot["problem"]),
            "```",
            "",
        ]

        if snapshot["final_answer_raw"]:
            lines.extend(
                [
                    "## Raw Final Answer",
                    "```text",
                    self._format_text_block(snapshot["final_answer_raw"]),
                    "```",
                    "",
                ]
            )

        lines.extend(["## Step Trace", ""])
        for record in step_records:
            lines.extend(
                [
                    f"### Step {record['step']}: {record['action_name']} ({record['action']})",
                    f"- Done: {record['done']}",
                    f"- Reward: {record['reward']}",
                    f"- Reward source: {record['reward_source']}",
                    f"- Requested action: {record.get('requested_action_name')} ({record.get('requested_action')})",
                    f"- Done reason: {record.get('done_reason')}",
                    f"- PRM estimated tokens: {record.get('prm_estimated_tokens')}",
                    f"- State before: `{record['state_before']}`",
                    f"- State after: `{record['state_after']}`",
                    "",
                ]
            )

            if record.get("generated_text"):
                lines.extend(
                    [
                        "Generated content:",
                        "```text",
                        self._format_text_block(record["generated_text"]),
                        "```",
                        "",
                    ]
                )

        lines.extend(["## Full Generation Trace", ""])
        if snapshot["generation_trace"]:
            for idx, trace in enumerate(snapshot["generation_trace"], 1):
                lines.extend(
                    [
                        f"### Generation {idx}",
                        "```text",
                        self._format_text_block(trace),
                        "```",
                        "",
                    ]
                )
        else:
            lines.extend(["(no generation trace recorded)", ""])

        lines.extend(["## LLM Calls", ""])
        if snapshot["llm_calls"]:
            for idx, call in enumerate(snapshot["llm_calls"], 1):
                lines.extend(
                    [
                        f"### LLM Call {idx}",
                        f"- Purpose: {call.get('purpose', 'unlabeled')}",
                        f"- Status: {call.get('status', 'unknown')}",
                        f"- Estimated prompt tokens: {call.get('estimated_prompt_tokens')}",
                        f"- Max prompt tokens: {call.get('max_prompt_tokens')}",
                        f"- Over budget: {call.get('over_budget')}",
                        f"- Attempts: {call.get('attempts')}",
                        f"- Elapsed seconds: {call.get('elapsed_seconds')}",
                        f"- Usage: `{call.get('usage')}`",
                        f"- Metadata: `{call.get('metadata')}`",
                        "",
                        "Prompt:",
                        "```text",
                        self._format_text_block(call.get("prompt")),
                        "```",
                        "",
                        "Response:",
                        "```text",
                        self._format_text_block(call.get("response")),
                        "```",
                        "",
                    ]
                )
        else:
            lines.extend(["(no LLM calls recorded)", ""])

        lines.extend(["## Thought Tree", ""])
        for node in snapshot["thought_tree"]:
            lines.extend(
                [
                    f"### Node {node['tid']} depth={node['depth']}",
                    f"- Parents: {node['parents']}",
                    f"- Children: {node['children']}",
                    f"- Score: `{node['score']}`",
                    "```text",
                    self._format_text_block(node["thought"]),
                    "```",
                    "",
                ]
            )

        with open(md_path, "w") as f:
            f.write("\n".join(lines))

        return md_path, json_path

    def _write_llm_generation_log(self, episode_count, episode_reward, step_records, final_reward=None, prm_reward=None):
        log_dir = os.path.join(self.save_folder, "llm_generation_logs")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        episode = self._episode_snapshot(episode_count, episode_reward, step_records, final_reward, prm_reward)
        calls = episode["llm_calls"]
        calls_by_id = {call.get("call_id"): call for call in calls}
        steps = []
        for record in step_records:
            call_ids = list(record.get("llm_call_ids", []))
            steps.append(
                {
                    "step": record["step"],
                    "action": record["action"],
                    "action_name": record["action_name"],
                    "requested_action": record.get("requested_action"),
                    "requested_action_name": record.get("requested_action_name"),
                    "reward": record["reward"],
                    "reward_source": record["reward_source"],
                    "done_reason": record.get("done_reason"),
                    "done": record["done"],
                    "state_before": record["state_before"],
                    "state_after": record["state_after"],
                    "generated_text": record.get("generated_text", ""),
                    "llm_call_ids": call_ids,
                    "llm_calls": [calls_by_id[call_id] for call_id in call_ids if call_id in calls_by_id],
                }
            )

        generation_log = {
            "episode": episode["episode"],
            "timestamp": episode["timestamp"],
            "dataset": episode["dataset"],
            "llm_name": episode["llm_name"],
            "problem": episode["problem"],
            "expected_answer": episode["expected_answer"],
            "final_answer": episode["final_answer"],
            "final_answer_raw": episode["final_answer_raw"],
            "episode_reward": episode["episode_reward"],
            "final_reward": episode["final_reward"],
            "prm_reward": episode["prm_reward"],
            "epsilon_after_episode": episode["epsilon_after_episode"],
            "summary": episode["llm_summary"],
            "steps": steps,
            "llm_calls": calls,
            "generation_trace": episode["generation_trace"],
            "thought_tree": episode["thought_tree"],
        }

        json_path = os.path.join(log_dir, f"episode_{episode_count:05d}.json")
        md_path = os.path.join(log_dir, f"episode_{episode_count:05d}.md")
        self._write_json_atomic(json_path, generation_log)

        summary = generation_log["summary"]
        lines = [
            f"# LLM Generation Log - Episode {episode_count}",
            "",
            "## Summary",
            f"- Dataset: {generation_log['dataset']}",
            f"- LLM: {generation_log['llm_name']}",
            f"- Calls: {summary['total_calls']}",
            f"- Empty responses: {summary['empty_responses']}",
            f"- Over-budget calls: {summary['over_budget_calls']}",
            f"- Preflight skips: {summary['skipped_preflight_calls']}",
            f"- Estimated prompt tokens: {summary['estimated_prompt_tokens']}",
            f"- Reported prompt tokens: {summary['reported_prompt_tokens']}",
            f"- Reported completion tokens: {summary['reported_completion_tokens']}",
            f"- Status counts: `{json.dumps(summary['status_counts'], sort_keys=True)}`",
            f"- Purpose counts: `{json.dumps(summary['purpose_counts'], sort_keys=True)}`",
            f"- Episode reward: {generation_log['episode_reward']}",
            f"- Final reward: {generation_log['final_reward']}",
            f"- PRM reward: {generation_log['prm_reward']}",
            f"- Expected answer: {generation_log['expected_answer']}",
            f"- Parsed final answer: {generation_log['final_answer']}",
            "",
            "## Problem",
            "```text",
            self._format_text_block(generation_log["problem"]),
            "```",
            "",
        ]

        for step in steps:
            lines.extend(
                [
                    f"## Step {step['step']}: {step['action_name']} ({step['action']})",
                    f"- Done: {step['done']}",
                    f"- Reward: {step['reward']}",
                    f"- Reward source: {step['reward_source']}",
                    f"- Requested action: {step.get('requested_action_name')} ({step.get('requested_action')})",
                    f"- Done reason: {step.get('done_reason')}",
                    f"- LLM call IDs: {step['llm_call_ids']}",
                    "",
                ]
            )
            if step.get("generated_text"):
                lines.extend(
                    [
                        "Generated thought text:",
                        "```text",
                        self._format_text_block(step["generated_text"]),
                        "```",
                        "",
                    ]
                )
            if not step["llm_calls"]:
                lines.extend(["(no LLM calls in this step)", ""])
                continue

            for call in step["llm_calls"]:
                usage = call.get("usage") or {}
                if call.get("max_prompt_tokens") is None:
                    budget_text = str(call.get("estimated_prompt_tokens"))
                else:
                    budget_text = f"{call.get('estimated_prompt_tokens')} / {call.get('max_prompt_tokens')}"
                lines.extend(
                    [
                        f"### Call {call.get('call_id')}: {call.get('purpose', 'unlabeled')}",
                        f"- Status: {call.get('status', 'unknown')}",
                        f"- Estimated prompt tokens: {budget_text}",
                        f"- Over budget: {call.get('over_budget')}",
                        f"- Prompt bytes: {call.get('prompt_bytes')}",
                        f"- Response characters: {len(call.get('response') or '')}",
                        f"- Usage: prompt={usage.get('prompt_tokens')} completion={usage.get('completion_tokens')}",
                        f"- Attempts: {call.get('attempts')}",
                        f"- Elapsed seconds: {call.get('elapsed_seconds')}",
                        f"- Metadata: `{json.dumps(call.get('metadata') or {}, sort_keys=True)}`",
                        "",
                    ]
                )
                if call.get("error"):
                    lines.extend([f"- Error: `{call.get('error')}`", ""])
                lines.extend(
                    [
                        "Prompt:",
                        "```text",
                        self._format_text_block(call.get("prompt")),
                        "```",
                        "",
                        "Response:",
                        "```text",
                        self._format_text_block(call.get("response")),
                        "```",
                        "",
                    ]
                )

        if generation_log["final_answer_raw"]:
            lines.extend(
                [
                    "## Raw Final Answer",
                    "```text",
                    self._format_text_block(generation_log["final_answer_raw"]),
                    "```",
                    "",
                ]
            )

        with open(md_path, "w") as f:
            f.write("\n".join(lines))

        index_entry = {
            "episode": episode_count,
            "timestamp": generation_log["timestamp"],
            "json_path": json_path,
            "md_path": md_path,
            "summary": summary,
            "episode_reward": generation_log["episode_reward"],
            "final_reward": generation_log["final_reward"],
            "prm_reward": generation_log["prm_reward"],
            "expected_answer": generation_log["expected_answer"],
            "final_answer": generation_log["final_answer"],
        }
        self._update_generation_index(log_dir, index_entry)
        return md_path, json_path
    
    def train(self):
        if self.start_episode >= self.num_episodes:
            self._tqdm_write(
                f"Checkpoint already completed {self.start_episode} episode(s); "
                f"--num_episodes is {self.num_episodes}."
            )
            return

        for episode_count in tqdm(range(self.start_episode, self.num_episodes)):
            print('\n\n\n')
            print('++++++++++++++++++++++++++++++++')
            print(f'++++++++Episode {episode_count}++++++++')
            print('++++++++++++++++++++++++++++++++')
            state, _ = self.env.reset()

            episode_problem = (self.env.problem, self.env.ans)
            episode_states = list()
            episode_actions = list()
            episode_rewards = list()
            step_records = list()

            step_count = 0
            start_flag = True
            learned_this_episode = False

            while True:
                episode_states.append(state)

                step_count += 1
                self.epsilon = max(self.min_epsilon,self.epsilon*self.epsilon_decay)

                action = self.choose_action(state)
                if start_flag and action == 3:
                    action = 0
                start_flag = False
                requested_action = action
                requested_action_name = self.env.action_space[requested_action]
                trace_start = len(self.env.core.thought_each_step)
                call_start = len(self.env.core.LLM.request_history)
                
                state_next,reward_ORM,done = self.env.step(requested_action)
                reward_ORM = self._finite_float(reward_ORM)
                executed_action_name = self.env.core.last_executed_action or requested_action_name
                action = {name: idx for idx, name in self.env.action_space.items()}[executed_action_name]
                action_name = executed_action_name

                if done:
                    reward = reward_ORM
                    reward_source = "ORM/final_answer"
                    prm_estimated_tokens = None
                else:
                    thoughts = self.env.core.thought_each_step
                    input_for_prm = self.PRM.covert_to_input(self.env.problem,thoughts)
                    prm_message = None
                    try:
                        rewards,n_token = self.PRM.get_step_scores(input_for_prm)
                    except Exception as exc:
                        rewards = []
                        n_token = getattr(input_for_prm, "estimated_tokens", None)
                        prm_message = f"Warning: PRM scoring failed ({exc}); using reward 0.0."
                    if len(rewards) == 0:
                        reward = 0.0
                        if prm_message is None:
                            prm_message = (
                                "Warning: PRM returned no step scores "
                                f"for {len(thoughts)} thought(s), {n_token} estimated tokens; "
                                "using reward 0.0."
                            )
                    else:
                        reward = self._finite_float(rewards[-1])
                        prm_message = f"{n_token}"
                    reward_source = "PRM"
                    prm_estimated_tokens = n_token

                    self._tqdm_write(prm_message)

                reward = self._finite_float(reward)
                exp = Experience(state,action,reward,done,state_next)
                self.memory.append(exp)

                episode_rewards.append(reward)
                new_traces = self.env.core.thought_each_step[trace_start:]
                new_llm_calls = self.env.core.LLM.request_history[call_start:]
                step_records.append(
                    {
                        "step": step_count,
                        "action": action,
                        "action_name": action_name,
                        "requested_action": requested_action,
                        "requested_action_name": requested_action_name,
                        "done_reason": self.env.core.last_done_reason,
                        "state_before": self._state_to_named_dict(state),
                        "state_after": self._state_to_named_dict(state_next),
                        "reward": self._json_safe(reward),
                        "reward_source": reward_source,
                        "prm_estimated_tokens": prm_estimated_tokens,
                        "done": done,
                        "generated_text": "\n\n".join(new_traces),
                        "llm_call_count": len(new_llm_calls),
                        "llm_call_ids": [call.get("call_id") for call in new_llm_calls],
                    }
                )

                if len(self.memory) >= self.batch_size:
                    self.learn()
                    learned_this_episode = True
                
                if done:
                    episode_actions.append(action)

                    episode_reward = np.mean(episode_rewards)
                    final_reward = reward_ORM
                    prm_step_rewards = [
                        record["reward"]
                        for record in step_records
                        if record["reward_source"] == "PRM"
                    ]
                    prm_reward = np.mean(prm_step_rewards) if prm_step_rewards else np.nan
                    self.rewards.append(episode_reward)
                    self.final_rewards.append(final_reward)
                    self.prm_rewards.append(prm_reward)
                    self.step_counts.append(step_count)

                    reward_window = self.rewards[max(0, len(self.rewards)-100):]
                    self._tqdm_write(
                        f"Episode {episode_count} | Step {step_count}\t"
                        f"Reward {episode_reward}\tFinal {final_reward}\tPRM {prm_reward}\t"
                        f"AvgReward {np.mean(reward_window)}\t{len(self.memory)}"
                    )

                    with open(os.path.join(self.save_folder, f"record_episode{episode_count}.pkl"), "wb") as f:
                        pickle.dump((episode_problem, episode_states, episode_actions, episode_rewards), f)
                    self._save_reward_arrays()
                    if self.LLM_generation_log_interval > 0 and (episode_count + 1) % self.LLM_generation_log_interval == 0:
                        md_path, json_path = self._write_llm_generation_log(episode_count, episode_reward, step_records, final_reward, prm_reward)
                        self._tqdm_write(f"Saved LLM generation log: {md_path} and {json_path}")
                    if self.episode_log_interval > 0 and (episode_count + 1) % self.episode_log_interval == 0:
                        md_path, json_path = self._write_episode_log(episode_count, episode_reward, step_records, final_reward, prm_reward)
                        self._tqdm_write(f"Saved episode log: {md_path} and {json_path}")

                    break
            
                episode_actions.append(action)
                state = state_next

            if (episode_count + 1) % self.save_interval == 0:
                self.save(episode_count + 1)

            if learned_this_episode:
                self.lr_scheduler.step()

            self.save_training_checkpoint(episode_count + 1)

if __name__ == "__main__":
    import argparse

    def str2bool(value):
        if isinstance(value, bool):
            return value
        value = value.lower()
        if value in {"yes", "true", "t", "1", "y"}:
            return True
        if value in {"no", "false", "f", "0", "n"}:
            return False
        raise argparse.ArgumentTypeError("boolean value expected")

    parser = argparse.ArgumentParser()
    parser.add_argument('--manual_seed', type=int, default=1, help='manual seed for reproducibility')
    parser.add_argument('--dataset', type=str, default='GPQA', help='dataset to use')
    parser.add_argument('--LLM_name', type=str, default='Qwen/Qwen2.5-14B-Instruct', help='name of the LLM')
    parser.add_argument('--LLM_api_base_url', type=str, default=None, help='OpenAI-compatible LLM API base URL, e.g. http://localhost:8000/v1. Defaults to LLM_API_BASE_URL.')
    parser.add_argument('--LLM_api_key', type=str, default=None, help='API key for the OpenAI-compatible LLM API. Defaults to LLM_API_KEY.')
    parser.add_argument('--LLM_api_timeout', type=float, default=60.0, help='LLM API timeout in seconds')
    parser.add_argument('--LLM_max_tokens', type=int, default=int(os.environ.get('LLM_MAX_TOKENS', 4096)), help='maximum completion tokens requested from the LLM')
    parser.add_argument('--LLM_temperature', type=float, default=float(os.environ.get('LLM_TEMPERATURE', 1.0)), help='LLM sampling temperature')
    parser.add_argument('--LLM_top_p', type=float, default=float(os.environ.get('LLM_TOP_P', 0.7)), help='LLM nucleus sampling value')
    parser.add_argument('--LLM_top_k', type=int, default=int(os.environ.get('LLM_TOP_K', 1)), help='LLM top-k sampling value; 1 keeps local GGUF decoding greedy when supported')
    parser.add_argument('--LLM_send_top_k', type=str2bool, default=str2bool(os.environ.get('LLM_SEND_TOP_K', 'true')), help='include top_k in OpenAI-compatible requests; automatically disabled if the server rejects it')
    parser.add_argument('--LLM_max_prompt_tokens', type=int, default=int(os.environ.get('LLM_MAX_PROMPT_TOKENS', 65536)), help='estimated prompt-token budget before skipping an LLM request; set below the server context size')
    parser.add_argument('--LLM_prompt_token_chars_per_token', type=float, default=float(os.environ.get('LLM_PROMPT_TOKEN_CHARS_PER_TOKEN', 3.0)), help='chars-per-token estimate used for prompt-budget preflight')
    parser.add_argument('--LLM_skip_oversize_prompts', type=str2bool, default=str2bool(os.environ.get('LLM_SKIP_OVERSIZE_PROMPTS', 'false')), help='skip prompts estimated to exceed --LLM_max_prompt_tokens instead of sending them to the API server')
    parser.add_argument('--problem_indexs_name', type=str, default='problem_indexs', help='folder to load the problem indexs')
    parser.add_argument('--problem_indexs_path', type=str, default=None, help='optional explicit path to indexs.pkl')
    parser.add_argument('--problem_indexs_mode', type=str, default='file', choices=['file', 'auto', 'all'], help='file: require indexs.pkl; auto: use indexs.pkl if present otherwise all problems; all: ignore indexs.pkl')
    parser.add_argument('--problem_indexs_fallback_model', type=str, default='Qwen2.5-14B-Instruct', help='reference model hard-problem index to use when --problem_indexs_mode=auto and the current model has no index; set to "" to disable')
    parser.add_argument('--random_problems', type=str2bool, default=True, help='whether to use random problems')
    parser.add_argument('--debug_verbose', type=str2bool, default=False, help='print full LLM prompts/responses into thoughts_log.txt')
    parser.add_argument('--answer_fallback_policy', type=str, default='strict', choices=['strict', 'random'], help='strict: unparseable choice/boolean answers receive no-random reward 0; random: fall back to a random answer like the original implementation')
    parser.add_argument('--PRM_name', type=str, default='MATH-Shepherd-Mistral-7B-PRM', help='name of the PRM')
    parser.add_argument('--PRM_api_url', type=str, default=None, help='PRM API base URL. Defaults to PRM_API_URL or the hosted benchmark endpoint.')
    parser.add_argument('--PRM_timeout', type=float, default=300.0, help='PRM API timeout in seconds')
    parser.add_argument('--PRM_max_retries', type=int, default=3, help='number of PRM API retries per scoring request')
    parser.add_argument('--max_depth', type=int, default=5, help='maximum depth of the tree')
    parser.add_argument('--max_width', type=int, default=5, help='maximum width of the tree')
    parser.add_argument('--save_interval', type=int, default=10, help='interval to save the model')
    parser.add_argument('--episode_log_interval', type=int, default=0, help='interval for detailed per-episode JSON/Markdown logs; 0 disables them')
    parser.add_argument('--LLM_generation_log_interval', type=int, default=int(os.environ.get('LLM_GENERATION_LOG_INTERVAL', 1)), help='interval for per-episode LLM-generation JSON/Markdown logs in llm_generation_logs; 0 disables them')
    parser.add_argument('--save_dir', type=str, default='model', help='folder to save the model')
    parser.add_argument('--resume_from', type=str, default=None, help='existing Dueling_DDQN run folder to resume from checkpoint_latest.pt')
    parser.add_argument('--reset_training_state', type=str2bool, default=False, help='clear checkpoints, rewards, episode logs, and saved episode records before starting from episode 0')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    parser.add_argument('--learning_rate_decay', type=float, default=0.5, help='decay rate of learning rate')
    parser.add_argument('--learning_rate_decay_interval', type=int, default=1000, help='interval to decay the learning rate')
    parser.add_argument('--gamma', type=float, default=0.9, help='discount factor')
    parser.add_argument('--num_episodes', type=int, default=3000, help='number of episodes to train')
    parser.add_argument('--batch_size', type=int, default=500, help='batch size for training')
    parser.add_argument('--start_epsilon', type=float, default=1.0, help='start epsilon for epsilon-greedy')
    parser.add_argument('--min_epsilon', type=float, default=0.0, help='minimum epsilon for epsilon-greedy')
    parser.add_argument('--epsilon_decay', type=float, default=0.9995, help='decay rate of epsilon')
    parser.add_argument('--buffer_size', type=int, default=500, help='size of the replay buffer')
    parser.add_argument('--target_update_interval', type=int, default=50, help='interval to update the target')
    args = parser.parse_args()

    agent = Dueling_DDQN_agent(**vars(args))
    agent.train()
