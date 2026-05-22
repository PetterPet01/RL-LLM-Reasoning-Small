try:
    import setproctitle
except ImportError:
    setproctitle = None
import os
os.chdir(os.path.split(os.path.realpath(__file__))[0])

import sys
sys.path.append('env')

import os
import pickle
import random
import shutil
import torch
import time
from collections import Counter
from tqdm import tqdm

from Dueling_DQN_net import Dueling_DQN
from env.RL_env import RLEnv

class test_Dueling_DDQN_agent():
    def __init__(self,
                 manual_seed,
                 test_dataset,
                 test_LLM_name,
                 test_LLM_api_base_url,
                 test_LLM_api_key,
                 test_LLM_api_timeout,
                 test_LLM_max_tokens,
                 test_LLM_temperature,
                 test_LLM_top_p,
                 test_LLM_top_k,
                 test_LLM_send_top_k,
                 test_LLM_max_prompt_tokens,
                 test_LLM_prompt_token_chars_per_token,
                 test_LLM_skip_oversize_prompts,
                 problem_indexs,
                 max_depth,
                 max_width,
                 train_dataset,
                 train_LLM_name,
                 model_dir,
                 model_name,
                 model_index,
                 save_dir,
                 self_consistency_trials,
                 ):

        print('Initializing...')

        # generate config
        config_data=locals()
        del config_data['self']
        time_data=time.strftime('%Y-%m-%d_%H-%M', time.localtime())
        config_data['code_dir']=os.path.split(os.path.realpath(__file__))[0]
        config_data['time']=time_data

        # random seed
        self.device='cuda' if torch.cuda.is_available() else 'cpu'

        self.test_dataset = test_dataset
        self.test_LLM_name = test_LLM_name
        self.test_LLM_api_base_url = test_LLM_api_base_url
        self.test_LLM_api_key = test_LLM_api_key
        self.test_LLM_api_timeout = test_LLM_api_timeout
        self.test_LLM_max_tokens = test_LLM_max_tokens
        self.test_LLM_temperature = test_LLM_temperature
        self.test_LLM_top_p = test_LLM_top_p
        self.test_LLM_top_k = test_LLM_top_k
        self.test_LLM_send_top_k = test_LLM_send_top_k
        self.test_LLM_max_prompt_tokens = test_LLM_max_prompt_tokens
        self.test_LLM_prompt_token_chars_per_token = test_LLM_prompt_token_chars_per_token
        self.test_LLM_skip_oversize_prompts = test_LLM_skip_oversize_prompts
        self.problem_indexs= problem_indexs
        self.problem_positions = {idx: pos for pos, idx in enumerate(problem_indexs)}
        self.max_depth = max_depth
        self.max_width = max_width
        self.random_seed = manual_seed
        self.self_consistency_trials = max(1, self_consistency_trials)
        self.env = RLEnv(dataset=self.test_dataset, is_test=True, LLM_name=self.test_LLM_name, problem_indexs=self.problem_indexs, max_depth=self.max_depth, max_width=self.max_width, random_problems=False, random_seed=self.random_seed, LLM_api_base_url=self.test_LLM_api_base_url, LLM_api_key=self.test_LLM_api_key, LLM_api_timeout=self.test_LLM_api_timeout, LLM_max_tokens=self.test_LLM_max_tokens, LLM_temperature=self.test_LLM_temperature, LLM_top_p=self.test_LLM_top_p, LLM_top_k=self.test_LLM_top_k, LLM_send_top_k=self.test_LLM_send_top_k, LLM_max_prompt_tokens=self.test_LLM_max_prompt_tokens, LLM_prompt_token_chars_per_token=self.test_LLM_prompt_token_chars_per_token, LLM_skip_oversize_prompts=self.test_LLM_skip_oversize_prompts)
        self.n_actions = len(self.env.action_space)
        self.n_features = len(self.env.observation_space)

        self.train_dataset = train_dataset
        self.train_LLM_name = train_LLM_name
        self.model_dir = model_dir
        self.model_name = model_name
        self.model_folder = os.path.join(self.model_dir, self.train_LLM_name.split("/")[-1], self.train_dataset, self.model_name)
        self.model_index = model_index
        
        self.Dueling_DQN = Dueling_DQN(input_size=self.n_features,output_size=self.n_actions).to(self.device).eval()
        self.Dueling_DQN.load_state_dict(torch.load(os.path.join(self.model_folder, f"model_episode{model_index}.pth")))

        self.save_dir = save_dir
        self.result_root = os.path.join(self.save_dir, 'train-'+self.train_dataset, 'train-'+self.train_LLM_name.split("/")[-1], 'test-'+self.test_dataset, 'test-'+self.test_LLM_name.split("/")[-1])
        run_prefix = f'{model_name}_index{model_index}_sc{self.self_consistency_trials}_'
        self.save_folder = os.path.join(self.result_root, run_prefix + 'time' + time_data)
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        current_run = os.path.basename(self.save_folder)
        ref_results = sorted(x for x in os.listdir(self.result_root) if x != current_run and x.startswith(run_prefix))
        if len(ref_results) != 0:
            self.ref_folder = os.path.join(self.result_root, ref_results[0])
        else:
            self.ref_folder = None

        print(self.test_LLM_name + ' on ' + self.test_dataset + ' initialized.')
    
    def choose_action(self,state):
        with torch.no_grad():
            state = torch.tensor(state,dtype=torch.float32).to(self.device)
            state = state.unsqueeze(0)
            actions_value = self.Dueling_DQN(state)
            action = torch.argmax(actions_value).item()

        return action

    def reset_problem(self, problem_index):
        self.env.current_problem = self.problem_positions[problem_index]
        return self.env.reset()

    def reset_current_problem(self):
        self.env.core.reset()
        if self.env.answer_type == "Choice":
            self.env.core.set_problem(self.env.problem, self.env.ans, self.env.choices)
        else:
            self.env.core.set_problem(self.env.problem, self.env.ans)

        observation = self.env.core.score[self.env.core.current_tid]
        return list(observation.values())

    def cached_record_is_usable(self, record_path):
        try:
            with open(record_path, "rb") as f:
                record = pickle.load(f)
        except Exception:
            return False

        if isinstance(record, dict):
            return len(record.get("trials", [])) >= self.self_consistency_trials

        return self.self_consistency_trials == 1 and isinstance(record, tuple)

    def copy_cached_result(self, problem_index):
        if self.ref_folder is None:
            return False

        record_name = f"record_problem{problem_index}.pkl"
        log_name = f"thoughts_log_problem{problem_index}.txt"
        src_record = os.path.join(self.ref_folder, record_name)
        src_log = os.path.join(self.ref_folder, log_name)

        if not os.path.exists(src_record) or not self.cached_record_is_usable(src_record):
            return False

        shutil.copy2(src_record, os.path.join(self.save_folder, record_name))
        if os.path.exists(src_log):
            shutil.copy2(src_log, os.path.join(self.save_folder, log_name))
        return True

    def answers_equivalent(self, answer_a, answer_b):
        try:
            return bool(self.env.core.equal(answer_a, answer_b))
        except Exception:
            return self.answer_key(answer_a) == self.answer_key(answer_b)

    def answer_key(self, answer):
        if self.env.answer_type == "Choice":
            return str(answer).strip().upper()
        if self.env.answer_type == "Boolean":
            return str(answer).strip().lower()
        if self.env.answer_type == "Numerical":
            try:
                return f"{float(answer):.12g}"
            except Exception:
                return str(answer).strip()
        return str(answer).strip()

    def majority_vote(self, answers, problem_index):
        groups = []
        for answer in answers:
            for group in groups:
                if self.answers_equivalent(answer, group["representative"]):
                    group["answers"].append(answer)
                    break
            else:
                groups.append({"representative": answer, "answers": [answer]})

        max_count = max(len(group["answers"]) for group in groups)
        best_groups = [group for group in groups if len(group["answers"]) == max_count]
        if len(best_groups) == 1:
            selected_group = best_groups[0]
        else:
            selected_group = random.Random(self.random_seed + int(problem_index)).choice(best_groups)

        answer_counts = Counter()
        for group in groups:
            answer_counts[self.answer_key(group["representative"])] = len(group["answers"])

        return selected_group["representative"], dict(answer_counts)

    def run_rollout(self, state):
        episode_states = list()
        episode_actions = list()
        start_flag = True

        while True:
            episode_states.append(state)

            action = self.choose_action(state)
            if start_flag and action == 3:
                action = 0
            start_flag = False

            state_next,reward_ORM,done = self.env.step(action)
            executed_action_name = self.env.core.last_executed_action or self.env.action_space[action]
            action = {name: idx for idx, name in self.env.action_space.items()}[executed_action_name]
            
            if done:
                episode_actions.append(action)

                q_token, a_token = self.env.core.LLM.get_token()

                return {
                    "states": episode_states,
                    "actions": episode_actions,
                    "answer": self.env.core.last_answer,
                    "reward_ORM": float(reward_ORM),
                    "q_token": q_token,
                    "a_token": a_token,
                }

            episode_actions.append(action)
            state = state_next
    
    def test(self):
        correct_list = list()
        for problem_index in tqdm(self.problem_indexs):

            if self.copy_cached_result(problem_index):
                continue

            log_file = open(os.path.join(self.save_folder,f'thoughts_log_problem{problem_index}.txt'),'w')
            sys.stdout = log_file

            state, _ = self.reset_problem(problem_index)

            episode_problem = (self.env.problem, self.env.ans)
            trials = list()

            for trial_idx in range(self.self_consistency_trials):
                print("\n\n\n")
                print('++++++++++++++++++++++++++++++++')
                print(f'++++++++Problem {problem_index} Trial {trial_idx}++++++++')
                print('++++++++++++++++++++++++++++++++')

                trial_state = state if trial_idx == 0 else self.reset_current_problem()
                trials.append(self.run_rollout(trial_state))

            answers = [trial["answer"] for trial in trials]
            final_answer, answer_counts = self.majority_vote(answers, problem_index)
            reward_ORM = float(self.env.core.equal(final_answer, self.env.ans))

            if reward_ORM == 1:
                correct_list.append(problem_index)

            q_token = sum(trial["q_token"] for trial in trials)
            a_token = sum(trial["a_token"] for trial in trials)

            record = {
                "episode_problem": episode_problem,
                "trials": trials,
                "answers": answers,
                "answer_counts": answer_counts,
                "final_answer": final_answer,
                "reward_ORM": reward_ORM,
                "q_token": q_token,
                "a_token": a_token,
                "self_consistency_trials": self.self_consistency_trials,
            }

            sys.stdout.flush()
            with open(os.path.join(self.save_folder, f"record_problem{problem_index}.pkl"), "wb") as f:
                pickle.dump(record, f)
            sys.stdout.close()
            sys.stdout = sys.__stdout__

        return correct_list


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
    parser.add_argument('--manual_seed', type=int, default=0, help='random seed')
    parser.add_argument('--test_dataset', type=str, default='MATH', help='dataset to test')
    parser.add_argument('--test_LLM_name', type=str, default='Qwen/Qwen2.5-14B-Instruct', help='LLM to test')
    parser.add_argument('--test_LLM_api_base_url', type=str, default=None, help='OpenAI-compatible test LLM API base URL, e.g. http://localhost:8000/v1. Defaults to LLM_API_BASE_URL.')
    parser.add_argument('--test_LLM_api_key', type=str, default=None, help='API key for the OpenAI-compatible test LLM API. Defaults to LLM_API_KEY.')
    parser.add_argument('--test_LLM_api_timeout', type=float, default=60.0, help='test LLM API timeout in seconds')
    parser.add_argument('--test_LLM_max_tokens', type=int, default=int(os.environ.get('LLM_MAX_TOKENS', 4096)), help='maximum completion tokens requested from the test LLM')
    parser.add_argument('--test_LLM_temperature', type=float, default=float(os.environ.get('LLM_TEMPERATURE', 1.0)), help='test LLM sampling temperature')
    parser.add_argument('--test_LLM_top_p', type=float, default=float(os.environ.get('LLM_TOP_P', 0.7)), help='test LLM nucleus sampling value')
    parser.add_argument('--test_LLM_top_k', type=int, default=int(os.environ.get('LLM_TOP_K', 1)), help='test LLM top-k sampling value')
    parser.add_argument('--test_LLM_send_top_k', type=str2bool, default=str2bool(os.environ.get('LLM_SEND_TOP_K', 'true')), help='include top_k in OpenAI-compatible test requests')
    parser.add_argument('--test_LLM_max_prompt_tokens', type=int, default=int(os.environ.get('LLM_MAX_PROMPT_TOKENS', 32768)), help='estimated prompt-token budget before skipping a test LLM request')
    parser.add_argument('--test_LLM_prompt_token_chars_per_token', type=float, default=float(os.environ.get('LLM_PROMPT_TOKEN_CHARS_PER_TOKEN', 3.0)), help='chars-per-token estimate used for prompt-budget preflight')
    parser.add_argument('--test_LLM_skip_oversize_prompts', type=str2bool, default=str2bool(os.environ.get('LLM_SKIP_OVERSIZE_PROMPTS', 'false')), help='skip prompts estimated to exceed --test_LLM_max_prompt_tokens')
    parser.add_argument('--max_depth', type=int, default=5, help='maximum depth of the tree')
    parser.add_argument('--max_width', type=int, default=5, help='maximum width of the tree')
    parser.add_argument('--train_dataset', type=str, default='MATH', help='dataset to train')
    parser.add_argument('--train_LLM_name', type=str, default='Qwen/Qwen2.5-14B-Instruct', help='LLM to train')
    parser.add_argument('--model_dir', type=str, default='model', help='folder to load the model')
    parser.add_argument('--model_name', type=str, default='Dueling_DDQN_2025-01-17_02-48-13', help='name of the model')
    parser.add_argument('--model_index', type=int, default=3000, help='index of the model')
    parser.add_argument('--save_dir', type=str, default='test', help='folder to save the test results')
    parser.add_argument('--self_consistency_trials', type=int, default=3, help='number of repeated rollouts per problem for majority-vote self-consistency')
    parser.add_argument('--max_threads', type=int, default=10, help='maximum number of parallel evaluation workers')
    args = parser.parse_args()


    if args.test_dataset == 'GSM8K':
        problem_indexs = list(range(1319))
    elif args.test_dataset == 'MATH':
        problem_indexs = list(range(5000))
    elif args.test_dataset == 'GPQA':
        problem_indexs = list(range(448))
    elif args.test_dataset == 'MMLU-STEM':
        problem_indexs = list(range(3153))
    elif args.test_dataset == 'StrategyQA':
        problem_indexs = list(range(687))


    # agent = test_Dueling_DDQN_agent(manual_seed=args.manual_seed, test_dataset=args.test_dataset, test_LLM_name=args.test_LLM_name, test_LLM_api_base_url=args.test_LLM_api_base_url, test_LLM_api_key=args.test_LLM_api_key, test_LLM_api_timeout=args.test_LLM_api_timeout, problem_indexs=problem_indexs, max_depth=args.max_depth, max_width=args.max_width, train_dataset=args.train_dataset, train_LLM_name=args.train_LLM_name, model_dir=args.model_dir, model_name=args.model_name, model_index=args.model_index, save_dir=args.save_dir)
    # correct_list = agent.test()
    # print(correct_list)


    import multiprocessing
    import numpy as np

    def test_partition(problem_index_partition):
        agent = test_Dueling_DDQN_agent(manual_seed=args.manual_seed, test_dataset=args.test_dataset, test_LLM_name=args.test_LLM_name, test_LLM_api_base_url=args.test_LLM_api_base_url, test_LLM_api_key=args.test_LLM_api_key, test_LLM_api_timeout=args.test_LLM_api_timeout, test_LLM_max_tokens=args.test_LLM_max_tokens, test_LLM_temperature=args.test_LLM_temperature, test_LLM_top_p=args.test_LLM_top_p, test_LLM_top_k=args.test_LLM_top_k, test_LLM_send_top_k=args.test_LLM_send_top_k, test_LLM_max_prompt_tokens=args.test_LLM_max_prompt_tokens, test_LLM_prompt_token_chars_per_token=args.test_LLM_prompt_token_chars_per_token, test_LLM_skip_oversize_prompts=args.test_LLM_skip_oversize_prompts, problem_indexs=problem_index_partition, max_depth=args.max_depth, max_width=args.max_width, train_dataset=args.train_dataset, train_LLM_name=args.train_LLM_name, model_dir=args.model_dir, model_name=args.model_name, model_index=args.model_index, save_dir=args.save_dir, self_consistency_trials=args.self_consistency_trials)
        agent.test()

        return True
    
    MAX_THREADS=max(1, min(args.max_threads, len(problem_indexs)))
    p = multiprocessing.Pool(MAX_THREADS)
    result=list()

    np.random.shuffle(problem_indexs)
    partition_size = len(problem_indexs)//MAX_THREADS
    for i in range(MAX_THREADS):
        if i == MAX_THREADS-1:
            result.append(p.apply_async(test_partition,args=(sorted(problem_indexs[i*partition_size:]),)))
        else:
            result.append(p.apply_async(test_partition,args=(sorted(problem_indexs[i*partition_size:(i+1)*partition_size]),)))

    for obj in tqdm(result):
        obj.get()
    
    p.close()
