import os
import requests
import time
import httpx
import math
from API_key import api_key_dict

class OverloadError(Exception):
    pass

class ContextWindowError(Exception):
    pass

class LLM_api():
    def __init__(self, 
                 content = "", 
                 model = "Qwen/Qwen2.5-7B-Instruct", 
                 max_tokens = 4096,
                 stop = None,
                 temperature = 1,
                 top_p = 0.7,
                 top_k = 1,
                 frequency_penalty = 0,
                 n = 1,
                 key_idx = 0,
                 api_base_url = None,
                 api_key = None,
                 timeout = 60,
                 max_prompt_tokens = None,
                 prompt_token_chars_per_token = 3.0,
                 skip_oversize_prompts = False,
                 send_top_k = None,
            ):

        self.payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ],
            "stream": False,
            "max_tokens": max_tokens,
            "stop": stop,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "frequency_penalty": frequency_penalty,
            "n": 1,
            "response_format": {"type": "text"},
            "tools": []
        }
        self.api_base_url = api_base_url or os.environ.get("LLM_API_BASE_URL")
        self.api_key = api_key or os.environ.get("LLM_API_KEY")
        self.timeout = timeout
        self.max_prompt_tokens = self._env_int("LLM_MAX_PROMPT_TOKENS", max_prompt_tokens)
        self.prompt_token_chars_per_token = float(
            os.environ.get("LLM_PROMPT_TOKEN_CHARS_PER_TOKEN", prompt_token_chars_per_token)
        )
        self.skip_oversize_prompts = self._env_bool("LLM_SKIP_OVERSIZE_PROMPTS", skip_oversize_prompts)
        if send_top_k is None:
            send_top_k = True
        self.send_top_k = self._env_bool("LLM_SEND_TOP_K", send_top_k)
        
        if self.api_base_url:
            self.backend = "openai_compatible"
            self.url = self._chat_completions_url(self.api_base_url)
            self.headers = {"Content-Type": "application/json"}
            if self.api_key:
                self.headers["Authorization"] = f"Bearer {self.api_key}"

        elif model != "gpt-4o-mini":
            self.backend = "silicon_flow"
            self.url = "https://api.siliconflow.cn/v1/chat/completions"
            keys = api_key_dict["silicon_flow_keys"]
            self.headers = {
                "Authorization": f"Bearer {keys[key_idx]}",
                "Content-Type": "application/json"
            }
            
        elif model == "gpt-4o-mini":
            from openai import AzureOpenAI

            self.backend = "azure_openai"
            if self.payload["temperature"] != 0:
                print("Warning: temperature is not 0, set to 0 if greedy decode")
                self.payload["temperature"] = 0
                ### OpenAI API donot have a top_k parameter, so we need to set temperature to 0
            self.HTTP_CLIENT = httpx.Client(proxy="http://127.0.0.1:8456")
            self.client = AzureOpenAI(
                api_key = api_key_dict['azure_key'],
                api_version = "2024-07-01-preview",
                azure_endpoint = api_key_dict['azure_endpoint'],
                http_client = self.HTTP_CLIENT,
                azure_deployment="gpt-4o-mini-2"
            )

        self.q_token = 0
        self.a_token = 0
        self.request_history = []
        self.request_count = 0
        self.last_request_info = {}

    @staticmethod
    def _env_int(name, value):
        if value is not None:
            return int(value)
        env_value = os.environ.get(name)
        if env_value in (None, ""):
            return None
        return int(env_value)

    @staticmethod
    def _env_bool(name, value):
        env_value = os.environ.get(name)
        if env_value is None:
            return bool(value)
        return env_value.lower() not in {"0", "false", "no", "off"}

    @staticmethod
    def _chat_completions_url(api_base_url):
        api_base_url = api_base_url.rstrip("/")
        if api_base_url.endswith("/chat/completions"):
            return api_base_url
        return api_base_url + "/chat/completions"

    def _estimate_prompt_tokens(self, content):
        chars_per_token = max(self.prompt_token_chars_per_token, 0.1)
        # Include a little chat-template overhead so near-limit prompts are
        # rejected before the server has to do it.
        return max(1, int(math.ceil(len(content.encode("utf-8")) / chars_per_token)) + 64)

    def _empty_response(self, prompt_tokens=0):
        return {
            "choices": [{"message": {"content": ""}}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 0},
        }

    @staticmethod
    def _error_text(error):
        message = str(error).lower()
        response = getattr(error, "response", None)
        if response is not None:
            try:
                message += " " + response.text.lower()
            except Exception:
                pass
        return message

    @staticmethod
    def _is_context_window_error(error):
        message = LLM_api._error_text(error)

        context_markers = [
            "exceeds the available context size",
            "context size",
            "context window",
            "maximum context",
            "prompt is too long",
            "too many tokens",
        ]
        return any(marker in message for marker in context_markers)

    def _prompt_preflight_info(self, prompt=None):
        if prompt is None:
            prompt = self.payload["messages"][0]["content"]
        estimated_tokens = self._estimate_prompt_tokens(prompt)
        over_budget = (
            self.max_prompt_tokens is not None
            and estimated_tokens > self.max_prompt_tokens
        )
        return {
            "estimated_prompt_tokens": estimated_tokens,
            "max_prompt_tokens": self.max_prompt_tokens,
            "prompt_token_chars_per_token": self.prompt_token_chars_per_token,
            "skip_oversize_prompts": self.skip_oversize_prompts,
            "over_budget": over_budget,
        }

    def _preflight_prompt(self):
        info = self._prompt_preflight_info()
        if (
            self.skip_oversize_prompts
            and info["over_budget"]
        ):
            raise ContextWindowError(
                f"estimated prompt tokens {info['estimated_prompt_tokens']} exceeds "
                f"LLM_MAX_PROMPT_TOKENS={self.max_prompt_tokens}"
            )
        return info

    def _openai_compatible_payload(self):
        payload = {
            "model": self.payload["model"],
            "messages": self.payload["messages"],
            "stream": False,
            "max_tokens": self.payload["max_tokens"],
            "temperature": self.payload["temperature"],
            "top_p": self.payload["top_p"],
            "frequency_penalty": self.payload["frequency_penalty"],
            "n": self.payload["n"],
        }
        if self.send_top_k and self.payload.get("top_k") is not None:
            payload["top_k"] = self.payload["top_k"]
        if self.payload["stop"] is not None:
            payload["stop"] = self.payload["stop"]
        return payload

    def reset_token(self):
        self.q_token = 0
        self.a_token = 0
        self.request_history = []
        self.request_count = 0
        self.last_request_info = {}

    def get_token(self):
        return self.q_token, self.a_token

    def get_response(self):
        try:
            request_info = self._preflight_prompt()
        except ContextWindowError as e:
            request_info = self._prompt_preflight_info()
            estimated_prompt_tokens = request_info["estimated_prompt_tokens"]
            request_info.update(
                {
                    "status": "skipped_preflight",
                    "attempts": 0,
                    "elapsed_seconds": 0.0,
                    "error": str(e),
                }
            )
            print(f"Warning: {e}; skipping LLM request and returning empty response")
            self.q_token += estimated_prompt_tokens
            response = self._empty_response(prompt_tokens=estimated_prompt_tokens)
            request_info["usage"] = response.get("usage") or {}
            self.last_request_info = request_info
            return response

        estimated_prompt_tokens = request_info["estimated_prompt_tokens"]
        if request_info["over_budget"]:
            print(
                "Warning: estimated prompt tokens "
                f"{estimated_prompt_tokens} exceeds "
                f"LLM_MAX_PROMPT_TOKENS={self.max_prompt_tokens}; "
                "sending request because LLM_skip_oversize_prompts=False"
            )

        t0 = time.time()
        max_try = 10
        try_num = 0
        max_sleep = 20
        sleep_time = 1
        request_info.update(
            {
                "status": "started",
                "attempts": 0,
                "elapsed_seconds": None,
                "error": None,
            }
        )
        while True:
            try_num += 1
            request_info["attempts"] = try_num
            if self.backend == "azure_openai":
                try:
                    response = self.client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "user", "content": self.payload["messages"][0]["content"]}
                        ],
                        max_tokens=self.payload["max_tokens"],
                        stop=self.payload["stop"],
                        temperature=self.payload["temperature"],
                        top_p=self.payload["top_p"],
                        frequency_penalty=self.payload["frequency_penalty"],
                        n=self.payload["n"],
                        timeout=self.timeout,
                    )
                    content = response.choices[0].message.content or ""
                    if response.usage is not None:
                        self.q_token += response.usage.prompt_tokens
                        self.a_token += response.usage.completion_tokens
                    if len(content) != 0:
                        response = {"choices": [{"message": {"content": content}}]}
                        request_info["status"] = "ok"
                        break
                except Exception as e:
                    if self._is_context_window_error(e):
                        print(f"Warning: LLM context window error: {e}; returning empty response")
                        self.q_token += estimated_prompt_tokens
                        response = self._empty_response(prompt_tokens=estimated_prompt_tokens)
                        request_info["status"] = "context_window_error"
                        request_info["error"] = str(e)
                        break
                    try:
                        if "content management policy" in str(e):
                            print("Warning: Blocked by content management policy")
                            response = {"choices": [{"message": {"content": ""}}]}
                            request_info["status"] = "blocked_by_policy"
                            request_info["error"] = str(e)
                            break
                    except:
                        pass
                    request_info["error"] = str(e)
                    print("try_num: ", try_num, "error", e)
                    time.sleep(sleep_time)
                    sleep_time = min(sleep_time * 2, max_sleep)
                
                if try_num >= max_try:
                    print("Warning: TOO MANY TRIES, EMPTY RESPONSE")
                    response = {"choices": [{"message": {"content": ""}}]}
                    request_info["status"] = "empty_after_retries"
                    break
            
            elif self.backend == "openai_compatible":
                try:
                    response = requests.request(
                        "POST",
                        self.url,
                        json=self._openai_compatible_payload(),
                        headers=self.headers,
                        timeout=self.timeout,
                    )
                    response.raise_for_status()
                    response = response.json()
                    content = response["choices"][0]["message"].get("content") or ""
                    usage = response.get("usage") or {}
                    self.q_token += usage.get("prompt_tokens", 0)
                    self.a_token += usage.get("completion_tokens", 0)
                    if len(content) != 0:
                        request_info["status"] = "ok"
                        break
                except Exception as e:
                    if self._is_context_window_error(e):
                        print(f"Warning: LLM context window error: {e}; returning empty response")
                        self.q_token += estimated_prompt_tokens
                        response = self._empty_response(prompt_tokens=estimated_prompt_tokens)
                        request_info["status"] = "context_window_error"
                        request_info["error"] = str(e)
                        break
                    if self.send_top_k and "top_k" in self._error_text(e):
                        print("Warning: server rejected top_k; retrying OpenAI-compatible request without top_k")
                        self.send_top_k = False
                        request_info["error"] = str(e)
                        request_info["top_k_disabled_after_error"] = True
                        continue
                    request_info["error"] = str(e)
                    print("try_num: ", try_num, "error", e)
                    time.sleep(sleep_time)
                    sleep_time = min(sleep_time * 2, max_sleep)

                if try_num >= max_try:
                    print("Warning: TOO MANY TRIES, EMPTY RESPONSE")
                    response = {"choices": [{"message": {"content": ""}}]}
                    request_info["status"] = "empty_after_retries"
                    break

            else:
                try:
                    response = requests.request("POST", self.url, json=self.payload, headers=self.headers, timeout=self.timeout)
                    response = response.json()
                    content = response["choices"][0]["message"]["content"] or ""
                    usage = response.get("usage") or {}
                    self.q_token += usage.get("prompt_tokens", 0)
                    self.a_token += usage.get("completion_tokens", 0)
                    if len(content) != 0:
                        request_info["status"] = "ok"
                        break
                except Exception as e:
                    if self._is_context_window_error(e):
                        print(f"Warning: LLM context window error: {e}; returning empty response")
                        self.q_token += estimated_prompt_tokens
                        response = self._empty_response(prompt_tokens=estimated_prompt_tokens)
                        request_info["status"] = "context_window_error"
                        request_info["error"] = str(e)
                        break
                    request_info["error"] = str(e)
                    # print("try_num: ", try_num, "error", e)
                    time.sleep(sleep_time)
                    sleep_time = min(sleep_time * 2, max_sleep)

                if try_num >= max_try:
                    print("Warning: TOO MANY TRIES, EMPTY RESPONSE")
                    response = {"choices": [{"message": {"content": ""}}]}
                    request_info["status"] = "empty_after_retries"
                    break

        request_info["elapsed_seconds"] = round(time.time() - t0, 3)
        request_info["usage"] = response.get("usage") or {}
        if request_info["status"] == "started":
            content = response["choices"][0]["message"].get("content") or ""
            request_info["status"] = "ok" if content else "empty_response"
        self.last_request_info = request_info
        return response
    
    def get_text(self, content = "", purpose=None, metadata=None):
        self.set_content(content)
        response = self.get_response()
        text = response['choices'][0]['message']['content']
        request_info = dict(self.last_request_info)
        usage = request_info.get("usage") or response.get("usage") or {}
        self.request_count += 1
        self.request_history.append(
            {
                "call_id": self.request_count,
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                "model": self.payload["model"],
                "backend": self.backend,
                "purpose": purpose or "unlabeled",
                "metadata": metadata or {},
                "prompt": content,
                "response": text,
                "prompt_characters": len(content),
                "prompt_bytes": len(content.encode("utf-8")),
                "estimated_prompt_tokens": request_info.get("estimated_prompt_tokens", self._estimate_prompt_tokens(content)),
                "max_prompt_tokens": request_info.get("max_prompt_tokens", self.max_prompt_tokens),
                "prompt_token_chars_per_token": request_info.get("prompt_token_chars_per_token", self.prompt_token_chars_per_token),
                "over_budget": request_info.get("over_budget", False),
                "skip_oversize_prompts": request_info.get("skip_oversize_prompts", self.skip_oversize_prompts),
                "send_top_k": self.send_top_k,
                "top_k": self.payload.get("top_k"),
                "status": request_info.get("status", "unknown"),
                "attempts": request_info.get("attempts"),
                "elapsed_seconds": request_info.get("elapsed_seconds"),
                "error": request_info.get("error"),
                "usage": usage,
            }
        )
        return text
    
    def extract_state(self, res_string):
        ## a post process for extracting the score. there are 8 of them,
        state_key = ["A1", "A2", "A3", "B1", "B2", "B3", "B4", "C1"]
        state = {}
        for key in state_key:
            location = res_string.find(key)
            if location != -1 and location + 4 < len(res_string):
                char = res_string[location + 4]
                state[key] = char
                if char in ["1", "2", "3", "4", "5"]:
                    state[key] = int(char)
                else:
                    state[key] = -1
            else:
                state[key] = -1
        return state
    
    def set_content(self, content):
        self.payload["messages"][0]["content"] = content

    def print_usage(self):
        print(f"Prompt tokens: {self.q_token}")
        print(f"Completion tokens: {self.a_token}")
    

if __name__ == "__main__":
    llm = LLM_api(model="meta-llama/Meta-Llama-3.1-70B-Instruct", key_idx=6)
    text = llm.get_text(content="123, 321")
    print(text)
