from llama_cpp import Llama
import numpy as np

model_path = "/AIClub_NAS/core_baotg/quanhm/models/math-shepherd-mistral-7b-prm.Q8_0.gguf"

llm = Llama(
    model_path=model_path,
    n_ctx=4096,
    n_gpu_layers=16,
    logits_all=True,
    verbose=False,

)

good_token = "+"
bad_token = "-"
step_tag = "ки"

candidate_tokens = llm.tokenize(f"{good_token} {bad_token}".encode(), add_bos=True)[1:]
step_tag_id = llm.tokenize(step_tag.encode(), add_bos=False)[-1]

print(candidate_tokens)  # should be close to [648, 387]
print(step_tag_id)       # should be close to 12902

def softmax2(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()

def score_prm(question, output):
    text = f"{question} {output}"
    tokens = llm.tokenize(text.encode("utf-8"), add_bos=True)

    llm.reset()
    llm.eval(tokens)

    # llm.scores[i] = logits after consuming tokens[i], i.e. prediction for next token
    scores = np.array(llm.scores)

    step_scores = []
    for i, tok in enumerate(tokens):
        if tok == step_tag_id:
            logits_pm = scores[i, candidate_tokens]
            prob_good = softmax2(logits_pm)[0]
            step_scores.append(float(prob_good))

    return step_scores