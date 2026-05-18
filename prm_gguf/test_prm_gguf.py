from llama_cpp import Llama
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


# MODEL_PATH = "/AIClub_NAS/core_baotg/quanhm/models/math-shepherd-mistral-7b-prm.Q8_0.gguf"
MODEL_PATH = "/AIClub_NAS/core_baotg/quanhm/models/math-shepherd-mistral-7b-prm.Q4_K_M.gguf"


GOOD_TOKEN = "+"
BAD_TOKEN = "-"
STEP_TAG = "ки"


@dataclass
class PRMCase:
    name: str
    question: str
    good_output: str
    bad_output: str
    min_good_final_score: float = 0.50
    max_bad_final_score: float = 0.50


def softmax2(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def load_prm(model_path: str) -> Llama:
    return Llama(
        model_path=model_path,
        n_ctx=4096,
        n_gpu_layers=32,
        logits_all=True,
        verbose=False,
    )


def get_token_ids(llm: Llama) -> Tuple[List[int], int]:
    candidate_tokens = llm.tokenize(
        f"{GOOD_TOKEN} {BAD_TOKEN}".encode("utf-8"),
        add_bos=True,
    )[1:]

    step_tag_id = llm.tokenize(
        STEP_TAG.encode("utf-8"),
        add_bos=False,
    )[-1]

    return candidate_tokens, step_tag_id


def score_prm(
    llm: Llama,
    question: str,
    output: str,
    candidate_tokens: List[int],
    step_tag_id: int,
    verbose: bool = False,
) -> List[float]:
    input_for_prm = f"{question} {output}"
    tokens = llm.tokenize(input_for_prm.encode("utf-8"), add_bos=True)

    llm.reset()
    llm.eval(tokens)

    scores = np.asarray(llm.scores)

    if len(scores) < len(tokens):
        raise RuntimeError(
            f"llm.scores shorter than tokens: scores={len(scores)}, tokens={len(tokens)}. "
            f"Make sure logits_all=True."
        )

    step_scores = []

    for i, tok in enumerate(tokens):
        if tok == step_tag_id:
            logits_pm = scores[i, candidate_tokens]
            prob_good = softmax2(logits_pm)[0]
            step_scores.append(float(prob_good))

            if verbose:
                plus_logit = float(logits_pm[0])
                minus_logit = float(logits_pm[1])
                print(
                    f"  step_tag at token index {i}: "
                    f"+ logit={plus_logit:.4f}, "
                    f"- logit={minus_logit:.4f}, "
                    f"P(+)={prob_good:.6f}"
                )

    return step_scores


def count_step_tags(llm: Llama, text: str, step_tag_id: int) -> int:
    tokens = llm.tokenize(text.encode("utf-8"), add_bos=True)
    return sum(tok == step_tag_id for tok in tokens)


def run_case(
    llm: Llama,
    case: PRMCase,
    candidate_tokens: List[int],
    step_tag_id: int,
    verbose: bool = False,
) -> bool:
    good_scores = score_prm(
        llm=llm,
        question=case.question,
        output=case.good_output,
        candidate_tokens=candidate_tokens,
        step_tag_id=step_tag_id,
        verbose=verbose,
    )

    bad_scores = score_prm(
        llm=llm,
        question=case.question,
        output=case.bad_output,
        candidate_tokens=candidate_tokens,
        step_tag_id=step_tag_id,
        verbose=verbose,
    )

    expected_good_steps = count_step_tags(llm, case.good_output, step_tag_id)
    expected_bad_steps = count_step_tags(llm, case.bad_output, step_tag_id)

    good_final = good_scores[-1] if good_scores else None
    bad_final = bad_scores[-1] if bad_scores else None

    step_count_ok = (
        len(good_scores) == expected_good_steps
        and len(bad_scores) == expected_bad_steps
    )

    separation_ok = (
        good_final is not None
        and bad_final is not None
        and good_final >= case.min_good_final_score
        and bad_final <= case.max_bad_final_score
        and good_final > bad_final
    )

    passed = step_count_ok and separation_ok

    print("=" * 100)
    print(f"CASE: {case.name}")
    print("-" * 100)
    print(f"Good scores: {[round(x, 6) for x in good_scores]}")
    print(f"Bad scores:  {[round(x, 6) for x in bad_scores]}")
    print()
    print(f"Expected good steps: {expected_good_steps}, got: {len(good_scores)}")
    print(f"Expected bad steps:  {expected_bad_steps}, got: {len(bad_scores)}")
    print()
    print(f"Good final score: {good_final}")
    print(f"Bad final score:  {bad_final}")
    print()
    print(f"Step count OK:  {step_count_ok}")
    print(f"Separation OK:  {separation_ok}")
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")

    return passed


def build_test_cases() -> List[PRMCase]:
    janet_question = (
        "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning "
        "and bakes muffins for her friends every day with four. She sells the remainder "
        "at the farmers' market daily for $2 per fresh duck egg. How much in dollars "
        "does she make every day at the farmers' market?"
    )

    janet_good = (
        "Step 1: Janet's ducks lay 16 eggs per day. ки\n"
        "Step 2: She eats three for breakfast every morning, so she has 16 - 3 = 13 eggs left. ки\n"
        "Step 3: She bakes muffins for her friends every day with four eggs, so she has 13 - 4 = 9 eggs left. ки\n"
        "Step 4: She sells the remainder at the farmers' market daily for $2 per fresh duck egg, "
        "so she makes 9 * $2 = $18 every day at the farmers' market. The answer is: 18 ки"
    )

    janet_bad = (
        "Step 1: Janet's ducks lay 16 eggs per day. ки\n"
        "Step 2: She eats three for breakfast every morning, so she has 16 - 3 = 13 eggs left. ки\n"
        "Step 3: She bakes muffins for her friends every day with four eggs, so she has 13 - 4 = 9 eggs left. ки\n"
        "Step 4: She sells the remainder at the farmers' market daily for $2 per fresh duck egg, "
        "so she makes 9 * $2 = $17 every day at the farmers' market. The answer is: 17 ки"
    )

    apples_question = (
        "Tom has 5 apples. He buys 7 more apples and then gives 3 apples to his friend. "
        "How many apples does Tom have left?"
    )

    apples_good = (
        "Step 1: Tom starts with 5 apples. ки\n"
        "Step 2: He buys 7 more, so he has 5 + 7 = 12 apples. ки\n"
        "Step 3: He gives away 3 apples, so he has 12 - 3 = 9 apples left. The answer is: 9 ки"
    )

    apples_bad = (
        "Step 1: Tom starts with 5 apples. ки\n"
        "Step 2: He buys 7 more, so he has 5 + 7 = 12 apples. ки\n"
        "Step 3: He gives away 3 apples, so he has 12 - 3 = 10 apples left. The answer is: 10 ки"
    )

    speed_question = (
        "A car travels 180 miles in 3 hours at a constant speed. "
        "What is the car's speed in miles per hour?"
    )

    speed_good = (
        "Step 1: Speed equals distance divided by time. ки\n"
        "Step 2: The distance is 180 miles and the time is 3 hours. ки\n"
        "Step 3: 180 / 3 = 60, so the speed is 60 miles per hour. The answer is: 60 ки"
    )

    speed_bad = (
        "Step 1: Speed equals distance divided by time. ки\n"
        "Step 2: The distance is 180 miles and the time is 3 hours. ки\n"
        "Step 3: 180 / 3 = 90, so the speed is 90 miles per hour. The answer is: 90 ки"
    )

    rectangle_question = (
        "A rectangle has length 12 cm and width 5 cm. What is its area?"
    )

    rectangle_good = (
        "Step 1: The area of a rectangle is length times width. ки\n"
        "Step 2: The length is 12 cm and the width is 5 cm. ки\n"
        "Step 3: 12 * 5 = 60, so the area is 60 square centimeters. The answer is: 60 ки"
    )

    rectangle_bad = (
        "Step 1: The area of a rectangle is length times width. ки\n"
        "Step 2: The length is 12 cm and the width is 5 cm. ки\n"
        "Step 3: 12 * 5 = 17, so the area is 17 square centimeters. The answer is: 17 ки"
    )

    return [
        PRMCase(
            name="Math-Shepherd model-card Janet example",
            question=janet_question,
            good_output=janet_good,
            bad_output=janet_bad,
            min_good_final_score=0.50,
            max_bad_final_score=0.50,
        ),
        PRMCase(
            name="Apples arithmetic",
            question=apples_question,
            good_output=apples_good,
            bad_output=apples_bad,
            min_good_final_score=0.50,
            max_bad_final_score=0.50,
        ),
        PRMCase(
            name="Speed arithmetic",
            question=speed_question,
            good_output=speed_good,
            bad_output=speed_bad,
            min_good_final_score=0.50,
            max_bad_final_score=0.50,
        ),
        PRMCase(
            name="Rectangle area",
            question=rectangle_question,
            good_output=rectangle_good,
            bad_output=rectangle_bad,
            min_good_final_score=0.50,
            max_bad_final_score=0.50,
        ),
    ]


def main():
    llm = load_prm(MODEL_PATH)

    candidate_tokens, step_tag_id = get_token_ids(llm)

    print("Candidate tokens:", candidate_tokens)
    print("Step tag id:", step_tag_id)
    print()

    expected_candidate_tokens = [648, 387]
    expected_step_tag_id = 12902

    tokenizer_ok = (
        candidate_tokens == expected_candidate_tokens
        and step_tag_id == expected_step_tag_id
    )

    print(f"Expected candidate tokens: {expected_candidate_tokens}")
    print(f"Expected step tag id: {expected_step_tag_id}")
    print(f"Tokenizer check: {'PASS' if tokenizer_ok else 'FAIL'}")
    print()

    if not tokenizer_ok:
        print("WARNING: Token IDs do not match the original Hugging Face tokenizer.")
        print("The test may still run, but scores may not be comparable.")
        print()

    cases = build_test_cases()

    results = []
    for case in cases:
        passed = run_case(
            llm=llm,
            case=case,
            candidate_tokens=candidate_tokens,
            step_tag_id=step_tag_id,
            verbose=False,
        )
        results.append(passed)

    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"Passed {sum(results)} / {len(results)} cases")

    if all(results):
        print("Overall result: PASS")
    else:
        print("Overall result: FAIL")


if __name__ == "__main__":
    main()