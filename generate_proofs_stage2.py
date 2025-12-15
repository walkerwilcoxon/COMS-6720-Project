import argparse
from pathlib import Path
import multiprocessing as mp
import os

import torch
import tomli_w
import tomli

from llm_utils import load_model, extract_problems, ParallelExecutor, verify_proof
from proof_strategies import (
    produce_proof_baseline,
    produce_proof_with_feedback,
    produce_proof_with_guidance,
    produce_guidance_commander,
)


def llm_worker(gpu_id, input_q, output_q, worker_args):
    print(f"Starting LLM worker {gpu_id}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    model, tokenizer = load_model(worker_args.model_id)

    c_model, c_tokenizer = None, None
    if worker_args.proof_strategy == "commander":
        print(f"Worker {gpu_id}: Loading DeepSeek Commander...")
        c_model, c_tokenizer = load_model("deepseek-ai/DeepSeek-Prover-V2-7B")

    while True:
        input_data = input_q.get()
        if input_data is None:
            break

        i, name, problem, prev_entry = input_data
        print(f"Retrying proof {name} ({i})")

        
        # Prepare feedback dict (common for both feedback and commander strategies)
        if worker_args.proof_strategy == "feedback" or worker_args.proof_strategy == "commander":
            errors_str = prev_entry.get("error", "")
            feedback_list = prev_entry.get("feedback", [])
            error_msgs = [f"{e.get('line', '?')}:{e.get('column', '?')}: {e.get('message', 'Unknown error')}" for e in feedback_list]
            
            if error_msgs:
                if errors_str:
                    errors_str += "\n"
                errors_str += "\n".join(error_msgs)

            feedback_dict = {
                "errors": errors_str,
                "warnings": [],
                "sorries": 0
            }
            prev_proof = prev_entry.get("proof", "")

        if worker_args.proof_strategy == "feedback":
            proof, outline, dt = produce_proof_with_feedback(
                model, tokenizer, problem, feedback=feedback_dict, prev_proof=prev_proof
            )
        elif worker_args.proof_strategy == "commander":
            guidance, t_commander = produce_guidance_commander(
                c_model,
                c_tokenizer,
                problem=problem,
                proof=prev_proof,
                feedback=feedback_dict
            )

            proof, outline, dt = produce_proof_with_guidance(
                model, tokenizer, problem, guidance=guidance
            )
            dt += t_commander
        else:
            proof, outline, dt = produce_proof_baseline(model, tokenizer, problem)

        if not proof.strip():
             verification_result = {
                 "verified": False,
                 "error": "Empty proof generated",
                 "feedback": []
             }
        else:
             verification_result = verify_proof(proof)
             if verification_result["verified"] and "sorry" in proof:
                 verification_result["verified"] = False
                 verification_result["error"] = "Proof contains 'sorry'"

        output = {
            "name": name,
            "time": int(dt),
            "verified": verification_result["verified"],
            "error": verification_result.get("error", ""),
            "feedback": verification_result.get("feedback", []),
            "proof": f"{proof}",
            "outline": f"{outline}",
        }
        
        if worker_args.proof_strategy == "commander" and "guidance" in locals():
             output["guidance"] = guidance

        output_q.put(output)

def main():

    model_map = {
            "Goedel": "Goedel-LM/Goedel-Prover-V2-8B",
            "Deepseek": "deepseek-ai/DeepSeek-Prover-V2-7B",
    }

    proof_strategies_map = {
        "pass2": produce_proof_baseline,
        "feedback": produce_proof_with_feedback,
        "commander": produce_proof_with_guidance,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--proof-strategy", choices=list(proof_strategies_map.keys()), required=True)
    parser.add_argument("--model", choices=list(model_map.keys()), default="Deepseek")
    parser.add_argument("--problem-set", default="test")
    args = parser.parse_args()

    args.model_id = model_map[args.model]

    input_path = Path(f"output/baseline_{args.model}_{args.problem_set}_solutions.txt")
    output_path = Path(f"output/{args.proof_strategy}_{args.model}_{args.problem_set}_solutions.txt")


    
    Path("output").mkdir(exist_ok=True)

    # 1. Read input file to find failed problems
    try:
        with open(input_path, "rb") as f:
            prev_results = tomli.load(f)["problem"]
    except FileNotFoundError:
        print(f"Input file {input_path} not found.")
        return

    failed_entries = {}
    baseline_success = 0
    total_problems = len(prev_results)

    for entry in prev_results:
        if not entry.get("verified", False):
            failed_entries[entry["name"]] = entry
        else:
            baseline_success += 1

    print(f"Baseline Summary: {baseline_success}/{total_problems} verified ({baseline_success / total_problems * 100:.2f}%)")
    print(f"Found {len(failed_entries)} failed problems to retry.")

    if not failed_entries:
        print("No failed problems to retry.")
        return

    all_problems = extract_problems(f"datasets/miniF2F/{args.problem_set}.lean")
    
    problems_to_retry = []
    for name, problem_str in all_problems:
        if name in failed_entries:
            problems_to_retry.append((name, problem_str, failed_entries[name]))
    
    print(f"Retrying {len(problems_to_retry)} problems.")

    output_json = []

    mp.set_start_method("spawn", force=True)
    num_workers = torch.cuda.device_count()
    executor = ParallelExecutor(num_workers, worker=llm_worker, worker_args=args)

    for i, (name, problem_str, prev_entry) in enumerate(problems_to_retry):
        executor.submit(i % num_workers, (i, name, problem_str, prev_entry))

    for i in range(len(problems_to_retry)):
        output = executor.gather()
        print(f"Completed Stage 2 for {output['name']} - Verified: {output['verified']}")

        output_json.sort(key=lambda x: x["name"])
        
        output_json.append(output)
        
        with open(output_path, "wb") as f:
            tomli_w.dump({"problem": output_json}, f, multiline_strings=True)

    executor.shutdown()
    total_generation_time = sum(p["time"] for p in output_json)

    num_improved = sum(1 for p in output_json if p["verified"])
    total_success = baseline_success + num_improved

    print(f"\n=== Cumulative Summary (Pass 1 + Pass 2) ===")
    print(f"Total Problems: {total_problems}")
    print(f"Baseline Success: {baseline_success}")
    print(f"Pass 2 Improved: {num_improved}")
    print(f"Total Success: {total_success}")
    print(f"Total Accuracy: {total_success / total_problems * 100:.2f}%")
    print(f"Absolute Improvement: +{num_improved / total_problems * 100:.2f}%")
    print(f"Relative Improvement: +{num_improved / baseline_success * 100:.2f}%")
    print(f"Total Generation Time (sum of dt for Pass 2): {total_generation_time}s")

if __name__ == "__main__":
    main()
