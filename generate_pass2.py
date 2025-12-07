import argparse
from pathlib import Path
import multiprocessing as mp
import os

import torch
import tomli_w
import tomli

from llm_utils import load_model, extract_problems, ParallelExecutor, verify_proof
from proof_producers import (
    produce_proof_benchmark,
    produce_proof_with_feedback,
    produce_proof_with_guidance,
    produce_guidance_commander,
)

model_map = {
    "Goedel": "Goedel-LM/Goedel-Prover-V2-8B",
    "Deepseek": "deepseek-ai/DeepSeek-Prover-V2-7B",
}


def llm_worker(gpu_id, input_q, output_q, worker_args):
    print(f"Starting LLM worker {gpu_id}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    llm_id = model_map[worker_args.model]
    model, tokenizer = load_model(llm_id)

    c_model, c_tokenizer = None, None
    if worker_args.commander:
        print(f"Worker {gpu_id}: Loading DeepSeek Commander...")
        c_model, c_tokenizer = load_model("deepseek-ai/DeepSeek-Prover-V2-7B")

    while True:
        input_data = input_q.get()
        if input_data is None:
            break

        i, name, problem, prev_entry = input_data
        print(f"Retrying proof {name} ({i})")

        
        # Prepare feedback dict (common for both feedback and commander strategies)
        if worker_args.feedback or worker_args.commander:
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

        if worker_args.feedback:
            proof, outline, dt = produce_proof_with_feedback(
                model, tokenizer, problem, feedback=feedback_dict, prev_proof=prev_proof
            )
        elif worker_args.commander:
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
            proof, outline, dt = produce_proof_benchmark(model, tokenizer, problem)

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
            "pass": 2, 
            "strategy": "feedback" if worker_args.feedback else ("commander" if worker_args.commander else "benchmark"),
            "time": int(dt),
            "verified": verification_result["verified"],
            "error": verification_result.get("error", ""),
            "feedback": verification_result.get("feedback", []),
            "proof": f"{proof}",
            "outline": f"{outline}",
        }
        
        if worker_args.commander and "guidance" in locals():
             output["guidance"] = guidance

        output_q.put(output)


class ArgsParallelExecutor(ParallelExecutor):
    def __init__(self, num_workers, worker, worker_args):
        self.num_workers = num_workers
        self.input_queues = []
        self.output_queue = mp.Queue()
        self.workers = []
        self.worker_args = worker_args

        for worker_id in range(num_workers):
            q = mp.Queue()
            self.input_queues.append(q)
            p = mp.Process(target=worker, args=(worker_id, q, self.output_queue, self.worker_args))
            p.start()
            self.workers.append(p)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feedback", action="store_true")
    parser.add_argument("--commander", action="store_true")
    parser.add_argument("--model", choices=list(model_map.keys()), default="Goedel")
    parser.add_argument("--input-file", default=None)
    parser.add_argument("--output-name", default=None)
    args = parser.parse_args()

    if args.feedback and args.commander:
        print("Error: Cannot use both --feedback and --commander")
        return

    # Configuration
    problem_set_name = "test"

    # Set defaults based on model if not provided
    if args.input_file is None:
        args.input_file = f"benchmark_{args.model}_50cases_solutions.txt"
    
    if args.output_name is None:
        args.output_name = f"benchmark_{args.model}_50cases_pass2"

    # Determine output filename based on strategy
    base_output_name = args.output_name
    
    if args.feedback:
        output_filename = f"{base_output_name}_feedback.txt"
    elif args.commander:
        output_filename = f"{base_output_name}_commander.txt"
    else:
        output_filename = f"{base_output_name}.txt"

    Path("output").mkdir(exist_ok=True)
    input_path = f"output/{args.input_file}"
    output_path = f"output/{output_filename}"

    # 1. Read input file to find failed problems
    try:
        with open(input_path, "rb") as f:
            prev_results = tomli.load(f)["proof"]
    except FileNotFoundError:
        print(f"Input file {input_path} not found.")
        return

    failed_entries = {}
    pass1_success = 0
    total_problems = len(prev_results)

    for entry in prev_results:
        if not entry.get("verified", False):
            failed_entries[entry["name"]] = entry
        else:
            pass1_success += 1

    print(f"Pass 1 Summary: {pass1_success}/{total_problems} verified ({pass1_success / total_problems * 100:.2f}%)")
    print(f"Found {len(failed_entries)} failed problems to retry.")

    if not failed_entries:
        print("No failed problems to retry.")
        return

    all_problems = extract_problems(f"datasets/miniF2F/{problem_set_name}.lean")
    
    problems_to_retry = []
    for name, problem_str in all_problems:
        if name in failed_entries:
            problems_to_retry.append((name, problem_str, failed_entries[name]))
    
    print(f"Retrying {len(problems_to_retry)} problems.")

    output_json = []

    mp.set_start_method("spawn", force=True)
    num_workers = torch.cuda.device_count()
    executor = ArgsParallelExecutor(num_workers, worker=llm_worker, worker_args=args)

    for i, (name, problem_str, prev_entry) in enumerate(problems_to_retry):
        executor.submit(i % num_workers, (i, name, problem_str, prev_entry))

    for _ in problems_to_retry:
        output = executor.gather()
        print(f"Completed Pass 2 for {output['name']} - Verified: {output['verified']}")
        
        output_json.append(output)
        
        with open(output_path, "wb") as f:
            tomli_w.dump({"proof": output_json}, f, multiline_strings=True)

    executor.shutdown()
    total_generation_time = sum(p["time"] for p in output_json)

    num_improved = sum(1 for p in output_json if p["verified"])
    total_success = pass1_success + num_improved

    print(f"\n=== Cumulative Summary (Pass 1 + Pass 2) ===")
    print(f"Total Problems: {total_problems}")
    print(f"Pass 1 Success: {pass1_success}")
    print(f"Pass 2 Improved: {num_improved}")
    print(f"Total Success: {total_success}")
    print(f"Total Accuracy: {total_success / total_problems * 100:.2f}%")
    print(f"Absolute Improvement: +{num_improved / total_problems * 100:.2f}%")
    print(f"Relative Improvement: +{num_improved / pass1_success * 100:.2f}%")
    print(f"Total Generation Time (sum of dt for Pass 2): {total_generation_time}s")

if __name__ == "__main__":
    main()
