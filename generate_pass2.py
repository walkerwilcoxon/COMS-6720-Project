import argparse
from pathlib import Path
import sys
import os
import multiprocessing as mp
import textwrap

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

# Configuration
input_filename = "benchmark_Goedel_50cases_solutions.txt"
output_filename = "benchmark_Goedel_50cases_pass2.txt"

# Can either be "Goedel" or "Deepseek"
llm_name = "Goedel"
problem_set_name = "test" 

if llm_name == "Goedel":
    llm_id = "Goedel-LM/Goedel-Prover-V2-8B"
elif llm_name == "Deepseek":
    llm_id = "deepseek-ai/DeepSeek-Prover-V2-7B"
else:
    raise ValueError(f"Unknown LLM: {llm_name}")


def llm_worker(gpu_id, input_q, output_q, worker_args):
    print(f"Starting LLM worker {gpu_id}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    model, tokenizer = load_model(llm_id)
    
    # If commander is needed, load it too (Always use DeepSeek)
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
        
        # Determine Strategy
        if worker_args.feedback:
            # 1. Feedback Strategy
            proof, outline, dt = produce_proof_with_feedback(
                model, tokenizer, problem, feedback=feedback_dict, prev_proof=prev_proof
            )
            
        elif worker_args.commander:
            # 2. Commander Strategy
            # Get Guidance
            guidance, t_commander = produce_guidance_commander(
                c_model, c_tokenizer,
                problem=problem,
                proof=prev_proof,
                feedback=feedback_dict
            )
            
            # Generate Proof with Guidance
            proof, outline, dt = produce_proof_with_guidance(
                model, tokenizer, problem, guidance=guidance
            )
            dt += t_commander # Add commander time
            
        else:
            # 3. Benchmark Strategy (Standard Retry)
            proof, outline, dt = produce_proof_benchmark(model, tokenizer, problem)

        # Verify
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

# We need to subclass ParallelExecutor to pass worker_args to the worker
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
            # Pass worker_args as the 4th argument to target
            p = mp.Process(target=worker, args=(worker_id, q, self.output_queue, self.worker_args))
            p.start()
            self.workers.append(p)

def main():
    parser = argparse.ArgumentParser(description="Generate Pass 2 Proofs")
    parser.add_argument("--feedback", action="store_true", help="Use previous feedback to correct proof")
    parser.add_argument("--commander", action="store_true", help="Use DeepSeek Commander agent to guide proof")
    args = parser.parse_args()
    
    if args.feedback and args.commander:
        print("Error: Cannot use both --feedback and --commander")
        return

    # Determine output filename based on strategy
    base_output_name = "benchmark_Goedel_50cases_pass2"
    
    if args.feedback:
        output_filename = f"{base_output_name}_feedback.txt"
    elif args.commander:
        output_filename = f"{base_output_name}_commander.txt"
    else:
        output_filename = f"{base_output_name}.txt"

    Path("output").mkdir(exist_ok=True)
    
    input_path = f"output/{input_filename}"
    output_path = f"output/{output_filename}"

    # 1. Read input file to find failed problems
    try:
        with open(input_path, "rb") as f:
            prev_results = tomli.load(f)["proof"]
    except FileNotFoundError:
        print(f"Input file {input_path} not found.")
        return

    # Identify failed problems
    failed_entries = {}
    for entry in prev_results:
        if not entry.get("verified", False):
            failed_entries[entry["name"]] = entry

    print(f"Found {len(failed_entries)} failed problems from Pass 1.")

    if not failed_entries:
        print("No failed problems to retry.")
        return

    # 2. Load the actual problem statements
    all_problems = extract_problems(f"datasets/miniF2F/{problem_set_name}.lean")
    
    problems_to_retry = []
    for name, problem_str in all_problems:
        if name in failed_entries:
            problems_to_retry.append((name, problem_str, failed_entries[name]))
    
    print(f"Retrying {len(problems_to_retry)} problems.")

    output_json = []

    mp.set_start_method("spawn", force=True)
    num_workers = torch.cuda.device_count()
    
    # Use custom executor to pass args
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

    num_improved = sum(1 for p in output_json if p["verified"])
    print(f"\n=== Pass 2 Summary ===")
    print(f"Total Retried: {len(problems_to_retry)}")
    print(f"Improved (Verified): {num_improved}")
    print(f"Success Rate (Pass 2): {num_improved / len(problems_to_retry) * 100:.2f}%")

if __name__ == "__main__":
    main()
