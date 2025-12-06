from pathlib import Path
import sys
import os
import multiprocessing as mp

import torch
import tomli_w
import tomli

from llm_utils import load_model, extract_problems, ParallelExecutor, verify_proof
from proof_producers import produce_proof_benchmark, produce_proof_with_feedback

# Usage: 
#   Model:
#     Godel model: python generate_proofs.py --goedel
#     Deepseek model: python generate_proofs.py --deepseek
#   Proof producer:
#     Benchmark producer: python generate_proofs.py --benchmark
#   Dataset:
#     Test: python generate_proofs.py --test
#     Valid: python generate_proofs.py --valid
#   Default arguments: python generate_proofs.py --goedel --benchmark --test

if "--deepseek" in sys.argv[1:]:
        llm_name, llm_id = ("Deekseek", "deepseek-ai/DeepSeek-Prover-V2-7B")
else:
    llm_name, llm_id = ("Goedel", "Goedel-LM/Goedel-Prover-V2-8B")

# TODO: Add more proof producers
producer_name, proof_producer = "benchmark", produce_proof_benchmark

if "--valid" in sys.argv[1:]:
    problem_set_name = "valid"
else:
    problem_set_name = "test"

num_gpus = torch.cuda.device_count()

def llm_worker(gpu_id, input_q, output_q):
    print(f"Starting LLM worker {gpu_id}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    device = torch.device("cuda:0")

    model, tokenizer = load_model(llm_id)

    while True:
        input = input_q.get()
        if input is None:
            break
        
        i, name, problem = input

        # 1. Initial Proof Generation
        proof, outline, dt = proof_producer(model, tokenizer, problem)
        
        # Check for empty proof (retry once with higher token limit)
        # if not proof.strip():
        #     print(f"Worker {gpu_id}: Empty proof generated for {name}. Retrying with more tokens...")

        #     proof, outline, dt_retry = proof_producer(model, tokenizer, problem)
        #     dt += dt_retry

        # 2. Verification
        if not proof.strip():
             verification_result = {
                 "verified": False,
                 "error": "Empty proof generated",
                 "feedback": []
             }
        else:
             verification_result = verify_proof(proof)
             # "sorry" is valid syntax but not a valid proof
             if verification_result["verified"] and "sorry" in proof:
                 verification_result["verified"] = False
                 verification_result["error"] = "Proof contains 'sorry'"

        verified = verification_result["verified"]
        
        # 3. Feedback & Repair (if initial verification failed)
        if not verified:
             errors_str = verification_result.get("error", "")
             feedback_list = verification_result.get("feedback", [])
             
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
             
             print(f"Worker {gpu_id}: Proof for {name} failed. Attempting fix...")
             
             # Generate fixed proof
             fixed_proof, fixed_outline, fixed_dt = produce_proof_with_feedback(
                 model, tokenizer, problem, feedback=feedback_dict, prev_proof=proof
             )
             
             # Verify fixed proof
             fixed_verification = verify_proof(fixed_proof)
             
             if fixed_verification["verified"] and "sorry" in fixed_proof:
                 fixed_verification["verified"] = False
                 fixed_verification["error"] = "Fixed proof contains 'sorry'"
             
             # Update to use fixed version
             proof = fixed_proof
             outline = fixed_outline
             dt += fixed_dt
             verification_result = fixed_verification
             verified = fixed_verification["verified"]
             
             status = "VERIFIED!" if verified else "still failed."
             print(f"Worker {gpu_id}: Fixed proof for {name} {status}")

        # 4. Output Construction
        output = {
            "name": name,
            "iteration": i + 1,
            "time": int(dt),
            "verified": verified,
            "error": verification_result.get("error", ""),
            "feedback": verification_result.get("feedback", []),
            "proof": f"{proof}",
            "outline": f"{outline}",
        }
        
        output_q.put(output)

def main():
    Path("output").mkdir(exist_ok=True)
    proof_filepath = f"output/{producer_name}_{llm_name}_{problem_set_name}_solutions_with_feedback.txt"

    try:
        with open(proof_filepath, "rb") as f:
            proofs = tomli.load(f)["proof"]
            all_proved_names = set([proof["name"] for proof in proofs])
            output_json = proofs
    except FileNotFoundError:
        all_proved_names = set()
        output_json = []

    problems_set = extract_problems(f"datasets/miniF2F/{problem_set_name}.lean")

    mp.set_start_method("spawn", force=True)
    executor = ParallelExecutor(num_gpus, worker=llm_worker)

    for i, (name, problem) in enumerate(problems_set):
        if name in all_proved_names:
            continue
        executor.submit(i % num_gpus, (i, name, problem))

    n = len(problems_set)
    for _ in problems_set:
        output = executor.gather()
        print(f"Completed proof {output['name']} in {int(output['time'])} seconds ({output['iteration']}/{n})")

        output_json.append(output)
        output_json.sort(key=lambda x: x["iteration"])
        
        with open(proof_filepath, "wb") as f:
            tomli_w.dump({"proof": output_json}, f, multiline_strings=True)

if __name__ == "__main__":
    main()
