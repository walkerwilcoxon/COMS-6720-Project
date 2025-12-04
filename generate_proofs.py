from pathlib import Path
import sys
import os
import multiprocessing as mp

import torch
import tomli_w
import tomli

from llm_utils import load_model, extract_problems, ParallelExecutor
from proof_producers import produce_proof_benchmark

# Can either be "benchmark", or another proof producer name
proof_producer_name = "benchmark"

# Can either be "Goedel" or "Deepseek"
llm_name = "Goedel"

# Can either be "test", "valid", or any file inside of "datasets/miniF2F"
problem_set_name = "incorrect_subset"



if proof_producer_name == "benchmark":
    proof_producer = produce_proof_benchmark
else:
    raise ValueError(f"Invalid proof producer name: {proof_producer_name}")


if llm_name == "Goedel":
    llm_id = "Goedel-LM/Goedel-Prover-V2-8B"
elif llm_name == "Deepseek":
    llm_id = "deepseek-ai/DeepSeek-Prover-V2-7B"
else:
    raise ValueError(f"Unknown LLM: {llm_name}")

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

        print(f"Starting proof {name} (i)")

        proof, outline, dt = proof_producer(model, tokenizer, problem)

        output = {}
        output["name"] = name
        output["iteration"] = i + 1
        output["time"] = int(dt)
        output["proof"] = f"{proof}"
        output["outline"] = f"{outline}"

        output_q.put(output)

def main():

    Path("output").mkdir(exist_ok=True)

    proof_filepath = f"output/{proof_producer_name}_{llm_name}_{problem_set_name}_solutions.txt"

    try:
        with open(proof_filepath, "rb") as f:
            proofs = tomli.load(f)["proof"]
            # Names of proofs that have already been proven
            already_proved_names = set([proof["name"] for proof in proofs])
            output_json = proofs
    except FileNotFoundError:
        already_proved_names = {}
        output_json = []
    
    problem_set = extract_problems(f"datasets/miniF2F/{problem_set_name}.lean")

    mp.set_start_method("spawn", force=True)

    # num_workers = number of gpus
    num_workers = torch.cuda.device_count()

    executor = ParallelExecutor(num_workers, worker=llm_worker)

    # Filter proofs that are already in the output file
    problem_set = list(filter(lambda problem: problem[0] not in already_proved_names, problem_set))

    for i, (name, problem) in enumerate(problem_set):
        executor.submit(i % num_workers, (i, name, problem))

    num_problems = len(problem_set)

    for _ in problem_set:
        output = executor.gather()

        print(f"Completed proof {output["name"]} in {int(output["time"])} seconds ({output["iteration"]}/{num_problems})")

        output_json.append(output)
        output_json.sort(key=lambda x: x["iteration"])
        
        with open(proof_filepath, "wb") as f:
            tomli_w.dump({"proof": output_json}, f, multiline_strings=True)

    executor.shutdown()

if __name__ == "__main__":
    main()
