from pathlib import Path
import sys
import os
import multiprocessing as mp

import torch
import tomli_w
import tomli
import argparse

from llm_utils import load_model, extract_problems, ParallelExecutor
from proof_strategies import produce_proof_baseline


def llm_worker(gpu_id, input_q, output_q, worker_args):
    print(f"Starting LLM worker {gpu_id}")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    device = torch.device("cuda:0")

    model, tokenizer = load_model(worker_args.llm_id)

    while True:
        input = input_q.get()
        if input is None:
            break

        
        i, name, problem = input

        print(f"Starting proof {name} ({i + 1})")

        proof, outline, dt = produce_proof_baseline(model, tokenizer, problem)

        output = {}
        output["name"] = name
        output["time"] = int(dt)
        output["proof"] = f"{proof}"
        output["outline"] = f"{outline}"

        output_q.put(output)

def main():
    model_map = {
        "Goedel": "Goedel-LM/Goedel-Prover-V2-8B",
        "Deepseek": "deepseek-ai/DeepSeek-Prover-V2-7B",
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(model_map.keys()), default="Deepseek")
    parser.add_argument("--problem-set", default="test")
    args = parser.parse_args()

    args.llm_id = model_map[args.model]

    Path("output").mkdir(exist_ok=True)

    proof_filepath = f"output/baseline_{args.model}_{args.problem_set}_solutions.txt"

    try:
        with open(proof_filepath, "rb") as f:
            proofs = tomli.load(f)["problem"]
            # Names of proofs that have already been proven
            already_proved_names = set([proof["name"] for proof in proofs])
            output_json = proofs
    except FileNotFoundError:
        already_proved_names = {}
        output_json = []
    
    problem_set = extract_problems(f"datasets/miniF2F/{args.problem_set}.lean")

    mp.set_start_method("spawn", force=True)

    # num_workers = number of gpus
    num_workers = torch.cuda.device_count()

    executor = ParallelExecutor(num_workers, worker=llm_worker, worker_args=args)

    # Filter proofs that are already in the output file
    problem_set = list(filter(lambda problem: problem[0] not in already_proved_names, problem_set))

    for i, (name, problem) in enumerate(problem_set):
        executor.submit(i % num_workers, (i, name, problem))

    num_problems = len(problem_set)

    for i in range(len(problem_set)):
        output = executor.gather()

        print(f"Completed proof {output["name"]} in {int(output["time"])} seconds ({i + 1}/{num_problems})")

        output_json.append(output)
        output_json.sort(key=lambda x: x["name"])
        
        with open(proof_filepath, "wb") as f:
            tomli_w.dump({"problem": output_json}, f, multiline_strings=True)

    executor.shutdown()

if __name__ == "__main__":
    main()
