from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
import os
import multiprocessing as mp

import torch
import tomli_w
import tomli

from llm_utils import load_model, get_f2f_problems
from proof_producers import produce_proof_benchmark

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
if False:
    pass
else:
    proof_producer = produce_proof_benchmark

if "--valid" in sys.argv[1:]:
    problem_set_name = "valid"
else:
    problem_set_name = "test"

num_gpus = torch.cuda.device_count()

class MultiGPUExecutor:
    def __init__(self, num_gpus):
        self.num_gpus = num_gpus
        self.input_queues = []
        self.output_queue = mp.Queue()
        self.workers = []

        for gpu_id in range(num_gpus):
            q = mp.Queue()
            self.input_queues.append(q)
            p = mp.Process(target=gpu_worker, args=(gpu_id, q, self.output_queue))
            p.start()
            self.workers.append(p)

    def submit(self, gpu_id, data):
        self.input_queues[gpu_id].put(data)

    def gather(self):
        return self.output_queue.get()

    def shutdown(self):
        for q in self.input_queues:
            q.put(None)
        for p in self.workers:
            p.join()

def gpu_worker(gpu_id, input_q, output_q):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    device = torch.device("cuda:0")

    model, tokenizer = load_model(llm_id)

    while True:
        input = input_q.get()
        if input is None:
            break

        
        i, name, problem = input

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

    try:
        with open(f"output/{llm_name}_{problem_set_name}_solutions.txt", "rb") as f:
            proofs = tomli.load(f)["proof"]
            # Names of proofs that have already been proven
            all_proved_names = set([proof["name"] for proof in proofs])
            output_json = proofs
    except FileNotFoundError:
        all_proved_names = {}
        output_json = []

    mp.set_start_method("spawn", force=True)

    problems = get_f2f_problems()
    
    problems_set = problems[problem_set_name]

    executor = MultiGPUExecutor(num_gpus)

    for i, (name, problem) in enumerate(problems_set):
        # Skip proofs that are already in the output file
        if name in all_proved_names:
            continue
        executor.submit(i % num_gpus, (i, name, problem))

    n = len(problems_set)

    for _ in problems_set:
        output = executor.gather()

        print(f"Completed proof {output["name"]} in {int(output["time"])} seconds ({output["iteration"]}/{n})")

        output_json.append(output)

        output_json.sort(key=lambda x: x["iteration"])
        with open(f"output/{llm_name}_{problem_set_name}_solutions.txt", "wb") as f:
            tomli_w.dump({"proof": output_json}, f, multiline_strings=True)

if __name__ == "__main__":
    main()
