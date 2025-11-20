from typing import List, Dict
from pathlib import Path
from functools import reduce
import re
import textwrap
import time

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, logging
import torch
import getpass


USER = getpass.getuser()

IMPORTS = """
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat
"""

def load_model(llm_id: str):
    logging.set_verbosity_error()
    tokenizer = AutoTokenizer.from_pretrained(llm_id, cache_dir=f"/work/classtmp/{USER}/models")
    model = AutoModelForCausalLM.from_pretrained(llm_id, device_map="auto", cache_dir=f"/work/classtmp/{USER}/models", torch_dtype=torch.bfloat16, trust_remote_code=True)
    if tokenizer.pad_token is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.pad_token = tokenizer.eos_token  # allow padding
        model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


# def revise_proof(model, tokenizer, proof, feedback, instructions=None):
#     if instructions == None:
#         instructions = """
#             Before producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan outlining the main proof steps and strategies.
#             The plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the construction of the final formal proof.
#         """.strip()
#     prompt = f"""
#     You are an expert in Lean. Your task is to revise the proof that is provided
    
#     {proof}
#     '''

#     {instructions}
#     """.strip()

#     chat = [
#         {"role": "user", "content": prompt},
#     ]

#     inputs = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_tensors="pt", padding=True, truncation=True).to(model.device)

#     start = time.time()
#     output_tokens = model.generate(inputs, max_new_tokens=8192)
#     output = tokenizer.batch_decode(output_tokens)[0]
#     dt = time.time() - start
#     return output, dt


def validate_proof(proof) -> dict[str, object]:
    proof_evalutation = None # evaluation

    return {
        "valid": False,
        "compiler errors": []
    }

def extract_problems(file) -> List[str]:
    problems = []
    with open(file) as f:
        lines = f.readlines()
        contents = reduce(lambda x, y: x + y, lines)

        regex = re.compile(r"theorem (\w*)\s[\s\S]*?sorry")

        matches = regex.finditer(contents)

        problems = [(match.group(1), match.group(0)) for match in matches]

    return problems

def get_f2f_problems() -> Dict[str, List[str]]:
    data_dir = Path(__file__).parent.resolve()
    test_file = Path(f"{data_dir}/datasets/miniF2F/test.lean")
    valid_file = Path(f"{data_dir}/datasets/miniF2F/valid.lean")

    test_problems = extract_problems(test_file)
    valid_problems = extract_problems(valid_file)

    return {
        "test": test_problems,
        "valid": valid_problems
    }

def extract_proof_and_outline(output: str) -> (str, str):
    match = re.search(r"(### Detailed Proof[\s\S]*)### Complete Lean 4 Proof\s*```lean4\s*(theorem[\s\S]*)```", output)

    # If there is no match, simply return an empty proof and the entire output as the outline
    if not match:
        return "", output

    proof = match.group(2)

    outline = match.group(1)

    return proof, outline

class ParallelExecutor:
    def __init__(self, num_gpus, worker):
        self.num_gpus = num_gpus
        self.input_queues = []
        self.output_queue = mp.Queue()
        self.workers = []

        for gpu_id in range(num_gpus):
            q = mp.Queue()
            self.input_queues.append(q)
            p = mp.Process(target=worker, args=(gpu_id, q, self.output_queue))
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