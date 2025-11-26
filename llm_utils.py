from typing import List, Dict
from subprocess import Popen, PIPE, STDOUT
import subprocess
import os
import tempfile
from pathlib import Path
from functools import reduce
import re
import textwrap
import time
import json
import multiprocessing as mp


from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, logging
import torch
import getpass

HOME_DIR = os.path.expanduser('~')

DEFAULT_LAKE_PATH = f'{HOME_DIR}/.elan/bin/lake'

DEFAULT_LEAN_WORKSPACE="/work/classtmp/wpw/COMS-6720-Project/mathlib4"

USER = getpass.getuser()

PROOF_IMPORTS = """
import Mathlib
import Aesop

set_option linter.all false
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


def verify_proof(proof, timeout=30) -> dict[str, object]:
    full_proof = f"{PROOF_IMPORTS}\n{proof}"

    # Write temporary Lean file
    with tempfile.NamedTemporaryFile(suffix=".lean", delete=False, mode="w") as f:
        f.write(full_proof)
        f.flush()
        tmp_path = f.name

    cmd = f"lake env lean --json {tmp_path}"

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            shell=True,
            cwd=DEFAULT_LEAN_WORKSPACE,
            timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return {
            "verified": False,
            "timeout": True,
            "messages": [],
        }

    output = result.stdout

    json_lines = [
        line.strip()
        for line in output.splitlines()
        if line.strip().startswith("{") and line.strip().endswith("}")
    ]

    error_list = []

    for json_line in json_lines:
        message_dict = json.loads(json_line)

        if message_dict["severity"] != "error":
            continue

        if message_dict["data"] == "No goals to be solved":
            continue

        error = {
            "line": message_dict["pos"]["line"],
            "column": message_dict["pos"]["column"],
            "feedback": message_dict["data"],
        }
        error_list.append(error)

    return {
        "verified": len(error_list) == 0,
        "timeout": False,
        "errors": error_list
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
    def __init__(self, num_workers, worker):
        self.num_workers = num_workers
        self.input_queues = []
        self.output_queue = mp.Queue()
        self.workers = []

        for worker_id in range(num_workers):
            q = mp.Queue()
            self.input_queues.append(q)
            p = mp.Process(target=worker, args=(worker_id, q, self.output_queue))
            p.start()
            self.workers.append(p)

    def submit(self, worker_id, data):
        self.input_queues[worker_id].put(data)

    def gather(self):
        return self.output_queue.get()

    def shutdown(self):
        for q in self.input_queues:
            q.put(None)
        for p in self.workers:
            p.join()