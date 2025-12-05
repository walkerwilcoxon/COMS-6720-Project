from typing import List, Dict
from pathlib import Path
from functools import reduce
import re
import textwrap
import time
import textwrap

from llm_utils import extract_proof_and_outline, PROOF_IMPORTS

max_new_tokens=8192

# This file contains all of the different proof strategies we will try


# Produces the benchmark proofs
def produce_proof_benchmark(model, tokenizer, problem):

    prompt = textwrap.dedent(f"""
    Complete the following Lean 4 code:

    ```lean4
    {PROOF_IMPORTS}

    {problem}
    ```

    Before producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan outlining the main proof steps and strategies.
    The plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the construction of the final formal proof.
    
    """).strip()

    chat = [
        {"role": "user", "content": prompt},
    ]

    inputs = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_tensors="pt", padding=True, truncation=True).to(model.device)

    start = time.time()
    output_tokens = model.generate(inputs)
    output = tokenizer.batch_decode(output_tokens)[0]
    dt = time.time() - start

    proof, outline = extract_proof_and_outline(output)

    return proof, outline, dt