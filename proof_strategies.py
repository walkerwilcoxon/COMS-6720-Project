from typing import List, Dict
from pathlib import Path
from functools import reduce
import re
import textwrap
import time
import textwrap

from llm_utils import extract_proof_and_outline, PROOF_IMPORTS

max_new_tokens=8192

# This file contains all of the different proof strategies we tried


# Produces the baseline proofs
def produce_proof_baseline(model, tokenizer, problem):

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
    output_tokens = model.generate(inputs, max_new_tokens=max_new_tokens)
    output = tokenizer.batch_decode(output_tokens)[0]
    dt = time.time() - start

    proof, outline = extract_proof_and_outline(output)

    return proof, outline, dt

def produce_proof_with_feedback(
    model,
    tokenizer,
    problem: str,
    feedback: dict,
    prev_proof: str,
    *,
    max_new_tokens: int = 8192,
    temperature: float = 0.2,
):
    """
    Agent 1 variant: Prover that corrects its own proof based on direct Lean feedback.
    """
    
    errors = feedback.get("errors", "")
    
    prompt = textwrap.dedent(f"""
    You are an expert Lean 4 prover.

    Your previous proof attempt failed with the following errors:
    
    ERRORS:
    {errors}
    
    PREVIOUS ATTEMPT:
    ```lean4
    {prev_proof}
    ```

    Your task is to fix the errors and produce a correct proof for the theorem below.
    
    Theorem and context:
    ```lean4
    {PROOF_IMPORTS}

    {problem}
    ```

    Output format:
    You must respond in the following exact structure:

    ### Detailed Proof
    (Your proof plan in natural language.)

    ### Complete Lean 4 Proof
    ```lean4
    theorem ... := by
      -- your proof here
    ```
    """).strip()

    chat = [
        {"role": "user", "content": prompt},
    ]

    inputs = tokenizer.apply_chat_template(
        chat,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)

    start = time.time()
    output_tokens = model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    output = tokenizer.batch_decode(output_tokens)[0]
    dt = time.time() - start

    proof, outline = extract_proof_and_outline(output)

    return proof, outline, dt


# === Multi-agent extensions ===

def produce_proof_with_guidance(
    model,
    tokenizer,
    problem: str,
    guidance: str | None = None,
    prev_proof: str | None = None,
    *,
    max_new_tokens: int = 8192,
    temperature: float = 0.2,
):
    
    # Format Commander guidance into a human-readable block for the prompt.
    if guidance is None:
        guidance_block = "(No specific guidance. Produce your best proof.)"
    else:
        guidance_block = guidance

    # Include previous proof attempt if relevant.
    # In this simplified mode, we assume we always restart from scratch (using the guidance),
    # but we could optionally include the previous proof if needed.
    # For now, we follow the previous logic where restart=True was default (hiding the old proof).
    previous_proof_block = "(No previous proof to reuse; start fresh.)"

    prompt = textwrap.dedent(f"""
    You are an expert Lean 4 prover.

    Your task is to produce a complete and correct Lean 4 proof for the theorem below.
    You must follow the Commander's guidance exactly. If the Commander says to avoid
    certain tactics, do not use them. If the Commander says to restart the proof,
    ignore all previous proof structures.

    Theorem and context:
    ```lean4
    {PROOF_IMPORTS}

    {problem}
    ```

    Commander guidance:
    {guidance_block}

    Previous attempt:
    {previous_proof_block}

    Before producing the Lean 4 code, provide a brief proof plan explaining your
    strategy given the guidance.

    Output format:
    You must respond in the following exact structure:

    ### Detailed Proof
    (Your proof plan in natural language.)

    ### Complete Lean 4 Proof
    ```lean4
    theorem ... := by
      -- your proof here
    ```
    """).strip()

    chat = [
        {"role": "user", "content": prompt},
    ]

    inputs = tokenizer.apply_chat_template(
        chat,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)

    start = time.time()
    output_tokens = model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    output = tokenizer.batch_decode(output_tokens)[0]
    dt = time.time() - start

    proof, outline = extract_proof_and_outline(output)

    return proof, outline, dt

def produce_guidance_commander(
    model,
    tokenizer,
    problem: str,
    proof: str,
    feedback: dict,
    *,
    max_new_tokens = 2048,
    temperature: float = 0.7,
) -> tuple[str, float]:
    """Agent 2: Commander / Strategist.

    Given the current theorem, Prover attempt, Lean feedback, and history,
    produce a structured guidance object for the next Prover call.
    """

    errors = feedback.get("errors", "")
    warnings = feedback.get("warnings", [])
    sorries = feedback.get("sorries", 0)

    prompt = textwrap.dedent(f"""
    You are an expert Lean 4 proof assistant.

    Your goal is to:
    1. Read the theorem.
    2. Read the Prover's Lean proof attempt.
    3. Read Lean's verification feedback.
    4. Diagnose what went wrong.
    5. Write a detailed direction for how the Prover should revise the proof next round.

    CORRECTION_DIRECTIONS
        1. If the error is "tactic '...' failed", check if the preconditions for that tactic are met. Suggest listing the hypotheses using `have` statements to ensure they match exactly.
        2. If the error involves "unknown identifier", it often means a lemma is missing from Mathlib imports or a variable name is typoed. Suggest searching for the correct lemma name or checking variable definitions.
        3. If the error is "maximum recursion depth" or "timeout", the proof search is too complex. Suggest breaking the proof into smaller `have` or `lemma` steps to guide the solver.
        4. If `simp` or `aesop` fails to close a goal, suggest unfolding definitions explicitly using `unfold` or providing specific lemmas to `simp [lemma_name]`.
        5. If the error is a type mismatch, explicitly state the expected type vs. the provided type and suggest a conversion tactic or a different theorem.
    END_CORRECTION_DIRECTIONS

    ---

    THEOREM:
    {problem}

    PROVER ATTEMPT:
    ```lean4
    {proof}
    ```
    
    Errors:
    {errors}

    Please provide a short, concise natural language plan to fix this proof.
    
    Guidelines:
    - Do NOT analyze the math in detail. Focus on the Lean tactics and structure.
    - Provide a numbered list of 2-3 concrete steps for the Prover to follow.
    - Refer to the CORRECTION_DIRECTIONS for specific strategies.
    - Do NOT write Lean code.
    - Do NOT repeat yourself.

    Put the steps in the plan block below, each step must be ordered starting from 1:

    PLAN

    ENDPLAN
    """).strip()

    chat = [
        {"role": "user", "content": prompt},
    ]

    inputs = tokenizer.apply_chat_template(
        chat,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)

    start = time.time()
    output_tokens = model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    
    dt = time.time() - start

    input_length = inputs.shape[1]
    new_tokens = output_tokens[0][input_length:]
    response_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return response_text, dt
