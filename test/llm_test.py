from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import getpass

USER = getpass.getuser()

# List of different LLMs that we may want to use or tryout. You can lookup each one on https://huggingface.co/
llm_ids = [
    "deepseek-ai/DeepSeek-Prover-V2-7B",
    "Goedel-LM/Goedel-Prover-V2-8B",
    "Goedel-LM/Goedel-Prover-SFT",
    "Goedel-LM/Goedel-Prover-DPO",
    "ByteDance-Seed/BFS-Prover",
    "internlm/internlm2-step-prover",
    "internlm/internlm2_5-step-prover",
    "AI-MO/Kimina-Prover-RL-1.7B"
]

# Select Deepseek LLM
llm_id = llm_ids[0]

tokenizer = AutoTokenizer.from_pretrained(llm_id, cache_dir=f"/work/classtmp/{USER}/models")
model = AutoModelForCausalLM.from_pretrained(llm_id, device_map="auto", cache_dir=f"/work/classtmp/{USER}/models", torch_dtype=torch.bfloat16, trust_remote_code=True)

# Inference

# This causes the outputs to be deterministic (comment out if this is not desired)
torch.manual_seed(30)

# This prompt was taken from the Deepseek page so it will likely not work with theother LLMs

theorem = """
theorem mathd_algebra_73
  (p q r x : ℂ)
  (h₀ : (x - p) * (x - q) = (r - p) * (r - q))
  (h₁ : x ≠ r) :
  x = p + q - r :=
begin
  sorry
end
"""

formal_statement = f"""
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

{theorem}
""".strip()

prompt = """
Complete the following Lean 4 code:

```lean4
{}
```

Before producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan outlining the main proof steps and strategies.
The plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the construction of the final formal proof.
""".strip()

chat = [
  {"role": "user", "content": prompt.format(formal_statement)},
]

inputs = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_tensors="pt", padding=True, truncation=True).to(model.device)

import time
start = time.time()
outputs = model.generate(inputs, max_new_tokens=8192)
print(tokenizer.batch_decode(outputs)[0])
print(time.time() - start)