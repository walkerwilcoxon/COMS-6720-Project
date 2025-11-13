<div align="center">
    <h1> <a href="http://blog.goedel-prover.com"> <strong>Setup for Goedel-Prover-V2</strong></a></h1>
</div>


## 1. Environment Setup

We follow [DeepSeek-Prover-V1.5](https://github.com/deepseek-ai/DeepSeek-Prover-V1.5), which uses Lean 4 version 4.9 and the corresponding Mathlib. Please refer to the following instructions to set up the environments.

### Requirements

* Supported platform: Linux
* Python 3.10

### Installation

1. **Install Lean 4**
  
You can uninstall Lean if you are using a different version and skip this step. It will automatically install the correct version in the following steps. Explicitly, it happens when running lake build in step 4 (you should have the required files lakefile.lean and lean-toolchain in mathlib4 after you clone the repository).

Or can install the correct version manually. 
   Follow the instructions on the [Lean 4 installation page](https://leanprover.github.io/lean4/doc/quickstart.html) to set up Lean 4.

2. **Clone the repository**

```sh
git clone --recurse-submodules https://github.com/Goedel-LM/Goedel-Prover-V2.git
cd Goedel-Prover-V2
```

3. **Install required packages**
Install conda if not installed:
```sh
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

Create conda virtual environment
```sh
conda env create -f goedelv2.yml
```

If you encounter installation error when installing packages (flash-attn), please run the following:

```sh
conda activate goedelv2
pip install torch==2.6.0
conda env update --file goedelv2.yml
```

4. **Build Mathlib4**

```sh
cd mathlib4
lake build
```

5. **Test Lean 4 and mathlib4 installation**

```sh
cd ..
python lean_compiler/repl_scheduler.py
```
If there is any error, reinstall Lean 4 and rebuild mathlib4.

If you have installed Lean and Mathlib for other projects and want to use the pre-installed things, note that you might need to modify `DEFAULT_LAKE_PATH` and `DEFAULT_LEAN_WORKSPACE` in `lean_compiler/repl_scheduler.py`.

## 2. Quick Start
You can directly use [Huggingface's Transformers](https://github.com/huggingface/transformers) for model inference.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
torch.manual_seed(30)

model_id = "Goedel-LM/Goedel-Prover-V2-32B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)


formal_statement = """
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat


theorem square_equation_solution {x y : ℝ} (h : x^2 + y^2 = 2*x - 4*y - 5) : x + y = -1 := by
  sorry
""".strip()

prompt = """
Complete the following Lean 4 code:

```lean4
{}```

Before producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan outlining the main proof steps and strategies.
The plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the construction of the final formal proof.
""".strip()

chat = [
  {"role": "user", "content": prompt.format(formal_statement)},
]

inputs = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)

import time
start = time.time()
outputs = model.generate(inputs, max_new_tokens=32768)
print(tokenizer.batch_decode(outputs))
print(time.time() - start)
```
