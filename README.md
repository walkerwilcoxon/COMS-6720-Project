
Install mathlib4

HTTPS:
```bash
git clone https://github.com/walkerwilcoxon/mathlib4.git
```
or
SSH: (requires ssh)
```bash
git clone git@github.com:walkerwilcoxon/mathlib4.git
```

## Generating Proofs

### Pass 1 (Initial Generation)
Run the baseline prover to generate the first set of proofs.
```bash
python generate_proofs.py
```

### Pass 2 (Retry Logic)
Use `generate_pass2.py` to retry problems that failed in Pass 1.

**1. Standard Retry (Pass@2)**
Re-runs the standard benchmark prover on failed problems.
```bash
python generate_pass2.py
```

**2. Feedback Retry**
Uses the error messages from Pass 1 to prompt the prover to fix its own mistake.
```bash
python generate_pass2.py --feedback
```

**3. Commander Guidance**
Uses a separate "Commander" agent (DeepSeek) to analyze the error and guide the prover.
```bash
python generate_pass2.py --commander
```

**LLM Selection & Naming Convention**
By default, the Goedel model runs the Pass 2 attempts. You can switch to DeepSeek by passing `--model Deepseek`.

The script automatically sets the input/output filenames based on the model:

You can override these defaults:
```bash
python generate_pass2.py --model Deepseek --input-file my_custom_input.txt --output-name my_output_base
```

Each strategy writes its own output file in `output/`, so the results remain separated.
