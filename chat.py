from extract_problems import get_f2f_problems
from llm_utils import load_model, produce_proof


problems = get_f2f_problems()

model, tokenizer = load_model("deepseek-ai/DeepSeek-Prover-V2-7B")

with open("solutions.txt", "a", encoding="utf-8") as f:
    for problem in problems["test"]:
        problem += "\n    sorry\nend"
        output, dt = produce_proof(model, tokenizer, problem)

        f.write(output + "\n\n\n")
        f.flush()
        break
