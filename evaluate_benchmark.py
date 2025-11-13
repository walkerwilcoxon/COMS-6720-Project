from llm_utils import load_model, produce_proof, get_f2f_problems

problems = get_f2f_problems()

# Select LLM. d: deepseek, g: goedel
llm = "g"

llm_ids = {
    "d": "deepseek-ai/DeepSeek-Prover-V2-7B",
    "g": "Goedel-LM/Goedel-Prover-V2-8B"
}

llm_id = llm_ids[llm]

model, tokenizer = load_model(llm_id)

with open("solutions.txt", "a") as f:
    for problem in problems["test"]:
        output, dt = produce_proof(model, tokenizer, problem)

        print([output, dt])

        f.write(output + "\n\n\n")
        f.flush()
