from llm_utils import load_model, produce_proof

import os

def main():
    print("Type 'reset' to start a new problem, or 'feedback' to refine the last proof.")

    while True:
        model_name = input("Select model: 'g' for Goedel (default) or 'd' for deepseek: ")

        if model_name == "g" or model_name == "":
            model_id = "Goedel-LM/Goedel-Prover-V2-8B"
            break
        elif model_name == "d":
            model_id = "deepseek-ai/DeepSeek-Prover-V2-7B"
            break
        else:
            print(f"Unknown model name: {model_name}")

    model, tokenizer = load_model(model_id)

    last_problem = None
    last_proof = None

    while True:
        user_choice = input("Enter mode ('reset', 'feedback', or 'quit'): ").strip().lower()

        if user_choice == "quit":
            break

        elif user_choice == "reset":
            input("Put the problem into the 'chat_input.txt' file. Then press ENTER.")
            if not os.path.exists("chat_input.txt"):
                print("File 'chat_input.txt' not found.")
                continue

            with open("chat_input.txt") as f:
                problem = f.read().strip()
                if not problem:
                    print("No problem found in file.")
                    continue

            print("Generating proof...")
            proof, dt = produce_proof(model, tokenizer, problem)
            last_problem, last_proof = problem, proof

            print("\n=== Proof Output ===")
            print(proof)
            print(f"\n(Took {dt:.0f} seconds)\n")

        elif user_choice == "feedback":
            if last_proof is None:
                print("No previous proof available.")
                continue

            print("Enter your feedback in 'chat_input.txt'. Then press ENTER")

            with open("chat_input.txt") as f:
                feedback = f.read().strip()

            if not feedback:
                print("No feedback found.")
                continue

            print("Revising proof...")
            corrected_proof, dt = produce_proof(
                model,
                tokenizer,
                last_problem,
                instructions=f"Revise the following proof based on user feedback:\n\n{feedback}\n\nOriginal proof:\n{last_proof}"
            )

            last_proof = corrected_proof
            print("\n=== Revised Proof ===")
            print(corrected_proof)
            print(f"\n(Took {dt:.0f} seconds)\n")

        else:
            print("Invalid command. Please type 'reset', 'feedback', or 'quit'.")


if __name__ == "__main__":
    main()
