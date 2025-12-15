from subprocess import Popen, PIPE, STDOUT
from pathlib import Path
import os
import subprocess
import pexpect
import sys
import json
import traceback
import multiprocessing as mp

import tomli
import tomli_w

from llm_utils import verify_proof, ParallelExecutor

# Either a filepath of the proof file or "output" to verify all files inside of the "output" directory
VERIFY_PATH = "output/benchmark_Deepseek_test_solutions.txt"

def proof_verifier_worker(worker_id, input_q, output_q):
    print(f"Starting worker {worker_id}")
    try:
        while True:
            input = input_q.get()

            if input == None:
                print(f"Terminating worker {worker_id}")
                return

            proof = input["proof"]

            print(f"Verifying Proof {input["name"]} ({input["iteration"]})")

            verified = True

            if proof == "":
                proof_verification = {
                    "verified": False,
                    "error": "Proof cut off",
                    "feedback": [],
                }
            else:
                proof_verification = verify_proof(input["proof"])

            output = {}

            output["name"] = input["name"]
            output["iteration"] = input["iteration"]
            output["time"] = input["time"]
            output["verified"] = proof_verification["verified"]
            if proof_verification["error"]:
                output["error"] = proof_verification["error"]
            if proof_verification["feedback"]:
                output["feedback"] = proof_verification["feedback"]
            output["proof"] = input["proof"]
            output["outline"] = input["outline"]

            output_q.put(output)
    except KeyboardInterrupt as e:
        pass



def verify_file(proof_filepath):
    try:
        with open(proof_filepath, "rb") as f:
            proofs = tomli.load(f)["proof"]
            # Names of proofs that have already been proven
            output_json = proofs
    except FileNotFoundError:
        output_json = []

    # num_workers = number of cpus
    num_workers = int(subprocess.run(["nproc"], capture_output=True, text=True).stdout.strip())

    executor = ParallelExecutor(num_workers, worker=proof_verifier_worker)

    for i, proof in enumerate(proofs):
        executor.submit(i % num_workers, proof)

    num_proofs = len(proofs)

    for i in range(len(proofs)):
        output = executor.gather()

        for j, proof in enumerate(output_json):
            if output["name"] == proof["name"]:
                output_json[j] = output
                break

        print(f"Verified proof {output["name"]} ({i + 1}/{num_proofs})")

        with open(proof_filepath, "wb") as f:
            tomli_w.dump({"proof": output_json}, f, multiline_strings=True)

    executor.shutdown()

def main():
    path = Path(VERIFY_PATH)
    if not path.exists():
        raise ValueError(f"File not found: {path}")
    if path.is_dir():
        files = [f for f in path.iterdir()]
        for file in files:
            verify_file(file)
    else:
        verify_file(VERIFY_PATH)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        pass