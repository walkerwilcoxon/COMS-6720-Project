#!/bin/bash

set -e

python generate_proofs_stage1.py --model="Deepseek" --problem-set="temp"
python verify_stage1_proofs.py --model="Deepseek" --problem-set="temp"
python generate_proofs_stage2.py --proof-strategy="feedback" --model="Deepseek" --problem-set="temp"