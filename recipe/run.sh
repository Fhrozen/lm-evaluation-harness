#!/usr/bin/env bash

set -euo pipefail

# Activate the conda environment
source ../.venv/bin/activate

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <pretrained_model_path>"
    exit 1
fi

pretrained=$1

# Run the Python script
accelerate launch -m lm_eval --model hf \
    --model_args pretrained=${pretrained},dtype="bfloat16",trust_remote_code=True \
    --tasks hellaswag,piqa,winogrande,arc_challenge,mmlu,truthfulqa_mc2,gsm8k \
    --batch_size auto:8 \
    --output_path results \
