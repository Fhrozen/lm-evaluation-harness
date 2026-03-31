#!/usr/bin/env bash

set -euo pipefail

# Activate the conda environment
source ../.venv/bin/activate

if [ "$#" -gt 2 ]; then
    echo "Usage: $0 <pretrained_model_path> <type_model>"
    exit 1
elif [ "$#" -eq 2 ]; then
    pretrained=$1
    type_model=$2
elif [ "$#" -eq 1 ]; then
    pretrained=$1
    type_model="base"
else
    echo "Usage: $0 <pretrained_model_path> [type_model]"
    exit 1
fi

# Original Base tasks: hellaswag,piqa,winogrande,arc_challenge,mmlu,truthfulqa_mc2,gsm8k
# Tasks from SmolLM2
base_tasks="hellaswag,arc_challenge,arc_easy,piqa,mmlu,commonsense_qa,triviaqa,winogrande,openbookqa,gsm8k"

# Tasks from https://huggingface.co/blog/codelion/optimal-model-architecture
base_tasks+=",truthfulqa_mc2"

# OBQA: openbookqa
# Tasks from MobileLLM-Pro paper
base_tasks+=",boolq,social_iqa,nq_open,longbench,longbench_e"

if [ "$type_model" = "base" ]; then
    tasks=${base_tasks}
elif [ "$type_model" = "instruct" ]; then
    echo "ERROR: Not implemented yet for instruct models. Please implement the tasks for instruct models in the script."
    exit 1
    tasks="${base_tasks},gsm8k,truthfulqa_mc2"
else
    echo "Invalid type_model: ${type_model}. Valid options are: base, instruct."
    exit 1
fi

# Run the Python script
accelerate launch \
    --num_machines 1 \
    -m lm_eval \
        --model hf \
        --model_args pretrained=${pretrained},dtype="bfloat16",trust_remote_code=True \
        --tasks ${tasks} \
        --batch_size auto:8 \
        --output_path results || exit 1

# Process the results
python ./local/extract_results.py --results-dir results \
    --model-id ${pretrained} \
    --output-file results/summarized_${type_model}_results.md || exit 1

echo "Done"
