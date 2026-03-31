import argparse
import glob
import json
import datetime
import os


def main():
    parser = argparse.ArgumentParser(description="Extract results from JSON files.")
    parser.add_argument("--results-dir", type=str, help="Path to the result directory.")
    parser.add_argument("--model-id", type=str, help="Model identifier.")
    parser.add_argument("--output-file", type=str, help="Path to the output JSON file.")
    args = parser.parse_args()

    subdir = args.model_id.replace("/", "__")
    today = datetime.date.today().strftime("%Y-%m-%d")

    # Select the most recent result file for the given model and date
    result_file = sorted(glob.glob(os.path.join(args.results_dir, subdir, f"results_{today}T*.json")))
    if len(result_file) == 0:
        print(f"No result file found for model {args.model_id} on date {today}.")
        return
    result_file = result_file[-1]  # Get the most recent file
    print(f"Extracting results from: {result_file}")
    results = {}

    with open(result_file, "r") as reader:
        results = json.load(reader)["results"]

    # Extract only the relevant metrics (e.g., accuracy) for each task
    extracted_results = {}
    value_results = []
    for task, metrics in results.items():
        if "mmlu" in task and task != "mmlu":
            continue  # Skip non-MMLU tasks that contain "mmlu" in their name
        if ("longbench" in task) and (task not in ["longbench", "longbench_e"]):
            continue  # Skip non-longbench tasks that contain "longbench" in their name

        if "acc,none" in metrics:
            extracted_results[task] = {"accuracy": metrics["acc,none"] * 100}  # Convert to percentage
            value_results.append(metrics["acc,none"] * 100)
        
        # For gsm8k, also extract the exact match score if available
        if "gsm8k" in task and "exact_match,strict-match" in metrics:
            extracted_results[task]=  {"exact_match": metrics["exact_match,strict-match"] * 100}  # Convert to percentage
            value_results.append(metrics["exact_match,strict-match"] * 100)

        if "longbench" in task and "score,none" in metrics:
            extracted_results[task] = {"score": metrics["score,none"] * 100}  # Convert to percentage
            value_results.append(metrics["score,none"] * 100)

    extracted_results["average"] = {"score": sum(value_results) / len(value_results) if value_results else None}

    # Append extracted results in Markdown
    if not os.path.exists(args.output_file):
        existing_data = "# Summary of Results\n\n"
        existing_data += "| Model | " + " | ".join(extracted_results.keys()) + " |\n"
        existing_data += "|---|" + "|".join(["---"] * len(extracted_results)) + "|\n"
        existing_data += "| ID | " + " | ".join([list(x)[0] for x in extracted_results.values()]) + " |\n"
    else:
        with open(args.output_file, "r") as reader:
            existing_data = reader.read()
        existing_data = existing_data.rstrip() + "\n"

    result_line = f"| {args.model_id} (Date: {today}) | " + " | ".join(["{:.02f}".format(list(x.values())[0]) for x in extracted_results.values()]) + " |\n"
    existing_data += result_line
    existing_data += "\n"
    
    print(existing_data)
    # return

    with open(args.output_file, "w") as writer:
        writer.write(existing_data)


if __name__ == "__main__":
    main()
