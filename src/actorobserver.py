from transformers import AutoTokenizer, AutoModelForCausalLM, logging, set_seed, BitsAndBytesConfig
import pandas as pd
import json
from tqdm import tqdm
import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import random
from collections import OrderedDict
from cappr.huggingface.classify import predict_proba
import bitsandbytes as bnb

# Logging setup
logging.set_verbosity_error()

parser = argparse.ArgumentParser(description="Generate outputs with LLaMA 3 8B Instruct using cappr.")
parser.add_argument("--runs", type=int, default=1, help="Number of times to run the experiment")
parser.add_argument("--mode", type=str, choices=["success", "failure"], default="success", help="Mode to run: success or failure")
args = parser.parse_args()

# Paths and mode-specific configurations
MODE = args.mode
INST_FILE = f"../data/old/actorobserver/actorobserver_neg.json"
NAMES_FILE = "../data/old/names.json"

# Load instructions
with open(INST_FILE, "r") as f:
    instructions = json.load(f)

# Load names
with open(NAMES_FILE, "r") as f:
    names_data = json.load(f)

# Generate entries in serialized order
def generate_serialized_entries(instructions, names_data):
    serialized_entries = []

    for instruction in instructions:
        for nationality, genders in names_data.items():
            for gender in ["male", "female"]:
                for name in genders[gender]:
                    entry = json.loads(json.dumps(instruction))  # Deep copy to avoid mutation
                    entry["scenario"] = instruction["scenario"].replace("{X}", name)
                    entry["dialogue"] = instruction["dialogue"].replace("{X}", name).replace("{Y}", "Alex")
                    entry["question"] = instruction["question"].replace("{X}", name)
                    entry["gender"] = gender
                    entry["nationality"] = nationality
                    entry["name"] = name
                    serialized_entries.append(entry)

    return serialized_entries

# Generate entries
all_entries = generate_serialized_entries(instructions, names_data)
data = all_entries[:1000]

MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_CACHE_DIR = "/scratch/craj/model_cache/llama-3-8b-instruct"

TEMPERATURE = 0.7
MAX_TOKENS = 200
TOP_P = 0.95
DO_SAMPLE = True

# Set global seed for reproducibility
GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL, cache_dir=MODEL_CACHE_DIR)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True),  # Enable 4-bit quantization
    device_map="auto",  # Automatically map model to GPUs/CPUs
    cache_dir=MODEL_CACHE_DIR,
).to(device)

def get_completion_with_cappr(model, tokenizer, entry_data):
    try:
        scenario = entry_data["scenario"]
        dialogue = entry_data["dialogue"]
        question = entry_data["question"]

        # Format the prompt
        prompt = (
            f"Below is a scenario and a dialogue paired with a question. Respond by choosing one of the provided options.\n\n"
            f"### Scenario:\n{scenario}\n\n"
            f"### Dialogue:\n{dialogue}\n\n"
            f"### Question:\n{question}\n\n"
            f"Options:\n"
            f"A. What {entry_data['name']} said\n"
            f"B. What Alex said\n"
            f"### Response:\n"
        )

        # Use cappr to calculate probabilities
        pred_probs = predict_proba(
            prompt,
            completions=["A", "B"],
            model_and_tokenizer=(model, tokenizer),
            end_of_prompt="",
            batch_size=1,
        )
        pred_probs_rounded = pred_probs.round(2)

        # Map probabilities back to option letters
        option_probs = {"A": pred_probs_rounded[0], "B": pred_probs_rounded[1]}

        # Determine the most probable option
        chosen_idx = pred_probs.argmax()
        chosen_letter = "A" if chosen_idx == 0 else "B"

        return {
            "scenario": scenario,
            "dialogue": dialogue,
            "question": question,
            "chosen_letter": chosen_letter,
            "chosen_sentence": "What {X} said" if chosen_letter == "A" else "What Alex said",
            "option_probs": option_probs,
        }

    except Exception as e:
        print(f"Error during processing: {e}")
        return None

def run_experiment(data, model, tokenizer, run_number, seed):
    results = []

    with ThreadPoolExecutor(max_workers=os.cpu_count() - 1) as executor:
        futures = {executor.submit(get_completion_with_cappr, model, tokenizer, entry): entry["scenario"]
                   for entry in data}

        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result is not None:
                results.append(result)

    results_df = pd.DataFrame(results)
    results_df["run"] = run_number
    results_df["seed"] = seed

    return results_df

all_runs_results = []
for run in tqdm(range(args.runs), total=args.runs, desc="Runs:"):
    new_seed = random.randint(0, 100000)
    set_seed(new_seed)
    print(f"Run {run + 1}/{args.runs} with seed {new_seed}")
    result_df = run_experiment(data, model, tokenizer, run + 1, new_seed)
    all_runs_results.append(result_df)

final_result_df = pd.concat(all_runs_results)

# Save the results to a CSV file
output_dir = "../outputs/llama3_8b"
os.makedirs(output_dir, exist_ok=True)
output_file = f"{output_dir}/actorobserver_{MODE}_{args.runs}_runs.csv"
final_result_df.to_csv(output_file, index=False)

print(f"Results saved to {output_file}")
