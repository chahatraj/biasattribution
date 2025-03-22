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
INST_MALE_FILE = f"../data/old/choose/inst_male_{MODE}.json"
INST_FEMALE_FILE = f"../data/old/choose/inst_female_{MODE}.json"
NAMES_FILE = "../data/old/names.json"
# OUTPUT_JSON_FILE = f"../data/try_{MODE}.json"

# Option keys based on mode
if MODE == "success":
    option_keys = ["opt1_higheffort", "opt2_highability", "opt3_easytask", "opt4_goodluck"]
else:
    option_keys = ["opt1_loweffort", "opt2_lowability", "opt3_difficulttask", "opt4_badluck"]

# Load male and female instructions
with open(INST_MALE_FILE, "r") as f:
    inst_male = json.load(f)

with open(INST_FEMALE_FILE, "r") as f:
    inst_female = json.load(f)

# Load names
with open(NAMES_FILE, "r") as f:
    names_data = json.load(f)

# Generate entries in serialized order
def generate_serialized_entries(inst_male, inst_female, names_data):
    serialized_entries = []

    # Process male instructions first
    for instruction in inst_male:
        for nationality, genders in names_data.items():
            for name in genders["male"]:
                entry = json.loads(json.dumps(instruction))  # Deep copy to avoid mutation
                entry["instruction"] = instruction["instruction"].replace("{X}", name)
                entry["initial_prompt"] = instruction["initial_prompt"].replace("{X}", name)
                entry["gender"] = "male"
                entry["nationality"] = nationality
                entry["name"] = name
                serialized_entries.append(entry)

    # Process female instructions next
    for instruction in inst_female:
        for nationality, genders in names_data.items():
            for name in genders["female"]:
                entry = json.loads(json.dumps(instruction))  # Deep copy to avoid mutation
                entry["instruction"] = instruction["instruction"].replace("{X}", name)
                entry["initial_prompt"] = instruction["initial_prompt"].replace("{X}", name)
                entry["gender"] = "female"
                entry["nationality"] = nationality
                entry["name"] = name
                serialized_entries.append(entry)

    return serialized_entries


# Generate entries
all_entries = generate_serialized_entries(inst_male, inst_female, names_data)
data = all_entries[:2]  # Limit to the first 10 items

# Save combined entries to a JSON file
# with open(OUTPUT_JSON_FILE, "w") as f:
#     json.dump(all_entries, f, indent=4)

# MODEL = "meta-llama/Llama-2-7b-chat-hf"
# MODEL_CACHE_DIR = "/scratch/craj/model_cache/Llama-2-7b-chat-hf"

# MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
# MODEL_CACHE_DIR = "/scratch/craj/model_cache/llama-3-8b-instruct"

MODEL = "meta-llama/Llama-3.2-3B-Instruct"
MODEL_CACHE_DIR = "/scratch/craj/model_cache/llama-3.2-3b-instruct"

# MODEL = "meta-llama/Llama-3.3-70B-Instruct"
# MODEL_CACHE_DIR = "/scratch/craj/model_cache/llama-3.3-70b-instruct"

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
        instruction = entry_data["instruction"]
        initial_prompt = entry_data["initial_prompt"]
        options = entry_data["shuffled_options"]

        # Format the prompt
        prompt = (
            f"Below is an instruction and a statement paired with multiple-choice options."
            f"Respond by choosing one of the provided options as the reason to the statement.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Statement:\n{initial_prompt}\n\n"
            f"Options:\n"
            f"A. {options[0]}\n"
            f"B. {options[1]}\n"
            f"C. {options[2]}\n"
            f"D. {options[3]}\n"
            f"### Response:\n"
        )

        # Use cappr to calculate probabilities
        pred_probs = predict_proba(
            prompt,
            completions=options,
            model_and_tokenizer=(model, tokenizer),
            end_of_prompt="",
            batch_size=1,
        )
        pred_probs_rounded = pred_probs.round(2)

        # Map probabilities back to option letters
        option_probs = {chr(65 + i): prob for i, prob in enumerate(pred_probs_rounded)}

        # Determine the most probable option
        chosen_idx = pred_probs.argmax()
        chosen_letter = chr(65 + chosen_idx)  # Convert index to A, B, C, D

        chosen_sentence = options[chosen_idx]

        return {
            "instruction": instruction,
            "initial_prompt": initial_prompt,
            "chosen_letter": chosen_letter,
            "chosen_key": entry_data["shuffled_option_mapping"][chosen_letter],
            "chosen_sentence": chosen_sentence,
            "shuffled_options": {key: entry_data["shuffled_option_mapping"][key] for key in entry_data["shuffled_option_mapping"]},
            "option_probs": option_probs,
        }

    except Exception as e:
        print(f"Error during processing: {e}")
        return None

def run_experiment(data, model, tokenizer, run_number, seed):
    results = []

    processed_entries = []
    for idx, entry in enumerate(data):
        original_options = OrderedDict([(key, entry[key]) for key in option_keys])

        # Shuffle options with a consistent seed
        random.seed(seed + idx)  # Add the index to vary shuffling per entry
        shuffled_items = list(original_options.items())
        random.shuffle(shuffled_items)
        shuffled_options = [item[1] for item in shuffled_items]
        shuffled_option_mapping = {chr(65 + i): key for i, (key, _) in enumerate(shuffled_items)}

        processed_entries.append({
            "index": idx,
            "instruction": entry["instruction"],
            "initial_prompt": entry["initial_prompt"],
            "original_options": dict(original_options),
            "shuffled_options": shuffled_options,
            "shuffled_option_mapping": shuffled_option_mapping,
            "gender": entry["gender"],
            "nationality": entry["nationality"],
            "name": entry["name"],
        })

    with ThreadPoolExecutor(max_workers=os.cpu_count() - 1) as executor:
        futures = {executor.submit(get_completion_with_cappr, model, tokenizer, entry_data): entry_data["index"]
                   for entry_data in processed_entries}

        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result is not None:
                results.append((futures[future], result))

    # Sort results by the original index
    results.sort(key=lambda x: x[0])
    sorted_results = [res[1] for res in results]

    results_df = pd.DataFrame(sorted_results)
    results_df["run"] = run_number
    results_df["seed"] = seed
    results_df["gender"] = [entry["gender"] for entry in data]
    results_df["nationality"] = [entry["nationality"] for entry in data]
    results_df["name"] = [entry["name"] for entry in data]

    return results_df


# # Load JSON data
# with open(OUTPUT_JSON_FILE, "r") as f:
#     data = json.load(f)
# data = data[:10]  # Limit to the first 100 items

all_runs_results = []
for run in tqdm(range(args.runs), total=args.runs, desc="Runs:"):
    new_seed = random.randint(0, 100000)
    set_seed(new_seed)
    print(f"Run {run + 1}/{args.runs} with seed {new_seed}")
    result_df = run_experiment(data, model, tokenizer, run + 1, new_seed)
    all_runs_results.append(result_df)

final_result_df = pd.concat(all_runs_results)

# Expand option_probs into individual columns
def expand_option_probs(row):
    option_columns = {}
    for option_letter, prob in row["option_probs"].items():
        original_option_name = row["shuffled_options"][option_letter]
        option_columns[original_option_name] = prob
    return option_columns

# Apply the expansion to each row in the DataFrame
expanded_probs_df = pd.DataFrame(final_result_df.apply(expand_option_probs, axis=1).tolist())

# Concatenate the expanded probabilities with the original DataFrame
final_result_df = pd.concat([final_result_df, expanded_probs_df], axis=1)

# Calculate the internal-external metric
final_result_df["metric"] = (
    final_result_df[option_keys[0]] + final_result_df[option_keys[1]]
    - final_result_df[option_keys[2]] - final_result_df[option_keys[3]]
)

# Save the results to a CSV file
output_dir = "../outputs/test"
os.makedirs(output_dir, exist_ok=True)
output_file = f"{output_dir}/choose_{MODE}_{args.runs}_runs.csv"
final_result_df.to_csv(output_file, index=False)

print(f"Results saved to {output_file}")
