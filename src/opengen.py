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

# Logging setup
logging.set_verbosity_error()

parser = argparse.ArgumentParser(description="Generate outputs with LLaMA 3 8B Instruct.")
parser.add_argument("--runs", type=int, default=1, help="Number of times to run the experiment")
parser.add_argument("--mode", type=str, choices=["success", "failure"], default="success", help="Mode to run: success or failure")
args = parser.parse_args()

# Paths and mode-specific configurations
MODE = args.mode
INST_MALE_FILE = f"../data/old/storytelling/inst_male_{MODE}.json"
INST_FEMALE_FILE = f"../data/old/storytelling/inst_female_{MODE}.json"
NAMES_FILE = "../data/old/names.json"

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
data = all_entries[:10]  # Limit to the first item

MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_CACHE_DIR = "/scratch/craj/model_cache/llama-3-8b-instruct"

TEMPERATURE = 0.7
MAX_TOKENS = 500
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

def get_model_response(model, tokenizer, entry_data):
    try:
        instruction = entry_data["instruction"]
        initial_prompt = entry_data["initial_prompt"]

        # # Format the prompt
        # prompt = (
        #     f"Below is an incomplete story. Continue the story after the provided context.\n\n"
        #     # f"### Instruction:\n{instruction}\n\n"
        #     f"### Statement:\n{initial_prompt}\n\n"
        #     f"### Response:\n"
        # )

        # Minimal prompt formatting
        prompt = (
            # f"{instruction}\n\n"
            f"{initial_prompt}\n\n"
            f"### Response:\n"
        )

        # Generate response
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
        outputs = model.generate(
            **inputs,
            max_length=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=DO_SAMPLE
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract the content after '### Response'
        if "### Response:" in response:
            response = response.split("### Response:")[1].strip()

        # Remove any additional sections starting with '###'
        response = response.split("###")[0].strip()
        
        return response

    except Exception as e:
        print(f"Error during processing: {e}")
        return None

def run_experiment(data, model, tokenizer, run_number, seed):
    results = []

    processed_entries = []
    for idx, entry in enumerate(data):
        processed_entries.append({
            "index": idx,
            "instruction": entry["instruction"],
            "initial_prompt": entry["initial_prompt"],
            "gender": entry["gender"],
            "nationality": entry["nationality"],
            "name": entry["name"],
        })

    with ThreadPoolExecutor(max_workers=os.cpu_count() - 1) as executor:
        futures = {executor.submit(get_model_response, model, tokenizer, entry_data): entry_data["index"]
                   for entry_data in processed_entries}

        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result is not None:
                results.append((futures[future], result))

    # Sort results by the original index
    results.sort(key=lambda x: x[0])
    sorted_results = [res[1] for res in results]

    results_df = pd.DataFrame(processed_entries)
    results_df["run"] = run_number
    results_df["seed"] = seed
    results_df["response"] = sorted_results

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
output_file = f"{output_dir}/opengen_{MODE}_{args.runs}_runs.csv"
final_result_df.to_csv(output_file, index=False)

print(f"Results saved to {output_file}")
