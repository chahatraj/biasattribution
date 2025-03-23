import cohere
import json
import argparse
import os
import random
import traceback
from tqdm import tqdm
import pandas as pd
import time

random.seed(42)

# TODO@Mahika: Change to 35
BATCH_SIZE = 1  # Max API calls per minute

# Initialize Cohere API
co = cohere.ClientV2("ydOmzRO8YiDre0qAmzV6WlZCiYxSHHFryib8Doq0")  # Replace with your actual API key

# Logging setup
parser = argparse.ArgumentParser(description="Test Cohere API response.")
parser.add_argument("--runs", type=int, default=1, help="Number of times to run the experiment")
parser.add_argument("--mode", type=str, choices=["success", "failure"], default="success", help="Mode to run: success or failure")
args = parser.parse_args()

MODE = args.mode
PROMPT_FILE = f"../data/mcq/{MODE}.json"
NAMES_FILE = "../data/names.json"

if MODE == "success":
    option_keys = ["opt1_higheffort", "opt2_highability", "opt3_easytask", "opt4_goodluck"]
else:
    option_keys = ["opt1_loweffort", "opt2_lowability", "opt3_difficulttask", "opt4_badluck"]

# Load prompt data
with open(PROMPT_FILE, "r") as f:
    prompt_data = json.load(f)

# Load names
with open(NAMES_FILE, "r") as f:
    names_data = json.load(f)

# Allowed countries and subregions
allowed_countries = {
    "American Names", "African Names", "Middle-Eastern Names", "Latin American Names"}
allowed_subregions = {"German", "French", "Italian", "Greek", "South Asian", "Chinese", "Japanese", "Southeast Asian"}

# Extract 3 male + 3 female names from each allowed country/subregion
def extract_names():
    selected_names = []
    for region, content in names_data.items():
        if region in allowed_countries:
            for gender in ["Male", "Female"]:
                name_list = content.get(gender, [])
                sampled = random.sample(name_list, min(3, len(name_list)))
                selected_names.extend([(region, gender.lower(), name) for name in sampled])

        elif region == "European Names":
            for subregion, genders in content.items():
                if subregion in allowed_subregions:
                    for gender in ["Male", "Female"]:
                        name_list = genders.get(gender, [])
                        sampled = random.sample(name_list, min(3, len(name_list)))
                        selected_names.extend([(subregion, gender.lower(), name) for name in sampled])

        elif region == "Asian Names":
            for subregion, genders in content.items():
                if subregion in allowed_subregions:
                    for gender in ["Male", "Female"]:
                        name_list = genders.get(gender, [])
                        sampled = random.sample(name_list, min(3, len(name_list)))
                        selected_names.extend([(subregion, gender.lower(), name) for name in sampled])
    return selected_names


all_names = extract_names()

# Only use education, workplace, healthcare + short + first 2 sets
allowed_domains = {"education", "workplace", "healthcare"}
serialized_entries = []

output_dir = "../outputs/cohere_aya_exp_32b/mahika"
os.makedirs(output_dir, exist_ok=True)
partial_file = f"{output_dir}/mcq_{MODE}_{args.runs}_runs_partial.csv"

# Load previous results if partial file exists
if os.path.exists(partial_file):
    print(f"🔁 Resuming from partial file: {partial_file}")
    prev_df = pd.read_csv(partial_file)
    results = prev_df.to_dict(orient="records")
    processed_keys = {(row["set_id"], row["name"]) for row in results}
else:
    results = []
    processed_keys = set()


for gender in ["male", "female"]:
    if gender not in prompt_data:
        continue
    for domain, lengths in prompt_data[gender].items():
        if domain not in allowed_domains:
            continue
        if "short" not in lengths:
            continue
        for prompt_idx, prompt in enumerate(lengths["short"][:2]):  # First 2 prompts
            for nat, g, name in all_names:
                if g != gender:
                    continue
                entry = json.loads(json.dumps(prompt))
                entry["instruction"] = prompt["instruction"].replace("{X}", name)
                entry["initial_prompt"] = prompt["initial_prompt"].replace("{X}", name)
                entry["gender"] = gender
                entry["domain"] = domain
                entry["length"] = "short"
                entry["nationality"] = nat
                entry["name"] = name
                entry["shuffled_options"] = random.sample(option_keys, len(option_keys))
                entry["set_id"] = f"{gender}_{domain}_short_set{prompt_idx}"  # 👈 Add this line
                serialized_entries.append(entry)


data = serialized_entries
# TODO@Mahika: Comment next line
data = serialized_entries[:2]

# TODO@Mahika: Change to 500
MAX_TO_PROCESS = 2  # 👈 Process only this many new entries

# Filter out already processed ones
unprocessed_data = [
    entry for entry in serialized_entries
    if (entry["set_id"], entry["name"]) not in processed_keys
]

# Pick only the next N unprocessed entries
data = unprocessed_data[:MAX_TO_PROCESS]


# Save intermediate data before sending to API
intermediate_df = pd.DataFrame(data)
intermediate_output_dir = "../outputs/intermediate/mahika"
os.makedirs(intermediate_output_dir, exist_ok=True)
intermediate_file = f"{intermediate_output_dir}/mcq_{MODE}_intermediate_entries.csv"
# intermediate_df.to_csv(intermediate_file, index=False)

# Check if intermediate file already exists
write_header = not os.path.exists(intermediate_file)
# Append to the intermediate file
intermediate_df.to_csv(intermediate_file, mode='a', header=write_header, index=False)
print(f"📝 Intermediate file saved to: {intermediate_file}")

# results = []

def test_cohere_prompt(entry_data, run_number, seed):
    try:
        instruction = entry_data.get("instruction", "No instruction provided")
        initial_prompt = entry_data.get("initial_prompt", "No initial prompt provided")
        shuffled_option_keys = entry_data.get("shuffled_options")

        if shuffled_option_keys is None:
            print("\n❌ ERROR: 'shuffled_options' key is missing.")
            return None

        options = [entry_data.get(key, f"Missing: {key}") for key in shuffled_option_keys]

        prompt = (
            f"Below is an instruction and a statement paired with multiple-choice options.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Statement:\n{initial_prompt}\n\n"
            f"Options:\n"
            f"A. {options[0]}\n"
            f"B. {options[1]}\n"
            f"C. {options[2]}\n"
            f"D. {options[3]}\n\n"
            f"ONLY respond with a single letter (A, B, C, or D)."
        )

        response = co.chat(
            model="c4ai-aya-expanse-32b",
            messages=[{"role": "user", "content": prompt}],
        )

        response_text = response.message.content[0].text.strip() if response.message and response.message.content else "No response"


        results.append({
            "run": run_number,
            "seed": seed,
            "set_id": entry_data.get("set_id"),
            "instruction": instruction,
            "initial_prompt": initial_prompt,
            "chosen_letter": response_text,
            "gender": entry_data["gender"],
            "domain": entry_data["domain"],
            "length": entry_data["length"],
            "nationality": entry_data["nationality"],
            "name": entry_data["name"],
            # "shuffled_options": entry_data.get("shuffled_options"),
            "opt_A": options[0],
            "opt_B": options[1],
            "opt_C": options[2],
            "opt_D": options[3],
            "original_option_mapping": {chr(65 + i): shuffled_option_keys[i] for i in range(len(shuffled_option_keys))}
        })

    except Exception as e:
        print("\n❌ ERROR in test_cohere_prompt")
        traceback.print_exc()

# Run the experiment
for run in tqdm(range(args.runs), total=args.runs, desc="Runs"):
    new_seed = random.randint(0, 100000)
    print(f"Run {run + 1}/{args.runs} | Seed: {new_seed}")
    for i, entry in enumerate(data):
        # Skip if already processed
        if (entry["set_id"], entry["name"]) in processed_keys:
            continue
        test_cohere_prompt(entry, run + 1, new_seed)
        # if (i + 1) % BATCH_SIZE == 0:
        #     print("\n⏳ Rate limit hit. Waiting for 60 seconds...\n")
        #     time.sleep(60)
        if (i + 1) % BATCH_SIZE == 0:
            # Save batch results
            partial_df = pd.DataFrame(results)
            output_dir = "../outputs/cohere_aya_exp_32b/mahika"
            os.makedirs(output_dir, exist_ok=True)
            partial_file = f"{output_dir}/mcq_{MODE}_{args.runs}_runs_partial.csv"
            partial_df.to_csv(partial_file, index=False)
            print(f"💾 Saved partial results after {i + 1} entries to: {partial_file}")
            print("\n⏳ Rate limit hit. Waiting for 60 seconds...\n")
            time.sleep(60)


# Save results
final_result_df = pd.DataFrame(results)
output_file = f"{output_dir}/mcq_{MODE}_{args.runs}_runs.csv"
final_result_df.to_csv(output_file, index=False)
print(f"✅ Results saved to: {output_file}")
