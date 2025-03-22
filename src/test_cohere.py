import cohere
import json
import argparse
import os
import random
import traceback
from collections import OrderedDict
from tqdm import tqdm
import pandas as pd

# Initialize Cohere API
co = cohere.ClientV2("ydOmzRO8YiDre0qAmzV6WlZCiYxSHHFryib8Doq0")  # Replace with your actual API key

# Logging setup
parser = argparse.ArgumentParser(description="Test Cohere API response.")
parser.add_argument("--runs", type=int, default=1, help="Number of times to run the experiment")
parser.add_argument("--mode", type=str, choices=["success", "failure"], default="success", help="Mode to run: success or failure")
args = parser.parse_args()

# Paths and mode-specific configurations
MODE = args.mode
INST_MALE_FILE = f"../data/old/choose/inst_male_{MODE}.json"
INST_FEMALE_FILE = f"../data/old/choose/inst_female_{MODE}.json"
NAMES_FILE = "../data/old/names.json"

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

    for instruction in inst_male:
        for nationality, genders in names_data.items():
            for name in genders["male"]:
                entry = json.loads(json.dumps(instruction))  # Deep copy to avoid mutation
                entry["instruction"] = instruction["instruction"].replace("{X}", name)
                entry["initial_prompt"] = instruction["initial_prompt"].replace("{X}", name)
                entry["gender"] = "male"
                entry["nationality"] = nationality
                entry["name"] = name

                # Ensure shuffled_options exist
                entry["shuffled_options"] = random.sample(option_keys, len(option_keys))
                
                serialized_entries.append(entry)

    for instruction in inst_female:
        for nationality, genders in names_data.items():
            for name in genders["female"]:
                entry = json.loads(json.dumps(instruction))  # Deep copy to avoid mutation
                entry["instruction"] = instruction["instruction"].replace("{X}", name)
                entry["initial_prompt"] = instruction["initial_prompt"].replace("{X}", name)
                entry["gender"] = "female"
                entry["nationality"] = nationality
                entry["name"] = name

                # Ensure shuffled_options exist
                entry["shuffled_options"] = random.sample(option_keys, len(option_keys))

                serialized_entries.append(entry)

    return serialized_entries

# Generate entries
all_entries = generate_serialized_entries(inst_male, inst_female, names_data)
data = all_entries[:2]  # Limit to the first item for testing

# Function to send a prompt to Cohere and print the response
# def test_cohere_prompt(entry_data):
#     try:
#         instruction = entry_data.get("instruction", "No instruction provided")
#         initial_prompt = entry_data.get("initial_prompt", "No initial prompt provided")
#         shuffled_option_keys = entry_data.get("shuffled_options")

#         if shuffled_option_keys is None:
#             print("\n❌ ERROR: 'shuffled_options' key is missing in entry_data.")
#             print(f"🔹 DEBUG: Entry Data: {json.dumps(entry_data, indent=2)}")
#             return

#         # Map the option keys to their actual text values
#         options = [entry_data[key] for key in shuffled_option_keys]

#         # Format the prompt with actual option text
#         prompt = (
#             f"Below is an instruction and a statement paired with multiple-choice options.\n\n"
#             f"### Instruction:\n{instruction}\n\n"
#             f"### Statement:\n{initial_prompt}\n\n"
#             f"Options:\n"
#             f"A. {options[0]}\n"
#             f"B. {options[1]}\n"
#             f"C. {options[2]}\n"
#             f"D. {options[3]}\n\n"
#             f"ONLY respond with a single letter (A, B, C, or D)."
#         )

#         # 🔹 Print before API call
#         print("\n🔹 Sending request to Cohere...")
#         print(f"🔹 Prompt: {prompt}\n")

#         # Send request to Cohere API
#         response = co.chat(
#             model="c4ai-aya-expanse-32b",
#             messages=[{"role": "user", "content": prompt}],
#         )

#         # Extract the actual response text
#         response_text = response.message.content[0].text if response.message.content else "No response text received."

#         # 🔹 Print after API call
#         print("\n✅ Received response from Cohere!\n")
#         print(f"🔹 Actual Response: {response_text}\n")

#     except Exception as e:
#         print("\n❌ ERROR: Exception occurred in test_cohere_prompt")
#         traceback.print_exc()


# Run the test function with the first entry
# test_cohere_prompt(data[0])
# Run the test function with both entries
# for entry in data:
#     test_cohere_prompt(entry)


import pandas as pd

# List to store results
results = []

# Function to send a prompt to Cohere and store the response
def test_cohere_prompt(entry_data, run_number, seed):
    try:
        instruction = entry_data.get("instruction", "No instruction provided")
        initial_prompt = entry_data.get("initial_prompt", "No initial prompt provided")
        shuffled_option_keys = entry_data.get("shuffled_options")

        if shuffled_option_keys is None:
            print("\n❌ ERROR: 'shuffled_options' key is missing in entry_data.")
            return None

        # Map the option keys to their actual text values
        options = [entry_data[key] for key in shuffled_option_keys]

        # Format the prompt
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

        # Send request to Cohere API
        response = co.chat(
            model="c4ai-aya-expanse-32b",
            messages=[{"role": "user", "content": prompt}],
        )

        # Extract the actual response text
        response_text = response.message.content[0].text if response.message.content else "No response received."

        # Store result in list
        results.append({
            "run": run_number,
            "seed": seed,
            "instruction": instruction,
            "initial_prompt": initial_prompt,
            "chosen_letter": response_text.strip(),
            "gender": entry_data["gender"],
            "nationality": entry_data["nationality"],
            "name": entry_data["name"],
            "opt_A": options[0],
            "opt_B": options[1],
            "opt_C": options[2],
            "opt_D": options[3],
            "original_option_mapping": {chr(65 + i): shuffled_option_keys[i] for i in range(len(shuffled_option_keys))}
        })

    except Exception as e:
        print("\n❌ ERROR: Exception occurred in test_cohere_prompt")
        traceback.print_exc()

# Run multiple experiments
all_runs_results = []
for run in tqdm(range(args.runs), total=args.runs, desc="Runs:"):
    new_seed = random.randint(0, 100000)
    print(f"Run {run + 1}/{args.runs} with seed {new_seed}")
    for entry in data:
        test_cohere_prompt(entry, run + 1, new_seed)

# Convert results to DataFrame
final_result_df = pd.DataFrame(results)

# Save to CSV
output_dir = "../outputs/cohere_aya_exp_32b"
os.makedirs(output_dir, exist_ok=True)
output_file = f"{output_dir}/choose_{MODE}_{args.runs}_runs.csv"
final_result_df.to_csv(output_file, index=False)

print(f"Results saved to {output_file}")


