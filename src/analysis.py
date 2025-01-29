# import pandas as pd

# # Load the CSV file into a DataFrame
# data_file_path = f"../outputs/llama3_8b/failure_full_1_runs.csv"
# df = pd.read_csv(data_file_path)

# # List of columns to calculate averages for
# columns_to_average = ['opt4_badluck', 'opt2_lowability', 'opt3_difficulttask', 'opt1_loweffort']

# # Calculate averages grouped by gender
# gender_averages = df.groupby('gender')[columns_to_average].mean()

# # Calculate averages grouped by nationality
# nationality_averages = df.groupby('nationality')[columns_to_average].mean()

# # Display the results
# print("Averages grouped by gender:")
# print(gender_averages)

# print("\nAverages grouped by nationality:")
# print(nationality_averages)

# # Optionally save results to CSV
# gender_averages.to_csv("../analysis/failure/gender_averages.csv")
# nationality_averages.to_csv("../analysis/failure/nationality_averages.csv")




# import pandas as pd
# import matplotlib.pyplot as plt

# # Load the CSV file into a DataFrame
# data_file_path = f"../outputs/llama3_8b/failure_full_1_runs.csv"
# df = pd.read_csv(data_file_path)

# # Add a column to extract the latter part of initial_prompt
# df['latter_prompt'] = df['initial_prompt'].apply(lambda x: x.split(maxsplit=1)[1] if len(x.split(maxsplit=1)) > 1 else '')

# # Group by gender, nationality, latter_prompt, and chosen_key
# grouped = df.groupby(['gender', 'nationality', 'latter_prompt', 'chosen_key']).size().unstack(fill_value=0)

# # Display the results
# print("Counts of chosen_key for specific gender+nationality combinations across different latter_prompt parts:")
# print(grouped)

# # Save the results to CSV
# output_path = "../analysis/failure/chosen_key.csv"
# grouped.to_csv(output_path)

# print(f"Analysis saved to {output_path}")

# # Visualizations for specific latter_prompts
# for latter_prompt in df['latter_prompt'].unique():
#     try:
#         specific_data = grouped.xs(latter_prompt, level='latter_prompt', drop_level=False)
#         specific_data.sum(level=['gender', 'nationality']).plot(kind='bar', stacked=True, figsize=(10, 6))
#         plt.title(f"Distribution of chosen_key for latter_prompt: '{latter_prompt}'")
#         plt.ylabel("Count")
#         plt.xlabel("Gender and Nationality")
#         plt.xticks(rotation=45)
#         plt.tight_layout()
#         plt.savefig(f"../figs/failure/{latter_prompt.replace(' ', '_')}_comparison.png")
#         plt.show()
#     except KeyError:
#         print(f"No data available for latter_prompt: {latter_prompt}")

# # Overall visualization across all latter_prompts
# overall_data = grouped.sum(level=['gender', 'nationality'])
# overall_data.plot(kind='bar', stacked=True, figsize=(10, 6))
# plt.title("Overall Distribution of chosen_key across all latter_prompts")
# plt.ylabel("Count")
# plt.xlabel("Gender and Nationality")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.savefig("../figs/failure/overall_comparison.png")
# plt.show()



import pandas as pd
import matplotlib.pyplot as plt
import itertools

# Load the CSV file into a DataFrame
data_file_path = f"../outputs/llama3_8b/failure_full_1_runs.csv"
df = pd.read_csv(data_file_path)

# Add a column to extract the latter part of initial_prompt
df['latter_prompt'] = df['initial_prompt'].apply(lambda x: x.split(maxsplit=1)[1] if len(x.split(maxsplit=1)) > 1 else '')

# Define the four columns to analyze
columns_to_analyze = ['opt1_loweffort', 'opt2_lowability', 'opt3_difficulttask', 'opt4_badluck']

# Define all possible gender and nationality combinations
genders = df['gender'].unique()
nationalities = df['nationality'].unique()
all_combinations = pd.MultiIndex.from_product([genders, nationalities],
                                              names=['gender', 'nationality'])

# Group by gender and nationality, ignoring latter_prompt, and calculate the mean for each column
grouped = df.groupby(['gender', 'nationality'])[columns_to_analyze].mean()

# Reindex to ensure all combinations are present, filling missing values with 0
grouped = grouped.reindex(all_combinations, fill_value=0)

# Display the results
print("Average values of opt1_higheffort, opt2_highability, opt3_easytask, opt4_goodluck for gender+nationality combinations:")
print(grouped)

# Save the results to CSV
output_path = "../analysis/failure/aggregated_options_analysis.csv"
grouped.to_csv(output_path)

print(f"Analysis saved to {output_path}")

# Visualization: Aggregated comparison across gender and nationality
grouped.plot(kind='bar', figsize=(12, 8))
plt.title("Comparison of Options by Gender and Nationality (Aggregated Across All Prompts)")
plt.ylabel("Average Value")
plt.xlabel("Gender and Nationality")
plt.xticks(rotation=45)
plt.legend(title="Options")
plt.tight_layout()
plt.savefig("../figs/failure/aggregated_options_comparison.png")
plt.show()
