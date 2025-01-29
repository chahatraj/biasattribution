import pandas as pd

# Input CSV file
input_file = "../outputs/llama3_8b/success_full_1_runs.csv"

# Load the CSV file into a DataFrame
data = pd.read_csv(input_file)

# Check if the required columns are present in the data
required_columns = ["opt1_higheffort", "opt2_highability", "opt3_easytask", "opt4_goodluck"]

if not all(col in data.columns for col in required_columns):
    raise ValueError(f"The input file must contain the following columns: {', '.join(required_columns)}")

# Calculate the metric explicitly
data["metric"] = (
    data["opt1_higheffort"] + data["opt2_highability"] - data["opt3_easytask"] - data["opt4_goodluck"]
)

# Output file to save the results
output_file = "../analysis/success_full_1_runs_with_metric.csv"

# Save the updated DataFrame to a new CSV file
data.to_csv(output_file, index=False)

print(f"Metric calculation completed and saved to {output_file}")
