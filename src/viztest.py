# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os

# # Load the datasets
# success_df = pd.read_csv("../outputs/llama3_8b/success_full_1_runs.csv")
# failure_df = pd.read_csv("../outputs/llama3_8b/failure_full_1_runs.csv")

# # Add a column to differentiate the datasets
# success_df["mode"] = "Success"
# failure_df["mode"] = "Failure"

# # Combine both datasets
# df = pd.concat([success_df, failure_df])

# # Create an output directory for saving plots
# output_dir = "../outputs/llama3_8b/plots"
# os.makedirs(output_dir, exist_ok=True)

# # Plot Attribution Disparity by Gender
# plt.figure(figsize=(8, 6))
# sns.boxplot(x="gender", y="metric", hue="mode", data=df)
# plt.title("Attribution Disparity by Gender")
# plt.xlabel("Gender")
# plt.ylabel("Metric")
# plt.savefig(os.path.join(output_dir, "attribution_disparity_gender.png"))
# plt.show()

# # Plot Attribution Disparity by Nationality
# plt.figure(figsize=(12, 6))
# sns.boxplot(x="nationality", y="metric", hue="mode", data=df)
# plt.xticks(rotation=45, ha="right")  # Rotate x labels for readability
# plt.title("Attribution Disparity by Nationality")
# plt.xlabel("Nationality")
# plt.ylabel("Metric")
# plt.savefig(os.path.join(output_dir, "attribution_disparity_nationality.png"))
# plt.show()





# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load the datasets
# success_df = pd.read_csv("../outputs/llama3_8b/success_full_1_runs.csv")
# failure_df = pd.read_csv("../outputs/llama3_8b/failure_full_1_runs.csv")

# # Add a column to differentiate the datasets
# success_df["mode"] = "Success"
# failure_df["mode"] = "Failure"

# # Combine both datasets
# df = pd.concat([success_df, failure_df])

# # Compute mean and standard deviation of the metric for each nationality
# nationality_stats = df.groupby("nationality")["metric"].agg(["mean", "std"]).reset_index()

# # Save the statistics as a CSV file
# nationality_stats.to_csv("../outputs/llama3_8b/nationality_attribution_stats.csv", index=False)

# # Visualization: Bar plot of mean Attribution Disparity by nationality
# plt.figure(figsize=(12, 6))
# sns.barplot(x="nationality", y="mean", data=nationality_stats, palette="coolwarm")
# plt.xticks(rotation=45, ha="right")
# plt.title("Mean Attribution Disparity by Nationality")
# plt.xlabel("Nationality")
# plt.ylabel("Mean Metric")
# plt.savefig("../outputs/llama3_8b/attribution_disparity_nationality_mean.png")
# plt.show()

# # Visualization: Error bars to show variation (std deviation)
# plt.figure(figsize=(12, 6))
# sns.barplot(x="nationality", y="mean", data=nationality_stats, palette="coolwarm")
# plt.errorbar(x=range(len(nationality_stats)), y=nationality_stats["mean"], yerr=nationality_stats["std"], fmt='o', color='black', capsize=5)
# plt.xticks(rotation=45, ha="right")
# plt.title("Attribution Disparity by Nationality (with Variation)")
# plt.xlabel("Nationality")
# plt.ylabel("Mean Metric ± Std Dev")
# plt.savefig("../outputs/llama3_8b/attribution_disparity_nationality_variation.png")
# plt.show()





# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os

# # Load the datasets
# success_df = pd.read_csv("../outputs/llama3_8b/success_full_1_runs.csv")
# failure_df = pd.read_csv("../outputs/llama3_8b/failure_full_1_runs.csv")

# # Add a column to differentiate the datasets
# success_df["mode"] = "Success"
# failure_df["mode"] = "Failure"

# # Combine both datasets
# df = pd.concat([success_df, failure_df])

# # Group by gender and nationality separately for Success and Failure
# success_stats = success_df.groupby(["gender", "nationality"])["metric"].agg(["mean", "std"]).reset_index()
# failure_stats = failure_df.groupby(["gender", "nationality"])["metric"].agg(["mean", "std"]).reset_index()

# # Save the results as CSV files
# output_dir = "../outputs/llama3_8b"
# os.makedirs(output_dir, exist_ok=True)

# success_stats.to_csv(f"{output_dir}/success_attribution_stats.csv", index=False)
# failure_stats.to_csv(f"{output_dir}/failure_attribution_stats.csv", index=False)

# # Visualization function
# def plot_attribution_disparity(data, mode, output_dir):
#     plt.figure(figsize=(14, 6))
#     sns.barplot(x="nationality", y="mean", hue="gender", data=data, palette="coolwarm")
#     plt.xticks(rotation=45, ha="right")
#     plt.title(f"Attribution Disparity by Gender & Nationality ({mode})")
#     plt.xlabel("Nationality")
#     plt.ylabel("Mean Metric")
#     plt.legend(title="Gender")
#     plt.savefig(f"{output_dir}/attribution_disparity_{mode.lower()}_gender_nationality.png")
#     plt.show()

# # Generate plots separately for Success and Failure
# plot_attribution_disparity(success_stats, "Success", output_dir)
# plot_attribution_disparity(failure_stats, "Failure", output_dir)





# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os

# # Load the datasets
# success_df = pd.read_csv("../outputs/llama3_8b/success_attribution_stats.csv")
# failure_df = pd.read_csv("../outputs/llama3_8b/failure_attribution_stats.csv")

# # Create output directory for saving plots
# output_dir = "../outputs/llama3_8b"
# os.makedirs(output_dir, exist_ok=True)

# # Pivot to get male and female values for easy difference calculation
# success_diff = success_df.pivot(index="nationality", columns="gender", values="mean")
# failure_diff = failure_df.pivot(index="nationality", columns="gender", values="mean")

# # Compute gender difference: Male - Female
# success_diff["difference"] = success_diff["male"] - success_diff["female"]
# failure_diff["difference"] = failure_diff["male"] - failure_diff["female"]

# # Reset index for plotting
# success_diff = success_diff.reset_index()
# failure_diff = failure_diff.reset_index()

# # Bar plot of gender differences in Attribution Disparity (Success)
# plt.figure(figsize=(12, 6))
# sns.barplot(x="nationality", y="difference", data=success_diff, palette="coolwarm")
# plt.xticks(rotation=45, ha="right")
# plt.axhline(0, color="black", linestyle="--")  # Reference line at 0
# plt.title("Gender Difference in Attribution Disparity (Success)")
# plt.xlabel("Nationality")
# plt.ylabel("Male - Female Metric Difference")
# plt.savefig(os.path.join(output_dir, "gender_diff_success.png"))
# plt.show()

# # Bar plot of gender differences in Attribution Disparity (Failure)
# plt.figure(figsize=(12, 6))
# sns.barplot(x="nationality", y="difference", data=failure_diff, palette="coolwarm")
# plt.xticks(rotation=45, ha="right")
# plt.axhline(0, color="black", linestyle="--")
# plt.title("Gender Difference in Attribution Disparity (Failure)")
# plt.xlabel("Nationality")
# plt.ylabel("Male - Female Metric Difference")
# plt.savefig(os.path.join(output_dir, "gender_diff_failure.png"))
# plt.show()

# # Heatmap of gender differences across nationalities (Success)
# plt.figure(figsize=(10, 6))
# sns.heatmap(success_diff.set_index("nationality")[["difference"]], annot=True, cmap="coolwarm", center=0)
# plt.title("Heatmap of Gender Difference in Attribution Disparity (Success)")
# plt.xlabel("")
# plt.ylabel("Nationality")
# plt.savefig(os.path.join(output_dir, "gender_diff_heatmap_success.png"))
# plt.show()

# # Heatmap of gender differences across nationalities (Failure)
# plt.figure(figsize=(10, 6))
# sns.heatmap(failure_diff.set_index("nationality")[["difference"]], annot=True, cmap="coolwarm", center=0)
# plt.title("Heatmap of Gender Difference in Attribution Disparity (Failure)")
# plt.xlabel("")
# plt.ylabel("Nationality")
# plt.savefig(os.path.join(output_dir, "gender_diff_heatmap_failure.png"))
# plt.show()





# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os

# # Load the datasets
# success_df = pd.read_csv("../outputs/llama3_8b/success_full_1_runs.csv")
# failure_df = pd.read_csv("../outputs/llama3_8b/failure_full_1_runs.csv")

# # Add a column to differentiate the datasets
# success_df["mode"] = "Success"
# failure_df["mode"] = "Failure"

# # Combine both datasets
# df = pd.concat([success_df, failure_df])

# # Group by gender and nationality separately for Success and Failure
# success_stats = success_df.groupby(["gender", "nationality"])["metric"].agg(["mean", "std"]).reset_index()
# failure_stats = failure_df.groupby(["gender", "nationality"])["metric"].agg(["mean", "std"]).reset_index()

# # Save the results as CSV files
# output_dir = "../outputs/llama3_8b"
# os.makedirs(output_dir, exist_ok=True)

# success_stats.to_csv(f"{output_dir}/success_attribution_stats.csv", index=False)
# failure_stats.to_csv(f"{output_dir}/failure_attribution_stats.csv", index=False)

# # Alternative visualization function
# def plot_alternative_visualizations(data, mode, output_dir):
#     plt.figure(figsize=(14, 6))
#     sns.pointplot(x="nationality", y="mean", hue="gender", data=data, dodge=True, markers=["o", "s"], capsize=0.2)
#     plt.xticks(rotation=45, ha="right")
#     plt.title(f"Point Plot of Attribution Disparity by Gender & Nationality ({mode})")
#     plt.xlabel("Nationality")
#     plt.ylabel("Mean Metric")
#     plt.legend(title="Gender")
#     plt.savefig(f"{output_dir}/pointplot_attribution_disparity_{mode.lower()}_gender_nationality.png")
#     plt.show()

#     plt.figure(figsize=(14, 6))
#     sns.violinplot(x="nationality", y="mean", hue="gender", data=data, split=True, inner="quartile", palette="coolwarm")
#     plt.xticks(rotation=45, ha="right")
#     plt.title(f"Violin Plot of Attribution Disparity by Gender & Nationality ({mode})")
#     plt.xlabel("Nationality")
#     plt.ylabel("Mean Metric")
#     plt.legend(title="Gender")
#     plt.savefig(f"{output_dir}/violinplot_attribution_disparity_{mode.lower()}_gender_nationality.png")
#     plt.show()

# # Generate alternative plots separately for Success and Failure
# plot_alternative_visualizations(success_stats, "Success", output_dir)
# plot_alternative_visualizations(failure_stats, "Failure", output_dir)






# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os

# # Load the success dataset
# success_df = pd.read_csv("../outputs/llama3_8b/success_full_1_runs.csv")

# # Extract the relevant columns
# option_columns = ["opt1_higheffort", "opt2_highability", "opt3_easytask", "opt4_goodluck"]

# # Compute selection percentages for each option, grouped by gender and nationality
# success_counts = success_df.groupby(["gender", "nationality"])["chosen_key"].value_counts(normalize=True).unstack(fill_value=0) * 100

# # Reset index for plotting
# success_counts = success_counts.reset_index()

# # Create output directory
# output_dir = "../outputs/llama3_8b"
# os.makedirs(output_dir, exist_ok=True)

# # Define a function for plotting option selection percentages
# def plot_option_selection(data, mode, output_dir):
#     plt.figure(figsize=(14, 6))
#     melted_data = data.melt(id_vars=["gender", "nationality"], var_name="Option", value_name="Percentage")
    
#     sns.barplot(x="nationality", y="Percentage", hue="Option", data=melted_data, palette="coolwarm")
#     plt.xticks(rotation=45, ha="right")
#     plt.title(f"Selection Percentage of Options by Gender & Nationality ({mode})")
#     plt.xlabel("Nationality")
#     plt.ylabel("Selection Percentage (%)")
#     plt.legend(title="Option")
#     plt.savefig(f"{output_dir}/option_selection_{mode.lower()}_gender_nationality.png")
#     plt.show()

# # Generate plot for option selection percentages
# plot_option_selection(success_counts, "Success", output_dir)





# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os

# # Load the datasets
# success_df = pd.read_csv("../outputs/llama3_8b/success_full_1_runs.csv")
# failure_df = pd.read_csv("../outputs/llama3_8b/failure_full_1_runs.csv")

# # Add a column to differentiate the datasets
# success_df["mode"] = "Success"
# failure_df["mode"] = "Failure"

# # Combine both datasets
# df = pd.concat([success_df, failure_df])

# # Group by gender, nationality, and mode (Success/Failure)
# stats = df.groupby(["gender", "nationality", "mode"])["metric"].agg(["mean", "std"]).reset_index()

# # Save the results as CSV file
# output_dir = "../outputs/llama3_8b"
# os.makedirs(output_dir, exist_ok=True)
# stats.to_csv(f"{output_dir}/combined_attribution_stats.csv", index=False)

# # Create figure with two subplots
# fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)

# # Success Plot (Left)
# sns.pointplot(
#     x="nationality", y="mean", hue="gender", data=stats[stats["mode"] == "Success"],
#     dodge=True, markers=["o", "s"], capsize=0.2, errwidth=1.5, palette="Blues", linewidth=1.5, ax=axes[0]
# )
# axes[0].set_title("Attribution Disparity (Success)", fontsize=16, fontweight="bold")
# axes[0].set_xlabel("Nationality", fontsize=14)
# axes[0].set_ylabel("Mean Attribution Disparity", fontsize=14)
# axes[0].tick_params(axis="x", rotation=45)
# axes[0].legend(title="Gender", fontsize=12)

# # Failure Plot (Right)
# sns.pointplot(
#     x="nationality", y="mean", hue="gender", data=stats[stats["mode"] == "Failure"],
#     dodge=True, markers=["D", "^"], capsize=0.2, errwidth=1.5, palette="Reds", linewidth=1.5, ax=axes[1]
# )
# axes[1].set_title("Attribution Disparity (Failure)", fontsize=16, fontweight="bold")
# axes[1].set_xlabel("Nationality", fontsize=14)
# axes[1].tick_params(axis="x", rotation=45)
# axes[1].legend(title="Gender", fontsize=12)

# # Adjust layout and save
# plt.tight_layout()
# plt.savefig(f"{output_dir}/subplots_attribution_disparity.png", dpi=300, bbox_inches="tight")
# plt.show()






# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os

# # Load the datasets
# success_df = pd.read_csv("../outputs/llama3_8b/success_full_1_runs.csv")
# failure_df = pd.read_csv("../outputs/llama3_8b/failure_full_1_runs.csv")

# # Add a column to differentiate the datasets
# success_df["mode"] = "Success"
# failure_df["mode"] = "Failure"

# # Combine both datasets
# df = pd.concat([success_df, failure_df])

# # Group by gender, nationality, and mode (Success/Failure)
# stats = df.groupby(["gender", "nationality", "mode"])["metric"].agg(["mean", "std"]).reset_index()

# # Save the results as CSV file
# output_dir = "../outputs/llama3_8b"
# os.makedirs(output_dir, exist_ok=True)
# stats.to_csv(f"{output_dir}/combined_attribution_stats.csv", index=False)

# # Create a single plot with 4 lines (Success-Male, Success-Female, Failure-Male, Failure-Female)
# plt.figure(figsize=(16, 8))

# # Define colors and markers
# palette = {"Success": "blue", "Failure": "red"}
# markers = {"male": "o", "female": "s"}

# # Plot each group separately
# for mode in ["Success", "Failure"]:
#     for gender in ["male", "female"]:
#         subset = stats[(stats["mode"] == mode) & (stats["gender"] == gender)]
#         sns.pointplot(
#             x="nationality", y="mean", data=subset, color=palette[mode], marker=markers[gender],
#             label=f"{mode} ({gender.capitalize()})", capsize=0.2, errwidth=1.5, linewidth=1.5
#         )

# plt.xticks(rotation=45, ha="right", fontsize=14)
# plt.yticks(fontsize=14)
# plt.title("Attribution Disparity by Gender & Nationality (Success vs. Failure)", fontsize=18, fontweight="bold")
# plt.xlabel("Nationality", fontsize=16)
# plt.ylabel("Mean Attribution Disparity", fontsize=16)
# plt.legend(title="Group", title_fontsize=14, fontsize=12)
# plt.grid(axis="y", linestyle="--", alpha=0.5)

# # Save and show
# plt.savefig(f"{output_dir}/pointplot_4lines_attribution_disparity.png", dpi=300, bbox_inches="tight")
# plt.show()





import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the datasets
success_df = pd.read_csv("../outputs/llama3_8b/success_full_1_runs.csv")
failure_df = pd.read_csv("../outputs/llama3_8b/failure_full_1_runs.csv")

# Add a column to differentiate the datasets
success_df["mode"] = "Success"
failure_df["mode"] = "Failure"

# Combine both datasets
df = pd.concat([success_df, failure_df])

# Group by gender, nationality, and mode (Success/Failure)
stats = df.groupby(["gender", "nationality", "mode"])["metric"].agg(["mean", "std"]).reset_index()

# Save the results as CSV file
output_dir = "../outputs/llama3_8b"
os.makedirs(output_dir, exist_ok=True)
stats.to_csv(f"{output_dir}/combined_attribution_stats.csv", index=False)

# Set aesthetic styles
sns.set_context("paper")  # Professional-looking style
sns.set_style("white")  # Clean background

# Create a single plot with 4 lines (Success-Male, Success-Female, Failure-Male, Failure-Female)
plt.figure(figsize=(16, 8))

# Define improved colors and markers
# palette = {"Success": "#44c0c7", "Failure": "#fa467f"}  # Blue for success, Red for failure
palette = {
    ("Success", "male"): "#44c0c7",  # Teal
    ("Success", "female"): "#fa467f",  # Purple
    ("Failure", "male"): "#8dc959",  # Pink
    ("Failure", "female"): "#9f78e3"  # Orange
}#fa6e6e
marker_style = "o"

# Plot each group separately with its own color
for (mode, gender), color in palette.items():
    subset = stats[(stats["mode"] == mode) & (stats["gender"] == gender)]
    sns.pointplot(
        x="nationality", y="mean", data=subset, color=color, marker=marker_style,
        label=f"{mode} ({gender.capitalize()})", capsize=0.2, errwidth=2.0, linewidth=2.5, markersize=10
    )

plt.title("Attribution Disparity by Gender & Nationality (Success vs. Failure)", fontsize=18, fontweight="bold")
plt.xticks(rotation=45, ha="right", fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel("", fontsize=18, fontweight="bold")
plt.ylabel("Mean Attribution Disparity", fontsize=18, fontweight="bold")
plt.legend(fontsize=14)  # Removes title from legend
plt.gca().spines["top"].set_linewidth(1.5)  # Removes top border
plt.gca().spines["right"].set_linewidth(1.5)  # Removes right border
plt.gca().spines["left"].set_linewidth(1.5)  # Thicker left border
plt.gca().spines["bottom"].set_linewidth(1.5)  # Thicker bottom border

# Save and show
plt.savefig(f"{output_dir}/pointplot_4lines_attribution_disparity_aesthetic.png", dpi=300, bbox_inches="tight")
plt.show()
