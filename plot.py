import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("scores.csv")

# Rename models for clarity
model_name_map = {
    "falcon_mixed": "Heterogeneous",
    "falcon_single": "Falcon_7B",
    "llama_single": "Llama_8B",
}
df["Model"] = df["Model"].replace(model_name_map)

# Melt for stacking
df_melted = df.melt(
    id_vars=["Model", "Iteration"],
    value_vars=["Relevance", "Household Benefit", "Firm Benefit"],
    var_name="Score Type",
    value_name="Score",
)

# Pivot so each bar (Model x Iteration) has 3 stacked components
pivot_df = df_melted.pivot_table(
    index=["Iteration", "Model"],
    columns="Score Type",
    values="Score",
    aggfunc="sum",
    fill_value=0,
).reset_index()

# Set plotting order
models = ["Heterogeneous", "Falcon_7B", "Llama_8B"]
score_types = ["Relevance", "Household Benefit", "Firm Benefit"]
colors = {
    "Heterogeneous": ["#1f77b4", "#6baed6", "#9ecae1"],  # blue shades
    "Falcon_7B": ["#ff7f0e", "#fdae6b", "#fdd0a2"],  # orange shades
    "Llama_8B": ["#2ca02c", "#74c476", "#a1d99b"],  # green shades
}

fig, ax = plt.subplots(figsize=(18, 8))

bar_width = 0.25
iterations = sorted(pivot_df["Iteration"].unique())
x = range(len(iterations))

for i, model in enumerate(models):
    subset = pivot_df[pivot_df["Model"] == model].set_index("Iteration")
    positions = [pos + i * bar_width for pos in x]

    bottom = [0] * len(subset)
    for j, score_type in enumerate(score_types):
        ax.bar(
            positions,
            subset[score_type],
            bar_width,
            label=f"{model} - {score_type}",
            color=colors[model][j],
            bottom=bottom,
        )
        bottom = bottom + subset[score_type].values

# X-axis formatting
ax.set_xticks([pos + bar_width for pos in x])
ax.set_xticklabels(iterations)
ax.set_xlabel("Iteration")
ax.set_ylabel("Score")
ax.set_title(
    "Policy Update Scores per Iteration (Stacked by Relevance, Household, Firm)"
)
ax.legend(title="Model & Score Type", loc="upper right")
plt.tight_layout()
plt.savefig("scores.png")
plt.show()
