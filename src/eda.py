"""Focused visual and tabular evidence for preprocessing choices."""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def create_eda_outputs(data: pd.DataFrame, figures_dir: Path, tables_dir: Path) -> None:
    """Create a small set of decision-oriented figures and auditable CSV summaries."""
    figures_dir.mkdir(parents=True, exist_ok=True); tables_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    missing = pd.DataFrame({"missing_count": data.isna().sum(), "missing_percent": data.isna().mean().mul(100)}).query("missing_count > 0").sort_values("missing_count", ascending=False)
    missing.to_csv(tables_dir / "missingness_before.csv")
    ax = missing["missing_percent"].sort_values().plot.barh(color="#3b82a0", figsize=(8, 3.5))
    ax.set(xlabel="Missing values (%)", ylabel="Feature", title="Missingness before treatment")
    plt.tight_layout(); plt.savefig(figures_dir / "missingness_before.png", dpi=220); plt.close()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(data=data, x="age", hue="income", bins=35, stat="density", common_norm=False, ax=axes[0], palette="Set2")
    axes[0].set_title("Age distribution by income group")
    sns.boxplot(data=data, x="income", y="hours_per_week", ax=axes[1], palette="Set2", hue="income", legend=False)
    axes[1].set_title("Hours worked by income group")
    plt.tight_layout(); plt.savefig(figures_dir / "numeric_distributions.png", dpi=220); plt.close()
    top = data["occupation"].value_counts(dropna=False).head(10).sort_values()
    ax = top.plot.barh(color="#d77f38", figsize=(8, 4.5)); ax.set(title="Ten most frequent occupation values", xlabel="Records", ylabel="Occupation")
    plt.tight_layout(); plt.savefig(figures_dir / "occupation_frequency.png", dpi=220); plt.close()
    numeric = data.select_dtypes("number")
    plt.figure(figsize=(7, 5)); sns.heatmap(numeric.corr(), annot=True, fmt=".2f", cmap="vlag", center=0, square=True)
    plt.title("Numerical-feature correlations")
    plt.tight_layout(); plt.savefig(figures_dir / "numeric_correlation.png", dpi=220); plt.close()
    data["income"].value_counts().rename_axis("income").reset_index(name="count").to_csv(tables_dir / "income_distribution.csv", index=False)
