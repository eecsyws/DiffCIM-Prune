import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Fill in your CSV path here
# Example: CSV_PATH = r"/path/to/your/data.csv"
# =========================
CSV_PATH = r"../../Experiment/Accuracy/Results/accuracy_results.csv"

# Output image names
FIG1_PATH = "accuracy_vs_sigma.png"
FIG2_PATH = "accuracy_vs_pruning.png"


def load_data(csv_path: str) -> pd.DataFrame:
    if not csv_path:
        raise ValueError("Please fill in CSV_PATH before running the script.")

    df = pd.read_csv(csv_path)

    # Standardize column names and string values
    df.columns = [c.strip() for c in df.columns]
    for col in ["weight_encode", "activation_encode"]:
        df[col] = df[col].astype(str).str.strip()

    for col in ["pruning_rate", "sigma", "accuracy"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def plot_figure_1(df: pd.DataFrame) -> None:
    """
    Figure 1:
    Fix pruning_rate = 0.0
    Plot four curves for the four encoding configurations
    x-axis: sigma
    y-axis: accuracy
    """
    df1 = df[df["pruning_rate"] == 0.0].copy()

    config_order = [
        ("twos_complement", "twos_complement", "TC / TC"),
        ("twos_complement", "differential", "TC / Diff"),
        ("differential", "twos_complement", "Diff / TC"),
        ("differential", "differential", "Diff / Diff"),
    ]

    # Light fresh color palette - distinguishable but cohesive
    colors = ['#89CFF0', '#FFB6C1', '#FFD700', '#98FB98']

    plt.figure(figsize=(8, 5))

    for idx, (weight_enc, act_enc, label) in enumerate(config_order):
        subset = df1[
            (df1["weight_encode"] == weight_enc)
            & (df1["activation_encode"] == act_enc)
        ].sort_values("sigma")

        plt.plot(
            subset["sigma"],
            subset["accuracy"],
            marker="o",
            label=label,
            color=colors[idx],
            linewidth=2.5,
            markersize=8,
            markeredgecolor=colors[idx],
            markeredgewidth=1.5
        )

    plt.xlabel("Sigma", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.title("Accuracy vs Sigma (Pruning Rate = 0.0)", fontsize=14)
    plt.legend(title="Encoding Config (Weight / Activation)", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG1_PATH, dpi=300)
    plt.show()


def plot_figure_2(df: pd.DataFrame) -> None:
    """
    Figure 2:
    Fix configuration = twos_complement / twos_complement
    Plot curves for sigma = 0 and sigma = 0.05
    x-axis: pruning_rate
    y-axis: accuracy
    """
    df2 = df[
        (df["weight_encode"] == "twos_complement")
        & (df["activation_encode"] == "twos_complement")
        & (df["sigma"].isin([0.0, 0.05]))
    ].copy()

    sigma_order = [0.0, 0.05]

    # Mint Mambo color palette
    colors = ['#7FD1B9', '#88D8B0']

    plt.figure(figsize=(8, 5))

    for idx, sigma in enumerate(sigma_order):
        subset = df2[df2["sigma"] == sigma].sort_values("pruning_rate")

        plt.plot(
            subset["pruning_rate"],
            subset["accuracy"],
            marker="o",
            label=f"Sigma = {sigma}",
            color=colors[idx],
            linewidth=2.5,
            markersize=8,
            markeredgecolor=colors[idx],
            markeredgewidth=1.5
        )

    plt.xlabel("Pruning Rate", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.title("Accuracy vs Pruning Rate (TC / TC)", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG2_PATH, dpi=300)
    plt.show()


def main():
    df = load_data(CSV_PATH)
    plot_figure_1(df)
    plot_figure_2(df)


if __name__ == "__main__":
    main()
