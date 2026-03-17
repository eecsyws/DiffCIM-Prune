import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Fill in your CSV path here
# Example: CSV_PATH = r"/path/to/your/data.csv"
# =========================
CSV_PATH = r"../../Experiment/Energy/Results/energy_results.csv"

# Output image name
FIG_PATH = "energy.png"


def load_data(csv_path: str) -> pd.DataFrame:
    if not csv_path:
        raise ValueError("Please fill in CSV_PATH before running the script.")

    df = pd.read_csv(csv_path)

    # Standardize column names
    df.columns = [c.strip() for c in df.columns]

    # Clean string columns
    for col in ["weight_encode", "activation_encode"]:
        df[col] = df[col].astype(str).str.strip()

    # Convert numeric columns
    for col in ["pruning_rate", "sigma", "total_1x1"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def plot_bar_chart(df: pd.DataFrame) -> None:
    """
    Compare encoding configurations including Diff/Diff with pruning.
    - First 4 bars: pruning_rate = 0.0
    - 5th bar: Diff / Diff with pruning_rate = 0.2
    """
    # First 4 bars: no pruning
    subset_no_prune = df[df["pruning_rate"] == 0.0].copy()

    # 5th bar: Diff/Diff with pruning
    subset_prune = df[(df["weight_encode"] == "differential")
                      & (df["activation_encode"] == "differential")
                      & (df["pruning_rate"] == 0.2)].copy()

    config_order = [
        ("twos_complement", "twos_complement", "TC / TC", 0.0),
        ("twos_complement", "differential", "TC / Diff", 0.0),
        ("differential", "twos_complement", "Diff / TC", 0.0),
        ("differential", "differential", "Diff / Diff", 0.0),
        ("differential", "differential", "Diff/Diff w.Prune", 0.2),
    ]

    labels = []
    values = []

    for weight_enc, act_enc, label, prune_rate in config_order:
        if prune_rate == 0.0:
            row = subset_no_prune[
                (subset_no_prune["weight_encode"] == weight_enc)
                & (subset_no_prune["activation_encode"] == act_enc)
            ].sort_values("sigma")
        else:
            row = subset_prune.sort_values("sigma")

        if row.empty:
            labels.append(label)
            values.append(float("nan"))
        else:
            labels.append(label)
            values.append(row.iloc[0]["total_1x1"])

    # Mint Mambo color palette (薄荷曼波)
    colors = ['#7FD1B9', '#88D8B0', '#B5EAD7', '#C7F9CC', '#95E1D3']

    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, values, color=colors, edgecolor='#2D5A4A', linewidth=1.5)

    plt.xlabel("Encoding Config (Weight / Activation)", fontsize=12)
    plt.ylabel("Total Conducting Cells (1×1 Interactions)", fontsize=12)
    plt.title("Total Conducting Cells", fontsize=14)
    plt.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    plt.tight_layout()

    # Remove horizontal grid lines
    plt.gca().yaxis.grid(False)

    # Add value labels on top of bars (without scientific notation)
    for bar, value in zip(bars, values):
        if pd.notna(value):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.02,
                f"{value/1e9:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight='bold',
                color='#2D5A4A'
            )

    # Calculate reduction ratio: TC/TC vs lowest bar (now index 4)
    lowest_idx = values.index(min([v for v in values if pd.notna(v)]))
    if values[0] > 0 and values[lowest_idx] > 0:
        reduction_ratio = values[0] / values[lowest_idx]
        # Add bold italic green text in empty space
        plt.text(0.98, 0.95, f'Array Energy Reduction: {reduction_ratio:.1f}×',
                 transform=plt.gca().transAxes,
                 ha='right', va='top', fontsize=14,
                 fontweight='bold', fontstyle='italic', color='#27AE60',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#27AE60', alpha=0.9))

    plt.savefig(FIG_PATH, dpi=300)
    plt.show()


def main():
    df = load_data(CSV_PATH)
    plot_bar_chart(df)


if __name__ == "__main__":
    main()
