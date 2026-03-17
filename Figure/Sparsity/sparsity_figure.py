import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =========================
# Fill in your CSV path here
# =========================
CSV_PATH = r"../../Experiment/Sparsity/Results/sparsity_results.csv"

# Output image name
FIG_PATH = "sparsity.png"


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
    for col in ["pruning_rate", "sigma", "avg_weight_density", "avg_activation_density"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def plot_bar_chart(df: pd.DataFrame) -> None:
    """
    5 bars with 3 sub-bars each:
    - Weight density, Activation density, Weight × Activation
    """
    # First 4: no pruning, 5th: with pruning
    subset_no_prune = df[df["pruning_rate"] == 0.0].copy()
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
    weight_density = []
    activation_density = []
    product_density = []

    for weight_enc, act_enc, label, prune_rate in config_order:
        if prune_rate == 0.0:
            row = subset_no_prune[
                (subset_no_prune["weight_encode"] == weight_enc)
                & (subset_no_prune["activation_encode"] == act_enc)
            ]
        else:
            row = subset_prune

        if row.empty:
            labels.append(label)
            weight_density.append(float("nan"))
            activation_density.append(float("nan"))
            product_density.append(float("nan"))
        else:
            labels.append(label)
            w_d = row.iloc[0]["avg_weight_density"]
            a_d = row.iloc[0]["avg_activation_density"]
            weight_density.append(w_d)
            activation_density.append(a_d)
            # Product: weight * activation density
            product_density.append(w_d * a_d if pd.notna(w_d) and pd.notna(a_d) else float("nan"))

    # Convert to arrays
    weight_density = np.array(weight_density)
    activation_density = np.array(activation_density)
    product_density = np.array(product_density)

    # Mint Mambo color palette (薄荷曼波)
    colors = ['#7FD1B9', '#88D8B0', '#B5EAD7', '#C7F9CC', '#95E1D3']

    # Sub-bar colors (薄荷曼波 - 高区分度)
    sub_colors = ['#2E8B57', '#7FD1B9', '#B5EAD7']  # 深绿、浅绿、极浅绿 with borders

    x = np.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(12, 6))

    # Plot grouped bars
    bars1 = plt.bar(x - width, weight_density, width, label='Weight', color=sub_colors[0], edgecolor='#1a5a3a', linewidth=2)
    bars2 = plt.bar(x, activation_density, width, label='Activation', color=sub_colors[1], edgecolor='#2D5A4A', linewidth=2)
    bars3 = plt.bar(x + width, product_density, width, label='Weight × Activation', color=sub_colors[2], edgecolor='#3D7A6A', linewidth=2)

    plt.xlabel("Encoding Config (Weight / Activation)", fontsize=12)
    plt.ylabel("Density (Probability of 1)", fontsize=12)
    plt.title("Per-Bit-Plane Density Comparison", fontsize=14)
    plt.xticks(x, labels, fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()

    # Add value labels on top of bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if pd.notna(height):
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.01,
                    f'{height:.2f}',
                    ha='center',
                    va='bottom',
                    fontsize=7,
                    fontweight='bold'
                )

    # Calculate ADC precision reduction: log2(first_product / last_product)
    first_product = product_density[0]
    last_product = product_density[-1]
    if pd.notna(first_product) and pd.notna(last_product) and last_product > 0:
        log2_ratio = np.log2(first_product / last_product)
        # Add text in empty space with light orange color
        plt.text(0.98, 0.95, f'Avg. ADC Precision Reduction: {log2_ratio:.1f} bit',
                 transform=plt.gca().transAxes,
                 ha='right', va='top', fontsize=14,
                 fontweight='bold', fontstyle='italic', color='#FFB347',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#FFB347', alpha=0.9))

    plt.savefig(FIG_PATH, dpi=300)
    plt.show()


def main():
    df = load_data(CSV_PATH)
    plot_bar_chart(df)


if __name__ == "__main__":
    main()
