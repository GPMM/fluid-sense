import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

# Carregar dados
run1 = pd.read_csv('Run1.csv')
run2 = pd.read_csv('Run2.csv')
sensor_cols = [col for col in run1.columns if col.startswith('a')]

# Correlação por sensor
print("Pearson por sensor:")
for col in sensor_cols:
    p, _ = pearsonr(run1[col], run2[col])
    print(f"{col}: {p:.2f}")

print("\nSpearman por sensor:")
for col in sensor_cols:
    s, _ = spearmanr(run1[col], run2[col])
    print(f"{col}: {s:.2f}")

# Correlação global
r1_flat = run1[sensor_cols].values.flatten()
r2_flat = run2[sensor_cols].values.flatten()
pearson_global, _ = pearsonr(r1_flat, r2_flat)
spearman_global, _ = spearmanr(r1_flat, r2_flat)

print(f"\nPearson global: {pearson_global:.2f}")
print(f"Spearman global: {spearman_global:.2f}")

# Scatter plot global
plt.figure(figsize=(8,6))
plt.scatter(r1_flat, r2_flat, alpha=0.5, s=10)
plt.plot([min(r1_flat), max(r1_flat)], [min(r1_flat), max(r1_flat)], color='red', linestyle='--')
plt.title(f'Run1 vs Run2\nPearson={pearson_global:.2f}, Spearman={spearman_global:.2f}')
plt.xlabel('Valores Run1')
plt.ylabel('Valores Run2')
plt.grid(True)
plt.show()
