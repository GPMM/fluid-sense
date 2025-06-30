import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# -------------------------
# 1. Carregar dados
# -------------------------
run1 = pd.read_csv('Run1.csv')
run2 = pd.read_csv('Run2.csv')

# Adiciona o identificador explicitamente (pode ser redundante dependendo do CSV)
run1['run'] = 'run1'
run2['run'] = 'run2'

# Juntar tudo
data = pd.concat([run1, run2], ignore_index=True)

# -------------------------
# 2. Selecionar colunas numéricas
# -------------------------
sensor_cols = [col for col in data.columns if col.startswith('a')]
X = data[sensor_cols]
y = data['run']

# -------------------------
# 3. Padronizar os dados
# -------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------
# 4. PCA
# -------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# -------------------------
# 5. Visualizar
# -------------------------
plt.figure(figsize=(8,6))
for run in ['run1', 'run2']:
    plt.scatter(
        X_pca[y == run, 0],
        X_pca[y == run, 1],
        label=run,
        alpha=0.6
    )

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA das Rodadas de Medidas')
plt.legend()
plt.grid(True)
plt.show()
