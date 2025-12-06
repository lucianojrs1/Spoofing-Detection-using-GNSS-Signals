import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ==============================================================================
# CONFIGURAÇÃO
# ==============================================================================
ARQUIVO_DATASET = 'dataset_FINAL_TREINO.csv'

# ==============================================================================
# EXECUÇÃO
# ==============================================================================

# 1. Carregar os dados
try:
    df = pd.read_csv(ARQUIVO_DATASET)
    print(f"Dataset carregado com sucesso! {len(df)} amostras.")
    print("-" * 30)
    print("Primeiras 5 linhas para conferência:")
    print(df.head())
    print("-" * 30)
except FileNotFoundError:
    print(f"ERRO: Não encontrei o arquivo '{ARQUIVO_DATASET}'. Verifique o nome.")
    exit()

# 2. Configurar o estilo dos gráficos
sns.set(style="whitegrid")

# Lista de colunas para plotar (todas menos a 'Classe')
colunas_para_plotar = [col for col in df.columns if col != 'Classe']
n_cols = len(colunas_para_plotar)

# Calcular tamanho do grid (ex: 2 linhas x 3 colunas)
n_rows_plot = (n_cols + 2) // 3  # Arredonda para cima
plt.figure(figsize=(15, 5 * n_rows_plot))

print("Gerando gráficos...")

for i, feature in enumerate(colunas_para_plotar):
    plt.subplot(n_rows_plot, 3, i + 1)
    
    # Histograma com curva de densidade (KDE)
    # Azul = Autêntico (0), Vermelho = Spoofing (1)
    sns.histplot(data=df, x=feature, hue='Classe', kde=True, 
                 palette={0: 'blue', 1: 'red'}, element="step", common_norm=False)
    
    plt.title(f'Distribuição: {feature}')
    plt.xlabel(feature)
    plt.ylabel('Contagem')

plt.tight_layout()
plt.suptitle(f'Análise do Dataset Final ({len(df)} amostras)', y=1.02, fontsize=16)
plt.show()

# 3. (Bônus) Boxplot Comparativo - Ótimo para ver outliers
print("Gerando Boxplots (visualização de separação)...")
plt.figure(figsize=(15, 5 * n_rows_plot))

for i, feature in enumerate(colunas_para_plotar):
    plt.subplot(n_rows_plot, 3, i + 1)
    
    sns.boxplot(x='Classe', y=feature, data=df, palette={0: 'blue', 1: 'red'})
    plt.title(f'Boxplot: {feature}')
    plt.xticks([0, 1], ['Autêntico (0)', 'Spoofing (1)'])

plt.tight_layout()
plt.show()

print("Concluído. Se as caixas azul e vermelha não se tocam, seu modelo vai acertar 100%.")