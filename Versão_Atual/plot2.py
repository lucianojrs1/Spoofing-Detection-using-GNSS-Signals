import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIGURAÇÃO ---
ARQUIVO_CSV = 'Versão_Atual/dataset_completo_sinais_sistemas.csv'

# Defina aqui as features "Sinais e Sistemas" que você quer destacar
features_para_plotar = [
    'Entropia_Espectral', 
    'Flatness_Espectral',
    'Kurt_Temp',          # Curtose Temporal
    'Fator_Crista',       # Crest Factor
    'Potencia'            # Para comparação
]

# --- CARREGAR DADOS ---
try:
    df = pd.read_csv(ARQUIVO_CSV)
    print(f"Dados carregados: {len(df)} amostras.")
except FileNotFoundError:
    print(f"ERRO: Não encontrei {ARQUIVO_CSV}. Gere o dataset primeiro.")
    exit()

# --- PREPARAÇÃO PARA PLOTAGEM ---
# Mapeamento de Classes para Nomes e Cores (Ajuste se usou apenas 0 e 1)
label_map = {0: 'Autêntico', 1: 'Overpowered (Bruto)', 2: 'Matched (Sutil)'}
color_map = {'Autêntico': 'green', 'Overpowered (Bruto)': 'red', 'Matched (Sutil)': 'orange'}

# Se no seu CSV as classes 1 e 2 estiverem juntas como '1', ajuste o mapa acima.
# Exemplo se usou apenas 0 e 1:
# label_map = {0: 'Autêntico', 1: 'Spoofing (Geral)'}
# color_map = {'Autêntico': 'green', 'Spoofing (Geral)': 'red'}

df['Cenario'] = df['Classe'].map(label_map)

# Configuração de Estilo Acadêmico
sns.set_style("whitegrid", {'grid.linestyle': '--'})
plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})

# --- LOOP DE PLOTAGEM PARA CADA FEATURE ---
for feature in features_para_plotar:
    if feature not in df.columns:
        print(f"Aviso: Feature '{feature}' não encontrada no CSV. Pulando.")
        continue

    print(f"Gerando gráfico para: {feature}...")
    
    # Cria uma figura com 2 subplots lado a lado (1 linha, 2 colunas)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Análise da Característica: {feature}", fontsize=16, y=1.02)

    # =====================================================================
    # GRÁFICO DA ESQUERDA: Evolução Temporal (Valor por Janela)
    # Mostra a estabilidade da feature ao longo dos chunks processados.
    # =====================================================================
    # Para o gráfico de linha ficar bonito, precisamos separar os dados e
    # criar um índice sequencial para cada cenário.
    
    start_idx = 0
    for label_code in sorted(label_map.keys()):
        nome_cenario = label_map[label_code]
        cor = color_map[nome_cenario]
        
        subset = df[df['Classe'] == label_code]
        if subset.empty: continue
        
        # Cria um índice x para plotar sequencialmente
        x_indices = np.arange(start_idx, start_idx + len(subset))
        
        ax1.plot(x_indices, subset[feature].values, 
                 label=nome_cenario, color=cor, linewidth=1.5, alpha=0.8)
        
        # Adiciona uma linha média para referência
        ax1.axhline(y=subset[feature].mean(), color=cor, linestyle=':', alpha=0.6)
        
        start_idx += len(subset) + 10 # Pequeno espaço entre cenários

    ax1.set_title("Evolução do Valor por Janela de Tempo")
    ax1.set_xlabel("Índice da Janela (Chunk)")
    ax1.set_ylabel("Valor da Feature")
    ax1.legend()
    
    # =====================================================================
    # GRÁFICO DA DIREITA: Distribuição (PDF via KDE)
    # Mostra a "assinatura estatística" da feature. (Estilo o gráfico que vc gostou)
    # =====================================================================
    sns.kdeplot(data=df, x=feature, hue='Cenario', palette=color_map, 
                fill=True, common_norm=False, alpha=0.3, linewidth=2, ax=ax2)
    
    ax2.set_title("Densidade de Probabilidade (PDF) da Feature")
    ax2.set_ylabel("Densidade")
    # ax2.set_xlabel é automático pelo nome da feature
    
    plt.tight_layout()
    
    # Salva o gráfico
    nome_arquivo = f"feature_analise_{feature}.png"
    plt.savefig(nome_arquivo, dpi=300, bbox_inches='tight')
    plt.show() # Comente se quiser gerar todos de uma vez sem travar

print("--- Processo Concluído ---")
print("Verifique as imagens geradas (feature_analise_*.png)")