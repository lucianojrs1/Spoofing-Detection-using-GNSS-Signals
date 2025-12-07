import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew

# --- CONFIGURAÇÃO ---
ARQUIVO_AUTENTICO = "dados/dadoslimpos/cleanStatic.bin" 
ARQUIVO_SPOOFING = "dados/Cenario 2/ds2.bin"

DTYPE = np.int16
CHUNK_SIZE = 1000000 

# --- AQUI ESTÁ O SEGREDO PARA NÃO TRAVAR O PC ---
# 500 chunks * 1 milhão de amostras = 500 milhões de amostras.
# Isso deve dar uns 30 a 60 segundos de sinal processado, o que é ótimo.
LIMITE_DE_LEITURA = 500 

def processar_arquivo(caminho, label):
    features_list = []
    print(f"Processando: {caminho} ...")
    
    # Contador para saber onde parar
    contador = 0 
    
    try:
        with open(caminho, 'rb') as f:
            while True:
                # --- PARADA DE EMERGÊNCIA (LIMITADOR) ---
                if contador >= LIMITE_DE_LEITURA:
                    print(f"--> Limite de {LIMITE_DE_LEITURA} pedaços atingido. Parando leitura!")
                    break
                
                # Lê I e Q juntos
                raw = np.fromfile(f, dtype=DTYPE, count=CHUNK_SIZE * 2)
                if len(raw) < 2: break 
                
                # Incrementa o contador
                contador += 1
                
                # Separa I (pares) e Q (ímpares)
                i_data = raw[0::2].astype(np.float32)
                q_data = raw[1::2].astype(np.float32)
                
                # Garante mesmo tamanho
                n = min(len(i_data), len(q_data))
                magnitude = np.sqrt(i_data[:n]**2 + q_data[:n]**2)
                
                # --- EXTRAÇÃO ---
                features_list.append({
                    'Potencia': np.mean(magnitude**2),
                    'Maximo': np.max(magnitude),
                    'DesvioPadrao': np.std(magnitude),
                    'Curtose': kurtosis(magnitude),
                    'Assimetria': skew(magnitude),
                    'Classe': label 
                })
                
    except FileNotFoundError:
        print(f"ERRO: Arquivo não encontrado: {caminho}")
        return pd.DataFrame()

    return pd.DataFrame(features_list)

# --- EXECUÇÃO ---
print("--- Iniciando Extração Rápida ---")
df_clean = processar_arquivo(ARQUIVO_AUTENTICO, label=0)
df_spoofed = processar_arquivo(ARQUIVO_SPOOFING, label=1)

if not df_clean.empty and not df_spoofed.empty:
    df_final = pd.concat([df_clean, df_spoofed], ignore_index=True)
    df_final.to_csv('features_parciais_sem_cn0.csv', index=False)
    print("Sucesso! CSV salvo. Gerando gráficos...")
    
    # Plotagem
    features_to_plot = ['Potencia', 'Maximo', 'DesvioPadrao', 'Curtose', 'Assimetria']
    sns.set(style="whitegrid")
    plt.figure(figsize=(15, 10))
    
    for i, col in enumerate(features_to_plot):
        plt.subplot(2, 3, i+1)
        sns.histplot(data=df_final, x=col, hue='Classe', kde=True, 
                     palette={0: 'blue', 1: 'red'}, element="step")
        plt.title(f'Distribuição: {col}')
        plt.xlabel(col)
    
    plt.tight_layout()
    plt.show()
else:
    print("Falha: Verifique os caminhos dos arquivos.")