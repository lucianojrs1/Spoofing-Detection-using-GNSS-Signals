import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

# --- CONFIGURAÇÃO ---
# Aponte para o arquivo binário do Cenário 4
ARQUIVO_ENTRADA = "Dados_GPS/cenario5/ds4.bin"  
ARQUIVO_SAIDA   = "features_cenario4.csv"

DTYPE = np.int16
CHUNK_SIZE = 1000000 
LIMITE_DE_LEITURA = 1000  # 1000 chunks para garantir uma boa amostragem do teste

def processar_unico_arquivo(caminho):
    features_list = []
    print(f"Processando: {caminho} ...")
    
    contador = 0 
    
    try:
        with open(caminho, 'rb') as f:
            while True:
                # Limitador
                if contador >= LIMITE_DE_LEITURA:
                    print(f"--> Limite de {LIMITE_DE_LEITURA} pedaços atingido.")
                    break
                
                # Lê I e Q juntos
                raw = np.fromfile(f, dtype=DTYPE, count=CHUNK_SIZE * 2)
                if len(raw) < 2: break 
                
                contador += 1
                
                # Separa I e Q
                i_data = raw[0::2].astype(np.float32)
                q_data = raw[1::2].astype(np.float32)
                
                # Magnitude
                n = min(len(i_data), len(q_data))
                magnitude = np.sqrt(i_data[:n]**2 + q_data[:n]**2)
                
                # --- EXTRAÇÃO ---
                features_list.append({
                    'Potencia': np.mean(magnitude**2),
                    'Maximo': np.max(magnitude),
                    'DesvioPadrao': np.std(magnitude),
                    'Curtose': kurtosis(magnitude),
                    'Assimetria': skew(magnitude)
                    # Não precisamos da coluna 'Classe' aqui, pois vamos 
                    # adicionar ela no script de fusão depois
                })
                
    except FileNotFoundError:
        print(f"ERRO: Arquivo não encontrado: {caminho}")
        return pd.DataFrame()

    return pd.DataFrame(features_list)

# --- EXECUÇÃO ---
print(f"--- Extraindo características do Cenário 4 ---")
df_cenario4 = processar_unico_arquivo(ARQUIVO_ENTRADA)

if not df_cenario4.empty:
    df_cenario4.to_csv(ARQUIVO_SAIDA, index=False)
    print(f"Sucesso! Arquivo '{ARQUIVO_SAIDA}' salvo com {len(df_cenario4)} linhas.")
    
    # Check rápido de sanidade (Mostra a média da Potência)
    potencia_media = df_cenario4['Potencia'].mean()
    print(f"\n[CHECK] Potência Média encontrada: {potencia_media:.2f}")
    print("Compare esse valor com o do seu gráfico de treino (Clean vs Spoof).")
    print("Se for 'baixo' (parecido com Clean), seu modelo vai falhar (o que é bom para o relatório).")
else:
    print("Falha na leitura.")