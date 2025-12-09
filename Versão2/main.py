import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.fft import fft
# from sklearn.preprocessing import StandardScaler <-- Removido, não precisamos mais

# --- CONFIGURAÇÃO ---
ARQUIVO_AUTENTICO = "Dados_GPS/cleanStatic.bin" 
ARQUIVO_SPOOFING = "Dados_GPS/cenario5/ds4.bin"

DTYPE = np.int16
CHUNK_SIZE = 1000000 
LIMITE_DE_LEITURA = 3500 
N_FFT = 2048 

def calcular_features_frequencia(i_data, q_data, n_fft):
    """
    Reconstrói o sinal complexo e extrai features do espectro.
    """
    sinal_complexo = i_data[:n_fft] + 1j * q_data[:n_fft]
    espectro = np.abs(fft(sinal_complexo))
    espectro = espectro[1:] # Remove DC
    
    fft_peak = np.max(espectro)
    fft_mean = np.mean(espectro)
    
    return fft_peak, fft_mean

def processar_arquivo(caminho, label):
    features_list = []
    print(f"Processando: {caminho} ...")
    contador = 0 
    
    try:
        with open(caminho, 'rb') as f:
            while True:
                if contador >= LIMITE_DE_LEITURA:
                    break
                
                raw = np.fromfile(f, dtype=DTYPE, count=CHUNK_SIZE * 2)
                if len(raw) < N_FFT * 2: break 
                
                contador += 1
                
                i_data = raw[0::2].astype(np.float32)
                q_data = raw[1::2].astype(np.float32)
                
                # --- PRE-PROCESSAMENTO: AGC (Mantido pois é essencial) ---
                magnitude = np.sqrt(i_data**2 + q_data**2)
                rms = np.sqrt(np.mean(magnitude**2))
                
                if rms > 0:
                    i_norm = i_data / rms
                    q_norm = q_data / rms
                    magnitude_norm = magnitude / rms
                else:
                    i_norm, q_norm, magnitude_norm = i_data, q_data, magnitude

                # --- Extração de Features ---
                fft_peak, fft_mean = calcular_features_frequencia(i_norm, q_norm, N_FFT)
                
                features_list.append({
                    'Potencia': np.mean(magnitude_norm**2),
                    'Maximo': np.max(magnitude_norm),
                    'DesvioPadrao': np.std(magnitude_norm),
                    'FFT_Peak': fft_peak,
                    'FFT_Mean': fft_mean,
                    'Classe': label 
                })
                
    except FileNotFoundError:
        print(f"ERRO: Arquivo não encontrado: {caminho}")
        return pd.DataFrame()

    return pd.DataFrame(features_list)

# --- EXECUÇÃO ---
print("--- Iniciando Extração (Valores Reais/Brutos) ---")
df_clean = processar_arquivo(ARQUIVO_AUTENTICO, label=0)
df_spoofed = processar_arquivo(ARQUIVO_SPOOFING, label=1)

if not df_clean.empty and not df_spoofed.empty:
    # 1. Junta os DataFrames
    df_final = pd.concat([df_clean, df_spoofed], ignore_index=True)
    
    # --- MUDANÇA: SALVAMENTO DIRETO (SEM STANDARD SCALER) ---
    # Agora estamos salvando os valores físicos reais
    nome_csv = 'features_brutas_com_fft.csv'
    df_final.to_csv(nome_csv, index=False)
    print(f"Sucesso! CSV com dados brutos salvo como: {nome_csv}")
    
    # --- PLOTAGEM (Dados Reais) ---
    # Adicionei FFT_Mean pois vimos que é a feature mais importante
    features_to_plot = ['Potencia', 'Maximo', 'DesvioPadrao', 'FFT_Peak', 'FFT_Mean']
    
    sns.set(style="whitegrid")
    plt.figure(figsize=(14, 10))
    
    for i, col in enumerate(features_to_plot):
        plt.subplot(3, 2, i+1)  
        # O eixo X agora mostrará o valor real (físico)
        sns.histplot(data=df_final, x=col, hue='Classe', kde=True, 
                     palette={0: 'blue', 1: 'red'}, element="step")
        plt.title(f'Distribuição Real: {col}')
        plt.xlabel(f'Valor Físico ({col})')
    
    plt.tight_layout()
    plt.show()
    
else:
    print("Falha: Verifique os caminhos dos arquivos.")