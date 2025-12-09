import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import StandardScaler

# --- CONFIGURAÇÃO ---
ARQUIVO_AUTENTICO = "Dados_GPS/cleanStatic.bin" 
#ARQUIVO_SPOOFING = "Dados_GPS/ds2.bin"
ARQUIVO_SPOOFING = "Dados_GPS/cenario5/ds4.bin"

DTYPE = np.int16
CHUNK_SIZE = 1000000 
LIMITE_DE_LEITURA = 3500 
N_FFT = 2048 # Tamanho da janela para análise de frequência (potência de 2 é melhor)

def calcular_features_frequencia(i_data, q_data, n_fft):
    """
    Reconstrói o sinal complexo e extrai features do espectro.
    Pega apenas os primeiros n_fft pontos para ser rápido.
    """
    # Reconstrói sinal complexo: Z = I + jQ
    # Pega apenas uma fatia para a FFT (muito mais rápido que fazer no chunk todo)
    sinal_complexo = i_data[:n_fft] + 1j * q_data[:n_fft]
    
    # Aplica FFT e pega a magnitude
    espectro = np.abs(fft(sinal_complexo))
    
    # Remove componente DC (índice 0) para evitar viés de offset
    espectro = espectro[1:]
    
    # --- Features Espectrais ---
    fft_peak = np.max(espectro)
    fft_mean = np.mean(espectro)
    
    # Centroide Espectral (Média ponderada das frequências)
    # Indica onde a energia está concentrada
    freqs = np.arange(1, len(espectro) + 1)
    centroide = np.sum(freqs * espectro) / np.sum(espectro)
    
    return fft_peak, fft_mean, centroide

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
                
                # Converte para float para processamento
                i_data = raw[0::2].astype(np.float32)
                q_data = raw[1::2].astype(np.float32)
                
                # --- PASSO CRUCIAL: NORMALIZAÇÃO (AGC) ---
                # 1. Calcula a magnitude (envelope)
                magnitude = np.sqrt(i_data**2 + q_data**2)
                
                # 2. Calcula a energia média (RMS) desse pedaço
                rms = np.sqrt(np.mean(magnitude**2))
                
                # 3. Normaliza tudo dividindo pelo RMS
                # Isso força todo sinal a ter Potência = 1.0
                if rms > 0:
                    i_norm = i_data / rms
                    q_norm = q_data / rms
                    magnitude_norm = magnitude / rms
                else:
                    i_norm, q_norm, magnitude_norm = i_data, q_data, magnitude

                # --- Domínio da Frequência (Feito no sinal JÁ normalizado) ---
                # Agora o pico da FFT mostra a concentração de energia, não o volume
                fft_peak, fft_mean, fft_centroid = calcular_features_frequencia(i_norm, q_norm, N_FFT)
                
                features_list.append({
                    # Tempo
                    'Potencia': np.mean(magnitude_norm**2), # Deve dar sempre aprox 1.0
                    'Maximo': np.max(magnitude_norm),       # Pico relativo à média
                    'DesvioPadrao': np.std(magnitude_norm),
                    'Curtose': kurtosis(magnitude_norm),
                    'Assimetria': skew(magnitude_norm),
                    # Frequência
                    'FFT_Peak': fft_peak,
                    'FFT_Mean': fft_mean,
                    'FFT_Centroide': fft_centroid,
                    # Label
                    'Classe': label 
                })
                
    except FileNotFoundError:
        print(f"ERRO: Arquivo não encontrado: {caminho}")
        return pd.DataFrame()

    return pd.DataFrame(features_list)
# --- EXECUÇÃO ---
print("--- Iniciando Extração com Domínio da Frequência ---")
df_clean = processar_arquivo(ARQUIVO_AUTENTICO, label=0)
df_spoofed = processar_arquivo(ARQUIVO_SPOOFING, label=1)

if not df_clean.empty and not df_spoofed.empty:
    # 1. Junta os DataFrames
    df_final = pd.concat([df_clean, df_spoofed], ignore_index=True)
    
    # 2. Separa as features (X) dos labels (y)
    colunas_features = [c for c in df_final.columns if c != 'Classe']
    X = df_final[colunas_features]
    y = df_final['Classe']
    
    # --- NORMALIZAÇÃO (SOLUÇÃO DO PROBLEMA) ---
    print("Normalizando dados (StandardScaler)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Cria um novo DataFrame normalizado para visualização e salvamento
    df_normalized = pd.DataFrame(X_scaled, columns=colunas_features)
    df_normalized['Classe'] = y # Devolve o label
    
    # Salva o arquivo pronto para ML
    df_normalized.to_csv('features_normalizadas_com_fft.csv', index=False)
    print("Sucesso! CSV normalizado salvo.")
    
    # --- PLOTAGEM (Dados Normalizados) ---
    features_to_plot = ['Potencia', 'Curtose', 'FFT_Peak', 'FFT_Centroide']
    
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 10))
    
    for i, col in enumerate(features_to_plot):
        plt.subplot(2, 2, i+1)
        # Note que agora o eixo X estará em escala de Desvio Padrão (ex: -2 a +2)
        sns.histplot(data=df_normalized, x=col, hue='Classe', kde=True, 
                     palette={0: 'blue', 1: 'red'}, element="step")
        plt.title(f'Distribuição Normalizada: {col}')
    
    plt.tight_layout()
    plt.show()
    
else:
    print("Falha: Verifique os caminhos dos arquivos.")