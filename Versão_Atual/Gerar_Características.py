import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew, entropy

# --- CONFIGURAÇÃO DOS ARQUIVOS ---
ARQ_AUTENTICO = "Dados_GPS/cleanStatic.bin"
ARQ_OVERPOWER = "Dados_GPS/ds2.bin"       # Cenário que você já tem
ARQ_MATCHED   = "Dados_GPS/cenario5/ds4.bin"       # Coloque o caminho do Matched-Power aqui (ex: ds3 ou ds4)
DTYPE = np.int16
CHUNK_SIZE = 1000000 
LIMITE_DE_LEITURA = 500 

def extrair_features_sinais_sistemas(sinal_complexo):
    """
    Aplica conceitos de Sinais e Sistemas para extrair assinatura do sinal.
    Input: Array numpy complexo (I + jQ)
    """
    features = {}
    
    # ---------------------------------------------------------
    # 1. ANÁLISE NO DOMÍNIO DO TEMPO (Momentos Estatísticos)
    # -------------------------------------                                                                                                                                                                                                                                 --------------------
    # O sinal é uma variável aleatória. Analisamos sua PDF (Função Densidade de Probabilidade).
    
    # Magnitude (Envelope)
    mag = np.abs(sinal_complexo)
    
    # Momento 1: Média (Componente DC / Bias)
    features['Media_Temp'] = np.mean(mag)
    
    # Momento 2: Variância (Potência AC)
    features['Var_Temp'] = np.var(mag)
    
    # Momento 3: Assimetria (Skewness - O quanto a PDF inclina para um lado)
    features['Skew_Temp'] = skew(mag)
    
    # Momento 4: Curtose (Kurtosis - O quanto a PDF tem caudas longas/outliers)
    features['Kurt_Temp'] = kurtosis(mag)
    
    # Fator de Crista (Pico / RMS): Indica quão "pontiagudo" é o sinal
    rms = np.sqrt(np.mean(mag**2))
    features['Fator_Crista'] = np.max(mag) / rms

    # ---------------------------------------------------------
    # 2. ANÁLISE NO DOMÍNIO DA FREQUÊNCIA (Fourier)
    # ---------------------------------------------------------
    # Transformamos o sinal para ver como a energia se distribui nas frequências.
    
    # FFT (Fast Fourier Transform) - Centralizada
    f_sinal = np.fft.fftshift(np.fft.fft(sinal_complexo))
    
    # PSD (Power Spectral Density - Densidade Espectral de Potência)
    # Representa a energia em cada frequência.
    psd = np.abs(f_sinal)**2
    
    # Normalizar PSD para tratar como uma distribuição de probabilidade
    psd_norm = psd / np.sum(psd)
    
    # Entropia Espectral (Shannon Entropy)
    # Mede a "incerteza" ou complexidade do espectro.
    # Sinal Spoofing muitas vezes é "mais limpo" ou filtrado diferente do ruído térmico real.
    features['Entropia_Espectral'] = entropy(psd_norm)
    
    # Flatness Espectral (Planicidade)
    # Média Geométrica da PSD / Média Aritmética da PSD
    # Se = 1, é ruído branco puro. Se < 1, o sinal tem "cor" (picos ou filtros).
    gmean = np.exp(np.mean(np.log(psd + 1e-10))) # +1e-10 evita log(0)
    amean = np.mean(psd)
    features['Flatness_Espectral'] = gmean / amean
    
    # Centroide Espectral (Média ponderada da frequência)
    # Onde está o "centro de massa" da energia do sinal?
    freqs = np.fft.fftshift(np.fft.fftfreq(len(sinal_complexo)))
    features['Centroide_Espectral'] = np.sum(freqs * psd_norm)

    return features

def processar_arquivo(caminho, label):
    features_list = []
    print(f"Processando: {caminho} ...")
    contador = 0 
    
    try:
        with open(caminho, 'rb') as f:
            while True:
                if contador >= LIMITE_DE_LEITURA: break
                
                raw = np.fromfile(f, dtype=DTYPE, count=CHUNK_SIZE * 2)
                if len(raw) < 2: break 
                contador += 1
                
                # Converter para Complexo (I + jQ) - Fundamental para Fourier correto
                i_data = raw[0::2].astype(np.float32)
                q_data = raw[1::2].astype(np.float32)
                
                # Garante mesmo tamanho e cria vetor complexo
                n = min(len(i_data), len(q_data))
                sinal_complexo = i_data[:n] + 1j * q_data[:n]
                
                # --- EXTRAÇÃO MATEMÁTICA PURA ---
                feats = extrair_features_sinais_sistemas(sinal_complexo)
                feats['Classe'] = label
                features_list.append(feats)
                
    except FileNotFoundError:
        print(f"ERRO: Arquivo não encontrado: {caminho}")
        return pd.DataFrame()

    return pd.DataFrame(features_list)

print("--- Iniciando Extração Sinais & Sistemas Completa ---")

# 1. Processa o Autêntico (Label 0)
df_clean = processar_arquivo(ARQ_AUTENTICO, label=0)

# 2. Processa o Overpowered (Label 1)
df_over = processar_arquivo(ARQ_OVERPOWER, label=1)

# 3. Processa o Matched-Power (Label 1)
# Note: Usamos Label 1 também, pois queremos que o modelo diga "É Spoofing", não importa qual tipo.
df_matched = processar_arquivo(ARQ_MATCHED, label=1) 

# --- JUNTA TUDO ---
dfs_para_concatenar = []
if not df_clean.empty: dfs_para_concatenar.append(df_clean)
if not df_over.empty:  dfs_para_concatenar.append(df_over)
if not df_matched.empty: dfs_para_concatenar.append(df_matched)

if dfs_para_concatenar:
    df_final = pd.concat(dfs_para_concatenar, ignore_index=True)
    
    # Embaralha os dados (importante para o treino não ficar viciado na ordem)
    df_final = df_final.sample(frac=1).reset_index(drop=True)
    
    df_final.to_csv('dataset_completo_sinais_sistemas.csv', index=False)
    print(f"Sucesso! Dataset gerado com {len(df_final)} amostras.")
    print("Agora seu modelo aprenderá a detectar ambos os ataques!")
else:
    print("Erro: Nenhum dado foi processado.")