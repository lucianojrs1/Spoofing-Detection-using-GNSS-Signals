import pandas as pd
import numpy as np
import librosa
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import skew, kurtosis, entropy
import os
from tqdm import tqdm

# ==========================================
# 1. CONFIGURAÇÕES
# ==========================================
ARQUIVO_TREINO = 'dataset_completo_sinais_sistemas.csv'
ARQUIVO_ENTRADA_BIN = 'ds4.bin'
ARQUIVO_SAIDA_CSV = 'dataset_extraido_ds4_10k.csv'  # <--- O arquivo que será gerado

# Parâmetros de Áudio
TAXA_AMOSTRAGEM = 16000
DTYPE_BIN = np.int16
TEMPO_JANELA = 1.0      # 1 segundo
QTD_AMOSTRAS = 40000    # Total de linhas no arquivo final

TAMANHO_JANELA_SAMPLES = int(TAXA_AMOSTRAGEM * TEMPO_JANELA)

# ==========================================
# 2. PREPARAR O MODELO (Para adicionar a predição no arquivo)
# ==========================================
print("--- Passo 1: Carregando modelo para classificar as novas amostras ---")
if os.path.exists(ARQUIVO_TREINO):
    df_train = pd.read_csv(ARQUIVO_TREINO)
    X_train = df_train.drop('Classe', axis=1)
    y_train = df_train['Classe']
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    print("Modelo treinado com sucesso.")
else:
    print(f"AVISO: {ARQUIVO_TREINO} não encontrado. O arquivo será gerado sem a coluna de predição.")
    rf_model = None

# ==========================================
# 3. EXTRAÇÃO EM MASSA
# ==========================================
print(f"\n--- Passo 2: Extraindo {QTD_AMOSTRAS} amostras de '{ARQUIVO_ENTRADA_BIN}' ---")

if not os.path.exists(ARQUIVO_ENTRADA_BIN):
    print(f"ERRO: Arquivo {ARQUIVO_ENTRADA_BIN} não encontrado.")
    exit()

# Mapeia o arquivo binário
dados_brutos = np.memmap(ARQUIVO_ENTRADA_BIN, dtype=DTYPE_BIN, mode='r')
total_samples = len(dados_brutos)

# Gera índices aleatórios
indices = np.random.randint(0, total_samples - TAMANHO_JANELA_SAMPLES, QTD_AMOSTRAS)

dados_extraidos = []

print("Processando e gerando arquivo...")
for idx in tqdm(indices):
    # Recorta o áudio bruto
    y_raw = np.array(dados_brutos[idx : idx + TAMANHO_JANELA_SAMPLES])
    
    # IMPORTANTE: Mantendo a escala "bruta" (sem normalizar para -1 a 1)
    # para tentar replicar a escala de Variância do seu dataset original.
    y = y_raw.astype(np.float32) 
    
    try:
        # Extração de Features
        S = np.abs(librosa.stft(y))
        
        feat = {
            'Media_Temp': np.mean(y),
            'Var_Temp': np.var(y),
            'Skew_Temp': skew(y),
            'Kurt_Temp': kurtosis(y),
            'Fator_Crista': (np.max(np.abs(y)) / np.sqrt(np.mean(y**2))) if np.mean(y**2) > 0 else 0,
            'Entropia_Espectral': entropy(np.mean(S, axis=1) / np.sum(np.mean(S, axis=1))),
            'Flatness_Espectral': np.mean(librosa.feature.spectral_flatness(S=S)),
            'Centroide_Espectral': np.mean(librosa.feature.spectral_centroid(S=S, sr=TAXA_AMOSTRAGEM))
        }
        dados_extraidos.append(feat)
        
    except ValueError:
        continue

# Cria o DataFrame
df_novo = pd.DataFrame(dados_extraidos)

# ==========================================
# 4. CLASSIFICAÇÃO E SALVAMENTO
# ==========================================
if rf_model is not None:
    # Garante a ordem das colunas igual ao treino
    df_novo = df_novo[X_train.columns]
    
    # Faz a predição
    predicoes = rf_model.predict(df_novo)
    
    # Adiciona a coluna 'Classe' (que agora é a predição do modelo)
    df_novo['Classe'] = predicoes

# Salva em CSV
print(f"\n--- Passo 3: Salvando arquivo ---")
df_novo.to_csv(ARQUIVO_SAIDA_CSV, index=False)
print(f"Sucesso! Arquivo gerado: {ARQUIVO_SAIDA_CSV}")
print(df_novo.head()) # Mostra as primeiras linhas
	
