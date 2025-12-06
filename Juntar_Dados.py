import pandas as pd
import numpy as np
import os

# ==============================================================================
# 1. CONFIGURAÇÃO - Nomes dos seus arquivos
# ==============================================================================
FILE_FEATURES = 'features_parciais_sem_cn0.csv'  # Suas estatísticas (Potencia, Curtose...)
FILE_CN0_CLEAN = 'cn0_clean.csv'                 # C/N0 do arquivo Limpo (NMEA)
FILE_CN0_SPOOF = 'cn0_ds2.csv'                 # C/N0 do arquivo Spoofing (NMEA)

# ==============================================================================
# 2. FUNÇÃO MÁGICA DE REALINHAMENTO
# ==============================================================================
def ajustar_cn0_ao_dataset(cn0_values, tamanho_alvo):
    """
    Estica ou encolhe o vetor de C/N0 para ter exatamente 'tamanho_alvo' linhas.
    Usa interpolação linear.
    """
    qtd_atual = len(cn0_values)
    
    # Cria eixo X antigo (ex: 0 a 8000)
    x_antigo = np.linspace(0, qtd_atual - 1, qtd_atual)
    
    # Cria eixo X novo (ex: 0 a 8000, mas com apenas 'tamanho_alvo' pontos)
    x_novo = np.linspace(0, qtd_atual - 1, tamanho_alvo)
    
    # Interpola (Redimensiona os dados)
    cn0_novo = np.interp(x_novo, x_antigo, cn0_values)
    
    return cn0_novo

# ==============================================================================
# 3. PROCESSO DE FUSÃO
# ==============================================================================
print("--- Iniciando Fusão Final ---")

# 1. Carregar Features
try:
    df_feat = pd.read_csv(FILE_FEATURES)
    print(f"Features carregadas: {len(df_feat)} linhas totais.")
except FileNotFoundError:
    print("ERRO: Arquivo de features não encontrado.")
    exit()

# 2. Separar Clean (Classe 0) e Spoof (Classe 1)
# Precisamos tratar cada grupo separado para não misturar os C/N0s
idx_clean = df_feat[df_feat['Classe'] == 0].index
idx_spoof = df_feat[df_feat['Classe'] == 1].index

tam_clean = len(idx_clean)
tam_spoof = len(idx_spoof)

print(f"Alvos para alinhar -> Clean: {tam_clean} linhas | Spoof: {tam_spoof} linhas")

# 3. Carregar e Ajustar C/N0 LIMPO
try:
    # Tenta ler CSV (assumindo que tem cabeçalho ou é só coluna)
    # Se seu CSV tiver cabeçalho 'CN0', use usecols=['CN0']. Se não, header=None.
    # Vou tentar ler genérico:
    df_c = pd.read_csv(FILE_CN0_CLEAN)
    vals_c = df_c.iloc[:, 0].values # Pega a primeira coluna numérica
    
    # Ajusta o tamanho para caber nas features limpas
    cn0_clean_final = ajustar_cn0_ao_dataset(vals_c, tam_clean)
    print("-> C/N0 Clean alinhado com sucesso.")
except Exception as e:
    print(f"[AVISO] Falha ao ler {FILE_CN0_CLEAN}: {e}")
    print("Preenchendo com zeros (pode afetar a acurácia).")
    cn0_clean_final = np.zeros(tam_clean)

# 4. Carregar e Ajustar C/N0 SPOOFING
try:
    df_s = pd.read_csv(FILE_CN0_SPOOF)
    vals_s = df_s.iloc[:, 0].values
    
    cn0_spoof_final = ajustar_cn0_ao_dataset(vals_s, tam_spoof)
    print("-> C/N0 Spoof alinhado com sucesso.")
except Exception as e:
    print(f"[AVISO] Falha ao ler {FILE_CN0_SPOOF}: {e}")
    cn0_spoof_final = np.zeros(tam_spoof)

# 5. Inserir no DataFrame Principal
# Criamos a coluna vazia
df_feat['CN0_NMEA'] = 0.0

# Colocamos os dados nos lugares certos usando os índices que salvamos antes
df_feat.loc[idx_clean, 'CN0_NMEA'] = cn0_clean_final
df_feat.loc[idx_spoof, 'CN0_NMEA'] = cn0_spoof_final

# 6. Salvar o Dataset de Ouro
nome_final = 'dataset_FINAL_TREINO.csv'
df_feat.to_csv(nome_final, index=False)

print("\n" + "="*50)
print(f"SUCESSO TOTAL! Arquivo gerado: {nome_final}")
print("Verifique as colunas. Deve ter: Potencia, Curtose... e CN0_NMEA.")
print("Agora é só treinar o Random Forest com este arquivo!")
print("="*50)