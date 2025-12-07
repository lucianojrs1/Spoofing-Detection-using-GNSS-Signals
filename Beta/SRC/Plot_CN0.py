import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime, time, timedelta

# ==============================================================================
# 1. CONFIGURAÇÃO (EDITE AQUI COM OS NOMES DOS SEUS ARQUIVOS)
# ==============================================================================
ARQUIVO_CLEAN = "dados/dadoslimpos/nmea_pvt.nmea"  # Caminho do NMEA Autêntico
ARQUIVO_SPOOF = "dados/Cenario 2/nmea_pvt.nmea"     # Caminho do NMEA Spoofing

# ==============================================================================
# 2. FUNÇÃO DE LEITURA (Reutilizando sua lógica)
# ==============================================================================
def ler_cn0_do_nmea(caminho_arquivo, label_nome):
    print(f"Lendo arquivo: {caminho_arquivo}...")
    
    gsv_re = re.compile(r'^\$(GP|GN|GL|GA|GB)GSV,')
    per_epoch = {}
    current_time = None
    
    # Tentativa de criar um timestamp artificial caso o arquivo não tenha GGA/RMC
    # ou para simplificar a plotagem por "Número de Epoch"
    epoch_counter = 0

    try:
        with open(caminho_arquivo, 'r', errors='ignore') as f:
            for line in f:
                line = line.strip()
                
                # Se encontrar sentença de Tempo (GGA/RMC), atualiza o relógio
                # (Simplificado: vamos usar apenas o contador de linhas GSV para sincronizar)
                
                if gsv_re.match(line):
                    epoch_counter += 1
                    parts = line.split(',')
                    try:
                        # Extrai satélites e SNRs
                        sat_data = parts[4:]
                        snrs_nesta_linha = []
                        
                        for i in range(0, len(sat_data), 4):
                            if i + 3 >= len(sat_data): break
                            snr_str = sat_data[i+3].strip()
                            if snr_str:
                                try:
                                    snrs_nesta_linha.append(float(snr_str))
                                except:
                                    pass
                        
                        # Se capturou SNRs, calcula a média deste instante
                        if snrs_nesta_linha:
                            media_instante = np.mean(snrs_nesta_linha)
                            # Usa um índice sequencial para facilitar a comparação visual
                            per_epoch[epoch_counter] = media_instante
                            
                    except Exception:
                        continue
                        
    except FileNotFoundError:
        print(f"[ERRO] Arquivo não encontrado: {caminho_arquivo}")
        return None

    # Transforma em Series do Pandas
    s = pd.Series(per_epoch)
    s.name = label_nome
    return s

# ==============================================================================
# 3. EXECUÇÃO E PLOTAGEM
# ==============================================================================

# A. Ler os dados
series_clean = ler_cn0_do_nmea(ARQUIVO_CLEAN, "Autêntico (Clean)")
series_spoof = ler_cn0_do_nmea(ARQUIVO_SPOOF, "Ataque (Spoofing)")

if series_clean is None or series_spoof is None:
    print("Parece que um dos arquivos falhou. Verifique os caminhos.")
    exit()

# B. Alinhamento de Tempo (Resetar para t=0)
# Vamos plotar por "Amostra/Epoch" para sobrepor as curvas
# (Assumindo que ambos foram gravados com taxas parecidas, ex: 1Hz)

# Cria DataFrame para facilitar o plot
plt.figure(figsize=(14, 8))

# --- GRÁFICO 1: LINHA DO TEMPO (OVERLAY) ---
plt.subplot(2, 1, 1)

# Plot Autêntico (Azul)
plt.plot(series_clean.index, series_clean.values, 
         label=f'Autêntico (Média: {series_clean.mean():.1f} dB)', 
         color='blue', alpha=0.7, linewidth=1.5)

# Plot Spoofing (Vermelho)
# Se o arquivo spoof for maior ou menor, o eixo X se ajusta
plt.plot(series_spoof.index, series_spoof.values, 
         label=f'Spoofing (Média: {series_spoof.mean():.1f} dB)', 
         color='red', alpha=0.8, linewidth=1.5)

plt.title("Comparação Temporal: Média do C/N0 (Todos os Satélites)", fontsize=14)
plt.xlabel("Tempo (Epochs / Amostras)")
plt.ylabel("C/N0 Médio (dB-Hz)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# --- GRÁFICO 2: HISTOGRAMA (DISTRIBUIÇÃO) ---
# Esse é MUITO IMPORTANTE para o Random Forest. Mostra que as médias são diferentes.
plt.subplot(2, 1, 2)

plt.hist(series_clean.values, bins=30, color='blue', alpha=0.5, label='Autêntico', density=True)
plt.hist(series_spoof.values, bins=30, color='red', alpha=0.5, label='Spoofing', density=True)

plt.title("Histograma de Distribuição do C/N0", fontsize=14)
plt.xlabel("C/N0 (dB-Hz)")
plt.ylabel("Densidade de Probabilidade")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

print("\nAnálise Rápida:")
print(f"Média Autêntico: {series_clean.mean():.2f} dB-Hz")
print(f"Média Spoofing:  {series_spoof.mean():.2f} dB-Hz")
print("Se houver uma diferença grande nas médias ou na 'tremedeira' (variância),")
print("seu modelo de IA vai detectar isso facilmente.")