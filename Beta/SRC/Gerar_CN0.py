import re
from collections import defaultdict, OrderedDict
from datetime import datetime, time, timedelta
import pandas as pd
import numpy as np
import os

# ==============================================================================
# CONFIGURAÇÃO - EDITE AQUI ANTES DE RODAR
# ==============================================================================

# 1. Caminho do arquivo de entrada (NMEA)
# Mude para o seu arquivo CLEAN na primeira vez, e SPOOF na segunda
NMEA_FILE = "Dados_GPS/cenario5/nmea_pvt.nmea"  
# NMEA_FILE = "dados/Cenario 2/nmea_pvt.nmea"

# 2. Nome do arquivo de saída (CSV para a IA)
# Mude para 'cn0_clean.csv' na primeira vez, e 'cn0_spoof.csv' na segunda
ARQUIVO_SAIDA = "cn0_ds4.csv"
# ARQUIVO_SAIDA = "cn0_spoof.csv"

# ==============================================================================

print(f"Lendo: {NMEA_FILE}")
print(f"Saída será: {ARQUIVO_SAIDA}")

gsv_re = re.compile(r'^\$(GP|GN|GL|GA|GB)GSV,')
gga_re = re.compile(r'^\$(GP|GN|GL|GA|GB)GGA,')
rmc_re = re.compile(r'^\$(GP|GN|GL|GA|GB)RMC,')

def parse_nmea_time(timestr):
    if not timestr: return None
    try:
        hh = int(timestr[0:2])
        mm = int(timestr[2:4])
        ss = float(timestr[4:])
        return time(hour=hh, minute=mm, second=int(ss), microsecond=int((ss - int(ss)) * 1e6))
    except: return None

per_epoch = OrderedDict()
current_time = None
last_epoch_dt = None

try:
    with open(NMEA_FILE, 'r', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('!'): continue

            if gga_re.match(line) or rmc_re.match(line):
                parts = line.split(',')
                if len(parts) > 1:
                    t = parse_nmea_time(parts[1])
                    if t:
                        dt = datetime.combine(datetime.today(), t)
                        if last_epoch_dt and dt < last_epoch_dt - timedelta(seconds=10):
                            dt += timedelta(days=1)
                        last_epoch_dt = dt
                        current_time = dt
                        if current_time not in per_epoch:
                            per_epoch[current_time] = {}
                continue

            if gsv_re.match(line):
                parts = line.split(',')
                try:
                    sat_data = parts[4:]
                    for i in range(0, len(sat_data), 4):
                        if i + 3 >= len(sat_data): break
                        prn = sat_data[i].strip()
                        snr_raw = sat_data[i+3].strip()
                        if prn == '' or not snr_raw: continue
                        
                        try: snr = float(snr_raw)
                        except: continue

                        # Se não tiver tempo associado (alguns logs são assim), cria tempo artificial
                        if current_time is None:
                            idx = len(per_epoch)
                            ts = datetime.now() + timedelta(seconds=idx)
                            per_epoch.setdefault(ts, {})[prn] = snr
                        else:
                            per_epoch.setdefault(current_time, {})[prn] = snr
                except: continue

except FileNotFoundError:
    print(f"ERRO: Arquivo não encontrado: {NMEA_FILE}")
    exit()

# Cria DataFrame
df_epochs = pd.DataFrame.from_dict(per_epoch, orient='index').sort_index()

# Calcula a MÉDIA de C/N0 de todos os satélites naquele instante
# Isso gera uma coluna única que representa a "qualidade global" do sinal
cn0_mean = df_epochs.mean(axis=1, skipna=True)

# Salva para CSV simples (apenas números, com cabeçalho 'CN0')
df_final = pd.DataFrame({'CN0': cn0_mean.values})
df_final.to_csv(ARQUIVO_SAIDA, index=False)

print("-" * 30)
print(f"SUCESSO! Arquivo '{ARQUIVO_SAIDA}' gerado com {len(df_final)} linhas.")
print("-" * 30)