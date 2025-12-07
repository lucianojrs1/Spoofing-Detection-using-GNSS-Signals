import pandas as pd
import numpy as np

# 1. Carregar os arquivos gerados acima
try:
    df_feat = pd.read_csv("features_cenario4.csv")
    df_cn0 = pd.read_csv("cn0_ds4.csv")
except FileNotFoundError:
    print("Rode os passos 1 e 2 primeiro!")
    exit()

# 2. Sincronizar (Interpolação)
print("Sincronizando...")
cn0_vals = df_cn0.iloc[:, 0].values
x_old = np.linspace(0, len(cn0_vals)-1, len(cn0_vals))
x_new = np.linspace(0, len(cn0_vals)-1, len(df_feat))

cn0_interp = np.interp(x_new, x_old, cn0_vals)

# 3. Montar Dataset Final
df_feat['CN0_NMEA'] = cn0_interp

# GABARITO: Sabemos que é Spoofing (Classe 1), 
# mas queremos ver se a IA consegue descobrir isso sozinha.
df_feat['Classe'] = 1 

df_feat.to_csv("dataset_TESTE_CENARIO4.csv", index=False)
print("PRONTO! Arquivo 'dataset_TESTE_CENARIO4.csv' criado.")
print("Agora rode o script de validação (validar_dataset_externo.py) apontando para este arquivo.")