import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Supondo que você ainda tenha o df_normalized carregado na memória
# Se não tiver, carregue o CSV novamente:
df_normalized = pd.read_csv('Spoofing-Detection-using-GNSS-Signals/Versão2/features_normalizadas_com_fft3.csv')

# Vamos plotar APENAS as duas features vencedoras
features_campeas = ['FFT_Mean', 'Maximo']

plt.figure(figsize=(12, 5))

for i, col in enumerate(features_campeas):
    plt.subplot(1, 2, i+1)
    sns.histplot(data=df_normalized, x=col, hue='Classe', kde=True, 
                 palette={0: 'blue', 1: 'red'}, element="step")
    plt.title(f'Distribuição: {col} (Feature Importante!)')

plt.tight_layout()
plt.show()