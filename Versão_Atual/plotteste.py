import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

# --- CONFIGURAÇÃO ---
ARQ_AUTENTICO = "Dados_GPS/cleanStatic.bin"
ARQ_OVERPOWER = "Dados_GPS/ds2.bin"
ARQ_MATCHED   = "Dados_GPS/cenario5/ds4.bin" # Ajuste se necessário

DTYPE = np.int16
AMOSTRAS_PARA_PLOTAR = 2000 # Pegamos um trecho curto para o zoom ficar bom

def ler_um_chunk(caminho):
    """Lê apenas o primeiro pedaço do arquivo para visualização."""
    try:
        with open(caminho, 'rb') as f:
            # Lê um pouco mais para garantir estatística na FFT
            raw = np.fromfile(f, dtype=DTYPE, count=20000) 
            
        i_data = raw[0::2].astype(np.float32)
        q_data = raw[1::2].astype(np.float32)
        
        # Cria complexo e normaliza (opcional, mas ajuda na visualização comparativa)
        sinal = i_data + 1j * q_data
        
        # Remove nível DC
        sinal = sinal - np.mean(sinal)
        return sinal
    except FileNotFoundError:
        print(f"Erro: {caminho} não achado.")
        return None

# Carrega uma amostra de cada
sinal_clean = ler_um_chunk(ARQ_AUTENTICO)
sinal_over  = ler_um_chunk(ARQ_OVERPOWER)
sinal_match = ler_um_chunk(ARQ_MATCHED)

sinais = {
    "Autêntico (Ruído Térmico)": sinal_clean,
    "Spoofing (Matched Power)": sinal_match,
    "Spoofing (Overpower)": sinal_over
}

# --- PLOTAGEM MATEMÁTICA ---
fig, ax = plt.subplots(3, 3, figsize=(18, 10))
fig.suptitle("Análise Sinais e Sistemas: Comparação nos Domínios", fontsize=16)

cols = ["Autêntico (Ruído Térmico)", "Spoofing (Matched Power)", "Spoofing (Overpower)"]

for i, (nome, sinal) in enumerate(sinais.items()):
    if sinal is None: continue
    
    # 1. DOMÍNIO DO TEMPO (Magnitude)
    # Mostra a "força" bruta do sinal
    tempo = np.arange(AMOSTRAS_PARA_PLOTAR)
    mag = np.abs(sinal[:AMOSTRAS_PARA_PLOTAR])
    
    ax[0, i].plot(tempo, mag, color='tab:blue', lw=1)
    ax[0, i].set_title(f"{nome}\nDomínio do Tempo (Envelope)")
    ax[0, i].set_xlabel("Amostras")
    ax[0, i].set_ylabel("Amplitude")
    ax[0, i].grid(True, alpha=0.3)
    
    # 2. DOMÍNIO DA FREQUÊNCIA (PSD/Welch)
    # Mostra ONDE a energia está concentrada
    # Usamos Welch para uma estimativa de PSD mais suave que a FFT pura
    freqs, psd = welch(sinal, nperseg=1024, return_onesided=False)
    # Shift para centralizar o 0 Hz
    freqs = np.fft.fftshift(freqs)
    psd = np.fft.fftshift(psd)
    
    # Plot em dB (escala logarítmica é padrão em telecom)
    psd_db = 10 * np.log10(psd + 1e-12)
    
    ax[1, i].plot(freqs, psd_db, color='tab:orange')
    ax[1, i].set_title("Domínio da Frequência (PSD)")
    ax[1, i].set_xlabel("Frequência Normalizada")
    ax[1, i].set_ylabel("Potência (dB)")
    ax[1, i].grid(True, alpha=0.3)
    
    # 3. HISTOGRAMA / PDF (Domínio da Amplitude)
    # Mostra a ESTATÍSTICA (Base para Curtose e Skewness)
    sns.histplot(np.abs(sinal), bins=50, kde=True, ax=ax[2, i], color='tab:green', stat="density")
    ax[2, i].set_title("PDF (Distribuição de Probabilidade)")
    ax[2, i].set_xlabel("Magnitude")
    ax[2, i].set_ylabel("Densidade")
    ax[2, i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()