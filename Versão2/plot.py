import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

# --- CONFIGURAÇÃO ---
ARQUIVO_CLEAN = "Dados_GPS/cleanStatic.bin" 
#ARQUIVO_SPOOFING = "Dados_GPS/ds2.bin"
ARQUIVO_SPOOF = "Dados_GPS/cenario5/ds4.bin"
N_AMOSTRAS = 100000 # Vamos pegar um pedaço maior para ver bem o espectro
FS = 25000000 # Taxa de amostragem estimada do TexBAT (25 MHz) - Ajuste se souber o exato

def ler_sinal_normalizado(caminho, n_amostras):
    """Lê o arquivo binário, converte para complexo e normaliza (AGC)."""
    try:
        # Lê raw data (Intercalado I, Q, I, Q...)
        raw = np.fromfile(caminho, dtype=np.int16, count=n_amostras * 2)
        if len(raw) == 0: return None
        
        # Separa I e Q e cria sinal complexo
        i_data = raw[0::2].astype(np.float32)
        q_data = raw[1::2].astype(np.float32)
        sinal = i_data + 1j * q_data
        
        # --- NORMALIZAÇÃO (AGC) ---
        # Divide pela energia média para remover diferenças de volume/ganho
        # Assim comparamos apenas a FORMA do sinal
        energia = np.sqrt(np.mean(np.abs(sinal)**2))
        if energia > 0:
            sinal = sinal / energia
            
        return sinal
    except FileNotFoundError:
        print(f"Erro: Arquivo {caminho} não encontrado.")
        return None

# --- LEITURA DOS SINAIS ---
print("Lendo sinais e aplicando AGC...")
sinal_clean = ler_sinal_normalizado(ARQUIVO_CLEAN, N_AMOSTRAS)
sinal_spoof = ler_sinal_normalizado(ARQUIVO_SPOOF, N_AMOSTRAS)

if sinal_clean is None or sinal_spoof is None:
    print("Falha na leitura. Verifique os caminhos.")
    exit()

# --- PLOTAGEM DE SINAIS E SISTEMAS ---
plt.figure(figsize=(14, 10))

# 1. DENSIDADE ESPECTRAL DE POTÊNCIA (PSD) - O Gráfico Mais Importante
# Este gráfico mostra "onde" está a energia na frequência.
# O Clean deve ser "plano" (ruído branco).
# O Spoof deve ter uma "barriga" ou pico (sinal colorido/coerente).
plt.subplot(2, 2, (1, 2)) # Ocupa a largura toda em cima
f_clean, Pxx_clean = welch(sinal_clean, fs=FS, nperseg=2048, return_onesided=False)
f_spoof, Pxx_spoof = welch(sinal_spoof, fs=FS, nperseg=2048, return_onesided=False)

# Ordena frequências para plotar (fftshift visual)
f_clean, Pxx_clean = np.fft.fftshift(f_clean), np.fft.fftshift(Pxx_clean)
f_spoof, Pxx_spoof = np.fft.fftshift(f_spoof), np.fft.fftshift(Pxx_spoof)

plt.semilogy(f_clean/1e6, Pxx_clean, label='Sinal Autêntico (Ruído)', color='blue', alpha=0.7)
plt.semilogy(f_spoof/1e6, Pxx_spoof, label='Sinal Spoofing (ds4)', color='red', alpha=0.7)
plt.title("Comparação Espectral (PSD) - O 'DNA' do Sinal")
plt.xlabel("Frequência (MHz)")
plt.ylabel("Densidade de Potência (dB/Hz)")
plt.legend()
plt.grid(True, which="both", ls="-")

# 2. DIAGRAMA DE CONSTELAÇÃO (Scatter Plot)
# Mostra a dispersão dos símbolos I/Q.
# Ruído puro é uma bola difusa. Sinal estruturado pode ter forma diferente.
plt.subplot(2, 2, 3)
plt.scatter(sinal_clean.real[:5000], sinal_clean.imag[:5000], s=1, color='blue', alpha=0.3)
plt.title("Constelação: Sinal Autêntico")
plt.xlabel("In-Phase (I)")
plt.ylabel("Quadrature (Q)")
plt.axis('equal')
plt.grid()

plt.subplot(2, 2, 4)
plt.scatter(sinal_spoof.real[:5000], sinal_spoof.imag[:5000], s=1, color='red', alpha=0.3)
plt.title("Constelação: Sinal Spoofing")
plt.xlabel("In-Phase (I)")
plt.ylabel("Quadrature (Q)")
plt.axis('equal')
plt.grid()

plt.tight_layout()
plt.show()