import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, welch

# --- 1. CONFIGURAÇÕES (AJUSTE AQUI) ---
ARQUIVO_CLEAN = "Dados_GPS/cleanStatic.bin" 
#ARQUIVO_SPOOFING = "Dados_GPS/ds2.bin"
ARQUIVO_SPOOF = "Dados_GPS/cenario5/ds4.bin"

FS = 25000000       # Taxa de Amostragem do TexBAT (25 MHz)
N_AMOSTRAS = 200000 # Quantidade de dados para carregar (suficiente para PSD e Espectrograma)
N_ZOOM = 300        # Quantidade de amostras para o gráfico de Tempo (Zoom)

# Configuração de estilo dos gráficos (para ficar bonito no relatório)
plt.rcParams.update({'font.size': 10, 'font.family': 'serif', 'axes.grid': True})

# --- 2. FUNÇÃO DE LEITURA E NORMALIZAÇÃO (AGC) ---
def ler_e_normalizar(caminho, n_amostras):
    """
    Lê o arquivo binário I/Q (int16), converte para complexo 
    e normaliza pela energia (RMS) para remover diferenças de ganho.
    """
    print(f"Lendo: {caminho}...")
    try:
        # TexBAT é int16 intercalado (I, Q, I, Q...)
        raw = np.fromfile(caminho, dtype=np.int16, count=n_amostras * 2)
        if len(raw) == 0:
            print(f"ERRO: Arquivo {caminho} está vazio ou não encontrado.")
            return None
        
        # Separa I e Q
        i_data = raw[0::2].astype(np.float32)
        q_data = raw[1::2].astype(np.float32)
        
        # Cria sinal complexo
        sinal = i_data + 1j * q_data
        
        # --- O PULO DO GATO: NORMALIZAÇÃO (AGC DIGITAL) ---
        # Calcula a energia média (RMS)
        rms = np.sqrt(np.mean(np.abs(sinal)**2))
        
        # Divide o sinal pelo RMS. Resultado: Potência média = 1.0
        if rms > 0:
            sinal_norm = sinal / rms
        else:
            sinal_norm = sinal
            
        return sinal_norm

    except FileNotFoundError:
        print(f"ERRO CRÍTICO: Não achei o arquivo em: {caminho}")
        return None

# --- 3. CARREGAMENTO DOS DADOS ---
sinal_clean = ler_e_normalizar(ARQUIVO_CLEAN, N_AMOSTRAS)
sinal_spoof = ler_e_normalizar(ARQUIVO_SPOOF, N_AMOSTRAS)

if sinal_clean is None or sinal_spoof is None:
    print("Parece que um dos arquivos falhou. Verifique os caminhos na linha 5 e 6.")
    exit()

print("Dados carregados e normalizados. Gerando plotagem...")

# --- 4. GERAÇÃO DOS GRÁFICOS (TRILOGIA) ---
fig = plt.figure(figsize=(14, 10), constrained_layout=True)
gs = fig.add_gridspec(2, 2) # Layout em grade 2x2

# --- GRÁFICO 1: DOMÍNIO DO TEMPO (A Ilusão) ---
# Mostra apenas um "zoom" para provar que visualmente são iguais
ax1 = fig.add_subplot(gs[0, 0])
t_axis = np.arange(N_ZOOM) # Eixo de amostras

# Plota a magnitude (envelope)
ax1.plot(t_axis, np.abs(sinal_clean[:N_ZOOM]), color='blue', alpha=0.6, label='Autêntico (Ruído)')
ax1.plot(t_axis, np.abs(sinal_spoof[:N_ZOOM]), color='red', alpha=0.6, linestyle='--', label='Spoofing (ds4)')

ax1.set_title("1. Domínio do Tempo (Amplitude Normalizada)")
ax1.set_xlabel("Amostras ($n$)")
ax1.set_ylabel("Magnitude $|x[n]|$")
ax1.legend(loc='upper right')
ax1.text(0.5, 0.5, "Indistinguível Visualmente\n(Potência Equivalente)", 
         transform=ax1.transAxes, ha='center', va='center', 
         bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))

# --- GRÁFICO 2: PSD - DOMÍNIO DA FREQUÊNCIA (A Revelação) ---
# Usa o método de Welch para estimar a densidade espectral
ax2 = fig.add_subplot(gs[0, 1])

# Calcula PSD para o Clean
f_c, Pxx_c = welch(sinal_clean, fs=FS, nperseg=2048, return_onesided=False)
f_c, Pxx_c = np.fft.fftshift(f_c), np.fft.fftshift(Pxx_c) # Centraliza o 0 Hz

# Calcula PSD para o Spoof
f_s, Pxx_s = welch(sinal_spoof, fs=FS, nperseg=2048, return_onesided=False)
f_s, Pxx_s = np.fft.fftshift(f_s), np.fft.fftshift(Pxx_s) # Centraliza o 0 Hz

# Plota em escala Logarítmica (dB)
ax2.semilogy(f_c/1e6, Pxx_c, color='blue', alpha=0.5, label='Autêntico (Plano)')
ax2.semilogy(f_s/1e6, Pxx_s, color='red', alpha=0.8, label='Spoofing (Pico)')

ax2.set_title("2. Densidade Espectral de Potência (PSD)")
ax2.set_xlabel("Frequência (MHz)")
ax2.set_ylabel("Densidade (dB/Hz)")
ax2.legend()
ax2.grid(True, which="both", linestyle='--', alpha=0.5)

# Anotação apontando para o pico (opcional, ajustável visualmente)
pico_max = np.max(Pxx_s)
ax2.annotate('Assinatura Espectral\n(Coerência do Ataque)', 
             xy=(0, pico_max), xytext=(3, pico_max),
             arrowprops=dict(facecolor='black', shrink=0.05))

# --- GRÁFICO 3: ESPECTROGRAMA DO SPOOFING (Tempo x Frequência) ---
# Mostra a persistência do sinal falso
ax3 = fig.add_subplot(gs[1, :]) # Ocupa a parte de baixo toda

# Calcula Espectrograma apenas do sinal Spoofing
f_spec, t_spec, Sxx = spectrogram(sinal_spoof, fs=FS, nperseg=512, noverlap=256, return_onesided=False)
Sxx_shifted = np.fft.fftshift(Sxx, axes=0)
f_shifted = np.fft.fftshift(f_spec)

# Plota mapa de calor (dB)
im = ax3.pcolormesh(t_spec*1000, f_shifted/1e6, 10 * np.log10(Sxx_shifted), shading='gouraud', cmap='inferno')

ax3.set_title("3. Espectrograma do Sinal Spoofing (Estabilidade Temporal)")
ax3.set_ylabel("Frequência (MHz)")
ax3.set_xlabel("Tempo (ms)")
cbar = fig.colorbar(im, ax=ax3, orientation='vertical', pad=0.01)
cbar.set_label('Potência (dB)')

# --- FINALIZAÇÃO ---
plt.suptitle("Análise de Sinais e Sistemas: Detecção de GPS Spoofing (Cenário Matched Power)", fontsize=14, fontweight='bold')
print("Gráfico gerado com sucesso!")
plt.show()