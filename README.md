# Spoofing-Detection-using-GNSS-Signals
Sistema de detecção de GPS Spoofing baseado em análise estatística e espectral de sinais, aplicado ao cenário DS4 (Static Matched-Power Position Push) do dataset TEXBAT (Texas Spoofing Test Battery).

O projeto explora conceitos fundamentais de Sinais e Sistemas e Processamento Digital de Sinais (PDS) para identificar ataques de spoofing altamente sofisticados, mesmo quando a potência do sinal falso é cuidadosamente ajustada para mimetizar o sinal autêntico.

📌 Contexto do Problema

O ataque de spoofing GPS consiste na injeção de sinais falsos que imitam os sinais legítimos dos satélites, desviando a navegação do receptor.

No cenário DS4 (Matched Power):

O atacante ajusta a potência do sinal falso para ser quase idêntica à do sinal legítimo.

O aumento de potência total na banda é inferior a 2 dB, tornando o ataque invisível a detectores simples (ex: AGC).

O sinal recebido passa a ser a soma de:

Um sinal autêntico dominado por ruído térmico (Gaussiano).

Um sinal falso determinístico, cuidadosamente modelado.

Esse cenário representa um dos casos mais difíceis de detecção.

🎯 Objetivo

Desenvolver um sistema de detecção de spoofing GPS capaz de identificar distorções estatísticas e espectrais sutis introduzidas pelo ataque DS4, mesmo quando:

A potência média é igualada.

A forma espectral aparenta ser visualmente semelhante.

O ataque foi projetado para enganar detectores tradicionais.

🧠 Fundamentação Teórica

A metodologia baseia-se na seguinte premissa:

Sinal autêntico: dominado por ruído térmico → distribuição aproximadamente Gaussiana.

Sinal com spoofing: contém componentes determinísticos → distorção da estatística do sinal, mesmo com potência igual.

Essas distorções são capturadas por métricas no:

Domínio do Tempo

Domínio da Frequência (FFT)

🛠️ Ferramentas Utilizadas

Python

NumPy – Processamento numérico

SciPy (FFT) – Análise espectral

Scikit-learn – Machine Learning (Random Forest)

Dataset: TEXBAT (Texas Spoofing Test Battery)

🔧 Metodologia
1. Reconstrução do Sinal Complexo

Os sinais são processados em banda base complexa:

s(n) = I(n) + jQ(n)


Essa representação preserva informações de amplitude e fase, essenciais para análise espectral.

2. Pré-processamento – Normalização RMS (Emulação de AGC)

Foi identificada uma discrepância de calibração no dataset:

O sinal autêntico (cleanStatic) apresenta maior amplitude média do que os sinais de spoofing.

Para eliminar esse viés:

Aplicou-se normalização por RMS (Root Mean Square), simulando um AGC de hardware.

Todos os sinais passam a ter potência média unitária.

🔎 Interpretação em Sinais e Sistemas:

Sistema não-linear e variante no tempo.

Força o classificador a aprender a morfologia estatística, e não a energia absoluta.

3. Extração de Features
Domínio do Tempo

Máximo (x_max): identifica picos construtivos da superposição de sinais.

Desvio Padrão (σ): captura variações sutis na distribuição do ruído.

Potência: usada apenas como verificação da normalização.

Domínio da Frequência (FFT)

FFT_Mean: média das magnitudes espectrais (piso de ruído).

FFT_Peak: maior pico espectral (energia concentrada / componentes determinísticas).

4. Classificação – Random Forest

Classificador não-linear e robusto.

Dataset com mais de 6.000 amostras de features.

Divisão:

Treinamento: 70%

Teste/Validação: 30% (2.012 amostras)

📊 Resultados
Desempenho Global

Acurácia: ~ 81.4% (≈ 81.6%)

Importância das Features
Feature	Importância
FFT_Mean	~55%
Desvio Padrão	~25%
FFT_Peak	~15%
Potência	~5%

🔎 Conclusões importantes:

A Média da Assinatura Espectral (FFT_Mean) é o principal discriminador.

O Desvio Padrão revela diferenças sutis entre ruído térmico e ruído artificial.

A baixa relevância da potência valida a eficácia da normalização RMS.

⚠️ Limitações

Erro total: ~18.6%

Falsos Positivos: ~15.8%

Falsos Negativos: ~21.7%

Essas limitações são intrínsecas ao cenário DS4, que:

É projetado para mimetizar o ruído térmico.

Cria uma zona cinzenta estatística, onde as distribuições se sobrepõem.

Torna impossível uma separação perfeita apenas com features de tempo e frequência.

📚 Conexão com Sinais e Sistemas

O projeto aplica diretamente conceitos clássicos da disciplina:

Sinais em Banda Base Complexa

Teorema de Parseval e Potência Média

Normalização RMS (AGC)

Transformada de Fourier (FFT)

Sistemas Não-Lineares e Variantes no Tempo

Classificação Não-Linear (Random Forest)

✅ Conclusão

Este projeto demonstra que é possível:

Transformar um receptor GPS comum em um detector de desvios estatísticos.

Detectar spoofing sofisticado mesmo quando a potência é cuidadosamente mascarada.

Utilizar Processamento Digital de Sinais + Machine Learning de forma complementar.

📌 Resultado-chave:
A Média da Assinatura Espectral mostrou-se a métrica mais eficaz para identificar contaminação determinística em sinais de navegação GPS.
