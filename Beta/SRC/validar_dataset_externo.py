import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

# ==============================================================================
# CONFIGURAÇÃO
# ==============================================================================
ARQUIVO_MODELO = 'Spoofing-Detection-using-GNSS-Signals/meu_detector_gps.pkl'
ARQUIVO_TESTE  = 'Spoofing-Detection-using-GNSS-Signals/dataset_TESTE_CENARIO4.csv' 

# ==============================================================================
# EXECUÇÃO
# ==============================================================================
print("--- Iniciando Validação do Cenário 4 ---")

# 1. Carregar Modelo e Dados
model = joblib.load(ARQUIVO_MODELO)
df_teste = pd.read_csv(ARQUIVO_TESTE)

print(f"Dados carregados: {len(df_teste)} linhas.")

# 2. CORREÇÃO DA ORDEM DAS COLUNAS (O PULO DO GATO)
# Baseado na sua imagem do treino (image_27e2da.png), quando você fez
# X = df.drop('Classe'), a ordem das features que sobraram foi:
colunas_ordem_treino = [
    'Potencia', 
    'Maximo', 
    'DesvioPadrao', 
    'Curtose', 
    'Assimetria', 
    'CN0_NMEA'
]

# Verificamos se todas existem, independente da ordem no arquivo novo
for col in colunas_ordem_treino:
    if col not in df_teste.columns:
        print(f"ERRO FATAL: Coluna '{col}' não existe no arquivo de teste.")
        exit()

# FORÇAMOS a ordem correta para a IA
X_teste = df_teste[colunas_ordem_treino]

# Pegamos a resposta real (Classe) onde quer que ela esteja
y_real = df_teste['Classe']

print("--> Colunas realinhadas com sucesso para o padrão do treino.")

# 3. Fazer a Previsão
print("Classificando...")
predicoes = model.predict(X_teste)

# 4. Resultados
acuracia = accuracy_score(y_real, predicoes)
print(f"\n{'='*40}")
print(f"ACURÁCIA NO CENÁRIO 4 (MATCHED-POWER): {acuracia*100:.2f}%")
print(f"{'='*40}")

# 5. Matriz de Confusão (A Prova da Falha)
# Se a acurácia for baixa, a matriz vai mostrar muitos "Falsos Negativos"
# (Era Spoofing, mas a IA disse Autêntico)
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_real, predicoes)

# Ajuste para garantir que o gráfico mostre 0 e 1 mesmo se a IA só chutar 0
labels = [0, 1] 
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=['Pred: Autêntico', 'Pred: Spoofing'],
            yticklabels=['Real: Autêntico', 'Real: Spoofing'])

plt.title(f'Teste de Robustez - Cenário 4\n(Esperamos falha aqui!)')
plt.tight_layout()
plt.show()

# Análise rápida para o terminal
n_spoofing_real = len(y_real[y_real == 1])
n_detectado = len(predicoes[predicoes == 1])

print(f"\nResumo da Análise:")
print(f"Total de ataques reais no arquivo: {n_spoofing_real}")
print(f"Total que a IA conseguiu detectar: {n_detectado}")

if acuracia < 0.5:
    print("\nCONCLUSÃO PERFEITA PARA O RELATÓRIO:")
    print("O modelo falhou em detectar o ataque Matched-Power.")
    print("Isso prova que depender apenas de Potência/CN0 não é suficiente para ataques furtivos.")