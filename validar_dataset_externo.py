import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

# ==============================================================================
# CONFIGURAÇÃO
# ==============================================================================
ARQUIVO_MODELO = 'meu_detector_gps.pkl'       # O cérebro treinado
ARQUIVO_NOVO_DATASET = 'dataset_NOVO_TESTE.csv' # O CSV do novo dataset processado

# VOCÊ SABE A RESPOSTA CORRETA DESSE DATASET?
# Se o dataset for misturado, deixe True e garanta que tem a coluna 'Classe'.
# Se for um teste cego (você não tem a coluna Classe), mude para False.
TEM_GABARITO = True 

# ==============================================================================
# EXECUÇÃO
# ==============================================================================
print("--- Iniciando Validação Externa ---")

# 1. Carregar o Modelo
try:
    model = joblib.load(ARQUIVO_MODELO)
    print("Modelo carregado com sucesso!")
except FileNotFoundError:
    print("ERRO: Não achei o arquivo .pkl. Rode o script de salvar modelo antes.")
    exit()

# 2. Carregar os Novos Dados
try:
    df_novo = pd.read_csv(ARQUIVO_NOVO_DATASET)
    print(f"Novos dados carregados: {len(df_novo)} linhas.")
except FileNotFoundError:
    print("ERRO: Não achei o CSV novo. Você processou os dados novos?")
    exit()

# 3. Preparar as Colunas (Garantir a mesma ordem do treino)
# O modelo precisa EXATAMENTE das colunas que usou no treino, menos a Classe.
colunas_esperadas = ['Potencia', 'Maximo', 'DesvioPadrao', 'Curtose', 'Assimetria', 'CN0_NMEA']

# Verifica se falta alguma
for col in colunas_esperadas:
    if col not in df_novo.columns:
        print(f"ERRO CRÍTICO: O novo dataset não tem a coluna '{col}'.")
        print("Você precisa rodar o script de fusão corretamente.")
        exit()

# Seleciona só o que interessa para a previsão
X_novo = df_novo[colunas_esperadas]

# 4. Fazer a Previsão
print("Classificando...")
predicoes = model.predict(X_novo)
probabilidades = model.predict_proba(X_novo)

# Conta quantos de cada tipo ele achou
contagem = pd.Series(predicoes).value_counts()
print("\n--- RESULTADO DA ANÁLISE ---")
print(f"Total analisado: {len(predicoes)}")
print(f"Detectado como Autêntico (0): {contagem.get(0, 0)}")
print(f"Detectado como Spoofing  (1): {contagem.get(1, 0)}")

# 5. Se tiver gabarito, calcula a nota
if TEM_GABARITO and 'Classe' in df_novo.columns:
    y_real = df_novo['Classe']
    acuracia = accuracy_score(y_real, predicoes)
    print(f"\n>>> ACURÁCIA NESTE DATASET: {acuracia*100:.2f}% <<<")
    
    # Matriz de Confusão do Novo Teste
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_real, predicoes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                xticklabels=['Pred: Autêntico', 'Pred: Spoofing'],
                yticklabels=['Real: Autêntico', 'Real: Spoofing'])
    plt.title(f'Validação em Dataset Externo\nAcurácia: {acuracia:.1%}')
    plt.tight_layout()
    plt.show()

# 6. Salvar as predições num arquivo (para você analisar depois)
df_novo['Predicao_IA'] = predicoes
df_novo['Confianca_Spoofing'] = probabilidades[:, 1] # Chance de ser spoofing

df_novo.to_csv('resultado_teste_externo.csv', index=False)
print(f"\nRelatório salvo em 'resultado_teste_externo.csv'.")