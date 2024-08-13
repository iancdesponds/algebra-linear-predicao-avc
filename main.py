import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import autograd.numpy as np_   # Thinly-wrapped version of Numpy
from autograd import grad
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

df = pd.read_csv('data/healthcare-dataset-stroke-data.csv')
df = df.dropna() # Dropa linhas com valores nulos
df = df[df["gender"] != "Other"] # Dropa linha onde gender = other
df.replace({"stroke": 0}, -1, inplace=True) # Substitui 0 por -1
df["stroke"].value_counts()

df_sem_stroke = df.drop(columns=["stroke", "id"])
df_sem_stroke = pd.get_dummies(df_sem_stroke, drop_first=True)

def datasets(df_sem_stroke, df):
    X_train, X_test, y_train, y_test = train_test_split(df_sem_stroke, df["stroke"], train_size=0.5) # Divide o dataset em treino e teste

    # Transforma o df em um numpy array
    X_train = X_train.to_numpy(dtype=float).T # Transpõe o array para que cada coluna seja um ponto
    X_test = X_test.to_numpy(dtype=float).T # Transpõe o array para que cada coluna seja um ponto
    y_train = y_train.to_numpy(dtype=float).T # Transpõe o array para que cada coluna seja um ponto
    y_test = y_test.to_numpy(dtype=float).T # Transpõe o array para que cada coluna seja um ponto
    return X_train, X_test, y_train, y_test

def loss( parametros ): # Essa função calcula o erro médio quadrático
    w, b, pontos, val = parametros
    est = w.T @ pontos + b
    mse = np_.mean( (est - val)**2)
    return mse

def accuracy(y_test, y_est):
    return np.mean(np.sign(y_test)==np.sign(y_est))

def calcula_acuracia(df_sem_stroke, df):
    X_train, X_test, y_train, y_test = datasets(df_sem_stroke, df)

    g = grad(loss)

    w = np.random.randn(15 ,1) # Vetor de pesos
    w_ = w # Vetor de pesos inicial
    b = 0.0 # Viés / bias
    alpha = 10**-5 # Taxa de aprendizado

    for n in range(10000):
        grad_ = g( (w, b, X_train, y_train) )
        w -= alpha*grad_[0]
        b -= alpha*grad_[1]

    y_est = w.T @ X_test + b # Estimativa
    acc = accuracy(y_test, y_est) # Acurácia
    print(f"Accuracy -> {acc}")
    print("Vetor de pesos inicial:")
    print(w_)
    print("Vetor de pesos final:")
    print(w)

calcula_acuracia(df_sem_stroke, df)

# Resultados com maiores acurácias encontrados para diferentes valores de w, gerados aleatoriamente

w = np.array([[-0.24729168], [ 0.19434991], [ 0.23773054], [-1.9376755 ], [-0.23193858], [-0.22393396], [-0.20263472], [-0.53630738], [-0.3324345 ], [ 0.77656376], [ 0.77629564], [ 0.08099956], [-0.49731736], [ 1.18405913], [ 1.15017974]])
res = "Accuracy -> 0.8577832110839446"

w = np.array([[ 1.27422892], [ 0.57788955], [-1.15704503], [ 1.14186603], [ 0.56864049], [ 0.1354915 ], [ 1.04160439], [ 0.11069598], [-0.12440809], [ 0.21981187], [-0.31045596], [ 0.13195825], [ 0.69889204], [-0.26012764], [ 0.05030163]])
res = "Accuracy -> 0.8944580277098615"

w = np.array([[1.27422892],[0.57788955],[-1.15704503],[1.14186603],[0.56864049],[0.1354915],[1.04160439],[0.11069598],[-0.12440809],[0.21981187],[-0.31045596],[0.13195825],[0.69889204],[-0.26012764],[0.05030163]])
res = "Accuracy -> 0.8973105134474327"

w = np.array([[1.27422892],[0.57788955],[-1.15704503],[1.14186603],[0.56864049],[0.1354915],[1.04160439],[0.11069598],[-0.12440809],[0.21981187],[-0.31045596],[0.13195825],[0.69889204],[-0.26012764],[0.05030163]])
res = "Accuracy -> 0.9070904645476773"

# Compara a acurácia do modelo com a acurácia de um modelo que sempre chuta -1

def calcula_acuracia_chute(df_sem_stroke, df):
    X_train, X_test, y_train, y_test = datasets(df_sem_stroke, df) # Divide o dataset em treino e teste

    y_est = np.ones(y_test.shape) * -1 # Chute
    acc = accuracy(y_test, y_est) # Acurácia
    print(f"Accuracy -> {acc}")

calcula_acuracia_chute(df_sem_stroke, df) # Calcula a acurácia do modelo que sempre chuta -1

# Árvore de decisão usando o pacote sklearn:
tree = DecisionTreeClassifier(criterion='entropy')

X_train, X_test, y_train, y_test = train_test_split(df_sem_stroke, df["stroke"], train_size=0.5)

# Usando o método .fit() para ajustar os parâmetros da árvore:
tree.fit(X_train, y_train)

# Podemos visualizar a árvore de decisão em uma figura!
plt.figure( figsize=(50,50) )
a = plot_tree(tree, feature_names=list(X_train.columns), fontsize=10, 
              node_ids=False, impurity=False, filled=True)

# Encontrando quais são os principais fatores que influenciam a decisão da árvore:
importances = tree.feature_importances_
indices = np.argsort(importances)[::-1]
print("Feature ranking:")
for f in range(X_train.shape[1]):
    print(f"{f+1}. feature {X_train.columns[indices[f]]} ({importances[indices[f]]})") # Printa a importância de cada feature

# Gráfico de barras para visualizar a importância de cada feature:
plt.figure()
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.show()