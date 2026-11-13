import numpy as np
import pandas as pd
number_data= pd.read_csv('train.csv')

Y= number_data['label'].values
X = number_data.iloc[:, 1:].values / 255.0
YY=pd.get_dummies(Y).values
alpha=0.1
reponse= input("Reprendre l'entraînement précédent ? (o/n) : ")
reprendre=(reponse=='o')



if reprendre:
    W = np.load('W.npy')
    b = np.load('b.npy')
else :
    W = np.zeros((784, 10))
    b = np.zeros((1, 10))

for i in range(1000):
    Z = X @ W + b
    exp_Z = np.exp(Z)
    sum = np.sum(exp_Z, axis=1, keepdims=True)
    A = exp_Z / sum
    m = X.shape[0]
    loss = -np.sum(YY * np.log(A)) / m
    dZ = A - YY
    dw = (1 / m) * (X.T @ dZ)
    db = (1 / m) * np.sum(dZ,axis=0,keepdims=True)

    W = W - alpha * dw
    b = b - alpha * db
    if i % 10 == 0:
        print(f"Itération {i}, Perte : {loss}")



np.save('W.npy', W)
np.save('b.npy', b)


predictions = np.argmax(A, axis=1)
vrai_chiffres = Y

comparaison = (predictions == vrai_chiffres)
nb_reussites = np.sum(comparaison)
accuracy = nb_reussites / m
print(f"Précision finale : {accuracy * 100:.2f}%")