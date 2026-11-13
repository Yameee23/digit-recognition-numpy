import numpy as np
import pandas as pd

number_data = pd.read_csv('train.csv')

X = number_data.iloc[:, 1:].values / 255
Y = number_data['label'].values
YY = pd.get_dummies(Y).values
alpha = 0.1
reponse = input("Reprendre l'entraînement précédent ? (o/n) : ")
reprendre = (reponse == 'o')

if reprendre:

    W1 = np.load('W1.npy')
    b1 = np.load('b1.npy')
    W2 = np.load('W2.npy')
    b2 = np.load('b2.npy')

else:
    W1 = np.random.randn(784, 64) * 0.01
    W2 = np.random.randn(64, 10) * 0.01
    b1 = np.zeros((1, 64))
    b2 = np.zeros((1, 10))

for i in range(1000):

    Z1 = X @ W1 + b1
    A1 = np.maximum(0, Z1)
    Z2 = A1 @ W2 + b2
    exp_Z2 = np.exp(Z2)
    sum = np.sum(exp_Z2, axis=1, keepdims=True)
    A2 = exp_Z2 / sum
    m = X.shape[0]
    loss = -np.sum(YY * np.log(A2)) / m

    dZ2 = A2 - YY
    dW2 = (1 / m) * (A1.T @ dZ2)
    db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * (Z1 > 0)
    dW1 = (1 / m) * (X.T @ dZ1)
    db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    if i % 10 == 0:
        print(f"Itération {i}, Perte : {loss}")

np.save('W1.npy', W1)
np.save('b1.npy', b1)
np.save('W2.npy', W2)
np.save('b2.npy', b2)

predictions = np.argmax(A2, axis=1)
vrai_chiffres = Y

comparaison = (predictions == vrai_chiffres)
nb_reussites = np.sum(comparaison)
accuracy = nb_reussites / m
print(f"Précision finale : {accuracy * 100:.2f}%")
