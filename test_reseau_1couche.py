import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

number_data = pd.read_csv('train.csv')
X = number_data.iloc[:,1:].values/255
Y = number_data['label'].values

W = np.load('W.npy')
b = np.load('b.npy')


index = 100
image_test = X[index:index+1]


Z_test = image_test @ W + b
exp_Z_test = np.exp(Z_test)
A_test = exp_Z_test / np.sum(exp_Z_test, axis=1, keepdims=True)
prediction = np.argmax(A_test)

print(f"L'IA dit que c'est un : {prediction}")
print(f"La vraie réponse est : {Y[index]}")

plt.imshow(image_test.reshape(28, 28), cmap='gray')
plt.title(f"Prédiction : {prediction}")
plt.show()