import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

number_data = pd.read_csv('train.csv')
X = number_data.iloc[:,1:].values/255
Y = number_data['label'].values

W1 = np.load('W1.npy')
b1 = np.load('b1.npy')
W2 = np.load('W2.npy')
b2 = np.load('b2.npy')


index = 100
image_test = X[index:index+1]
Z1_test=image_test @ W1 + b1
A1_test= np.maximum(0,Z1_test)
Z2_test = A1_test @ W2 + b2
exp_Z2_test = np.exp(Z2_test)
sum_test = np.sum(exp_Z2_test, axis=1, keepdims=True)
A2_test = exp_Z2_test / sum_test

prediction = np.argmax(A2_test)

print(f"L'IA dit que c'est un : {prediction}")
print(f"La vraie réponse est : {Y[index]}")

plt.imshow(image_test.reshape(28, 28), cmap='gray')
plt.title(f"Prédiction : {prediction}")
plt.show()