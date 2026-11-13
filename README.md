# Reconnaissance de chiffres avec des réseaux de neurones (NumPy)
 
## Objectif
 
Mon objectif durant ce Projet n'était pas d'obtenir la meilleure précision possible (un framework y arriverait en quelques lignes et en mieux) mais de comprendre les mécanismes d'un réseau de neurones en les implémentant moi-même.
J'ai donc implémenté :
- forward pass (produits matriciels, biais)
- fonctions d'activation : ReLU et softmax
- fonction de coût : cross-entropy
- backpropagation : calcul des gradients couche par couche
- descente de gradient et mise à jour des poids.
 
## Dataset
 
J'ai utilisé le dataset MNIST au format CSV, que j'ai pris depuis la compétition Kaggle *Digit Recognizer* :
https://www.kaggle.com/competitions/digit-recognizer/data
 
Pour utiliser mon code il faut télécharger `train.csv` (42 000 images, 28×28) et le placer à la racine du projet.
 
## Les deux versions

j'ai donc testé avec 1000 iterations
 
| | 1 couche | 2 couches |
|---|---|---|
| Architecture | 784 → 10 (softmax) | 784 → 64 (ReLU) → 10 (softmax) |
| Paramètres | 7 850 | 50 890 |
| Itérations | 1000|1000 |
| Précision | 90.62% | 92,29 % |
 
La version à 1 couche est en réalité une régression softmax : sans couche cachée, elle ne peut apprendre que des frontières linéaires entre les classes. L'ajout d'une couche cachée de 64 neurones avec ReLU permet au modèle de combiner des motifs intermédiaires avant de décider, ce qui explique le gain de précision.
 
## Installation

```bash
pip install numpy pandas matplotlib
```
 
## Utilisation
 
**1. Entraîner un modèle**

```bash
python reseau_2couche.py
```
 
Le script demande s'il faut reprendre l'entraînement précédent :
- `n` → repart de poids aléatoires (écrase le modèle sauvegardé)
- `o` → recharge les poids depuis les fichiers `.npy` et continue

Les poids sont sauvegardés en fin d'entraînement.

**IMPORTANT : pour la première utilisation, répondre `n`** (les fichiers `.npy` n'existent pas encore).

**2. Tester le modèle**

```bash
python test_reseau_2couche.py
```
 
Charge les poids sauvegardés et prédit le chiffre d'une image, avec affichage.
 
## Structure
 
```
reseau_1couche.py         # réseau 1 couche (régression softmax), entraînement
test_reseau_1couche.py    # test du réseau 1 couche
reseau_2couche.py         # réseau 2 couches, entraînement
test_reseau_2couche.py    # inférence à partir des poids sauvegardés
```
 
## Limites et améliorations prévues
 
- La précision est mesurée sur les données d'entraînement. Un split train/test est nécessaire pour évaluer la capacité réelle de généralisation ,de plus le chiffre sur des images jamais vues sera plus bas.
- Le biais de la couche de sortie était initialement mal calculé (`np.sum` sans `axis`), ce qui donnait une seule correction appliquée aux 10 classes. Corrigé.
- Pistes : augmenter le nombre de neurones cachés, tester d'autres valeurs de learning rate, ajouter une seconde couche cachée.
 
