import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from mlxtend.plotting import plot_decision_regions

#zadatak 1

X, y = make_classification(n_samples = 200, n_features = 2, n_redundant = 0, n_informative = 2, random_state = 213, n_clusters_per_class = 1, class_sep = 1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 5)

X1_train = X_train[:, 0]
X2_train = X_train[:, 1]
plt.scatter(X1_train, X2_train, c = y_train, cmap = 'magma', label = 'Train data')
plt.scatter(X_test[:, 0], X_test[:, 1], c = y_test, cmap = 'viridis', marker = 'X', label = 'Test data')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.show()

LogRegression_model = LogisticRegression()
LogRegression_model.fit(X_train, y_train)

coef = LogRegression_model.coef_.T
intercept = LogRegression_model.intercept_[0]
print(f'Coefficient: {coef}, intercept: {intercept}')
plot_decision_regions(X_train, y_train, LogRegression_model)
plt.scatter(X1_train, X2_train, c = y_train, cmap = 'Blues')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

y_test_p = LogRegression_model.predict(X_test)
cm = confusion_matrix(y_test, y_test_p)
print('Matrica zabune: ', cm)
disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_test_p))
disp.plot()
plt.show()

print('Accuracy: ', accuracy_score(y_test, y_test_p))
print('Precision: ', precision_score(y_test, y_test_p))
print('Recall: ', recall_score(y_test, y_test_p))

correct = np.where(y_test_p == y_test)[0]
wrong = np.where(y_test_p != y_test)[0]
plt.scatter(X_test[correct, 0], X_test[correct, 1], c = 'green', label = 'Correct classification')
plt.scatter(X_test[wrong, 0], X_test[wrong, 1], c = 'black', label = 'Incorrect classification')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.show()





















import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report

# Rječnik za ispis imena vrsta na grafu
labels = {0: 'Adelie', 1: 'Chinstrap', 2: 'Gentoo'}

def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # Postavljanje granica grafa
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    
    # Predikcija klase za svaku točku mreže
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    
    # Crtanje regija (pozadina)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    
    # Crtanje stvarnih podataka
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=colors[idx],
                    marker=markers[idx], edgecolor='w',
                    label=labels[cl])
    
    plt.xlabel('Duljina kljuna (mm)')
    plt.ylabel('Duljina peraje (mm)')
    plt.legend(loc='upper left')
    plt.title('Regije odluke (Trening skup)')
    plt.show()

# 1. Učitavanje podataka (pazi na putanju datoteke!)
df = pd.read_csv('penguins.csv')

# 2. Čišćenje podataka
df = df.drop(columns=['sex'])
df.dropna(axis=0, inplace=True)

# 3. Kodiranje i pretvorba u CIJELE BROJEVE (Rješava ValueError)
df['species'] = df['species'].replace({'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2})
df['species'] = df['species'].astype(int) 

# 4. Određivanje značajki i cilja
X = df[['bill_length_mm', 'flipper_length_mm']].to_numpy()
y = df['species'].values # 1D niz

# 5. Podjela na train i test skup
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# 6. Prikaz broja primjera (Zadatak a)
train_classes, train_counts = np.unique(y_train, return_counts=True)
plt.bar(train_classes, train_counts, tick_label=['Adelie', 'Chinstrap', 'Gentoo'])
plt.title('Broj primjera po klasi (Train skup)')
plt.ylabel('Broj primjera')
plt.show()

# 7. Izgradnja modela (Zadatak b i c)
# Povećavamo max_iter na 1000 da izbjegnemo ConvergenceWarning
LogRegression_model = LogisticRegression(max_iter=1000)
LogRegression_model.fit(X_train, y_train)

print('Koeficijenti modela (theta_1, theta_2):\n', LogRegression_model.coef_)
print('Odsječak (theta_0):\n', LogRegression_model.intercept_)

# 8. Crtanje regija odluke (Zadatak d)
plot_decision_regions(X_train, y_train, classifier=LogRegression_model)

# 9. Testiranje i metrike (Zadatak e)
y_test_p = LogRegression_model.predict(X_test)

print('\nTočnost (Accuracy):', accuracy_score(y_test, y_test_p))
print('\nClassification Report:\n', classification_report(y_test, y_test_p))

# Prikaz matrice zabune
cm = confusion_matrix(y_test, y_test_p)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Adelie', 'Chinstrap', 'Gentoo'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Matrica zabune (Testni skup)')
plt.show()