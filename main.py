import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, recall_score, f1_score, precision_score, confusion_matrix

# Load data menjadi data frame
dataset = pd.read_csv("C:\\Users\\relax\\PycharmProjects\\Data-Science\\dataset\\breast-cancer-wisconsin.data", header=None)

# Memasukkan nama fitur kedalam dataset
dataset.columns = [
    "Sample_code_number",
    "Clump_thickness",
    "Uniformity_of_cell_size",
    "Uniformity_of_cell_shape",
    "Marginal_adhesion",
    "Single_epithelial_cell_size",
    "Bare_nuclei",
    "Bland_chromatin",
    "Normal_nucleoli",
    "Mitoses",
    "Class"
]

# Menghapus fitur yang tidak dapat digunakan
columns_to_drop = ['Sample_code_number']
dataset = dataset.drop(columns_to_drop, axis=1)

# Mengganti nilai '?' menjadi NaN
dataset.replace("?", np.nan, inplace=True)

# Mengisi nilai NaN dengan nilai median
median_value = dataset['Bare_nuclei'].median()
dataset['Bare_nuclei'] = dataset['Bare_nuclei'].fillna(median_value)

# Mengubah tipe data kolom 'Bare_nuclei' menjadi float
dataset['Bare_nuclei'] = dataset['Bare_nuclei'].astype(float)

# Membuang data duplikat
dataClean = dataset.drop_duplicates()

# Menampilkan distribusi status pasien kanker payudara
plt.figure(figsize=(7, 6))
dataClean['Class'].value_counts().plot(kind='bar', rot=0)
plt.xlabel("Status pasien kanker payudara", labelpad=14)
plt.ylabel("Jumlah", labelpad=14)
plt.title("Status pasien kanker payudara", y=1.02)
plt.show()

# Menampilkan korelasi antar fitur
correlation = dataClean.corr()
plt.figure(figsize=(12, 12))
plt.title("Heatmap Korelasi antar Fitur", y=1.02, fontdict={'size': 24})
sns.heatmap(correlation.round(2), annot=True, vmax=1, square=True, cmap='RdYlGn_r')
plt.show()

# Menampilkan boxplot untuk melihat adanya outlier
plt.figure(figsize=(20, 20))
dataClean.plot(kind='box', subplots=True, layout=(5, 6), sharex=False, figsize=(20, 20), title='Figure 1: Data distributions of all features')
plt.show()

# Menampilkan boxplot dengan dasar target-attribute (Class) untuk setiap fitur/atribut
for column in dataClean.columns[:-1]:
    sns.boxplot(x='Class', y=column, data=dataClean)
    plt.title(f'Box-plot {column} berdasarkan Class')
    plt.show()

# Menampilkan deskripsi data yang sudah dibersihkan
print(dataClean.describe())
print(dataClean.info())

# Menentukan label data
X_norm = dataClean.drop('Class', axis=1).values
y = dataClean['Class']

# Membagi data menjadi data training dan data testing
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.3, random_state=42)

# Membangun Model Gaussian Naive Bayes
clean_classifier_nb = GaussianNB()
clean_classifier_nb.fit(X_train, y_train)
y_pred_nb = clean_classifier_nb.predict(X_test)

# Evaluate the Gaussian NB model
print("\nGaussian NB Model:")
print("Classification Report:")
print(classification_report(y_test, y_pred_nb))
print("Accuracy:", round(accuracy_score(y_test, y_pred_nb), 3))

# Membangun Model Decision Tree
clf_dt = DecisionTreeClassifier(random_state=42)
clf_dt.fit(X_train, y_train)
y_pred_dt = clf_dt.predict(X_test)

# Evaluate the Decision Tree model
print("\nDecision Tree Model:")
print("Classification Report:")
print(classification_report(y_test, y_pred_dt))
print("Accuracy:", round(accuracy_score(y_test, y_pred_dt), 3))

# Membangun Model Random Forest
clf_rf = RandomForestClassifier(random_state=42)
clf_rf.fit(X_train, y_train)
y_pred_rf = clf_rf.predict(X_test)

# Evaluate the Random Forest model
print("\nRandom Forest Model:")
print("Classification Report:")
print(classification_report(y_test, y_pred_rf))
print("Accuracy:", round(accuracy_score(y_test, y_pred_rf), 3))

# Tampilkan bar-chart untuk membandingkan akurasi tiap model
models = ['Gaussian NB', 'Decision Tree', 'Random Forest']
accuracies = [accuracy_score(y_test, y_pred_nb), accuracy_score(y_test, y_pred_dt), accuracy_score(y_test, y_pred_rf)]
plt.figure(figsize=(10, 6))
sns.barplot(x=models, y=accuracies)
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Comparison of Model Accuracies')
plt.show()

# Tampilkan Confusion Matrix dari model Random Forest
cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix (Random Forest Model)')
plt.xlabel('True')
plt.ylabel('Predict')
plt.show()
