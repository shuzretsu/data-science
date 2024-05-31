import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, recall_score, f1_score, precision_score, confusion_matrix
import missingno

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

# Info dataset
dataset.info()

# Deskripsi dataset
print(dataset.describe())

# Distribusi atribut 'Clump_thickness'
sns.set_theme(font_scale=1.0)
dataset['Clump_thickness'].value_counts().plot(kind='bar', figsize=(7, 6), rot=0)
plt.xlabel("Clump_thickness", labelpad=14)
plt.ylabel("Jumlah", labelpad=14)
plt.title("Distribusi Clump_thickness", y=1.02)
plt.show()

# Distribusi atribut 'Uniformity_of_cell_size'
sns.set_theme(font_scale=1.0)
dataset['Uniformity_of_cell_size'].value_counts().plot(kind='bar', figsize=(7, 6), rot=0)
plt.xlabel("Uniformity_of_cell_size", labelpad=14)
plt.ylabel("Jumlah", labelpad=14)
plt.title("Distribusi Uniformity_of_cell_size", y=1.02)
plt.show()

# Distribusi atribut 'Marginal_adhesion'
dataset['Marginal_adhesion'].value_counts().plot(kind='bar', figsize=(7, 6), rot=0)
plt.xlabel("Marginal_adhesion", labelpad=14)
plt.ylabel("Jumlah", labelpad=14)
plt.title("Distribusi Marginal_adhesion", y=1.02)
plt.show()

# Distribusi atribut 'Single_epithelial_cell_size'
dataset['Single_epithelial_cell_size'].value_counts().plot(kind='bar', figsize=(7, 6), rot=0)
plt.xlabel("Single_epithelial_cell_size", labelpad=14)
plt.ylabel("Jumlah", labelpad=14)
plt.title("Distribusi Single_epithelial_cell_size", y=1.02)
plt.show()

# Distribusi atribut 'Bare_nuclei'
dataset['Bare_nuclei'].value_counts().plot(kind='bar', figsize=(7, 6), rot=0)
plt.xlabel("Bare_nuclei", labelpad=14)
plt.ylabel("Jumlah", labelpad=14)
plt.title("Distribusi Bare_nuclei", y=1.02)
plt.show()

# Distribusi atribut 'Bland_chromatin'
dataset['Bland_chromatin'].value_counts().plot(kind='bar', figsize=(7, 6), rot=0)
plt.xlabel("Bland_chromatin", labelpad=14)
plt.ylabel("Jumlah", labelpad=14)
plt.title("Distribusi Bland_chromatin", y=1.02)
plt.show()

# Distribusi atribut 'Normal_nucleoli'
dataset['Normal_nucleoli'].value_counts().plot(kind='bar', figsize=(7, 6), rot=0)
plt.xlabel("Normal_nucleoli", labelpad=14)
plt.ylabel("Jumlah", labelpad=14)
plt.title("Distribusi Normal_nucleoli", y=1.02)
plt.show()

# Distribusi atribut 'Mitoses'
dataset['Mitoses'].value_counts().plot(kind='bar', figsize=(7, 6), rot=0)
plt.xlabel("Mitoses", labelpad=14)
plt.ylabel("Jumlah", labelpad=14)
plt.title("Distribusi Mitoses", y=1.02)
plt.show()

# Distribusi atribut 'Class'
dataset['Class'].value_counts().plot(kind='bar', figsize=(7, 6), rot=0)
plt.xlabel("Class", labelpad=14)
plt.ylabel("Jumlah", labelpad=14)
plt.title("Distribusi Class", y=1.02)
plt.show()

# Distribusi atribut 'Sample_code_number'
sample_code_number = dataset['Sample_code_number'].value_counts()
scn_dict = {'SCN yg digunakan hanya 1 kali': 0, 'SCN yg digunakan 2 kali': 0}

for key, count in sample_code_number.items():
    if count > 2:
        scn_dict[str(key)] = count
    if count == 1:
        scn_dict['SCN yg digunakan hanya 1 kali'] += 1
    if count == 2:
        scn_dict['SCN yg digunakan 2 kali'] += 1

x = list(scn_dict.keys())
y = list(scn_dict.values())
sns.set_theme(font_scale=1.0)
plt.figure(figsize=(10, 6))
plt.bar(x, y)
plt.xlabel('Sample_code_number', labelpad=14)
plt.ylabel('Jumlah', labelpad=14)
plt.title("Distribusi Sample_code_number (Semi-grouped)", fontdict={'size': 16}, y=1.08)
plt.xticks(rotation=60)
plt.ylim(0, 50)
plt.show()

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

# Mengganti tanda '?' dengan NaN
dataset.replace("?", np.nan, inplace=True)

# Info dataset
dataset.info()

# Deteksi nilai null
dataset.isnull().sum()

# Mendapatkan kolom dengan nilai null
print(dataset.loc[:, dataset.isnull().any()].columns)

# Mengubah tipe data kolom 'Bare_nuclei' menjadi float
dataset['Bare_nuclei'] = dataset['Bare_nuclei'].astype(float)
dataset.info()

# Visualisasi nilai null
missingdata_df = dataset.columns[dataset.isnull().any()].tolist()
missingno.matrix(dataset[missingdata_df])
plt.show()

# Mengisi nilai NaN dengan nilai median
median_value = dataset['Bare_nuclei'].median()
dataset['Bare_nuclei'] = dataset['Bare_nuclei'].fillna(median_value)

# Hitung ulang nilai null pada dataset
print(dataset.isnull().sum())

# Menampilkan data duplikat
print("All Duplicate Rows:")
print(dataset[dataset.duplicated(keep=False)])

# Menghapus data duplikat
dataClean = dataset.drop_duplicates()
print("All Duplicate Rows (clean):")
print(dataClean[dataClean.duplicated(keep=False)])

# Info dataset
dataClean.info()

# Menampilkan bar-chart dari distribusi status pasien kanker payudara
dataClean['Class'].value_counts().plot(kind='bar', figsize=(7, 6), rot=0)
plt.xlabel("Status pasien kanker payudara", labelpad=14)
plt.ylabel("Jumlah", labelpad=14)
plt.title("Status pasien kanker payudara", y=1.02)
plt.show()
print(dataClean['Class'].value_counts())

# Menampilkan Korelasi antar Fitur
correlation = dataClean.corr()
plt.figure(figsize=(12, 12))
plt.title("Heatmap Korelasi antar Fitur", y=1.02, fontdict={'size': 24})
sns.heatmap(correlation.round(2), annot=True, vmax=1, square=True, cmap='RdYlGn_r')
plt.show()

# Menampilkan Boxplot untuk melihat adanya Outlier
ax = dataClean.plot(kind='box', subplots=True, layout=(5, 6), sharex=False, figsize=(20, 20), title='Figure 1: Data distributions of all features')
plt.show()

# Tampilkan Box-plot dengan dasar target-attribute (Class) untuk setiap fitur/atribut
for column in dataClean.columns[:-1]:
    sns.boxplot(x='Class', y=column, data=dataClean)
    plt.title(f'Box-plot {column} berdasarkan Class')
    plt.show()

# Tampilkan deskripsi data yang sudah dibersihkan
print(dataClean.describe())

# Menentukan Label Data
X_norm = dataClean.drop('Class', axis=1).values
y = dataClean['Class']

# Split data training dan data testing (70:30)
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.3, random_state=5)

# Menampilkan ukuran dataset Training dan Testing
print(X_train.shape)
print(X_test.shape)

# Implementasi model Naive Bayes
modelNB = GaussianNB()
modelNB.fit(X_train, y_train)
y_predNB = modelNB.predict(X_test)

# Tampilkan Akurasi model Naive Bayes
print('Accuracy of Naive Bayes model : ', accuracy_score(y_test, y_predNB))
print('Recall of Naive Bayes model : ', recall_score(y_test, y_predNB, average="macro"))
print('F1-Score of Naive Bayes model : ', f1_score(y_test, y_predNB, average="macro"))
print('Precision of Naive Bayes model : ', precision_score(y_test, y_predNB, average="macro"))
print('\n Clasification report:\n', classification_report(y_test, y_predNB))
print('\n Confusion matrix:\n', confusion_matrix(y_test, y_predNB))

# Implementasi model Decision Tree
modelDT = DecisionTreeClassifier()
modelDT.fit(X_train, y_train)
y_predDT = modelDT.predict(X_test)

# Tampilkan Akurasi model Decision Tree
print('Accuracy of Decision Tree model : ', accuracy_score(y_test, y_predDT))
print('Recall of Decision Tree model : ', recall_score(y_test, y_predDT, average="macro"))
print('F1-Score of Decision Tree model : ', f1_score(y_test, y_predDT, average="macro"))
print('Precision of Decision Tree model : ', precision_score(y_test, y_predDT, average="macro"))
print('\n Clasification report:\n', classification_report(y_test, y_predDT))
print('\n Confusion matrix:\n', confusion_matrix(y_test, y_predDT))

# Implementasi model Random Forest
modelRF = RandomForestClassifier()
modelRF.fit(X_train, y_train)
y_predRF = modelRF.predict(X_test)

# Tampilkan Akurasi model Random Forest
print('Accuracy of Random Forest model : ', accuracy_score(y_test, y_predRF))
print('Recall of Random Forest model : ', recall_score(y_test, y_predRF, average="macro"))
print('F1-Score of Random Forest model : ', f1_score(y_test, y_predRF, average="macro"))
print('Precision of Random Forest model : ', precision_score(y_test, y_predRF, average="macro"))
print('\n Clasification report:\n', classification_report(y_test, y_predRF))
print('\n Confusion matrix:\n', confusion_matrix(y_test, y_predRF))

