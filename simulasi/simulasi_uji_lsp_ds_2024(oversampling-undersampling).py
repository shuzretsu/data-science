# -*- coding: utf-8 -*-
"""Salinan Simulasi_Uji_LSP_DS_2024.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1AxhSTdS987JzZ9auPaSbeb2SgYPI3dex

### **Dataset Simulasi Uji LSP Data Science UDINUS 2024 (Winconsin Breast Cancer)**: [https://bit.ly/dataset-simulasi-lsp-udinus-2024](https://bit.ly/dataset-simulasi-lsp-udinus-2024)

### <b>Daftar Isi</b>
* [1) Mengumpulkan Data](#h1)
* [2) Menelaah Data](#h2)
* [3) Memvalidasi Data](#h3)
* [4) Menetukan Object Data](#h4)
* [5) Membersihkan Data](#h5)
* [6) Mengkonstruksi Data](#h6)
* [7) Menentukan Label Data](#h7)
* [8) Membangung Model](#h8)
* [9) Mengevaluasi Hasil Pemodelan](#h9)
* [10) Optimasi Model Klasifikasi](#h10)

## <b>1) Mengumpulkan Data</b> <a class="anchor" id="h1"></a>
"""

# Load library yang diperlukan
import pandas as pd

# Load data menjadi data frame
dataset = pd.read_csv("/content/breast-cancer-wisconsin_(simulasi-uji-LSP).csv", header=None)

dataset

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

# Lakukan pengecekan apakah dataset sudah benar dengan menampilkan 5 data teratas
dataset.head()



"""## <b>2) Menelaah Data</b> <a class="anchor" id="h2"></a>"""

# Menampilkan informasi dari file dataset
dataset.info()
"""
total kolum yang ada pada dataset terdiri dari 11

"""

# Menampilkan deskripsi dari file dataset
dataset.describe()

import seaborn as sns
import matplotlib.pyplot as plt

# tuliskan kode program untuk Menampilkan distribusi kelas dari semua fitur
# dibawah adalah contoh kode program untuk fitur 1

sns.set_theme(font_scale=1.0)
dataset["Clump_thickness"].value_counts().plot(kind='bar', figsize=(7, 6), rot=0)
plt.xlabel("Clump_thickness", labelpad=14)
plt.ylabel("Jumlah", labelpad=14)
plt.title("Distribusi", y=1.02)
plt.show()

# Tampilkan bar-chart dari distribusi atribut Uniformity_of_cell_size
sns.set_theme(font_scale=1.0)
dataset["Uniformity_of_cell_size"].value_counts().plot(kind='bar', figsize=(7, 6), rot=0)
plt.xlabel("Uniformity_of_cell_size", labelpad=14)
plt.ylabel("Jumlah", labelpad=14)
plt.title("Distribusi", y=1.02)
plt.show()

# 1. distribusi atribut Uniformity_of_cell_shape
sns.set_theme(font_scale=1.0)
dataset["Uniformity_of_cell_size"].value_counts().plot(kind='bar', figsize=(7, 6), rot=0)
plt.xlabel("Uniformity_of_cell_size", labelpad=14)
plt.ylabel("Jumlah", labelpad=14)
plt.title("Distribusi", y=1.02)
plt.show()

# 2. distribusi atribut Marginal_adhesion
sns.set_theme(font_scale=1.0)
dataset["Marginal_adhesion"].value_counts().plot(kind='bar', figsize=(7, 6), rot=0)
plt.xlabel("Marginal_adhesion", labelpad=14)
plt.ylabel("Jumlah", labelpad=14)
plt.title("Distribusi", y=1.02)
plt.show()

# 3. distribusi atribut Single_epithelial_cell_size sns.set_theme(font_scale=1.0)
dataset["Single_epithelial_cell_size"].value_counts().plot(kind='bar', figsize=(7, 6), rot=0)
plt.xlabel("Single_epithelial_cell_size", labelpad=14)
plt.ylabel("Jumlah", labelpad=14)
plt.title("Distribusi", y=1.02)
plt.show()

# 4. distribusi atribut Bare_nuclei
dataset["Bare_nuclei"].value_counts().plot(kind='bar', figsize=(7, 6), rot=0)
plt.xlabel("Bare_nuclei", labelpad=14)
plt.ylabel("Jumlah", labelpad=14)
plt.title("Distribusi", y=1.02)
plt.show()

# 5. distribusi atribut Bland_chromatin
dataset["Bland_chromatin"].value_counts().plot(kind='bar', figsize=(7, 6), rot=0)
plt.xlabel("Bland_chromatin", labelpad=14)
plt.ylabel("Jumlah", labelpad=14)
plt.title("Distribusi", y=1.02)
plt.show()

# 6. distribusi atribut Bland_chromatin
dataset["Bland_chromatin"].value_counts().plot(kind='bar', figsize=(7, 6), rot=0)
plt.xlabel("Bland_chromatin", labelpad=14)
plt.ylabel("Jumlah", labelpad=14)
plt.title("Distribusi", y=1.02)
plt.show()

# 7. distribusi atribut Normal_nucleoli
dataset["Normal_nucleoli"].value_counts().plot(kind='bar', figsize=(7, 6), rot=0)
plt.xlabel("Normal_nucleoli", labelpad=14)
plt.ylabel("Jumlah", labelpad=14)
plt.title("Distribusi", y=1.02)
plt.show()

# 8. distribusi atribut Mitoses
dataset["Mitoses"].value_counts().plot(kind='bar', figsize=(7, 6), rot=0)
plt.xlabel("Mitoses", labelpad=14)
plt.ylabel("Jumlah", labelpad=14)
plt.title("Distribusi", y=1.02)
plt.show()

# 9. distribusi atribut Sample_code_number
dataset["Sample_code_number"].value_counts().plot(kind='bar', figsize=(7, 6), rot=0)
plt.xlabel("Sample_code_number", labelpad=14)
plt.ylabel("Jumlah", labelpad=14)
plt.title("Distribusi", y=1.02)
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
sample_code_number = dataset['Sample_code_number'].value_counts()

scn_dict = {
  'SCN yg digunakan hanya 1 kali': 0,
  'SCN yg digunakan 2 kali': 0,
}

for key, count in sample_code_number.items():
  if count > 2:
    scn_dict[str(key)] = count
  if count == 1:
    scn_dict['SCN yg digunakan hanya 1 kali'] += 1
  if count == 2:
    scn_dict['SCN yg digunakan 2 kali'] += 1

x = list(scn_dict.keys())
y = list(scn_dict.values())
# Tampilkan bar-chart dari distribusi atribut Sample_code_number seperti contoh di-bawah
sns.set_theme(font_scale=1.0)
plt.figure(figsize=(10, 6))
plt.bar(x, y)
plt.xlabel('Sample_code_number', labelpad=14)
plt.ylabel('Jumlah', labelpad=14)
plt.title("Distribusi Sample_code_number (Semi-grouped)", fontdict={'size': 16}, y=1.08)
plt.xticks(rotation=60)
plt.ylim(0, 50)

"""## <b>3) Memvalidasi Data</b> <a class="anchor" id="h3"></a>

## <b>4) Menentukan Objek Data</b> <a class="anchor" id="h4"></a>
"""

import numpy as np

"""<h4>⬇ <b style="color:orange;">Instruksi</b>: Ubah "<b style="color:yellow;">[fix_me]</b>" dengan code yang benar!</h4>"""

#menghapus fitur yang tidak dapat digunakan
dataset = pd.read_csv("/content/breast-cancer-wisconsin_(simulasi-uji-LSP).csv", header=None)
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
columns_to_drop = ['Sample_code_number']
dataset = dataset.drop(columns_to_drop, axis=1)

dataset.replace('?', np.nan, inplace=True)
dataset.info()

dataset.isnull().sum()

"""## <b>5) Membersihkan Data</b> <a class="anchor" id="h5"></a>"""

# menghitung nilai Null pada dataset
dataset.isnull().sum()

# mendeteksi keberadaan nilai Null
dataset.loc[:, dataset.isnull().any()].columns

# Mengubah Type data dari salah satu fitur
dataset['Bare_nuclei'] = dataset['Bare_nuclei'].astype(float)

# Tampilkan lagi Informasi dari data
dataset.info()

import missingno

# Memvisualisasikan keberadaan nilai Null
missingdata_df = dataset.columns[dataset.isnull().any()].tolist()
missingno.matrix(dataset[missingdata_df])

missingno.matrix(dataset)

# diiskan jawaban masing-masing
median_value = dataset['Bare_nuclei'].median()
dataset['Bare_nuclei'] = dataset['Bare_nuclei'].fillna(median_value)

# Hitung ulang nilai Null pada dataset
print(dataset.isnull().sum())

# Menampilkan data duplikat
print("All Duplicate Rows:")
dataset[dataset.duplicated(keep=False)]

# Menghapus data duplikat, menyimpan data dalam variabel dataClean
dataClean = dataset.drop_duplicates()
print("All Duplicate Rows:")
dataClean[dataClean.duplicated(keep=False)]

dataset.info()

dataClean.info()

"""## <b>6) Menkonstruksi Data</b> <a class="anchor" id="h2"></a>"""

plot_data = dataClean['Class'].value_counts()
print(plot_data)
sns.set_theme(font_scale=1.0)
plot_data.plot(kind='bar', figsize=(7, 6), rot=0)
plt.xlabel("Status Pasien kanker payudara", labelpad=14)
plt.ylabel("Jumlah", labelpad=14)
plt.title("Distribusi data status kanker payudara", y=1.02);

for i, counts in enumerate(plot_data):
  plt.text(i, (counts + 1), str(counts), ha='center')

plt.show()

# Menampilkan Korelasi antar Fitur
correlation = dataClean.corr()
plt.figure(figsize=(12, 12))
plt.title("Heatmap Korelasi antar Fitur", y=1.02, fontdict={'size': 24})
sns.heatmap(
  correlation.round(2),
  annot = True,
  vmax = 1,
  square = True,
  cmap = 'RdYlGn_r'
)

plt.show()

# Menampilkan Boxplot untuk melihat adanya Outlayer
ax = dataClean.plot(
  kind='box',
  subplots=True,
  layout=(5, 6),
  sharex=False,
  figsize=(20, 20),
  title='figure 1: Data distributions of all features'
)

plt.show()

import pandas as pd
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 5, figsize=(20, 10))  # Adjust layout and figure size as needed
axes = axes.ravel()  # Flatten the axes for loop iteration

# Looping through features (excluding "Class")
for i, column in enumerate(dataClean.columns):
    if column != 'Class':
        dataClean.boxplot(ax=axes[i], column=column, by='Class')
        axes[i].set_title(f'{column}')

# Add a main title
fig.suptitle('Boxplot setiap fitur pada target Class', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to prevent overlapping elements
plt.show()

# Tampilkan deskripsi data yang sudah dibersihkan
dataset.describe()

dataClean.info()

dataClean

"""## <b>7) Menentukan Label Data</b> <a class="anchor" id="h7"></a>"""

from sklearn.model_selection import train_test_split

X = dataClean.drop('Class', axis=1).values
y = dataClean['Class']

# perbandingan data training dan data testing adalah 70 : 30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

"""## <b>8) Membangun Model</b> <a class="anchor" id="h8"></a>"""

# import library pemodelan yang digunakan
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt

clean_classifier_nb = GaussianNB()
clean_classifier_nb.fit(X_train, y_train)

clean_classifier_dt = DecisionTreeClassifier(random_state=42)
clean_classifier_dt.fit(X_train, y_train)

clean_classifier_rf = RandomForestClassifier(n_estimators=100, random_state=42)
clean_classifier_rf.fit(X_train, y_train)

"""## <b>9) Mengevaluasi Hasil Pemodelan</b> <a class="anchor" id="h9"></a>"""

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, roc_auc_score, confusion_matrix, precision_score

def evaluation(Y_test, Y_pred):
  acc = accuracy_score(Y_test, Y_pred)
  rcl = recall_score(Y_test, Y_pred, average='weighted')
  f1 = f1_score(Y_test, Y_pred, average='weighted')
  ps = precision_score(Y_test, Y_pred, average='weighted')

  metric_dict = {
    'accuracy': round(acc, 3),
    'recall': round(rcl, 3),
    'F1 score': round(f1, 3),
    'Precision score': round(ps, 3)
  }

  return print(metric_dict)

y_pred_nb = clean_classifier_nb.predict(X_test)

# Evaluate the Gaussian NB model
print("\nGaussian NB Model:")
accuracy_nb = round(accuracy_score(y_test, y_pred_nb), 3)
print("Accuracy:", accuracy_nb)
print("Classification Report:")
print(classification_report(y_test, y_pred_nb))

evaluation(y_test, y_pred_nb)

y_pred_dt = clean_classifier_dt.predict(X_test)

# Evaluate the Decision Tree model
print("\nDecision Tree Model:")
accuracy_dt = round(accuracy_score(y_test, y_pred_dt), 3)
print("Accuracy:", accuracy_dt)
print("Classification Report:")
print(classification_report(y_test, y_pred_dt))

evaluation(y_test, y_pred_dt)

y_pred_rf = clean_classifier_rf.predict(X_test)

# Evaluate the Random Forest model
print("\nRandom Forest Model:")
accuracy_rf = round(accuracy_score(y_test, y_pred_rf), 3)
print("Accuracy:", accuracy_rf)
print("Classification Report:")
print(classification_report(y_test, y_pred_rf))

evaluation(y_test, y_pred_rf)

# Tampilkan bar-chart untuk membandingkan akurasi tiap model
models = ['Gaussian Naive Bayes', 'Decision Tree', 'Random Forest']
accuracies = [accuracy_nb, accuracy_dt, accuracy_rf]

plt.figure(figsize=(10, 6))
plt.bar(models, accuracies)
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Model Comparison')
plt.show()

"""## <b>10) Optimasi Model Klasifikasi</b> <a class="anchor" id="h10"></a>

# 11) Optimasi Undersampling
"""

#UNDERSAMPLING
# Import RandomUnderSampler
from imblearn.under_sampling import RandomUnderSampler
x = dataClean.drop('Class', axis=1).values
y = dataClean['Class']
# Assuming X and y are your features and labels
rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(X,y)

import pandas as dataResampled
dataResampled = pd.DataFrame(X_res, columns=dataClean.drop('Class', axis=1).columns)
dataResampled['Class'] = y_res
dataResampled
plot_data = pd.DataFrame({'Class ': y_res}).value_counts()
print(plot_data)
sns.set_theme(font_scale=1.0)
plot_data.plot(kind='bar', figsize=(7, 6), rot=0)
plt.xlabel("Status Pasien kanker payudara", labelpad=14)
plt.ylabel("Jumlah", labelpad=14)
plt.title("Distribusi data status kanker payudara", y=1.02);

for i, counts in enumerate(plot_data):
  plt.text(i, (counts + 1), str(counts), ha='center')

plt.show()

# Menampilkan Korelasi antar Fitur
correlation = dataResampled.corr()
plt.figure(figsize=(12, 12))
plt.title("Heatmap Korelasi antar Fitur", y=1.02, fontdict={'size': 24})
sns.heatmap(
  correlation.round(2),
  annot = True,
  vmax = 1,
  square = True,
  cmap = 'RdYlGn_r'
)

plt.show()

dataset.describe()
dataClean.info()
dataClean

from sklearn.model_selection import train_test_split
X = dataResampled.drop('Class', axis=1).values
y = dataResampled['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# import library pemodelan yang digunakan
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
clean_classifier_nb = GaussianNB()
clean_classifier_nb.fit(X_train, y_train)

clean_classifier_dt = DecisionTreeClassifier(random_state=42)
clean_classifier_dt.fit(X_train, y_train)

clean_classifier_rf = RandomForestClassifier(n_estimators=100, random_state=42)
clean_classifier_rf.fit(X_train, y_train)

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, roc_auc_score, confusion_matrix, precision_score

def evaluation(Y_test, Y_pred):
  acc = accuracy_score(Y_test, Y_pred)
  rcl = recall_score(Y_test, Y_pred, average='weighted')
  f1 = f1_score(Y_test, Y_pred, average='weighted')
  ps = precision_score(Y_test, Y_pred, average='weighted')

  metric_dict = {
    'accuracy': round(acc, 3),
    'recall': round(rcl, 3),
    'F1 score': round(f1, 3),
    'Precision score': round(ps, 3)
  }

  return print(metric_dict)

y_pred_nb = clean_classifier_nb.predict(X_test)
# Optimasi Model Klasifikasi
cm = confusion_matrix(y_test, y_pred_nb)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix GNB')
plt.xlabel('True')
plt.ylabel('Predict')
plt.show()

# Evaluate the Gaussian NB model
print("\nGaussian NB Model:")
accuracy_nb = round(accuracy_score(y_test, y_pred_nb), 3)
print("Accuracy:", accuracy_nb)
print("Classification Report:")
print(classification_report(y_test, y_pred_nb))
evaluation(y_test, y_pred_nb)

evaluation(y_test, y_pred_nb)
y_pred_dt = clean_classifier_dt.predict(X_test)
# Optimasi Model Klasifikasi
cm = confusion_matrix(y_test, y_pred_dt)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix DTM')
plt.xlabel('True')
plt.ylabel('Predict')
plt.show()

# Evaluate the Decision Tree model
print("\nDecision Tree Model:")
accuracy_dt = round(accuracy_score(y_test, y_pred_dt), 3)
print("Accuracy:", accuracy_dt)
print("Classification Report:")
print(classification_report(y_test, y_pred_dt))

y_pred_rf = clean_classifier_rf.predict(X_test)
# Optimasi Model Klasifikasi
cm = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix RF')
plt.xlabel('True')
plt.ylabel('Predict')
plt.show()
# Evaluate the Random Forest model
print("\nRandom Forest Model:")
accuracy_rf = round(accuracy_score(y_test, y_pred_rf), 3)
print("Accuracy:", accuracy_rf)
print("Classification Report:")
print(classification_report(y_test, y_pred_rf))
evaluation(y_test, y_pred_rf)

# Tampilkan bar-chart untuk membandingkan akurasi tiap model
models = ['Gaussian Naive Bayes', 'Decision Tree', 'Random Forest']
accuracies = [accuracy_nb, accuracy_dt, accuracy_rf]

plt.figure(figsize=(10, 6))
# Pass 'models' as x-coordinates and 'accuracies' as heights
bars = plt.bar(models, accuracies, color=['blue', 'green', 'red'])  # Added colors for clarity

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval - 0.05, round(yval, 2), ha='center', va='bottom', color='white', fontweight='bold')

plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Model Comparison')
plt.show()
print(accuracies)

"""# 12) Optimasi Oversampling"""

"""from imblearn.over_sampling import RandomOverSampler

x = dataClean.drop('Class', axis=1)
y = dataClean['Class']

# Assuming X and y are your features and labels
ros = RandomOverSampler(random_state=42)
X_ros, y_ros = ros.fit_resample(X,y)
"""
from imblearn.over_sampling import RandomOverSampler

# Use the correct feature matrix and target variable
X = dataClean.drop('Class', axis=1)  # Do not call .values here
y = dataClean['Class']

# Assuming X and y are your features and labels
ros = RandomOverSampler(random_state=42)
X_ros, y_ros = ros.fit_resample(X,y)  # Rename second output to y_ros for clarity

import pandas as dataResampled
dataResampled = pd.DataFrame(X_ros, columns=dataClean.drop('Class', axis=1).columns)
dataResampled['Class'] = y_ros
dataResampled
plot_data = pd.DataFrame({'Class ': y_ros}).value_counts()
print(plot_data)
sns.set_theme(font_scale=1.0)
plot_data.plot(kind='bar', figsize=(7, 6), rot=0)
plt.xlabel("Status Pasien kanker payudara", labelpad=14)
plt.ylabel("Jumlah", labelpad=14)

plt.title("Distribusi data status kanker payudara", y=1.02);

for i, counts in enumerate(plot_data):
  plt.text(i, (counts + 1), str(counts), ha='center')

plt.show()

# Menampilkan Korelasi antar Fitur
correlation = dataResampled.corr()
plt.figure(figsize=(12, 12))
plt.title("Heatmap Korelasi antar Fitur", y=1.02, fontdict={'size': 24})
sns.heatmap(
  correlation.round(2),
  annot = True,
  vmax = 1,
  square = True,
  cmap = 'RdYlGn_r'
)

plt.show()

dataset.describe()
dataClean.info()
dataClean

from sklearn.model_selection import train_test_split
X = dataResampled.drop('Class', axis=1).values
y = dataResampled['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
clean_classifier_nb = GaussianNB()
clean_classifier_nb.fit(X_train, y_train)

clean_classifier_dt = DecisionTreeClassifier(random_state=42)
clean_classifier_dt.fit(X_train, y_train)

clean_classifier_rf = RandomForestClassifier(n_estimators=100, random_state=42)
clean_classifier_rf.fit(X_train, y_train)

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, roc_auc_score, confusion_matrix, precision_score

def evaluation(Y_test, Y_pred):
  acc = accuracy_score(Y_test, Y_pred)
  rcl = recall_score(Y_test, Y_pred, average='weighted') # Changed y_pred to Y_pred
  f1 = f1_score(Y_test, Y_pred, average='weighted') # Changed y_pred to Y_pred
  ps = precision_score(Y_test, Y_pred, average='weighted') # Changed y_pred to Y_pred
  metric_dict = {
    'accuracy': round(acc, 3),
    'recall': round(rcl, 3),
    'F1 score': round(f1, 3),
    'Precision score': round(ps, 3)
  }
  return print(metric_dict)

y_pred_nb = clean_classifier_nb.predict(X_test)
# Optimasi Model Klasifikasi
cm = confusion_matrix(y_test, y_pred_nb)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix GNB')
plt.xlabel('True')
plt.ylabel('Predict')
plt.show()

# Evaluate the Gaussian NB model
print("\nGaussian NB Model:")
accuracy_nb = round(accuracy_score(y_test, y_pred_nb), 3)
print("Accuracy:", accuracy_nb)
print("Classification Report:")
print(classification_report(y_test, y_pred_nb))
evaluation(y_test, y_pred_nb)

evaluation(y_test, y_pred_nb)
y_pred_dt = clean_classifier_dt.predict(X_test)
# Optimasi Model Klasifikasi
cm = confusion_matrix(y_test, y_pred_dt)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix DTM')
plt.xlabel('True')
plt.ylabel('Predict')
plt.show()

# Evaluate the Decision Tree model
print("\nDecision Tree Model:")
accuracy_dt = round(accuracy_score(y_test, y_pred_dt), 3)
print("Accuracy:", accuracy_dt)
print("Classification Report:")
print(classification_report(y_test, y_pred_dt))

y_pred_rf = clean_classifier_rf.predict(X_test)
# Optimasi Model Klasifikasi
cm = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix RF')
plt.xlabel('True')
plt.ylabel('Predict')
plt.show()
# Evaluate the Random Forest model
print("\nRandom Forest Model:")
accuracy_rf = round(accuracy_score(y_test, y_pred_rf), 3)
print("Accuracy:", accuracy_rf)
print("Classification Report:")
print(classification_report(y_test, y_pred_rf))
evaluation(y_test, y_pred_rf)

# Tampilkan bar-chart untuk membandingkan akurasi tiap model
models = ['Gaussian Naive Bayes', 'Decision Tree', 'Random Forest']
accuracies = [accuracy_nb, accuracy_dt, accuracy_rf]

plt.figure(figsize=(10, 6))
# Pass 'models' as x-coordinates and 'accuracies' as heights
bars = plt.bar(models, accuracies, color=['blue', 'green', 'red'])  # Added colors for clarity

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval - 0.05, round(yval, 2), ha='center', va='bottom', color='white', fontweight='bold')

plt.xlabel('Models')
plt.ylabel('Accuracy (%)')
plt.title('Model Comparison')
plt.show()
print(accuracies)