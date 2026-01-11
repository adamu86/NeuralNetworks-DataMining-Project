import os
import random
from itertools import product
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

import seaborn as sns
import tensorflow as tf
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

warnings.filterwarnings("ignore")


# ustalenie ziarna dla powtarzalności wyników
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# katalog na informacje o zbiorze
os.makedirs("info", exist_ok=True)

# wyczyszczenie katalogu z analizą zbioru
for filename in os.listdir("info"):
    file_path = os.path.join("info", filename)
    if os.path.isfile(file_path):
        os.remove(file_path)

# wczytanie danych
df = pd.read_csv("apple_quality.csv")
features = ["Size", "Weight", "Sweetness", "Crunchiness", "Juiciness", "Ripeness", "Acidity"]
labels = ["Quality"]

# informacje o zbiorze
print("Rozmiar zbioru:", df.shape)
print("\nFragment zbioru:\n", df.head(10))
print("\nStatystyki opisowe cech numerycznych:\n", df[features].describe())
print("\nStatystyki opisowe klas:\n", df[labels].describe())
print("\nLiczebność klas:\n", df[labels].value_counts())
print("\nProcentowy udział klas:\n", df['Quality'].value_counts(normalize=True) * 100)
print("\nTypy danych i brakujące wartości:", df.info())
print("\nLiczba duplikatów:", df.duplicated().sum())

# wyświetlenie części danych przed czyszczeniem
print("\nDane przed czyszczeniem:")
print(df.head())

# wyczyszczenie danych
df = df.drop(columns=['A_id'])
df = df.dropna()
df = df.drop_duplicates()
df["Acidity"] = df["Acidity"].astype(float)
df["Quality"] = df["Quality"].map({"good": 1, "bad": 0})

# wyświetlenie części danych po czyszczeniu
print("\nDane po czyszczeniu:")
print(df.head())

# zapis oczyszczonych danych
df.to_csv("apple_quality_cleaned.csv", index=False)

# dalsza analiza po czyszczeniu
# macierz korelacji
corr = df.corr()['Quality'].sort_values()
print("\n", corr)
plt.figure()
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.tight_layout()
plt.savefig("info/correlation_matrix.png")
plt.clf()
plt.close()

# histogramy cech
for feature in features:
    plt.figure()
    sns.histplot(df[feature], kde=True)
    plt.title(f'Rozkład cechy: {feature}')
    plt.savefig(f"info/histogram_{feature}.png")
    plt.clf()
    plt.close()

# boxplot cecha vs jakość
for feature in features:
    plt.figure()
    sns.boxplot(x='Quality', y=feature, data=df)
    plt.title(f'{feature} vs Quality')
    plt.savefig(f"info/boxplot_{feature}_vs_Quality.png")
    plt.clf()
    plt.close()

# scatterplot'y wszystkich cech
for feature in features:
    plt.figure(figsize=(6, 4))

    # dodanie lekkiego jitter na osi y
    y_jitter = df['Quality'] + np.random.uniform(-0.05, 0.05, size=df.shape[0])

    sns.scatterplot(x=df[feature], y=y_jitter, hue=df['Quality'], palette={0: 'red', 1: 'green'}, alpha=0.7)
    plt.title(f'{feature} vs Quality')
    plt.xlabel(feature)
    plt.ylabel('Quality')
    plt.yticks([0, 1], ['Bad', 'Good'])
    plt.legend(title='Quality')
    plt.tight_layout()
    plt.savefig(f"info/scatter_{feature}_vs_Quality.png")
    plt.clf()
    plt.close()

# katalog na wyniki
os.makedirs("results_nn", exist_ok=True)

# wyczyszczenie katalogu z wynikami
for filename in os.listdir("results_nn"):
    file_path = os.path.join("results_nn", filename)
    if os.path.isfile(file_path):
        os.remove(file_path)

# wczytanie oczyszczonych danych
df = pd.read_csv("apple_quality_cleaned.csv")

# cechy
X = df[features]

# etykiety
y = df[labels]

# wydzielenie zbioru testowego - 20%
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

print("Rozmiar zbioru testowego:", X_test.shape[0])
print("Rozkład klas (test):")
print(y_test.value_counts())
print()

# z pozostałych danych - wydzielenie zbioru walidacyjnego (25% z pozostałych = 20% całości)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.25,
    random_state=RANDOM_STATE,
    stratify=y_temp
)

print("Rozmiar zbioru treningowego:", X_train.shape[0])
print("Rozkład klas (train):")
print(y_train.value_counts())
print()

print("Rozmiar zbioru walidacyjnego:", X_val.shape[0])
print("Rozkład klas (val):")
print(y_val.value_counts())
print()

# normalizacja danych
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)












# Sztuczne Sieci Neuronowe




# funkcja tworząca model
def create_mlp_model(input, hidden_layers, activation='relu', dropout_rates=None, optimizer='adam'):
    model = Sequential()

    # pętla po warstwach
    for i, (units, dropout_rate) in enumerate(zip(hidden_layers, dropout_rates)):
        if i == 0:
            model.add(Dense(units, activation=activation, input_dim=input.shape[1]))
        else:
            model.add(Dense(units, activation=activation))

        model.add(Dropout(dropout_rate))

    # warstwa wyjściowa
    model.add(Dense(1, activation='sigmoid'))

    # kompilacja modelu
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

# early stopping
early_stop = EarlyStopping(
    monitor='val_loss', # monitorujemy stratę na zb. walidacyjnym
    patience=30, # liczba epok bez poprawy, po której zatrzymujemy trening
    restore_best_weights=True # przywrócenie najlepszej wartości monitorowanej metryki
)

# funkcja tworząca krzywe uczenia
def plot_metric(history, metric, val_metric, title, ylabel, xlabel='Epoka', alias=None):
    plt.plot(history.history[metric], label=f'Zbiór treningowy')
    plt.plot(history.history[val_metric], label=f'Zbiór walidacyjny')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(f"results_nn/{title.replace(' ', '_')}{('_' + alias) if alias else ''}.png")
    plt.clf()
    plt.close()

# funkcja tworząca macierz pomyłek
def plot_confusion_matrix(model, X_test, y_test, labels=["Bad", "Good"], title="CM", alias=None):
    # predykcje modelu
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)

    # macierz pomyłek
    cm = confusion_matrix(y_test, y_pred)

    # zapis macierzy
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='Reds')
    plt.title(title)
    plt.savefig(f"results_nn/{title.replace(' ', '_')}{('_' + alias) if alias else ''}.png")
    plt.clf()
    plt.close()

    return y_pred

# tworzenie modeli
models = {
    "8-4": create_mlp_model(X_train_scaled, [8, 4], dropout_rates=[0.1, 0]),
    "16-8": create_mlp_model(X_train_scaled, [16, 8], dropout_rates=[0.1, 0.1]),
    "24-12": create_mlp_model(X_train_scaled, [24, 12], dropout_rates=[0.2, 0.1]),
    "32-16": create_mlp_model(X_train_scaled, [32, 16], dropout_rates=[0.3, 0.2]),
    "40-20-10": create_mlp_model(X_train_scaled, [40, 20, 10], dropout_rates=[0.2, 0.1, 0.1]),
    "48-24-12": create_mlp_model(X_train_scaled, [48, 24, 12], dropout_rates=[0.3, 0.2, 0.2]),
    "56-28-14": create_mlp_model(X_train_scaled, [56, 28, 14], dropout_rates=[0.3, 0.3, 0.2]),
    "64-32-16": create_mlp_model(X_train_scaled, [64, 32, 16], dropout_rates=[0.4, 0.3, 0.2]),
    "72-36-18-9": create_mlp_model(X_train_scaled, [72, 36, 18, 9], dropout_rates=[0.4, 0.2, 0.2, 0.1]),
    "80-40-20-10": create_mlp_model(X_train_scaled, [80, 40, 20, 10], dropout_rates=[0.4, 0.3, 0.2, 0.1]),
    "88-44-22-11": create_mlp_model(X_train_scaled, [88, 48, 24, 11], dropout_rates=[0.5, 0.3, 0.2, 0.1]),
    "96-48-24-12": create_mlp_model(X_train_scaled, [96, 48, 24, 12], dropout_rates=[0.5, 0.4, 0.2, 0.1])
}

# tablica na wyniki
results = []

# pętla po modelach
for name, model in models.items():
    print(f"\nModel {name}")

    # podsumowanie struktury modelu
    model.summary()

    # uczenie modelu
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=256,
        batch_size=16,
        # callbacks=[early_stop],
        verbose=2
    )

    # ewaluacja na zbiorze testowym
    test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test)

    # dokładność
    plot_metric(history, metric='accuracy', val_metric='val_accuracy', title='LC - Accuracy', ylabel='Dokładność', alias=name)

    # strata
    plot_metric(history, metric='loss', val_metric='val_loss', title='LC - Loss', ylabel='Strata', alias=name)

    # macierz pomyłek i predykcje
    y_pred = plot_confusion_matrix(model, X_test_scaled, y_test, alias=name)

    # raport klasyfikacji
    report = classification_report(y_test, y_pred, target_names=["Bad", "Good"], output_dict=True)

    # zapis wyników do listy
    results.append({
        'Model': name,
        'Test Loss': test_loss,
        'Test Accuracy': test_accuracy,
        'Precision (Good)': report['Good']['precision'],
        'Recall (Good)': report['Good']['recall'],
        'F1-Score (Good)': report['Good']['f1-score'],
        'Precision (Bad)': report['Bad']['precision'],
        'Recall (Bad)': report['Bad']['recall'],
        'F1-Score (Bad)': report['Bad']['f1-score']
    })

# zapis tabeli wyników wszystkich modeli
results_df = pd.DataFrame(results)
results_df.to_csv("results_nn/classification_results.csv", index=False)

# porównanie dokładności modeli
plt.figure(figsize=(8,5))
for result in results:
    plt.bar(result['Model'], result['Test Accuracy'])
plt.title("Porównanie dokładności modeli")
plt.xlabel("Model")
plt.ylabel("Test Accuracy")
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig("results_nn/accuracy_comparison.png")
plt.clf()
plt.close()

# analiza ROC/AUC dla każdej sieci
plt.figure(figsize=(8,6))
for name, model in models.items():
    y_pred_prob = model.predict(X_test_scaled).ravel()  # predykcje prawdopodobieństwa
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Model {name} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.title("Krzywe ROC modeli")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.savefig("results_nn/roc_auc.png")
plt.clf()
plt.close()












# Eksploracja Danych




# katalog na wyniki
os.makedirs("results_ml", exist_ok=True)

# wyczyszczenie katalogu z wynikami
for filename in os.listdir("results_ml"):
    file_path = os.path.join("results_ml", filename)
    if os.path.isfile(file_path):
        os.remove(file_path)

# podział zbioru na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

# info o liczebności próbek
print("Rozmiar zbioru treningowego:", X_train.shape[0])
print("\nRozmiar zbioru testowego:", X_test.shape[0])

# info o liczebności klas
print("\nLiczebność klas w zbiorze treningowym:")
print(y_train.value_counts())
print("\nLiczebność klas w zbiorze testowym:")
print(y_test.value_counts())

# normalizacja danych
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# słownik na modele, na razie tylko te, przy których nie stroimy hiperparametrów
models = {
    'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
    'Quadratic Discriminant Analysis': QuadraticDiscriminantAnalysis(),
    'Gaussian Naive Bayes': GaussianNB()
}

# liczba foldów (do walidacji krzyżowej)
folds = 10

# funkcja pomocnicza zwracająca najlepszy hiperparametr (tylko do k-NN, Logistic Regression, SVM)
def pick_best(results, param_name):
    df = pd.DataFrame(results, columns=[param_name, 'Accuracy'])
    best_value = df.loc[df['Accuracy'].idxmax(), param_name]
    return np.round(best_value, 5), df



# k-NN
# tworzymy listę nieparzystych k (1 - sqrt(l. probek w treningowym)); lista na wyniki cv
k_range = list(range(1, int(np.round(np.sqrt(X_train.shape[0]))) + 1, 2))
cv_knn_results = []

# wykonujemy walidację krzyżową dla każdego k
for k in k_range:
    model = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(model, X_train_scaled, y_train, cv=folds, scoring='accuracy')
    cv_knn_results.append([k, scores.mean()])

# dodanie modelu k-nn z najlepszym k do słownika z modelami
best_k, df_cv_knn_results = pick_best(cv_knn_results, 'k')
models.update({f"k-NN, k={best_k}": KNeighborsClassifier(n_neighbors=best_k)})

# zapis do pliku CSV
df_cv_knn_results.to_csv("results_ml/cv_knn_results.csv", index=False)



# Regresja Logistyczna
# lista wartości C do przetestowania; lista na wyniki walidacji krzyżowej
C_range = np.logspace(-3, 3, 20)
cv_lr_results = []

# walidacja krzyżowa dla każdego C - strojenie
for C in C_range:
    model = LogisticRegression(C=C, random_state=RANDOM_STATE, max_iter=1000)
    scores = cross_val_score(model, X_train_scaled, y_train, cv=folds, scoring='accuracy')
    cv_lr_results.append([C, scores.mean()])

# ramka na wyniki; wybór najlepszego C; dodanie najlepszego modelu do słownika
best_C_lr, df_cv_lr_results = pick_best(cv_lr_results, 'C')
models.update({f"Logistic Regression, C={best_C_lr}": LogisticRegression(C=best_C_lr, random_state=RANDOM_STATE, max_iter=1000)})

# zapis do pliku CSV
df_cv_lr_results.to_csv("results_ml/cv_lr_results.csv", index=False)



# SVM
# lista na wyniki walidacji krzyżowej
cv_svm_results = []

# walidacja krzyżowa dla każdego C - strojenie
for C in C_range:
    model = SVC(C=C, probability=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X_train_scaled, y_train, cv=folds, scoring='accuracy')
    cv_svm_results.append([C, scores.mean()])

# ramka wyników; wybór najlepszego C; dodanie najlepszego modelu do słownika
best_C_svm, df_cv_svm_results = pick_best(cv_svm_results, 'C')
models.update({f"SVM best C={best_C_svm:.4f}": SVC(C=best_C_svm, probability=True, random_state=RANDOM_STATE)})

# zapis do pliku CSV
df_cv_svm_results.to_csv("results_ml/cv_svm_results.csv", index=False)



# Bagging Classifier
# zakresy parametrów
param_dist = {
    'n_estimators': [50, 100, 150, 200, 250],
    'max_samples': [0.5, 0.75, 1.0]
}

# RandomizedSearchCV i dopasowanie
random_search_bag = RandomizedSearchCV(
    estimator=BaggingClassifier(estimator=DecisionTreeClassifier(random_state=RANDOM_STATE), random_state=RANDOM_STATE),
    param_distributions=param_dist,
    n_iter=15,
    scoring='accuracy',
    cv=folds,
    random_state=RANDOM_STATE
)
random_search_bag.fit(X_train_scaled, y_train)

# najlepsze parametry
best_params = random_search_bag.best_params_
best_score = random_search_bag.best_score_

# zapis najlepszego modelu do słownika
best_bag = BaggingClassifier(
    n_estimators=int(best_params['n_estimators']),
    max_samples=float(best_params['max_samples']),
    estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
    random_state=RANDOM_STATE
)
models.update({f"Bagging, n={best_params['n_estimators']}, samples={best_params['max_samples']}": best_bag})

# zapis wyników wszystkich testowanych kombinacji
df_cv_bag_results = pd.DataFrame({
    'n_estimators': random_search_bag.cv_results_['param_n_estimators'].data,
    'max_samples': random_search_bag.cv_results_['param_max_samples'].data,
    'Accuracy': random_search_bag.cv_results_['mean_test_score']
})
df_cv_bag_results.to_csv("results_ml/cv_bag_results.csv", index=False)



# Random Forest
# zakresy parametrów
param_dist = {
    'n_estimators': [50, 100, 150, 200, 250],
    'max_depth': [1, 2, 3, 4, 5, None],
    'min_samples_leaf': [1, 2, 5, 10, 20],
    'min_samples_split': [2, 5, 10, 20]
}

# RandomizedSearchCV i dopasowanie
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=RANDOM_STATE),
    param_distributions=param_dist,
    n_iter=30,
    scoring='accuracy',
    cv=folds,
    random_state=RANDOM_STATE
)
random_search.fit(X_train_scaled, y_train)

# najlepsze parametry
best_params = random_search.best_params_
best_score = random_search.best_score_

# zapis najlepszego modelu do słownika
best_rf = RandomForestClassifier(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    min_samples_leaf=best_params['min_samples_leaf'],
    min_samples_split=best_params['min_samples_split'],
    random_state=RANDOM_STATE
)
models.update({f"Random Forest, n={best_params['n_estimators']}, depth={best_params['max_depth']}, leaf={best_params['min_samples_leaf']}, split={best_params['min_samples_split']}": best_rf})

# zapis wyników wszystkich testowanych kombinacji
df_cv_rf_results = pd.DataFrame({
    'n_estimators': random_search.cv_results_['param_n_estimators'].data,
    'max_depth': random_search.cv_results_['param_max_depth'].data,
    'min_samples_leaf': random_search.cv_results_['param_min_samples_leaf'].data,
    'min_samples_split': random_search.cv_results_['param_min_samples_split'].data,
    'Accuracy': random_search.cv_results_['mean_test_score']
})
df_cv_rf_results.to_csv("results_ml/cv_rf_results.csv", index=False)



# Gradient Boosting
# zakresy parametrów
param_dist = {
    'n_estimators': [50, 100, 150, 200],
    'learning_rate': [0.001, 0.01, 0.05, 0.1],
    'max_depth': [1, 2, 3, 4, 5, None]
}

# RandomizedSearchCV i dopasowanie
random_search_gb = RandomizedSearchCV(
    estimator=GradientBoostingClassifier(random_state=RANDOM_STATE),
    param_distributions=param_dist,
    n_iter=30,
    scoring='accuracy',
    cv=folds,
    random_state=RANDOM_STATE
)
random_search_gb.fit(X_train_scaled, y_train)

# najlepsze parametry
best_params = random_search_gb.best_params_
best_score = random_search_gb.best_score_

# zapis najlepszego modelu do słownika
best_gb = GradientBoostingClassifier(
    n_estimators=best_params['n_estimators'],
    learning_rate=best_params['learning_rate'],
    max_depth=best_params['max_depth'],
    random_state=RANDOM_STATE
)
models.update({f"Gradient Boosting, n={best_params['n_estimators']}, depth={best_params['max_depth']}, lr={best_params['learning_rate']}": best_gb})

# zapis wyników wszystkich testowanych kombinacji
df_cv_gb_results = pd.DataFrame({
    'n_estimators': random_search_gb.cv_results_['param_n_estimators'].data,
    'max_depth': random_search_gb.cv_results_['param_max_depth'].data,
    'learning_rate': random_search_gb.cv_results_['param_learning_rate'].data,
    'Accuracy': random_search_gb.cv_results_['mean_test_score']
})
df_cv_gb_results.to_csv("results_ml/cv_gb_results.csv", index=False)



# lista na wyniki
results = []

# pętla po modelach, uczenie każdego modelu
for name, model in models.items():
    print(f"\nModel: {name}")

    # uczenie
    model.fit(X_train_scaled, y_train)

    # predykcje
    y_pred = model.predict(X_test_scaled)
    y_pred_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test_scaled)

    # dokładność
    acc = accuracy_score(y_test, y_pred)

    # raport klasyfikacji
    report = classification_report(y_test, y_pred, output_dict=True)

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision (Good)": report.get("1", {}).get("precision", np.nan),
        "Recall (Good)": report.get("1", {}).get("recall", np.nan),
        "F1-Score (Good)": report.get("1", {}).get("f1-score", np.nan),
        "Precision (Bad)": report.get("0", {}).get("precision", np.nan),
        "Recall (Bad)": report.get("0", {}).get("recall", np.nan),
        "F1-Score (Bad)": report.get("0", {}).get("f1-score", np.nan)
    })

    # macierz pomyłek
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Bad", "Good"])
    disp.plot(cmap='Reds')
    plt.title(f"Macierz pomyłek - {name}")
    plt.savefig(f"results_ml/cm_{name.replace(' ', '_')}.png")
    plt.close()

    # ROC/AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')

# finalny wykres ROC/AUC
plt.plot([0,1], [0,1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Krzywe ROC modeli")
plt.legend(loc="lower right")
plt.savefig("results_ml/roc_auc.png")
plt.clf()
plt.close()

# tabela wyników
results_df = pd.DataFrame(results)
results_df.to_csv("results_ml/classification_results.csv", index=False)

# wykres porównania dokładności
plt.figure(figsize=(8,5))
plt.bar(results_df['Model'], results_df['Accuracy'], color='skyblue')
plt.title("Porównanie dokładności modeli")
plt.ylabel("Accuracy")
plt.ylim(0,1)
plt.xticks(rotation=90)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig("results_ml/accuracy_comparison.png")
plt.clf()
plt.close()