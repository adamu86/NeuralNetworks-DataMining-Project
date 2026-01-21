import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
import seaborn as sns
from scipy.stats import shapiro
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import PolynomialFeatures, PowerTransformer, QuantileTransformer, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

warnings.filterwarnings("ignore")


# ustalenie ziarna dla powtarzalności wyników
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

# katalog na informacje o zbiorze
os.makedirs("info", exist_ok=True)

# wczytanie danych
df = pd.read_csv("apple_quality.csv")
features = df.drop(columns=["Quality", "A_id"]).columns.tolist()
labels = ["Quality"]

# informacje o zbiorze
print("Rozmiar zbioru:", df.shape)
print("\nFragment zbioru:\n", df.head(10))
print("\nStatystyki opisowe cech numerycznych:\n", df.drop(columns=['A_id']).describe(include='all').fillna(''))
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
# macierz korelacji (Pearson)
corr = df.corr()['Quality'].sort_values()
print("\n", corr)
plt.figure()
sns.heatmap(df.corr(method='pearson'), annot=True, cmap='coolwarm')
plt.tight_layout()
plt.savefig("info/correlation_matrix_pearson.png")
plt.clf()
plt.close()

# macierz korelacji (Spearman)
corr = df.corr()['Quality'].sort_values()
print("\n", corr, '\n')
plt.figure()
sns.heatmap(df.corr(method='spearman'), annot=True, cmap='coolwarm')
plt.tight_layout()
plt.savefig("info/correlation_matrix_spearman.png")
plt.clf()
plt.close()

# wartości odstające
print("\nWartości odstające:\n")

for feature in features:
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5*IQR
    upper = Q3 + 1.5*IQR
    count = df[(df[feature] < lower) | (df[feature] > upper)].shape[0]

    print(f"{feature} - outliery: {count} ({count / df.shape[0] * 100:.2f}%)")

# skośne cechy
skewed_features = ["Juiciness", "Sweetness", "Weight", "Crunchiness"]

pt = PowerTransformer(method='yeo-johnson')
df[skewed_features] = pt.fit_transform(df[skewed_features])

# histogramy cech
normality = []
for feature in features:
    plt.figure()
    sns.histplot(df[feature], kde=True)
    stat, p = shapiro(df[feature])
    normality.append({'Feature': feature, 'Statistic': stat, 'p_value': p})
    plt.title(f'Rozkład cechy: {feature} (p={p:.3f}, stat={stat:.3f})')
    plt.savefig(f"info/histogram_{feature}.png")
    plt.clf()
    plt.close()

pd.DataFrame(normality).to_csv("info/normal_distributions.csv", index=False)

# boxplot cecha vs jakość
for feature in features:
    plt.figure()
    sns.boxplot(x='Quality', y=feature, data=df)
    plt.title(f'{feature} vs Quality')
    plt.savefig(f"info/boxplot_{feature}_vs_Quality.png")
    plt.clf()
    plt.close()



# katalog na wyniki
os.makedirs("results_ml", exist_ok=True)

# wczytanie oczyszczonych danych
df = pd.read_csv("apple_quality_cleaned.csv")

# cechy
X = df[features]

# etykiety
y = df[labels]

# podział zbioru na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

# info o liczebności próbek
print("\nRozmiar zbioru treningowego:", X_train.shape[0])
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

# wyliczamy MI (cecha vs klasa)
mi = mutual_info_classif(
    X,
    y,
    n_neighbors=5,
    random_state=42
)

mi_df = pd.DataFrame({
    "feature": features,
    "mi_score": mi
}).sort_values("mi_score", ascending=False)

print("\nMI (cecha - klasa):\n", mi_df)

# MI między cechami
mi_matrix = pd.DataFrame(index=features, columns=features, dtype=float)

# liczenie MI dla każdej pary cech
for f1 in features:
    for f2 in features:
        if f1 == f2:
            mi_matrix.loc[f1, f2] = 0
        else:
            mi_matrix.loc[f1, f2] = mutual_info_regression(
                df[[f1]], df[f2], random_state=42
            )[0]

# wizualizacja
print("\nMI (cecha - cecha):\n", mi_matrix)






# słownik na modele, na razie tylko te, przy których nie stroimy hiperparametrów
models = {
    'LDA': LinearDiscriminantAnalysis(),
    'QDA': QuadraticDiscriminantAnalysis(),
    'GaussianNB': GaussianNB()
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
models.update({f"LR, C={best_C_lr}": LogisticRegression(C=best_C_lr, random_state=RANDOM_STATE, max_iter=1000)})

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
models.update({f"SVM, C={best_C_svm:.4f}": SVC(C=best_C_svm, probability=True, random_state=RANDOM_STATE)})

# zapis do pliku CSV
df_cv_svm_results.to_csv("results_ml/cv_svm_results.csv", index=False)

# Drzewo decyzyjne
# zakresy parametrów
param_dist_tree = {
    'max_depth': [1, 2, 3, 4, 5, 6, 8, 10, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 5, 10]
}

# RandomizedSearchCV dla drzewa decyzyjnego
random_search_tree = RandomizedSearchCV(
    estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
    param_distributions=param_dist_tree,
    n_iter=20,
    scoring='accuracy',
    cv=folds,
    random_state=RANDOM_STATE
)

random_search_tree.fit(X_train_scaled, y_train)

# najlepsze parametry
best_params_tree = random_search_tree.best_params_
best_score_tree = random_search_tree.best_score_

# zapis najlepszego modelu do słownika
best_tree = DecisionTreeClassifier(
    max_depth=best_params_tree['max_depth'],
    min_samples_split=best_params_tree['min_samples_split'],
    min_samples_leaf=best_params_tree['min_samples_leaf'],
    random_state=RANDOM_STATE
)
models.update({f"Decision Tree, depth={best_params_tree['max_depth']}, split={best_params_tree['min_samples_split']}, leaf={best_params_tree['min_samples_leaf']}": best_tree})

# zapis wyników wszystkich kombinacji do CSV
df_cv_tree_results = pd.DataFrame({
    'max_depth': random_search_tree.cv_results_['param_max_depth'].data,
    'min_samples_split': random_search_tree.cv_results_['param_min_samples_split'].data,
    'min_samples_leaf': random_search_tree.cv_results_['param_min_samples_leaf'].data,
    'Accuracy': random_search_tree.cv_results_['mean_test_score']
})
df_cv_tree_results.to_csv("results_ml/cv_tree_results.csv", index=False)



# Bagging
# zakresy parametrów
param_dist = {
    'n_estimators': [50, 100, 150, 200, 250],
    'max_samples': [0.5, 0.75, 1.0]
}

# RandomizedSearchCV i dopasowanie
random_search_bag = RandomizedSearchCV(
    estimator=BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=RANDOM_STATE), 
        random_state=RANDOM_STATE
    ),
    param_distributions=param_dist,
    n_iter=20,
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
    'max_depth': [1, 2, 3, 4, 5, 6, 8, 10, None],
    'min_samples_leaf': [1, 2, 5, 10, 20],
    'min_samples_split': [2, 5, 10, 20]
}

# RandomizedSearchCV i dopasowanie
random_search_rf = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=RANDOM_STATE),
    param_distributions=param_dist,
    n_iter=20,
    scoring='accuracy',
    cv=folds,
    random_state=RANDOM_STATE
)
random_search_rf.fit(X_train_scaled, y_train)

# najlepsze parametry
best_params = random_search_rf.best_params_
best_score = random_search_rf.best_score_

# zapis najlepszego modelu do słownika
best_rf = RandomForestClassifier(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    min_samples_leaf=best_params['min_samples_leaf'],
    min_samples_split=best_params['min_samples_split'],
    random_state=RANDOM_STATE
)
models.update({f"RF, n={best_params['n_estimators']}, depth={best_params['max_depth']}, leaf={best_params['min_samples_leaf']}, split={best_params['min_samples_split']}": best_rf})

# zapis wyników wszystkich testowanych kombinacji
df_cv_rf_results = pd.DataFrame({
    'n_estimators': random_search_rf.cv_results_['param_n_estimators'].data,
    'max_depth': random_search_rf.cv_results_['param_max_depth'].data,
    'min_samples_leaf': random_search_rf.cv_results_['param_min_samples_leaf'].data,
    'min_samples_split': random_search_rf.cv_results_['param_min_samples_split'].data,
    'Accuracy': random_search_rf.cv_results_['mean_test_score']
})
df_cv_rf_results.to_csv("results_ml/cv_rf_results.csv", index=False)



# Gradient Boosting
# zakresy parametrów
param_dist = {
    'n_estimators': [50, 100, 150, 200, 250],
    'learning_rate': [0.001, 0.01, 0.05, 0.1],
    'max_depth': [1, 2, 3, 4, 5, 6, 8, 10, None]
}

# RandomizedSearchCV i dopasowanie
random_search_gb = RandomizedSearchCV(
    estimator=GradientBoostingClassifier(random_state=RANDOM_STATE),
    param_distributions=param_dist,
    n_iter=20,
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
models.update({f"GB, n={best_params['n_estimators']}, depth={best_params['max_depth']}, lr={best_params['learning_rate']}": best_gb})

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

    # predykcje; jeśli model ma predict_proba, to bierzemy prawdopodobieństwa
    y_pred = model.predict(X_test_scaled)
    if hasattr(model, "predict_proba"):
        y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_pred_prob = model.decision_function(X_test_scaled)

    # dokładność
    accuracy = accuracy_score(y_test, y_pred)

    # raport klasyfikacji
    report = classification_report(y_test, y_pred, output_dict=True)

    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision (Good)': report['1']['precision'],
        'Recall (Good)': report['1']['recall'],
        'F1-Score (Good)': report['1']['f1-score'],
        'Precision (Bad)': report['0']['precision'],
        'Recall (Bad)': report['0']['recall'],
        'F1-Score (Bad)': report['0']['f1-score']
    })

    # przygotowanie wykresu z dwoma osiami
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # większy tekst do czytelnosci
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'legend.fontsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12
    })

    # macierz pomyłek
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Bad", "Good"])
    disp.plot(ax=axes[0], cmap='Reds', colorbar=False)
    axes[0].set_title("Confusion Matrix")

    # krzywa ROC
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    axes[1].plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
    axes[1].plot([0,1], [0,1], linestyle='--', color='gray')
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC AUC")
    axes[1].legend(loc="lower right")

    # zapis pliku
    plt.suptitle(f"{name} Summary")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"results_ml/{name} Summary.png")
    plt.close()

    mpl.rcParams.update(mpl.rcParamsDefault)

# finalny wykres ROC/AUC
for name, model in models.items():
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test_scaled)[:, 1]
    else:
        probs = model.decision_function(X_test_scaled)
    fpr, tpr, _ = roc_curve(y_test.values.ravel(), probs)
    auc_score = auc(fpr, tpr)
    plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={auc_score:.3f})")

plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC AUC")
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("results_ml/ROC AUC.png", dpi=300, bbox_inches='tight')
plt.close()

# tabela wyników
results_df = pd.DataFrame(results)
results_df = results_df.round(4)

# sortowanie według Accuracy malejąco
results_df = results_df.sort_values(by='Accuracy', ascending=False)

results_df.to_csv("results_ml/classification_results.csv", index=False)

# wykres słupkowy dokładności modeli
plt.bar(results_df['Model'], results_df['Accuracy'], color='skyblue')
plt.title("Models Accuracy")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.xticks(rotation=85)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("results_ml/Accuracy Barplot.png")
plt.clf()
plt.close()