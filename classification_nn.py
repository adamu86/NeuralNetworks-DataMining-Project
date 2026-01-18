import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
import seaborn as sns
import tensorflow as tf
from scipy.stats import shapiro
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import L1, L2, L1L2
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

warnings.filterwarnings("ignore")


# ustalenie stałego ziarna (powtarzalnosci wyników)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# katalog na informacje o zbiorze
os.makedirs("info", exist_ok=True)

# wczytanie danych
df = pd.read_csv("apple_quality.csv")
features = ["Size", "Weight", "Sweetness", "Crunchiness", "Juiciness", "Ripeness", "Acidity"]
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
print("\n", corr)
plt.figure()
sns.heatmap(df.corr(method='spearman'), annot=True, cmap='coolwarm')
plt.tight_layout()
plt.savefig("info/correlation_matrix_spearman.png")
plt.clf()
plt.close()

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
os.makedirs("results_nn", exist_ok=True)

# wczytanie oczyszczonych danych
df = pd.read_csv("apple_quality_cleaned.csv")

# cechy
X = df[features]

# etykiety
y = df[labels]

# wydzielenie zbioru treningowego i testowego
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# normalizacja danych
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# funkcja tworząca model
def create_mlp_model(input, hidden_layers, activation='relu', dropout_rates=None, l1_rates=None, l2_rates=None, optimizer='adam'):
    model = Sequential()

    # liczba warstw
    n_layers = len(hidden_layers)

    # jeśli nie podamy regularyzacji
    if dropout_rates is None:
        dropout_rates = [0.0] * n_layers
    if l1_rates is None:
        l1_rates = [None] * n_layers
    if l2_rates is None:
        l2_rates = [None] * n_layers

    # pętla po warstwach
    for i, (units, dropout_rate, l1_rate, l2_rate) in enumerate(zip(hidden_layers, dropout_rates, l1_rates, l2_rates)):
        # obiekt regularyzacji L1/L2/L1L2
        if l1_rate is not None and l2_rate is not None:
            reg = L1L2(l1=l1_rate, l2=l2_rate)
        elif l1_rate is not None:
            reg = L1(l1_rate)
        elif l2_rate is not None:
            reg = L2(l2_rate)
        else:
            reg = None

        # dodanie warstwy w sieci,
        if i == 0:
            model.add(Dense(units, activation=activation, input_dim=input.shape[1], kernel_regularizer=reg))
        else:
            model.add(Dense(units, activation=activation, kernel_regularizer=reg))

        # dodanie dropout
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

# funkcja tworząca wykresy/macierz pomyłek i ROC
def plot_model_summary(history, model, X_test, y_test, title, filename, labels=["Bad", "Good"]):
    # większy tekst do czytelnosci
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'legend.fontsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12
    })

    # predykcje
    y_pred_prob = model.predict(X_test).ravel()
    y_pred = (y_pred_prob > 0.5).astype(int)

    # macierz pomyłek
    cm = confusion_matrix(y_test, y_pred)

    # przygotowanie ROC
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    # układ wykresów: 2x2
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # górny wiersz - krzywe uczenia
    axs[0, 0].plot(history.history['accuracy'], label='Train Accuracy')
    axs[0, 0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axs[0, 0].set_title('Learning Curve (Accuracy)')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Accuracy')
    axs[0, 0].legend()

    axs[0, 1].plot(history.history['loss'], label='Train Loss')
    axs[0, 1].plot(history.history['val_loss'], label='Val Loss')
    axs[0, 1].set_title('Learning Curve (Loss)')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Loss')
    axs[0, 1].legend()

    # dolny wiersz - macierz pomyłek i ROC
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=axs[1, 0], cmap='Reds', colorbar=False)
    axs[1, 0].set_title('Confusion Matrix')

    axs[1, 1].plot(fpr, tpr, color='blue', label=f'AUC = {roc_auc:.2f}')
    axs[1, 1].plot([0, 1], [0, 1], color='gray', linestyle='--')
    axs[1, 1].set_title('ROC Curve')
    axs[1, 1].set_xlabel('False Positive Rate')
    axs[1, 1].set_ylabel('True Positive Rate')
    axs[1, 1].legend(loc='lower right')

    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"results_nn/{filename}.png")
    plt.clf()
    plt.close()

    mpl.rcParams.update(mpl.rcParamsDefault)

    return y_pred

# słownik na modele
models = {}

# konfiguracje modeli
config = {
    (4,): {"dropout": [0.0], "l1": [0.0], "l2": [0.0]},
    (8,): {"dropout": [0.0], "l1": [0.0], "l2": [0.0]},
    (32,): {"dropout": [0.05], "l1": [0.0001], "l2": [0.0005]},
    (64,): {"dropout": [0.1], "l1": [0.0002], "l2": [0.001]},
    (16,8): {"dropout": [0.05,0.05], "l1": [0.00005]*2, "l2": [0.00025]*2},
    (32,16): {"dropout": [0.05,0.05], "l1": [0.0001]*2, "l2": [0.0005]*2},
    (64,32): {"dropout": [0.1,0.05], "l1": [0.0002]*2, "l2": [0.001]*2},
    (128,64): {"dropout": [0.15,0.1], "l1": [0.0003]*2, "l2": [0.002]*2},
    (64,32,16): {"dropout": [0.1,0.05,0.05], "l1": [0.00015]*3, "l2": [0.00075]*3},
    (128,64,32): {"dropout": [0.15,0.1,0.05], "l1": [0.0003]*3, "l2": [0.002]*3},
    (256,128,64): {"dropout": [0.2,0.15,0.1], "l1": [0.0004]*3, "l2": [0.004]*3},
    (512,256,128): {"dropout": [0.25,0.2,0.15], "l1": [0.0005]*3, "l2": [0.005]*3},
}

# optymalizatory
optimizers = [
    'adam',
    'adamw',
    'adamax',
    'nadam',
    'rmsprop',
    'lion'
]

# tworzenie modeli
for opt in [optimizers[0]]:
    for layers, params in config.items():
        models[f"{list(layers)} {opt}"] = create_mlp_model(
            X_train_scaled,
            layers,
            dropout_rates=params["dropout"],
            l1_rates=params["l1"],
            l2_rates=params["l2"],
            optimizer=opt
        )

# tablica na wyniki
results = []

# pętla po modelach
for name, model in models.items():
    print(f"\nModel {name}")

    # podsumowanie struktury modelu
    model.summary()

    # mechanizm wczesnego zatrzymania
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        min_delta=0.001,
        restore_best_weights=True,
        verbose=1,
    )

    # uczenie modelu
    history = model.fit(
        X_train_scaled, y_train,
        validation_split=0.2,
        epochs=1000,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=2
    )

    # ewaluacja na zbiorze testowym
    test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test)

    # predykcje i wykresy
    y_pred = plot_model_summary(history, model, X_test_scaled, y_test, title=f"Model/Optimizer: {name}", filename=f"{name} Summary")

    # raport klasyfikacji
    report = classification_report(y_test, y_pred, target_names=["Bad", "Good"], output_dict=True)

    # zapis wyników do listy (metryki dla klasy good i bad)
    results.append({
        'Model/Optimizer': name,
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

# zaokrąglenie (do czytelności)
results_df = results_df.round(4)

# sortowanie, najpierw po Test Accuracy malejąco, potem po Test Loss rosnąco
results_df = results_df.sort_values(by=['Test Accuracy', 'Test Loss'], ascending=[False, True])
results_df.to_csv("results_nn/classification_results.csv", index=False)

# wykres słupkowy dokładności
plt.figure()
plt.bar(results_df['Model/Optimizer'], results_df['Test Accuracy'])
plt.xticks(rotation=85, ha='right')
plt.ylabel("Test Accuracy")
plt.title("Models Accuracy")
plt.tight_layout()
plt.savefig("results_nn/Accuracy Barplot.png", dpi=300, bbox_inches='tight')
plt.close()

# krzywa roc i auc
plt.figure()
for name, model in models.items():
    y_prob = model.predict(X_test_scaled).ravel()
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={roc_auc:.3f})")

plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC AUC")
plt.legend()
plt.grid(True)
plt.savefig("results_nn/ROC AUC.png", dpi=300, bbox_inches='tight')
plt.close()

# wybieramy najlepszy model i wydobywamy jego architekture
best_model_name = results_df.iloc[0]['Model/Optimizer']
best_layers = eval(best_model_name.split(']')[0] + ']')
best_params = config[tuple(best_layers)]

# lista na wyniki
results_opt = []

models = {}

for opt in optimizers:
    print(f"\nOptimizer {opt}")

    # tworzymy nowy model na podstawie najlepszego
    new_model = create_mlp_model(
        X_train_scaled,
        hidden_layers=best_layers,
        dropout_rates=best_params["dropout"],
        l1_rates=best_params["l1"],
        l2_rates=best_params["l2"],
        optimizer=opt
    )

    # wczesne zatrzymanie
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        min_delta=0.0005,
        restore_best_weights=True,
        verbose=1,
    )

    # zmniejszanie lr jeśli model utknie
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.000001,
        verbose=1
    )

    # uczenie
    history = new_model.fit(
        X_train_scaled, y_train,
        validation_split=0.2,
        epochs=1000,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        verbose=2
    )

    # ewaluacja na zbiorze testowym
    test_loss, test_accuracy = new_model.evaluate(X_test_scaled, y_test)

    # predykcje i wykresy
    y_pred = plot_model_summary(history, new_model, X_test_scaled, y_test, title=f"Best Model - {best_layers} {opt} optimizer", filename=f"Best Model - {best_layers} {opt} optimizer")

    # raport klasyfikacji
    report = classification_report(y_test, y_pred, target_names=["Bad", "Good"], output_dict=True)

    # zapis wyników
    results_opt.append({
        'Optimizer': opt,
        'Test Loss': test_loss,
        'Test Accuracy': test_accuracy,
        'Precision (Good)': report['Good']['precision'],
        'Recall (Good)': report['Good']['recall'],
        'F1-Score (Good)': report['Good']['f1-score'],
        'Precision (Bad)': report['Bad']['precision'],
        'Recall (Bad)': report['Bad']['recall'],
        'F1-Score (Bad)': report['Bad']['f1-score']
    })

    models[opt] = new_model

# zapis wyników do CSV
results_opt_df = pd.DataFrame(results_opt)
results_opt_df = results_opt_df.round(4)

# sortowanie po Test Accuracy malejąco i Test Loss rosnąco
results_opt_df = results_opt_df.sort_values(by=['Test Accuracy', 'Test Loss'], ascending=[False, True])

# zapis wynikow
results_opt_df.to_csv("results_nn/best_model_optimizers.csv", index=False)

# wykres słupkowy dokładności najlepszych optymalizatorów
plt.figure()
plt.bar(results_opt_df['Optimizer'], results_opt_df['Test Accuracy'])
plt.xticks(rotation=45, ha='right')
plt.ylabel("Test Accuracy")
plt.title("Best Model Accuracy - Optimizers")
plt.tight_layout()
plt.savefig("results_nn/Best Model Optimizers.png", dpi=300, bbox_inches='tight')
plt.close()

# krzywa roc i auc
plt.figure()
for name, model in models.items():
    probs = model.predict(X_test_scaled).ravel()
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc_score = auc(fpr, tpr)
    plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={auc_score:.3f})")

plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Best Model ROC AUC - Optimizers")
plt.legend()
plt.grid(True)
plt.savefig("results_nn/Best Model ROC AUC.png", dpi=300, bbox_inches='tight')
plt.close()