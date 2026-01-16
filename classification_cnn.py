import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.callbacks import EarlyStopping
import random

# ustawienie ziarna
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# katalog na wyniki
os.makedirs("results_cnn", exist_ok=True)

# wczytanie danych
(X_train, y_train), (X_test, y_test) = datasets.cifar100.load_data(label_mode='fine')

print("Train:", X_train.shape, "Test:", X_test.shape)
print("Przykład etykiety:", y_train[0])

# rzut oka na kilka obrazków
plt.figure(figsize=(8,4))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(X_train[i])
    plt.axis('off')
plt.tight_layout()
plt.savefig("results_cnn/podglad_probek.png")
plt.close()

# normalizacja
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# flatten etykiety
y_train = y_train.flatten()
y_test = y_test.flatten()

# konwersja do one-hot
y_train_cat = tf.keras.utils.to_categorical(y_train, 100)
y_test_cat = tf.keras.utils.to_categorical(y_test, 100)

def plot_metric(history, metric, val_metric, title, ylabel):
    plt.plot(history.history[metric], label='train')
    plt.plot(history.history[val_metric], label='val')
    plt.title(title)
    plt.xlabel('epoch')
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(f"results_cnn/{title.replace(' ', '_')}.png")
    plt.close()

def evaluate_model(model, name):
    print(f"\nModel {name} ocena testowa")
    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")

    # predykcja
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    # raport
    report = classification_report(y_test, y_pred, output_dict=True)
    return {
        'Model': name,
        'Accuracy': test_acc,
        'Precision (avg)': report['weighted avg']['precision'],
        'Recall (avg)': report['weighted avg']['recall'],
        'F1 (avg)': report['weighted avg']['f1-score']
    }

# MODELE CNN
models_dict = {}

# Model 1 – mały CNN
models_dict["cnn_small"] = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.3),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(100, activation='softmax')
])

# Model 2 – większa sieć
models_dict["cnn_medium"] = models.Sequential([
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),

    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.3),

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(100, activation='softmax'),
])

# Model 3 – duży CNN
models_dict["cnn_large"] = models.Sequential([
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),

    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.3),

    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.4),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(100, activation='softmax')
])

results = []

for name, model in models_dict.items():
    print(f"\nModel: {name}")

    # kompilacja
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # podsumowanie struktury modelu
    model.summary()

    # mechanizm wczesnego zatrzymania
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        min_delta=0.001,
        restore_best_weights=True,
        verbose=1,
    )

    # uczenie
    history = model.fit(
        X_train, y_train_cat,
        validation_split=0.2,
        epochs=1000,
        batch_size=64,
        callbacks=[early_stopping],
        verbose=1
    )

    # krzywe uczenia
    plot_metric(history, 'accuracy', 'val_accuracy', f'Accuracy {name}', 'accuracy')
    plot_metric(history, 'loss', 'val_loss', f'Loss {name}', 'loss')

    res = evaluate_model(model, name)
    results.append(res)

# zapis wyników
df_results = pd.DataFrame(results)
df_results.to_csv("results_cnn/results.csv", index=False)

print(df_results)
