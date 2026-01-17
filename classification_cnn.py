import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


# ustawienie ziarna
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# katalog na wyniki
os.makedirs("results_cnn", exist_ok=True)

# liczba klas
classes = 100

# wczytanie danych
(X_train, y_train), (X_test, y_test) = datasets.cifar100.load_data(label_mode='coarse' if classes == 20 else 'fine')

print("Train:", X_train.shape, "Test:", X_test.shape)
print("Przykład etykiety:", y_train[0])

# podgląd kilku obrazków
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

# podział na trening i walidację
X_train_sub, X_val, y_train_sub, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train
)

tf.config.optimizer.set_jit('autoclustering')

# augementacja danych
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=True
)
datagen.fit(X_train_sub)

def plot_metric(history, metric, val_metric, title, ylabel):
    plt.plot(history.history[metric], label=f'Train {ylabel}')
    plt.plot(history.history[val_metric], label=f'Val {ylabel}')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(f"results_cnn/{title}{' (coarse)' if classes == 20 else ' (fine)'}.png")
    plt.close()

def evaluate_model(model, name):
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")

    # predykcja
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    # raport
    report = classification_report(y_test, y_pred, output_dict=True)

    return {
        'Model': name,
        'Test Loss': test_loss,
        'Test Accuracy': test_acc,
        'Precision (avg)': report['weighted avg']['precision'],
        'Recall (avg)': report['weighted avg']['recall'],
        'F1 (avg)': report['weighted avg']['f1-score']
    }

# słownik na modele
models_dict = {}

# model 1
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
    layers.Dense(classes, activation='softmax')
])

# model 2
models_dict["cnn_medium"] = models.Sequential([
    layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=(32,32,3)),
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
    layers.Dense(classes, activation='softmax'),
])

# model 3
models_dict["cnn_large"] = models.Sequential([
    layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=(32,32,3)),
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
    layers.Dense(classes, activation='softmax')
])

# model 4
models_dict["cnn_xlarge"] = models.Sequential([
    layers.Conv2D(64, 3, padding='same', activation='relu', input_shape=(32,32,3)),
    layers.BatchNormalization(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),

    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.35),

    layers.Conv2D(256, 3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(256, 3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.45),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(classes, activation='softmax')
])

# lista na wyniki
results = []

# pętla po modelach
for name, model in models_dict.items():
    print(f"\nModel: {name}")

    # kompilacja
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # podsumowanie struktury modelu
    model.summary()

    # mechanizm wczesnego zatrzymania
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        min_delta=0.0005,
        restore_best_weights=True,
        verbose=1,
    )

    # zmniejszanie lr jeśli model utknie
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )

    # uczenie
    history = model.fit(
        datagen.flow(X_train_sub, y_train_sub, batch_size=128),
        validation_data=(X_val, y_val),
        epochs=100,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    # krzywe uczenia
    plot_metric(history, 'accuracy', 'val_accuracy', f'{name} Accuracy', 'Accuracy')
    plot_metric(history, 'loss', 'val_loss', f'{name} Loss', 'Loss')

    res = evaluate_model(model, name)
    results.append(res)

    # zapis wyników po każdym modelu
    df_results = pd.DataFrame(results)
    df_results = df_results.round(4)
    df_results.to_csv(f"results_cnn/{name} Results{' (coarse)' if classes == 20 else ' (fine)'}.csv", index=False)

# zapis wyników
df_results = pd.DataFrame(results)
df_results.to_csv(f"results_cnn/results{' (coarse)' if classes == 20 else ' (fine)'}.csv", index=False)
