import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


# ustawienie stałego ziarna
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
plt.savefig("results_cnn/Overview.png")
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
    rotation_range=10, # losowa rotacja w zakresie +-10 stopni
    width_shift_range=0.05, # przesunięcie obrazu w poziomie do 5% szerekości
    height_shift_range=0.05, # przesunięcie obrazu w pionie do 5% wysokości
    shear_range=0.05, # lekkie ścięcie obrazu w losowym kierunku
    zoom_range=0.05, # losowe przybliżenie/oddalenie obrazu o +-5%
    fill_mode='nearest' # sposób wypełniania brakujących pikseli po przeksztalceniach
)
datagen.fit(X_train_sub)

# funkcja generująca krzywe uczenia (loss albo accuracy)
def plot_metric(history, metric, val_metric, title, ylabel):
    plt.plot(history.history[metric], label=f'Train {ylabel}')
    plt.plot(history.history[val_metric], label=f'Val {ylabel}')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(f"results_cnn/{title}{' (coarse)' if classes == 20 else ' (fine)'}.png")
    plt.close()

# ewaluacja modelu (metryki)
def evaluate_model(model, name):
    # ewaluacja
    test_loss, test_acc = model.evaluate(X_test, y_test)

    # predykcja
    y_pred = np.argmax(model.predict(X_test), axis=1)

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

# pętla po modelach w slowniku
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
        min_delta=0.001,
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

    # pomiar czasu
    start_time = time.time()

    # uczenie
    history = model.fit(
        datagen.flow(X_train_sub, y_train_sub, batch_size=64),
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr],
        epochs=100,
        shuffle=True,
        verbose=1
    )

    # koniec pomiaru czasu
    end_time = time.time()
    elapsed = end_time - start_time

    # krzywe uczenia
    plot_metric(history, 'accuracy', 'val_accuracy', f'{name} Accuracy', 'Accuracy')
    plot_metric(history, 'loss', 'val_loss', f'{name} Loss', 'Loss')

    # epoka najlepszego wyniku na walidacji
    best_epoch = np.argmin(history.history['val_loss']) + 1

    # ewaluacja i dodanie wyników do listy
    res = evaluate_model(model, name)
    res['Train Time (s)'] = round(elapsed, 2)
    res['Best Epoch'] = best_epoch
    results.append(res)

# zapis wszystkich wyników
df_results = pd.DataFrame(results)
df_results = df_results.round(4)
df_results.to_csv(f"results_cnn/results{' (coarse)' if classes == 20 else ' (fine)'}.csv", index=False)