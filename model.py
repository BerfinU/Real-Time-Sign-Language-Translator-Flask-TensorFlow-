import numpy as np
import os
import shutil
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from itertools import product
from sklearn import metrics
import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

def backup_existing_model():
    if os.path.exists('best_model_temp.keras'):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f'best_model_backup_{timestamp}.keras'
        shutil.copy2('best_model_temp.keras', backup_name)
        print(f"Eski model yedeklendi: {backup_name}")
        return backup_name
    return None

PATH = os.path.join('data_optimized')

all_actions = np.array(os.listdir(PATH))
print(f"Veri klasöründe bulunan eylemler: {all_actions}")

def detect_frame_count(action_path):
    sequences = os.listdir(action_path)
    if sequences:
        first_seq_path = os.path.join(action_path, sequences[0])
        if os.path.isdir(first_seq_path):
            frames = [f for f in os.listdir(first_seq_path) if f.endswith('.npy')]
            return len(frames)
    return 0

sample_action = None
for action in all_actions:
    action_path = os.path.join(PATH, action)
    if os.path.isdir(action_path):
        sample_action = action
        break

if sample_action:
    action_path = os.path.join(PATH, sample_action)
    frames = detect_frame_count(action_path)
    print(f"Otomatik algılanan kareler: {frames}")
else:
    print("HATA: Geçerli bir işlem bulunamadı!")
    exit()

available_actions = []
for action in all_actions:
    action_path = os.path.join(PATH, action)
    if os.path.isdir(action_path):
        sequences_in_action = [d for d in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, d))]
        if len(sequences_in_action) > 0:
            first_seq_path = os.path.join(action_path, '0')
            if os.path.isdir(first_seq_path):
                frames_in_seq = len([f for f in os.listdir(first_seq_path) if f.endswith('.npy')])
                if frames_in_seq == frames:
                    available_actions.append(action)
                    print(f"✓ '{action}': {frames_in_seq} frames OK")

actions = np.array(available_actions)
print(f"\nEylemleri kullanma: {actions}")
print(f"Toplam eylemler: {len(actions)}")

label_map = {label:num for num, label in enumerate(actions)}

landmarks, labels = [], []
for action in actions:
    print(f"Yükleniyor: {action}")
    action_path = os.path.join(PATH, action)
    available_sequences = [d for d in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, d))]
    
    for sequence_dir in available_sequences:
        sequence_path = os.path.join(PATH, action, sequence_dir)
        frame_files = [f for f in os.listdir(sequence_path) if f.endswith('.npy')]
        
        if len(frame_files) >= frames:
            temp = []
            valid_sequence = True
            
            for frame in range(frames):
                file_path = os.path.join(sequence_path, str(frame) + '.npy')
                try:
                    npy = np.load(file_path)
                    temp.append(npy)
                except FileNotFoundError:
                    valid_sequence = False
                    break
            
            if valid_sequence and len(temp) == frames:
                landmarks.append(temp)
                labels.append(label_map[action])

print(f"Toplam sekanslar: {len(landmarks)}")

X, Y = np.array(landmarks), to_categorical(labels).astype(int)
print(f"X şekli: {X.shape}, Y şekli: {Y.shape}")

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42, stratify=Y)

backup_file = backup_existing_model()

model = Sequential([
    LSTM(256, return_sequences=True, activation='tanh'),
    BatchNormalization(),
    Dropout(0.3),
    
    LSTM(512, return_sequences=True, activation='tanh'),
    BatchNormalization(), 
    Dropout(0.4),
    
    LSTM(256, return_sequences=False, activation='tanh'),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(64, activation='relu'),
    Dropout(0.2),
    
    Dense(len(actions), activation='softmax')
])

optimizer = Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-7
)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy']
)

print("\nModel Özeti:")
model.summary()

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.7,
        patience=8,
        min_lr=0.0001,
        verbose=1
    ),
    
    ModelCheckpoint(
        filepath='best_model_temp.keras',
        monitor='val_categorical_accuracy',
        save_best_only=True,
        verbose=1
    )
]

print("\nTraining başlıyor...")
print(f"Veri: {len(actions)} eylem, {len(landmarks)} dizi")
print(f"Kare sayısı: {frames}")

# Eğitim başlatılıyor   
history = model.fit( 
    X_train, Y_train,
    epochs=150,
    batch_size=16, 
    validation_data=(X_test, Y_test),
    callbacks=callbacks, 
    verbose=1 
)

try: 
    from tensorflow.keras.models import load_model
    model = load_model('best_model_temp.keras')
    print("En iyi model yüklendi")
except:
    print("Best model yüklenemedi, son modeli kullanıyor")

# Test verileri ile modelin doğruluğunu kontrol et
predictions = np.argmax(model.predict(X_test), axis=1) 
test_labels = np.argmax(Y_test, axis=1) 

accuracy = metrics.accuracy_score(test_labels, predictions)

print(f"\nSONUÇLAR:")
print(f"Test Accuracy: {accuracy:.4f}")

from sklearn.metrics import classification_report, confusion_matrix
print("\n📋 Classification Report:")
print(classification_report(test_labels, predictions, target_names=actions))

print("\nZor Algılanan Kelimeler:")
for i, action in enumerate(actions):
    action_indices = np.where(test_labels == i)[0]
    if len(action_indices) > 0:
        action_predictions = predictions[action_indices]
        correct_predictions = np.sum(action_predictions == i)
        accuracy_for_action = correct_predictions / len(action_indices)
        
        if accuracy_for_action < 0.8:
            print(f"{action}: {accuracy_for_action:.2f} accuracy")
        elif accuracy_for_action < 0.9:
            print(f"{action}: {accuracy_for_action:.2f} accuracy")

model.save('best_model_temp.keras')
print(f"\nModel kaydedildi: best_model_temp.keras")

np.save('actions.npy', actions)
print(f"Actions kaydedildi: {actions}")

if backup_file:
    print(f"\nEski model yedeği: {backup_file}")
    print("Eğer yeni model kötüyse: mv {backup_file} best_model_temp.keras")

print(f"\nTraining tamamlandı!")
print(f"Kare sayısı: {frames}")
print(f"Final accuracy: {accuracy:.4f}")

try:
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history.history['categorical_accuracy'], label='Training')
    plt.plot(history.history['val_categorical_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    cm = confusion_matrix(test_labels, predictions)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig('training_analysis.png', dpi=300, bbox_inches='tight')
    print("Training analizi kaydedildi: training_analysis.png")
    
except ImportError:
    print(" Matplotlib yok, grafik oluşturulamadı") 