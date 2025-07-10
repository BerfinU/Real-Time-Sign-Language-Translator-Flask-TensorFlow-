import eventlet
eventlet.monkey_patch() #Python'un bazı modüllerini daha hızlı çalışması için değiştirir
import os
import base64 #Görüntüleri string olarak alıp çözümlemek için
import numpy as np
import cv2
import mediapipe as mp
import json
from flask import Flask, jsonify, request #Web sunucusu
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from collections import deque #FIFO-kareleri tutmak için
import time


mp_holistic = mp.solutions.holistic
global_holistic = mp_holistic.Holistic(
    min_detection_confidence=0.75, #Güven eşiğini %75 olarak belirledim, yani daha az emin olan noktaları almıyorum.
    min_tracking_confidence=0.75
)

# Flask uygulamasını başlatıyorum.
app = Flask(__name__)
app.config['SECRET_KEY'] = 'sign_language_translator_2024' #gizli anahtar
CORS(app) #Tarayıcıların farklı domain’lerden API’ye erişebilmesi için CORS’u açıyorum.
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

model = None # Modeli ve aksiyonları global değişkenlerde tutuyorum
actions = None  # Aksiyonlar (kelimeler) global değişkenlerde tutuyorum
frame_buffers = {} # Her kullanıcı için frame verilerini tutmak için
single_letters = ['r', 'i', 'n', 'a', 'f', 'c', 'e', 'b', 't']

def load_model_safely():
    global model, actions 
    
    # Model yükleme
    model_files = [f for f in os.listdir('.') if f.lower().endswith('.keras')
                   and ('model' in f.lower() or 'best' in f.lower())]
    if not model_files:
        print("Model dosyası bulunamadı.")
        return False
    
    try:
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        model_file = model_files[0]
        model = load_model(model_file, compile=False)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print(f"Model yüklendi: {model_file}")
    except Exception as e:
        print(f"Model yükleme hatası: {e}")
        return False

    # Actions yükleme
    try:
        if os.path.exists('actions.npy'):
            actions = np.load('actions.npy')
        elif os.path.exists('actions.json'):
            with open('actions.json', 'r', encoding='utf-8') as f:
                actions = np.array(json.load(f))
        else:
            print("Actions dosyası bulunamadı.")
            return False
        print(f"Actions yüklendi: {len(actions)} adet")
    except Exception as e:
        print(f"Actions yükleme hatası: {e}")
        return False

    return True

load_model_safely()

@app.route('/') ## Kullanıcı tarayıcıda ana sayfayı açtığında bu fonksiyonu çalışıyor
def index():
    return open('index.html', encoding='utf-8').read()

def extract_keypoints_from_frame(frame_data_base64):
    try:
        if 'data:image' in frame_data_base64:
            frame_data_base64 = frame_data_base64.split(',', 1)[1]
        img_bytes = base64.b64decode(frame_data_base64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR) 
        if frame is None:
            return None, False

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = global_holistic.process(rgb)
        rgb.flags.writeable = True

        hands_detected = (results.left_hand_landmarks is not None or 
                         results.right_hand_landmarks is not None)

        lh = (np.array([[l.x, l.y, l.z] for l in results.left_hand_landmarks.landmark]).flatten()
              if results.left_hand_landmarks else np.zeros(63))
        rh = (np.array([[l.x, l.y, l.z] for l in results.right_hand_landmarks.landmark]).flatten()
              if results.right_hand_landmarks else np.zeros(63))
        combined_keypoints = np.concatenate([lh, rh])
        
        return combined_keypoints, hands_detected

    except Exception as e:
        print(f"Keypoint extraction error: {e}")
        return None, False

@socketio.on('connect')
def handle_connect():
    session_id = request.sid # Her bağlanan kullanıcıya unique bir session ID'si alıyorum
    frame_buffers[session_id] = {
        'keypoints': [], # El/vücut koordinatları
        'sentence': [], # Oluşturulan cümle
        'last_prediction': "", # Son tahmin
        'current_word': "",
        'word_display_time': 0, # Gösterim süresi
        'last_word_time': 0,
        'word_stability_count': 0,
        'last_clear_time': 0,
        'confidence_threshold': 0.85,  # Güven eşiği %85
        'word_cooldown': 2.5,  
        'required_stability': 2   # Geçerlilik için 2 tekrar
    }
    emit('status', {'msg': 'Bağlantı başarılı!'})

@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    if sid in frame_buffers:
        del frame_buffers[sid]

@socketio.on('start_camera')
def handle_start_camera():
    emit('camera_status', {'msg': 'Kamera başlatıldı!'})

# Gelen görüntü karesini işleyip el verilerini çıkarıyorum
@socketio.on('process_frame')
def handle_process_frame(data):
    session_id = request.sid
    try:
        if session_id not in frame_buffers:
            return
        
        frame_data = data.get('frame', '')
        if not frame_data:
            return

        buffer = frame_buffers[session_id]
        current_time = time.time()

        # Keypoint ve el algılama
        keypoints, hands_detected = extract_keypoints_from_frame(frame_data)
        if keypoints is None:
            return

        # Eller görünmüyorsa temizlik yap
        if not hands_detected:
            if current_time - buffer.get('last_clear_time', 0) > 1.0:
                buffer['current_word'] = ""
                buffer['word_stability_count'] = 0
                buffer['last_clear_time'] = current_time
                emit('current_word_update', {
                    'current_word': "",
                    'stability': 0,
                    'confidence': 0.0,
                    'hands_detected': False
                })
            return

        buffer['keypoints'].append(keypoints)

        # Frame durumu bildir
        emit('frame_status', {
            'frames_collected': len(buffer['keypoints']),
            'frames_needed': 20,
            'hands_detected': hands_detected
        })

        # 20 frame toplandığında tahmin yap
        if len(buffer['keypoints']) == 20:
            # Frame sayısını 35'e genişlet (Streamlit'teki gibi)
            keypoints_expanded = buffer['keypoints'].copy()
            while len(keypoints_expanded) < 35:
                keypoints_expanded.append(buffer['keypoints'][-1])
            keypoints_array = np.array(keypoints_expanded)

            # Tahmin yap
            if model is not None and actions is not None:
                try:
                    prediction = model.predict(keypoints_array[np.newaxis, :, :], verbose=0)
                    max_pred_value = float(np.amax(prediction))
                    max_pred_index = int(np.argmax(prediction))
                    
                    print(f"Güven: {max_pred_value:.3f}, İndeks: {max_pred_index}, El Algılandı: {hands_detected}")

                    # Confidence kontrolü - sadece eller varsa ve yüksek güven varsa
                    if (hands_detected and 
                        max_pred_value > buffer['confidence_threshold'] and 
                        max_pred_index < len(actions)):
                        
                        predicted_action = str(actions[max_pred_index])

                        # Stability kontrolü
                        if buffer['current_word'] == predicted_action:
                            buffer['word_stability_count'] += 1
                        else:
                            buffer['current_word'] = predicted_action
                            buffer['word_stability_count'] = 1
                            buffer['word_display_time'] = current_time

                        # Kararlı tahmin kontrolü
                        if buffer['word_stability_count'] >= buffer['required_stability']:
                            time_passed = current_time - buffer['last_word_time']

                            # Cooldown kontrolü
                            if (buffer['last_prediction'] != predicted_action and 
                                time_passed > buffer['word_cooldown']):

                                # Kelime ekle
                                buffer['sentence'].append(predicted_action)
                                buffer['last_prediction'] = predicted_action
                                buffer['last_word_time'] = current_time
                                buffer['word_stability_count'] = 0

                                word_type = "Harf" if predicted_action in single_letters else "Kelime"
                                print(f"Eklenen {word_type}: {predicted_action} (conf: {max_pred_value:.3f})")
                                
                                emit('prediction_result', {
                                    'type': 'kelime',
                                    'word': predicted_action,
                                    'confidence': max_pred_value,
                                    'word_type': word_type
                                })

                        # Mevcut kelimeyi güncelle
                        emit('current_word_update', {
                            'current_word': buffer['current_word'],
                            'stability': buffer['word_stability_count'],
                            'required_stability': buffer['required_stability'],
                            'confidence': max_pred_value,
                            'hands_detected': True,
                            'is_stable': buffer['word_stability_count'] >= buffer['required_stability']
                        })

                    else:
                        # Düşük güven veya el yok
                        if buffer['current_word'] and current_time - buffer['word_display_time'] > 2.0:
                            buffer['current_word'] = ""
                            buffer['word_stability_count'] = 0
                            emit('current_word_update', {
                                'current_word': "",
                                'stability': 0,
                                'confidence': max_pred_value,
                                'hands_detected': hands_detected
                            })

                except Exception as prediction_error:
                    print(f"Tahmin hatası: {prediction_error}")

            # Keypoints'i temizle
            buffer['keypoints'] = []

    except Exception as e:
        print(f"Frame işleme hatası: {e}")

@socketio.on('manual_add_word')
def handle_manual_add():
    """Manuel kelime ekleme"""
    session_id = request.sid
    if session_id not in frame_buffers:
        return
    
    buffer = frame_buffers[session_id]
    if buffer['current_word'] and buffer['current_word'] != buffer['last_prediction']:
        buffer['sentence'].append(buffer['current_word'])
        buffer['last_prediction'] = buffer['current_word']
        buffer['last_word_time'] = time.time()
        buffer['word_stability_count'] = 0
        
        emit('prediction_result', {
            'type': 'manuel',
            'word': buffer['current_word']
        })
        emit('status', {'msg': f"'{buffer['current_word']}' manuel olarak eklendi!"})

@socketio.on('update_params')
def handle_update_params(data):
    """Parametreleri güncelle"""
    buf = frame_buffers.get(request.sid)
    if not buf:
        return
    
    buf['confidence_threshold'] = float(data.get('confidence_threshold', buf['confidence_threshold']))
    buf['required_stability'] = int(data.get('required_stability', buf['required_stability']))
    buf['word_cooldown'] = float(data.get('word_cooldown', buf['word_cooldown']))
    
    emit('status', {'msg': f"Parametreler güncellendi: Güven={buf['confidence_threshold']:.2f}, Stabilite={buf['required_stability']}, Cooldown={buf['word_cooldown']:.1f}s"})

@socketio.on('reset_sentence')
def handle_reset_sentence():
    sid = request.sid
    if sid in frame_buffers:
        buf = frame_buffers[sid]
        buf['sentence'] = []
        buf['last_prediction'] = ""
        buf['current_word'] = ""
        buf['keypoints'] = []
        buf['word_stability_count'] = 0
        buf['last_word_time'] = 0
        buf['last_clear_time'] = 0
        emit('sentence_reset', {'message': 'Sistem sıfırlandı'})
        emit('current_word_update', {
            'current_word': "",
            'stability': 0,
            'confidence': 0.0,
            'hands_detected': False
        })

@socketio.on('get_sentence')
def handle_get_sentence():
    """Mevcut cümleyi al"""
    buf = frame_buffers.get(request.sid)
    if buf:
        sentence = buf['sentence'].copy()
        if sentence and sentence[0] not in single_letters:
            sentence[0] = sentence[0].capitalize()
        
        emit('sentence_update', {
            'sentence': sentence,
            'sentence_text': ' '.join(sentence) if sentence else ""
        })

@socketio.on('ping')
def handle_ping():
    emit('pong')

if __name__ == '__main__':
    print(f"Adres: http://localhost:5000")
    if model is not None and actions is not None:
        print("Sistem hazır!")
    else:
        print("Model dosyaları eksik!")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)