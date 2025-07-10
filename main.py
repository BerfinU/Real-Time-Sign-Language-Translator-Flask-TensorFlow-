import numpy as np
import os
import string
import mediapipe as mp
import cv2
from my_functions import *
from tensorflow.keras.models import load_model
import language_tool_python
import time

PATH = os.path.join('data_optimized')

try:
    actions = np.load('actions.npy')
    print(f"Yüklenmiş hareketler: {actions}")
except FileNotFoundError:
    print("HATA: actions.npy bulunamadı!")
    exit()

try:
    model = load_model('best_model_temp.keras')
    print("Model başarıyla yüklendi")
except:
    print("HATA: best_model_temp.keras yüklenemedi.")
    exit()

tool = language_tool_python.LanguageToolPublicAPI('en-UK')

sentence = []
keypoints = []
grammar_result = []

last_prediction = ""         
confidence_threshold = 0.85  
current_word = ""           
word_display_time = 0      
word_cooldown = 2.5         
last_word_time = 0
word_stability_count = 0    
required_stability = 2     

single_letters = ['r', 'i', 'n', 'a', 'f', 'c', 'e', 'b', 't']

def adapt_keypoints_to_10_frames(keypoints_35):
    """35 frame'lik veriyi 10 frame'e dönüştür"""
    if len(keypoints_35) < 35:
        return keypoints_35
    
    indices = np.linspace(0, len(keypoints_35)-1, 10, dtype=int)
    return [keypoints_35[i] for i in indices]

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Kameraya erişim sağlanamıyor.")
    exit()

with mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic:
    while cap.isOpened():
        current_time = time.time()
        _, image = cap.read()
        results = image_process(image, holistic)
        image = draw_landmarks(image, results)
        
        keypoints.append(keypoint_extraction(results))

        if len(keypoints) == 20:
            keypoints_expanded = keypoints.copy()
            while len(keypoints_expanded) < 35:
                keypoints_expanded.append(keypoints[-1])
            
            keypoints_array = np.array(keypoints_expanded)
            prediction = model.predict(keypoints_array[np.newaxis, :, :], verbose=0)
            keypoints = [] 

            max_pred_value = np.amax(prediction)
            max_pred_index = np.argmax(prediction)
            
            print(f"Güven: {max_pred_value:.3f}, Öngörülen: {actions[max_pred_index] if max_pred_index < len(actions) else 'BİLİNMİYOR'}")

            if max_pred_value > confidence_threshold:
                if max_pred_index < len(actions):
                    predicted_action = actions[max_pred_index]
                    
                    if current_word == predicted_action:
                        word_stability_count += 1
                    else:
                        current_word = predicted_action
                        word_stability_count = 1
                        word_display_time = current_time
                    
                    if word_stability_count >= required_stability:
                        time_passed = current_time - last_word_time
                        
                        if (last_prediction != predicted_action and time_passed > word_cooldown):

                            sentence.append(predicted_action)
                            last_prediction = predicted_action
                            last_word_time = current_time
                            word_stability_count = 0 
                            
                            word_type = "Harf" if predicted_action in single_letters else "Kelime"
                            print(f"Eklenen {word_type}: {predicted_action} (conf: {max_pred_value:.3f}, stabil: {required_stability}x)")
            else:
                if current_time - word_display_time > 2.0:
                    current_word = ""
                    word_stability_count = 0

        if len(sentence) > 10:
            sentence = sentence[-10:]

        key = cv2.waitKey(1) & 0xFF
        
        if key == 32:
            sentence = []
            last_prediction = ""
            current_word = ""
            grammar_result = []
            keypoints = []
            word_stability_count = 0
            print("🔄 Sıfırla")
            
        elif key == 13:
            if sentence:
                text = ' '.join(sentence)
                grammar_result = tool.correct(text).split()
                print(f"Dilbilgisi: {' '.join(grammar_result)}")
                
        elif key == ord('q'):
            break
        
        elif key == ord('c'):
            if current_word and current_word != last_prediction:
                sentence.append(current_word)
                last_prediction = current_word
                last_word_time = current_time
                word_stability_count = 0
                print(f"Manuel olarak eklendi: {current_word}")
        
        elif key == ord('d'): 
            confidence_threshold = max(0.5, confidence_threshold - 0.05)
            print(f"Eşik değeri düşürüldü: {confidence_threshold:.2f}")
            
        elif key == ord('u'):  
            confidence_threshold = min(0.95, confidence_threshold + 0.05)
            print(f"Eşik değeri yükseltildi:{confidence_threshold:.2f}")

        if sentence:
            if sentence[0] not in single_letters:
                sentence[0] = sentence[0].capitalize()

        cv2.putText(image, f"Frames: {len(keypoints)}/20 | Threshold: {confidence_threshold:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        
        if current_word:
            cv2.putText(image, f"Kararlılık: {word_stability_count}/{required_stability}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
        
        if current_word:
            word_type = "Harf" if current_word in single_letters else "Kelime"
            display_color = (0, 100, 255) if current_word in single_letters else (0, 255, 255)
            
            if word_stability_count >= required_stability:
                display_color = (0, 255, 0)
            
            cv2.putText(image, f"{word_type}: {current_word.upper()}", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, display_color, 3, cv2.LINE_AA)
        
        cooldown_remaining = max(0, word_cooldown - (current_time - last_word_time))
        if cooldown_remaining > 0:
            cv2.putText(image, f"Cooldown: {cooldown_remaining:.1f}s", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 2, cv2.LINE_AA)
        
        cv2.putText(image, "DENGELI MOD - Space:Reset | C:Add | D/U:Threshold | Q:Quit", 
                    (10, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        display_sentence = grammar_result if grammar_result else sentence
        if display_sentence:
            sentence_text = ' '.join(display_sentence)
            textsize = cv2.getTextSize(sentence_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
            text_X_coord = (image.shape[1] - textsize[0]) // 2
            cv2.putText(image, sentence_text, (text_X_coord, 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)

        cv2.imshow('Camera', image)

        if cv2.getWindowProperty('Camera', cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    tool.close()