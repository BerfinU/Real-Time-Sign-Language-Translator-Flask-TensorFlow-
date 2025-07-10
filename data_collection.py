import os
import numpy as np
import cv2
import mediapipe as mp
from itertools import product
import time
from collections import deque

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def image_process(image, model): # Görüntüyü işleme fonksiyonu
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image.flags.writeable = False 
    results = model.process(image)
    image.flags.writeable = True 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results 

def draw_landmarks(image, results): 
    if not image.flags.writeable:
        image = image.copy()
    
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
    )
    
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
    )
    
    return image

def keypoint_extraction(results):
    if results.left_hand_landmarks:
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
    else:
        lh = np.zeros(21*3)
    
    if results.right_hand_landmarks:
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(21*3)
    
    return np.concatenate([lh, rh])

class LandmarkStabilizer:
    def __init__(self, window_size=3):
        self.landmark_history = deque(maxlen=window_size)
    
    def add_landmarks(self, keypoints):
        self.landmark_history.append(keypoints)
    
    def get_stabilized_landmarks(self):
        if len(self.landmark_history) == 0:
            return np.zeros(126)
        history_array = np.array(list(self.landmark_history))
        return np.mean(history_array, axis=0)

def quality_check(results):
    score = 0
    if results.left_hand_landmarks:
        landmarks = [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]
        if not all(lm == [0, 0, 0] for lm in landmarks):
            score += 0.5
    
    if results.right_hand_landmarks:
        landmarks = [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]
        if not all(lm == [0, 0, 0] for lm in landmarks):
            score += 0.5
    
    return score

def has_hand_detected(results):
    """El algılandı mı kontrol et"""
    return results.left_hand_landmarks is not None or results.right_hand_landmarks is not None

actions = np.array([
    'dinlediğiniz', 'için'
])

sequences = 100
frames = 40
min_quality = 0.4
fps_target = 15

PATH = os.path.join('data_optimized')

for action, sequence in product(actions, range(sequences)):
    try:
        os.makedirs(os.path.join(PATH, action, str(sequence)))
    except:
        pass

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

print(f"Data collection: {len(actions)} actions, {sequences} sequences, {frames} frames")

with mp.solutions.holistic.Holistic(min_detection_confidence=0.85, min_tracking_confidence=0.85) as holistic:
    
    for action_idx, action in enumerate(actions):
        print(f"\nCollecting: {action} ({action_idx+1}/{len(actions)})")
        
        print(f"'{action}' kelimesi için hazır olduğunuzda SPACE tuşuna basın...")
        while True:
            _, image = cap.read()
            image = image.copy()
            image, results = image_process(image, holistic)
            image = draw_landmarks(image, results)
            
            quality = quality_check(results)
            
            cv2.putText(image, f'Kelime: {action} ({action_idx+1}/{len(actions)})',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(image, f'Hazir oldugunuzda SPACE tuşuna basin',
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(image, f'Quality: {quality:.1f}',
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if quality > 0.4 else (0, 0, 255), 2)
            
            cv2.imshow('Data Collection', image)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 32:  
                break
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit()
        
        for sequence in range(sequences):
            stabilizer = LandmarkStabilizer()
            sequence_data = []
            quality_scores = []
            
            print(f"Sequence {sequence+1}/{sequences} - El algılanıyor...")
            while True:
                _, image = cap.read()
                image = image.copy()
                image, results = image_process(image, holistic)
                image = draw_landmarks(image, results)
                
                quality = quality_check(results)
                
                cv2.putText(image, f'{action} - Seq {sequence+1}/{sequences}',
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(image, 'Elinizi kameraya gösterin...',
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(image, f'Quality: {quality:.1f}',
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if quality > 0.4 else (0, 0, 255), 2)
                
                cv2.imshow('Data Collection', image)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()
                
                if has_hand_detected(results) and quality > min_quality:
                    print(f"El algılandı! Kayıt başlıyor...")
                    break
            
            for frame in range(frames):
                _, image = cap.read()
                image = image.copy()
                image, results = image_process(image, holistic)
                image = draw_landmarks(image, results)
                
                quality = quality_check(results)
                quality_scores.append(quality)
                
                progress = (frame + 1) / frames
                cv2.rectangle(image, (10, 120), (10 + int(400 * progress), 150), (0, 255, 0), -1)
                
                cv2.putText(image, f'{action} - Frame {frame+1}/{frames}',
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(image, f'Seq {sequence+1}/{sequences}',
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(image, f'Quality: {quality:.1f}',
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if quality > 0.4 else (0, 0, 255), 2)
                
                cv2.imshow('Data Collection', image)
                cv2.waitKey(1)
                
                keypoints = keypoint_extraction(results)
                stabilizer.add_landmarks(keypoints)
                stabilized_keypoints = stabilizer.get_stabilized_landmarks()
                sequence_data.append(stabilized_keypoints)
                
                time.sleep(1.0 / fps_target)
            
            if quality_scores and np.mean(quality_scores) > min_quality:
                for frame_idx, keypoints in enumerate(sequence_data):
                    frame_path = os.path.join(PATH, action, str(sequence), str(frame_idx))
                    np.save(frame_path, keypoints)
                print(f"Saved seq {sequence+1}")
            else:
                print(f"Skipped seq {sequence+1} - low quality")
                sequence -= 1
        
        print(f"'{action}' kelimesi tamamlandı! ({action_idx+1}/{len(actions)})")

cap.release()
cv2.destroyAllWindows()
print("Tüm kelimeler tamamlandı!")