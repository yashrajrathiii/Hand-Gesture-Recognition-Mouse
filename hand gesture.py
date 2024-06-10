import cv2
import mediapipe as mp
import math
import numpy as np
from pynput.mouse import Controller as MouseController
import pyautogui
from pycaw.pycaw import AudioUtilities, ISimpleAudioVolume


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mouse = MouseController()

cap = cv2.VideoCapture(0)
previous_distance = None
volume = 0.5
dragging = False
drag_start = None

def set_system_volume(volume_level):
    sessions = AudioUtilities.GetAllSessions()
    for session in sessions:
        volume = session._ctl.QueryInterface(ISimpleAudioVolume)
        volume.SetMasterVolume(volume_level, None)

def change_volume(volume, direction):
    if direction == "increase":
        new_volume = min(volume + 0.1, 1.0)
        set_system_volume(new_volume)
        return new_volume
    elif direction == "decrease":
        new_volume = max(volume - 0.1, 0.0)
        set_system_volume(new_volume)
        return new_volume
    else:
        return volume

def perform_drag_and_drop(action, x, y):
    if action == "drag":
        pyautogui.moveTo(x, y)
        pyautogui.dragTo(x, y, button='left')
    elif action == "drop":
        pyautogui.moveTo(x, y)
        pyautogui.mouseUp(button='left')

def perform_left_click():
    mouse.click(pyautogui.position(), button='left')

def perform_right_click():
    mouse.click(pyautogui.position(), button='right')

def move_cursor(x, y):
    screen_x, screen_y = pyautogui.position()
    new_x = screen_x + int(x * 10)
    new_y = screen_y + int(y * 10)
    pyautogui.moveTo(new_x, new_y)

def perform_scroll(direction):
    if direction == "up":
        mouse.scroll(0, 1)
    elif direction == "down":
        mouse.scroll(0, -1)

def open_on_screen_keyboard():
    pyautogui.hotkey('win', 'r')
    pyautogui.write("osk")
    pyautogui.press("enter")

with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract the landmarks for calculation
                index = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                middle = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

                # Move cursor based on hand position
                move_cursor(index.x, index.y)

                distance = math.sqrt((thumb.x - index.x) ** 2 + (thumb.y - index.y) ** 2)

                if previous_distance:
                    if distance > previous_distance + 0.02:
                        volume = change_volume(volume, "increase")
                        print("Volume Increased")
                    elif distance < previous_distance - 0.02:
                        volume = change_volume(volume, "decrease")
                        print("Volume Decreased")

                previous_distance = distance

                if thumb.x < index.x:
                    perform_left_click()
                    print("Left Click")
                else:
                    perform_right_click()
                    print("Right Click")

                # Detect scroll gestures
                if thumb.y < index.y and middle.y < index.y:
                    perform_scroll("up")
                    print("Scroll Up")
                elif thumb.y > index.y and middle.y > index.y:
                    perform_scroll("down")
                    print("Scroll Down")
                 # Detect the gesture for opening the on-screen keyboard
                if thumb.x < index.x and thumb.y < index.y:
                    open_on_screen_keyboard()
                    print("On-Screen Keyboard Opened")
                    
        cv2.imshow('Hand Gesture Detection', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    open_on_screen_keyboard()

cap.release()
cv2.destroyAllWindows()
