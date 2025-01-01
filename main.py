import cv2
from deepface import DeepFace

# Start video capture from the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to capture image")
        break
    
    # Analyze the frame for emotion
    try:
        analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        dominant_emotion = analysis[0]['dominant_emotion']
        
        # Display the dominant emotion on the frame
        cv2.putText(frame, f'Emotion: {dominant_emotion}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    except Exception as e:
        print(f"Error in emotion detection: {e}")
    
    # Display the resulting frame
    cv2.imshow('Webcam - Emotion Detection', frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()