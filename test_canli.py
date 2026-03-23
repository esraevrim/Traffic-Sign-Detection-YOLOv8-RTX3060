from ultralytics import YOLO
import cv2

# 1. Kendi eğittiğin modeli yükle
# 'best.pt' dosyasının tam yolunu buraya yazmalısın
model = YOLO('C:/Projects/TrafficSignProject/runs/detect/train/weights/best.pt')

# 2. Kamerayı aç (0 genelde varsayılan kameradır)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if success:
        # Modelin kameradan gelen görüntüyü tahmin etmesini sağla
        # conf=0.5 demek, %50'den az emin olduğun şeyleri gösterme demek
        results = model.predict(frame, conf=0.5, show=True)

        # 'q' tuşuna basınca döngüden çık
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()