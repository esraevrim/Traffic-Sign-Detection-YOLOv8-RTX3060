from ultralytics import YOLO

def main():
    # 1. Modeli yükle (Nano model hızlıdır, Small model daha zekidir)
    model = YOLO('yolov8s.pt') 

    # 2. Eğitimi başlat
    # data: senin .yaml dosyanın tam yolu
    # epochs: tur sayısı (kendi kartın olduğu için 100 yapabilirsin)
    # imgsz: resim boyutu
    # device: 0 (RTX 3060'ı kullan demek)
    results = model.train(
        data='C:\Projects\TrafficSignProject\datasets\data.yaml', 
        epochs=100, 
        imgsz=640, 
        batch=16, 
        device=0
    )

if __name__ == '__main__':
    main()