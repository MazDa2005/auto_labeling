import cv2
import os
from pathlib import Path

def save_video_frames_incremental(video_path, output_folder, fps=1):
    """
    Сохраняет кадры, автоматически находя следующий свободный номер.
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    existing_files = list(output_folder.glob("frame_*.jpg"))
    if existing_files:
        numbers = []
        for f in existing_files:
            try:
                num = int(f.stem.split('_')[1])
                numbers.append(num)
            except (IndexError, ValueError):
                continue
        start_number = max(numbers) + 1 if numbers else 1
    else:
        start_number = 1
    
    print(f"Начинаю с номера: {start_number}")
    
    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps/fps) if fps > 0 else 1
    
    frame_count = 0
    saved_count = start_number
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frame_filename = f"frame_{saved_count:06d}.jpg"
            frame_path = output_folder / frame_filename
            cv2.imwrite(str(frame_path), frame)
            saved_count += 1
            
            if saved_count % 10 == 0:
                print(f"Сохранено {saved_count - start_number} кадров...")
        
        frame_count += 1
    
    cap.release()
    print(f"\nГотово! Добавлено {saved_count - start_number} кадров в папку: {output_folder}")
    print(f"Всего кадров в папке: {len(list(output_folder.glob('frame_*.jpg')))}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Путь к видеофайлу")
    parser.add_argument("--output", required=True, help="Папка для сохранения кадров")
    parser.add_argument("--fps", type=int, default=5, help="Кадров в секунду")
    args = parser.parse_args()

    save_video_frames_incremental(args.video, args.output, args.fps)
