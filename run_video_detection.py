from ultralytics import RTDETR
import cv2
import numpy as np
from lane_detection import detect_ego_lane
import os
import argparse

# Prioriza modelos exportados para máxima performance de CPU
if os.path.exists('rtdetr-l_int8_openvino_model'):
    MODEL_PATH = 'rtdetr-l_int8_openvino_model'
    print("🚀 Usando modelo compilado (OpenVINO INT8) para máxima performance na CPU!")
elif os.path.exists('rtdetr-l_openvino_model'):
    MODEL_PATH = 'rtdetr-l_openvino_model'
    print("🚀 Usando modelo compilado (OpenVINO) para máxima performance na CPU!")
elif os.path.exists('rtdetr-l.onnx'):
    MODEL_PATH = 'rtdetr-l.onnx'
    print("🚀 Usando modelo compilado (ONNX) para máxima performance na CPU!")
elif os.path.exists('runs/detect/train/weights/best.pt'):
    MODEL_PATH = 'runs/detect/train/weights/best.pt' 
else:
    print("⚠️ Modelo compilado ou treinado não encontrado.")
    print("⚠️ Usando o modelo pré-treinado padrão 'rtdetr-l.pt' em modo PyTorch (Lento).")
    MODEL_PATH = 'rtdetr-l.pt'

def get_traffic_light_color(frame, box):
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0: return "Unknown"
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lower_red1, upper_red1 = np.array([0, 70, 50]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([160, 70, 50]), np.array([180, 255, 255])
    lower_yellow, upper_yellow = np.array([15, 70, 50]), np.array([35, 255, 255])
    lower_green, upper_green = np.array([40, 70, 50]), np.array([90, 255, 255])
    mask_red = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1), cv2.inRange(hsv, lower_red2, upper_red2))
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    counts = {"Red": cv2.countNonZero(mask_red), "Yellow": cv2.countNonZero(mask_yellow), "Green": cv2.countNonZero(mask_green)}
    max_color = max(counts, key=counts.get)
    if counts[max_color] > 5: return max_color
    return "Unknown"

def main():
    parser = argparse.ArgumentParser(description='Advanced Traffic Light Detection')
    parser.add_argument('--video', type=str, default='video-teste.mp4', help='Caminho para o vídeo de teste')
    parser.add_argument('--skip-frames', type=int, default=0, help='Quantidade de frames para pular (Frame Skipping)')
    parser.add_argument('--imgsz', type=int, default=640, help='Resolução de inferência (menor = mais rápido)')
    parser.add_argument('--save', action='store_true', help='Salvar o resultado em output.mp4')
    parser.add_argument('--display-scale', type=float, default=0.5, help='Escala de tamanho da janela de exibição (0.5 = metade do tamanho)')
    args = parser.parse_args()
    
    model = RTDETR(MODEL_PATH)
    cap = cv2.VideoCapture(args.video)
    
    out_video = None
    if args.save:
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Prevenção contra erro de leitura do OpenCV (as vezes retorna 0 FPS)
        if fps == 0 or fps is None: fps = 30.0
            
        print(f"🎥 Configurando gravação do vídeo: {width}x{height} a {fps} FPS")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))
        
        if not out_video.isOpened():
            print("⚠️ Erro: Não foi possível iniciar a gravação em .mp4 (Problema de Codec no OpenCV). Tentando salvar em .avi...")
            fourcc_avi = cv2.VideoWriter_fourcc(*'XVID')
            out_video = cv2.VideoWriter('output.avi', fourcc_avi, fps, (width, height))
    
    state_history = {}
    HISTORY_LEN = 5
    frame_count = 0
    
    last_lane_center_x = -1
    last_boxes_data = []
    
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break
            
            annotated_frame = frame.copy()
            
            if frame_count % (args.skip_frames + 1) == 0:
                last_lane_center_x, _ = detect_ego_lane(frame)
                results = model.track(frame, conf=0.3, persist=True, tracker="bytetrack.yaml", verbose=False, imgsz=args.imgsz)
                
                last_boxes_data.clear()
                if results[0].boxes is not None and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                    class_ids = results[0].boxes.cls.int().cpu().tolist()
                    names = results[0].names
                    for box, tid, cid in zip(boxes, track_ids, class_ids):
                        class_name = names[cid]
                        if class_name == "traffic light":
                            class_name = get_traffic_light_color(frame, box)
                        last_boxes_data.append((box, tid, class_name))
            
            target_light_id = None
            min_dist = float('inf')
            
            for box, track_id, class_name in last_boxes_data:
                center_x = (box[0] + box[2]) / 2
                if last_lane_center_x != -1:
                    dist_to_lane = abs(center_x - last_lane_center_x)
                    if dist_to_lane < min_dist:
                        min_dist = dist_to_lane
                        target_light_id = track_id
            
            for box, track_id, class_name in last_boxes_data:
                x1, y1, x2, y2 = map(int, box)
                
                if track_id not in state_history:
                    state_history[track_id] = [class_name] if class_name != "Unknown" else ["Unknown"]
                else:
                    if class_name != "Unknown": state_history[track_id].append(class_name)
                    if len(state_history[track_id]) > HISTORY_LEN: state_history[track_id].pop(0)
                
                smoothed_state = max(set(state_history[track_id]), key=state_history[track_id].count)
                color = (0, 255, 0) if smoothed_state == "Green" else ((0, 0, 255) if smoothed_state == "Red" else (0, 255, 255))
                
                thickness = 4 if track_id == target_light_id else 2
                label = f"ID:{track_id} {smoothed_state}"
                
                if track_id == target_light_id and last_lane_center_x != -1:
                    label = f"TARGET " + label
                    cv2.line(annotated_frame, (int((x1+x2)/2), int(y2)), (last_lane_center_x, frame.shape[0]), (255, 0, 255), 2)
                    
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)
            
            if last_lane_center_x != -1:
                cv2.line(annotated_frame, (last_lane_center_x, frame.shape[0]//2 + 50), (last_lane_center_x, frame.shape[0]), (0, 255, 255), 2)
                cv2.circle(annotated_frame, (last_lane_center_x, frame.shape[0]//2 + 50), 10, (0, 255, 255), -1)

            if args.save and out_video is not None:
                out_video.write(annotated_frame)

            display_frame = cv2.resize(annotated_frame, (0, 0), fx=args.display_scale, fy=args.display_scale)
            cv2.imshow("Advanced Traffic Light Detection (Optimized)", display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord("q"): break
            frame_count += 1
            
    except KeyboardInterrupt:
        print("\n⏹️ Execução interrompida pelo usuário (CTRL+C). Salvando vídeo...")
    finally:
        cap.release()
        if out_video is not None:
            out_video.release()
            print("✅ Vídeo salvo com sucesso!")
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()