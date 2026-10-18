from ultralytics import RTDETR
import cv2
import numpy as np
from lane_detection import detect_ego_lane
from collections import Counter, deque
import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time

# Ordem de preferência quando --model não é informado: primeiro o modelo treinado no
# LISA (classes Green/Red/Yellow), depois o pré-treinado COCO (classe "traffic light").
# Dentro de cada um, a versão exportada mais rápida para CPU. Os modelos INT8 ficam de
# fora: a quantização padrão do RT-DETR zerou as detecções nos testes (ver relatorio.md),
# então só são usados se passados explicitamente em --model.
MODEL_CANDIDATES = [
    'runs/detect/train/weights/best_openvino_model',
    'runs/detect/train/weights/best.onnx',
    'runs/detect/train/weights/best.pt',
    'rtdetr-l_openvino_model',
    'rtdetr-l_openvino_model_640',
    'rtdetr-l.onnx',
]
DEFAULT_MODEL = 'rtdetr-l.pt'

COLOR_CLASSES = {"Green", "Red", "Yellow"}
TRAFFIC_LIGHT_CLASSES = COLOR_CLASSES | {"traffic light"}

HISTORY_LEN = 5   # leituras (inferências) usadas na votação da cor
TRACK_TTL = 30    # inferências sem ver um ID antes de descartar o histórico dele

STATE_COLORS = {"Green": (0, 255, 0), "Red": (0, 0, 255), "Yellow": (0, 255, 255), "Unknown": (200, 200, 200)}


def pick_model(explicit):
    if explicit:
        return explicit
    for path in MODEL_CANDIDATES:
        if os.path.exists(path):
            return path
    print("⚠️ Modelo compilado ou treinado não encontrado.")
    print("⚠️ Usando o modelo pré-treinado padrão 'rtdetr-l.pt' em modo PyTorch (Lento).")
    return DEFAULT_MODEL


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


def smoothed_state(readings):
    """Voto majoritário das últimas leituras. "Unknown" também vota, para que uma cor
    antiga não persista quando as leituras novas são desconhecidas. Empate: vence a
    leitura mais recente entre as empatadas."""
    counts = Counter(readings)
    best = max(counts.values())
    for reading in reversed(readings):
        if counts[reading] == best:
            return reading


def select_target(lights, lane_center_x, frame_width, max_dist, margin):
    """Escolhe o semáforo mais alinhado ao centro da faixa.
    Retorna (track_id ou None, motivo) — None quando não dá para decidir."""
    if lane_center_x is None:
        return None, "faixa nao detectada"
    if not lights:
        return None, "nenhum semaforo"
    ranked = sorted((abs((box[0] + box[2]) / 2 - lane_center_x), tid) for box, tid in lights)
    best_dist, best_id = ranked[0]
    if best_dist > max_dist * frame_width:
        return None, "nenhum semaforo alinhado a faixa"
    if len(ranked) > 1 and ranked[1][0] - best_dist < margin * frame_width:
        return None, f"ambiguo (IDs {best_id} e {ranked[1][1]})"
    return best_id, "ok"


def file_sha256(path):
    """SHA-256 do arquivo de pesos (ou de todos os arquivos, se for uma pasta exportada)."""
    files = [path] if os.path.isfile(path) else sorted(
        os.path.join(root, f) for root, _, names in os.walk(path) for f in names)
    digest = hashlib.sha256()
    for f in files:
        with open(f, 'rb') as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b''):
                digest.update(chunk)
    return digest.hexdigest()


def environment_info():
    versions = {"python": platform.python_version(), "opencv": cv2.__version__}
    for pkg in ["ultralytics", "torch", "torchvision", "numpy", "openvino", "onnxruntime", "lap"]:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = None
    cpu = platform.processor()
    if os.path.exists('/proc/cpuinfo'):
        with open('/proc/cpuinfo') as fh:
            cpu = next((l.split(':', 1)[1].strip() for l in fh if l.startswith('model name')), cpu)
    return {"versions": versions, "platform": platform.platform(), "cpu": cpu, "cpu_count": os.cpu_count()}


def timing_stats(values_s):
    if not values_s:
        return None
    ms = np.array(values_s) * 1000
    return {"n": len(ms), "mean_ms": round(float(ms.mean()), 1), "p50_ms": round(float(np.percentile(ms, 50)), 1),
            "p95_ms": round(float(np.percentile(ms, 95)), 1), "max_ms": round(float(ms.max()), 1)}


def draw_hud(img, lines, scale):
    y = int(30 * scale)
    for text, color in lines:
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, 2)
        cv2.rectangle(img, (5, y - h - 6), (15 + w, y + 6), (0, 0, 0), -1)
        cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, color, 2)
        y += h + int(16 * scale)


def main():
    parser = argparse.ArgumentParser(description='Advanced Traffic Light Detection')
    parser.add_argument('--video', type=str, default='video-teste.mp4', help='Caminho para o vídeo de teste')
    parser.add_argument('--model', type=str, default=None, help='Pesos/modelo exportado (padrão: escolha automática)')
    parser.add_argument('--device', type=str, default='cpu', help="Dispositivo de inferência: 'cpu' ou índice da GPU (ex.: 0)")
    parser.add_argument('--skip-frames', type=int, default=0, help='Quantidade de frames para pular (Frame Skipping)')
    parser.add_argument('--imgsz', type=int, default=640, help='Resolução de inferência (menor = mais rápido)')
    parser.add_argument('--conf', type=float, default=0.3, help='Confiança mínima das detecções')
    parser.add_argument('--lane-roi-bottom', type=float, default=1.0,
                        help='Fração da altura onde termina a busca de faixa (ex.: 0.72 para ignorar o capô/painel)')
    parser.add_argument('--target-max-dist', type=float, default=0.15,
                        help='Distância horizontal máxima semáforo-faixa para ser TARGET (fração da largura)')
    parser.add_argument('--target-margin', type=float, default=0.05,
                        help='Diferença mínima entre o 1º e o 2º candidato para não ser ambíguo (fração da largura)')
    parser.add_argument('--max-frames', type=int, default=0, help='Processa só os N primeiros frames (0 = todos)')
    parser.add_argument('--save', action='store_true', help='Salvar o resultado em vídeo (ver --output)')
    parser.add_argument('--output', type=str, default='output.mp4', help='Arquivo do vídeo salvo com --save')
    parser.add_argument('--log', type=str, default=None,
                        help='Arquivo JSON com versões, pesos, comando e métricas (padrão: resultados/<video>_<data>.json)')
    parser.add_argument('--no-display', action='store_true', help='Não abre janela (útil para medir desempenho)')
    parser.add_argument('--display-scale', type=float, default=0.5, help='Escala de tamanho da janela de exibição (0.5 = metade do tamanho)')
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"❌ Não foi possível abrir o vídeo: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    model_path = pick_model(args.model)
    print(f"🧠 Modelo: {model_path} | dispositivo: {args.device} | imgsz: {args.imgsz}")
    model = RTDETR(model_path)
    names = model.names
    light_class_ids = [cid for cid, name in names.items() if name in TRAFFIC_LIGHT_CLASSES]
    if not light_class_ids:
        sys.exit(f"❌ O modelo não tem classe de semáforo: {names}")

    out_video = None
    output_path = None
    if args.save:
        print(f"🎥 Configurando gravação do vídeo: {width}x{height} a {fps} FPS")
        output_path = args.output
        out_video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        if not out_video.isOpened():
            print("⚠️ Erro: Não foi possível iniciar a gravação em .mp4 (Problema de Codec no OpenCV). Tentando salvar em .avi...")
            output_path = os.path.splitext(args.output)[0] + '.avi'
            out_video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

    state_history = {}   # track_id -> {"readings": deque, "last_seen": índice da inferência}
    frame_count = 0
    inference_count = 0

    lane_center_x, lane_y, lane_status = None, height // 2 + 50, 'none'
    lights = []          # [(box, track_id)] da última inferência
    target_id, target_reason = None, "nenhum semaforo"

    frame_times, inference_times = [], []
    lane_status_counts = Counter()
    target_reason_counts = Counter()
    target_state_counts = Counter()
    lights_per_inference = []
    hud_scale = max(1.0, width / 900)

    try:
        while True:
            success, frame = cap.read()
            if not success: break
            if args.max_frames and frame_count >= args.max_frames: break

            t_frame = time.perf_counter()
            is_inference = frame_count % (args.skip_frames + 1) == 0

            if is_inference:
                lane_center_x, lane_y, lane_status = detect_ego_lane(frame, args.lane_roi_bottom)

                t_inf = time.perf_counter()
                results = model.track(frame, conf=args.conf, persist=True, tracker="bytetrack.yaml", verbose=False,
                                      imgsz=args.imgsz, device=args.device, classes=light_class_ids)
                inference_times.append(time.perf_counter() - t_inf)

                lights.clear()
                if results[0].boxes is not None and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                    class_ids = results[0].boxes.cls.int().cpu().tolist()
                    for box, tid, cid in zip(boxes, track_ids, class_ids):
                        reading = names[cid] if names[cid] in COLOR_CLASSES else get_traffic_light_color(frame, box)
                        # O histórico só recebe leituras novas: os frames pulados reaproveitam
                        # a detecção anterior e não podem votar de novo.
                        entry = state_history.setdefault(tid, {"readings": deque(maxlen=HISTORY_LEN)})
                        entry["readings"].append(reading)
                        entry["last_seen"] = inference_count
                        lights.append((box, tid))

                for tid in [t for t, e in state_history.items() if inference_count - e["last_seen"] > TRACK_TTL]:
                    del state_history[tid]

                target_id, target_reason = select_target(lights, lane_center_x, width,
                                                         args.target_max_dist, args.target_margin)
                lights_per_inference.append(len(lights))
                lane_status_counts[lane_status] += 1
                target_reason_counts[target_reason.split(' (')[0]] += 1
                if target_id is not None:
                    target_state_counts[smoothed_state(state_history[target_id]["readings"])] += 1
                inference_count += 1

            annotated_frame = frame.copy()
            for box, track_id in lights:
                x1, y1, x2, y2 = map(int, box)
                state = smoothed_state(state_history[track_id]["readings"])
                color = STATE_COLORS[state]
                thickness = 4 if track_id == target_id else 2
                label = f"ID:{track_id} {state}"
                if track_id == target_id:
                    label = "TARGET " + label
                    cv2.line(annotated_frame, (int((x1+x2)/2), int(y2)), (lane_center_x, lane_y), (255, 0, 255), 2)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)

            if lane_center_x is not None:
                cv2.line(annotated_frame, (lane_center_x, lane_y), (lane_center_x, height), (0, 255, 255), 2)
                cv2.circle(annotated_frame, (lane_center_x, lane_y), 10, (0, 255, 255), -1)

            frame_times.append(time.perf_counter() - t_frame)

            lane_text = {"ok": "detectada", "partial": "parcial (so uma borda) - centro desconhecido",
                         "none": "nao detectada", "invalid": "inconsistente - descartada"}[lane_status]
            if target_id is not None:
                target_state = smoothed_state(state_history[target_id]["readings"])
                target_text, target_color = f"Alvo: ID {target_id} {target_state}", STATE_COLORS[target_state]
            else:
                target_text, target_color = f"Alvo: indeterminado - {target_reason}", (0, 165, 255)
            recent = frame_times[-30:]
            hud = [
                (f"Faixa: {lane_text}", (255, 255, 255) if lane_status == 'ok' else (0, 165, 255)),
                (target_text, target_color),
                (f"Proc: {len(recent) / sum(recent):.1f} FPS (video {fps:.0f}) | {args.device}, imgsz {args.imgsz}", (255, 255, 255)),
                ("Inferencia neste quadro" if is_inference else "Quadro pulado: deteccao anterior", (180, 180, 180)),
            ]
            draw_hud(annotated_frame, hud, hud_scale)

            if out_video is not None:
                out_video.write(annotated_frame)

            if not args.no_display:
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
            print(f"✅ Vídeo salvo em {output_path}")
        cv2.destroyAllWindows()

    # Tempo real exige que cada quadro seja processado dentro do intervalo entre quadros
    # do vídeo. O tempo medido inclui faixa, inferência+tracking, cor e desenho; não inclui
    # leitura/gravação do vídeo nem a janela de exibição.
    frame_interval_ms = 1000 / fps
    frame_stats = timing_stats(frame_times)
    realtime = None
    if frame_stats:
        realtime = {
            "frame_interval_ms": round(frame_interval_ms, 1),
            "processing_fps": round(len(frame_times) / sum(frame_times), 2),
            "mean_within_interval": frame_stats["mean_ms"] <= frame_interval_ms,
            "p95_within_interval": frame_stats["p95_ms"] <= frame_interval_ms,
        }
        print(f"⏱️ Processamento: {realtime['processing_fps']} FPS (vídeo: {fps:.1f} FPS) | "
              f"quadro médio {frame_stats['mean_ms']} ms, p95 {frame_stats['p95_ms']} ms | "
              f"inferência média {timing_stats(inference_times)['mean_ms']} ms")

    log_path = args.log or os.path.join(
        'resultados', f"{os.path.splitext(os.path.basename(args.video))[0]}_{datetime.datetime.now():%Y%m%d_%H%M%S}.json")
    os.makedirs(os.path.dirname(log_path) or '.', exist_ok=True)
    run_log = {
        "timestamp": datetime.datetime.now().isoformat(timespec='seconds'),
        "command": " ".join(["python"] + sys.argv),
        "args": vars(args),
        "environment": environment_info(),
        "model": {"path": model_path, "sha256": file_sha256(model_path) if os.path.exists(model_path) else None,
                  "classes": names, "traffic_light_classes": [names[c] for c in light_class_ids]},
        "video": {"path": args.video, "width": width, "height": height, "fps": fps},
        "output_video": output_path,
        "frames_processed": frame_count,
        "inferences": inference_count,
        "timing": {"per_frame": frame_stats, "inference_and_tracking": timing_stats(inference_times)},
        "realtime": realtime,
        "traffic_light_detections": {
            "total": int(sum(lights_per_inference)),
            "inferences_with_any": int(sum(n > 0 for n in lights_per_inference)),
        },
        "per_inference_counts": {
            "lane_status": dict(lane_status_counts),
            "target_reason": dict(target_reason_counts),
            "target_state": dict(target_state_counts),
        },
    }
    with open(log_path, 'w') as fh:
        json.dump(run_log, fh, indent=2, ensure_ascii=False, default=str)
    print(f"📝 Registro da execução salvo em {log_path}")


if __name__ == "__main__":
    main()
