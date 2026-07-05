 #!/usr/bin/env python3
#!/usr/bin/env python3
"""RDK X5 车机系统 - MIPI 摄像头 + YOLOv5s BPU 推理 + 音量亮度调节"""
import os, sys, time, math, signal, json, cv2, numpy as np, requests, threading, subprocess
from flask import Flask, Response, render_template_string, jsonify, request, send_file, redirect

sys.path.insert(0, '/app/pydev_demo/')

# 集成 car_control（手动泊车）
sys.path.insert(0, '/userdata/rdkstudio/projects/studio-default-project/car_control')
try:
    from car_control import Car as _Car
    _car = _Car(default_speed=60)
    _car_available = True
    print('[CAR] control initialized')
except Exception as _e:
    _car = None
    _car_available = False
    print(f'[CAR] control unavailable: {_e}')

HAVE_MIPI = False; HAVE_DNN = False
try:
    from hobot_vio import libsrcampy as srcampy
    HAVE_MIPI = True
except: pass
try:
    import hbm_runtime
    from hobot_dnn import pyeasy_dnn as dnn
    import utils.preprocess_utils as pre_utils
    import utils.postprocess_utils as post_utils
    import utils.draw_utils as draw
    import utils.common_utils as common
    HAVE_DNN = True
except: pass
HAVE_FACE_MESH = False
try:
    import mediapipe as mp
    HAVE_FACE_MESH = True
except: pass

HTTP_PORT = 8080
# ===== 视距感知 (单目测距 + active-low 蜂鸣) =====
BUZZER_GPIO = 401
HEADLIGHT_GPIOS = [402, 387]  # 车灯 LED 正极:X5 GPIO 402 + 387(负极均接 GND)
BUZZER_BEEP_ON_MS = 200
BUZZER_BEEP_OFF_MS = 300
DANGER_DISTANCE_M = 0.5
REAL_HEIGHT_M = 1.7
DEFAULT_FOCAL_Y_PX = 778.0
VR_FOCAL_FILE = "/root/focal.json"
VR_PERSON_CONF = 0.3

def load_focal_y():
    if os.path.exists(VR_FOCAL_FILE):
        try:
            with open(VR_FOCAL_FILE) as f:
                d = json.load(f)
                return float(d.get("focal_y_px", DEFAULT_FOCAL_Y_PX))
        except Exception:
            pass
    return DEFAULT_FOCAL_Y_PX

vr_focal_y_px = load_focal_y()
vr_buzzer = None
_user_buzzer_hold = False  # 用户按住喇叭按钮时为 True;mipi_capture_loop 看到就不再 off(),让用户控制
# ===== 哨兵模式:开启后只要检测到人就触发蜂鸣 =====
sentinel_mode = False
sentinel_buzzer_on = True  # 哨兵模式蜂鸣器开关:默认开(检测到人时是否响)
fatigue_paused = False  # 哨兵模式下暂停疲劳检测(EAR/MAR/人眼闭合/打哈欠)

class Buzzer:
    def __init__(self, gpio_num):
        self.gpio = gpio_num
        self.base = f"/sys/class/gpio/gpio{gpio_num}"
        self._alarm = threading.Event()
        self._lock = threading.Lock()
        self._init_gpio()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
    def _init_gpio(self):
        if not os.path.exists(self.base):
            with open("/sys/class/gpio/export", "w") as f:
                f.write(str(self.gpio))
            time.sleep(0.05)
        with open(f"{self.base}/direction", "w") as f:
            f.write("out")
        self._set(1)
        print(f"[VR] Buzzer GPIO {self.gpio} init HIGH (silent)")
    def _set(self, val):
        with self._lock:
            with open(f"{self.base}/value", "w") as f:
                f.write(str(val))
    def _loop(self):
        while True:
            try:
                if self._alarm.is_set():
                    self._set(0)
                    time.sleep(BUZZER_BEEP_ON_MS / 1000.0)
                    self._set(1)
                    time.sleep(BUZZER_BEEP_OFF_MS / 1000.0)
                else:
                    self._set(1)
                    time.sleep(0.05)
            except Exception as _e:
                # 防御:不让单次 I/O 错误杀掉守护线程(daemon thread 死了不会被察觉)
                print(f"[VR] _loop transient err: {_e!r}, continuing", flush=True)
                time.sleep(0.2)
    def on(self): self._alarm.set()
    def off(self):
        self._alarm.clear()
        self._set(1)
# ===== 车灯 (GPIO 402 控制 LED 正极,HIGH=亮) =====
class Headlight:
    """多 GPIO 联动车灯:一个开关同时控制所有 LED。"""
    def __init__(self, gpios):
        self.gpios = list(gpios)
        self.bases = [f"/sys/class/gpio/gpio{g}" for g in self.gpios]
        self._lock = threading.Lock()
        self.state = False
        self._init_gpio()
    def _init_gpio(self):
        for g, base in zip(self.gpios, self.bases):
            if not os.path.exists(base):
                try:
                    with open("/sys/class/gpio/export", "w") as f:
                        f.write(str(g))
                    time.sleep(0.05)
                except Exception as e:
                    print(f"[LIGHT] export {g} failed: {e}")
            try:
                with open(f"{base}/direction", "w") as f:
                    f.write("out")
            except Exception as e:
                print(f"[LIGHT] direction {g} failed: {e}")
        self._set(0)
        print(f"[LIGHT] Headlight GPIOs {self.gpios} init LOW (off)")
    def _set(self, val):
        with self._lock:
            for base in self.bases:
                try:
                    with open(f"{base}/value", "w") as f:
                        f.write(str(val))
                except Exception as e:
                    print(f"[LIGHT] set {base}={val} failed: {e}")
    def on(self):
        self._set(1); self.state = True
    def off(self):
        self._set(0); self.state = False
    def toggle(self):
        if self.state: self.off()
        else: self.on()
        return self.state

headlight = None

# ===== 红外避障(左后 / 右后, 直接连 RDK X5) =====
# 红外避障模块是 3.3V active-low 数字输出: 有障碍 = LOW, 无障碍 = HIGH。
# 模块自带 10K 上拉到 VCC(板上 LM393 开漏输出 + 10K 上拉), 可直接接 3.3V GPIO, 无需分压。
IR_LEFT_REAR_GPIO = 399
IR_RIGHT_REAR_GPIO = 400

class InfraredRadar:
    def __init__(self, gpio_num, name):
        self.gpio = gpio_num
        self.name = name
        self.base = f"/sys/class/gpio/gpio{gpio_num}"
        self._init_gpio()
    def _init_gpio(self):
        if not os.path.exists(self.base):
            try:
                with open("/sys/class/gpio/export", "w") as f:
                    f.write(str(self.gpio))
                time.sleep(0.05)
            except Exception as e:
                print(f"[IR/{self.name}] export {self.gpio} failed: {e}")
        try:
            with open(f"{self.base}/direction", "w") as f:
                f.write("in")
            print(f"[IR/{self.name}] GPIO {self.gpio} ready (in, active-low)")
        except Exception as e:
            print(f"[IR/{self.name}] direction set failed: {e}")
    def read_raw(self):
        # 原始电平: 0 = 有障碍(LOW), 1 = 无障碍(HIGH)
        try:
            with open(f"{self.base}/value", "r") as f:
                return int(f.read().strip())
        except Exception:
            return -1
    def read_blocked(self):
        # 反转: 0 = 无障碍, 1 = 有障碍
        raw = self.read_raw()
        return 0 if raw == 0 else 1
    def cleanup(self):
        try:
            with open("/sys/class/gpio/unexport", "w") as f:
                f.write(str(self.gpio))
        except Exception:
            pass

ir_left_rear = InfraredRadar(IR_LEFT_REAR_GPIO, 'left_rear')
ir_right_rear = InfraredRadar(IR_RIGHT_REAR_GPIO, 'right_rear')

# ── Leaflet 离线资源目录 ──
LEAFLET_DIR = '/userdata/leaflet'

# ── 离线地图预载进度跟踪 ──
preload_state = {
    'running': False,
    'total': 0,
    'downloaded': 0,
    'skipped': 0,
    'errors': 0,
    'current_zoom': 0,
    'start_time': 0,
    'message': ''
}

# 石家庄市及周边市中心坐标
SJZ_CENTER = (38.0428, 114.5149)

car_data = {
    'speed': 0, 'battery': 85, 'gear': 'P',
    'signal_left': False, 'signal_right': False,
    'radar': [0, 0, 0, 0], 'warning': False, 'warning_text': '',
    'mode': 'drive', 'temp': 26.5, 'weather': '☀ --', 'fuel': 68, 'odo': 12543,
    # ── 红外避障(0=无障碍, 1=有障碍) ──
    'infrared': {
        'left_rear': 0,
        'right_rear': 0,
        'left_rear_raw': 1,        # 原始电平: 0=有障碍(LOW), 1=无障碍(HIGH)
        'right_rear_raw': 1,
        'left_rear_gpio': IR_LEFT_REAR_GPIO,
        'right_rear_gpio': IR_RIGHT_REAR_GPIO,
    },
    # ── 倒车影像与红外避障联动 ──
    'rear_alarm': False,           # 任一后向红外触发
    'rear_alarm_text': '',         # 前端 revWarn banner 文本: '左后方有障碍' / '右后方有障碍' / '后方两侧有障碍'
}

# 音量/亮度状态
settings = {'volume': 70, 'brightness': 80}

# ===== 疲劳检测 (MediaPipe FaceMesh + EAR/MAR) =====
FATIGUE_EAR_THRESH = 0.21
FATIGUE_MAR_THRESH = 0.60
FATIGUE_BLINK_MIN_MS = 80
FATIGUE_BLINK_MAX_MS = 400
FATIGUE_YAWN_MIN_MS = 800
FATIGUE_DROWSY_MS = 1500
FATIGUE_YAWN_WINDOW_S = 30
FATIGUE_YAWN_TRIGGER_N = 3
FATIGUE_FRAME_SKIP = 1
FATIGUE_DETECT_W = 640
FATIGUE_DETECT_H = 360
# MediaPipe FaceMesh 关键点索引
# 观察者视角:左眼=263..,右眼=33..,嘴外轮廓=61/291/0/17/405/320
_LM_L_EYE = [263, 385, 387, 362, 380, 373]
_LM_R_EYE = [33, 160, 158, 133, 153, 144]
_LM_MOUTH = [61, 291, 0, 17, 405, 320]
# 画框用完整轮廓 (比 EAR/MAR 的 6 点多, 框更大更准)
_LM_L_EYE_BBOX = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
_LM_R_EYE_BBOX = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
_LM_MOUTH_BBOX = [61, 185, 40, 37, 0, 267, 270, 409, 291, 308, 415, 13, 82, 87, 14, 17, 84, 91, 146, 321, 375, 405, 314]

fatigue_state = {
    'face': False, 'ear': 0.0, 'mar': 0.0,
    'eye_closed': False, 'mouth_open': False,
    'eye_boxes': None, 'mouth_box': None,
    # ── 累计/状态机字段 ──
    'blinks': 0,
    'yawns': 0,
    'perclos': 0.0,
    'state': 'normal',                 # normal / drowsy / yawning
    'alarm': False,
    'drowsy_hold_ms': 0,
    # ── 内部边沿检测时间戳(0 表示当前未触发) ──
    '_eye_closed_since_ms': 0,
    '_mouth_open_since_ms': 0,
    '_yawn_window': [],
    '_event_acked': False,  # 用户按了'取消警报'后置位,事件退出前不再置 alarm=True
}
_face_mesh = None
_fm_last_t = 0.0

def init_face_mesh():
    global _face_mesh
    if not HAVE_FACE_MESH or _face_mesh is not None:
        return _face_mesh
    try:
        _face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.3, min_tracking_confidence=0.3)
        print('[FATIGUE] MediaPipe FaceMesh initialized')
    except Exception as e:
        print(f'[FATIGUE] init failed: {e}')
    return _face_mesh

def _lm_xy(lms, idx, w, h):
    p = lms[idx]
    return p.x * w, p.y * h

def _bbox(lms, indices, w, h, pad=0.05):
    xs = [lms[i].x * w for i in indices]
    ys = [lms[i].y * h for i in indices]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    dx = (x2 - x1) * pad
    dy = (y2 - y1) * pad
    return [round(max(0, x1-dx), 1), round(max(0, y1-dy), 1),
            round(min(w, x2+dx), 1), round(min(h, y2+dy), 1)]

def _ear_from_points(eye_idx, lms, w, h):
    pts = [_lm_xy(lms, i, w, h) for i in eye_idx]
    p1, p2, p3, p4, p5, p6 = pts
    num = math.hypot(p2[0]-p6[0], p2[1]-p6[1]) + math.hypot(p3[0]-p5[0], p3[1]-p5[1])
    den = 2.0 * math.hypot(p1[0]-p4[0], p1[1]-p4[1])
    if den <= 0:
        return 0.0
    return num / den

def _perclos_window_append(now_ms, is_closed):
    win = fatigue_state.get('_perclos_win', [])
    win.append((now_ms, 1 if is_closed else 0))
    cutoff = now_ms - 60000
    fatigue_state['_perclos_win'] = [(t, c) for (t, c) in win if t > cutoff]
    if len(fatigue_state['_perclos_win']) > 0:
        fatigue_state['perclos'] = round(
            sum(c for _, c in fatigue_state['_perclos_win']) / len(fatigue_state['_perclos_win']), 3)

def process_fatigue(bgr_small, now_ms):
    global _fm_last_t
    if not HAVE_FACE_MESH or _face_mesh is None:
        return
    _fm_last_t = now_ms / 1000.0
    h, w = bgr_small.shape[:2]
    rgb = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2RGB)
    res = _face_mesh.process(rgb)
    has_face = bool(res.multi_face_landmarks)
    fatigue_state['face'] = has_face
    if not has_face:
        # 失帧:清空瞬时状态但保留 blinks/yawns/perclos 历史
        fatigue_state['ear'] = 0.0
        fatigue_state['mar'] = 0.0
        fatigue_state['eye_closed'] = False
        fatigue_state['mouth_open'] = False
        fatigue_state['eye_boxes'] = None
        fatigue_state['mouth_box'] = None
        fatigue_state['_eye_closed_since_ms'] = 0
        fatigue_state['_mouth_open_since_ms'] = 0
        fatigue_state['drowsy_hold_ms'] = 0
        return
    lms = res.multi_face_landmarks[0].landmark
    if len(lms) < 468:
        return
    ear = (_ear_from_points(_LM_L_EYE, lms, w, h) + _ear_from_points(_LM_R_EYE, lms, w, h)) / 2.0
    mar = _ear_from_points(_LM_MOUTH, lms, w, h)
    fatigue_state['ear'] = round(ear, 3)
    fatigue_state['mar'] = round(mar, 3)
    eye_closed_now = ear < FATIGUE_EAR_THRESH
    mouth_open_now = mar > FATIGUE_MAR_THRESH
    fatigue_state['eye_closed'] = eye_closed_now
    fatigue_state['mouth_open'] = mouth_open_now
    fatigue_state['eye_boxes'] = [
        _bbox(lms, _LM_L_EYE_BBOX, w, h),
        _bbox(lms, _LM_R_EYE_BBOX, w, h),
    ]
    fatigue_state['mouth_box'] = _bbox(lms, _LM_MOUTH_BBOX, w, h)

    # ── 闭眼状态机:边沿累计 + blink 计数 + drowsy 判定 ──
    if eye_closed_now:
        if fatigue_state['_eye_closed_since_ms'] == 0:
            fatigue_state['_eye_closed_since_ms'] = now_ms
            fatigue_state['_event_acked'] = False  # 新事件开始,清 ack
        held = now_ms - fatigue_state['_eye_closed_since_ms']
        fatigue_state['drowsy_hold_ms'] = held
        if held >= FATIGUE_DROWSY_MS and not fatigue_state.get('_event_acked', False):
            fatigue_state['state'] = 'drowsy'
            fatigue_state['alarm'] = True
    else:
        if fatigue_state['_eye_closed_since_ms'] > 0:
            held = now_ms - fatigue_state['_eye_closed_since_ms']
            if FATIGUE_BLINK_MIN_MS <= held <= FATIGUE_BLINK_MAX_MS:
                fatigue_state['blinks'] += 1
            fatigue_state['_eye_closed_since_ms'] = 0
        fatigue_state['drowsy_hold_ms'] = 0

    # ── 哈欠状态机:持续张嘴 FATIGUE_YAWN_MIN_MS 算一次 yawn ──
    if mouth_open_now:
        if fatigue_state['_mouth_open_since_ms'] == 0:
            fatigue_state['_mouth_open_since_ms'] = now_ms
        yawn_held = now_ms - fatigue_state['_mouth_open_since_ms']
        if yawn_held >= FATIGUE_YAWN_MIN_MS and not fatigue_state.get('_event_acked', False):
            fatigue_state['state'] = 'yawning'
    else:
        if fatigue_state['_mouth_open_since_ms'] > 0:
            held = now_ms - fatigue_state['_mouth_open_since_ms']
            fatigue_state['_mouth_open_since_ms'] = 0
            if held >= FATIGUE_YAWN_MIN_MS:
                fatigue_state['yawns'] += 1
                fatigue_state['_yawn_window'].append(now_ms)
        if fatigue_state['state'] == 'yawning' and not eye_closed_now:
            fatigue_state['state'] = 'normal'

    # ── 30s 哈欠窗口:>= N 次触发报警 ──
    yawn_cutoff = now_ms - FATIGUE_YAWN_WINDOW_S * 1000
    fatigue_state['_yawn_window'] = [t for t in fatigue_state['_yawn_window'] if t > yawn_cutoff]
    if len(fatigue_state['_yawn_window']) >= FATIGUE_YAWN_TRIGGER_N and not fatigue_state.get('_event_acked', False):
        fatigue_state['state'] = 'yawning'
        fatigue_state['alarm'] = True

    # ── 恢复正常时清 alarm + _event_acked ──
    if (not eye_closed_now) and (not mouth_open_now) and \
       fatigue_state['drowsy_hold_ms'] == 0 and \
       len(fatigue_state['_yawn_window']) < FATIGUE_YAWN_TRIGGER_N and \
       fatigue_state['state'] != 'drowsy':
        fatigue_state['alarm'] = False
        if fatigue_state['state'] == 'yawning':
            fatigue_state['state'] = 'normal'
        fatigue_state['_event_acked'] = False  # 事件真正退出,清 ack,允许下次事件重新报警

    # PERCLOS 滑窗(60s 闭眼占比)
    _perclos_window_append(now_ms, eye_closed_now)

app = Flask(__name__)
running = True
demo_mode = False

# MIPI camera + YOLO shared state
latest_frame = None
latest_detections = {'count': 0, 'objects': [], 'fps': 0}
frame_lock = threading.Lock()
det_lock = threading.Lock()
cam = None

TILE_DIR = '/userdata/map_tiles'

# ── YOLO 模型配置 ──
YOLO_MODEL_PATH = '/opt/hobot/model/x5/basic/yolov5s_672x672_nv12.bin'
LABEL_PATH = '/app/pydev_demo/09_web_display_camera_sample/coco_classes.names'
STRIDES = np.array([8, 16, 32], dtype=np.int32)
ANCHORS = np.array([
    [10, 13], [16, 30], [33, 23],
    [30, 61], [62, 45], [59, 119],
    [116, 90], [156, 198], [373, 326]
], dtype=np.float32).reshape(3, 3, 2)

# ── COCO 80 类 → 车机业务化三分类:人 / 车辆 / 障碍物 ──
CAT_COLORS = {"人": (0, 255, 0), "车辆": (255, 0, 0), "障碍物": (0, 0, 255)}

def classify_coco(cid):
    if cid == 0:
        return "人"
    if cid in (1, 2, 3, 5, 7):
        return "车辆"
    return "障碍物"



def load_model():
    if not HAVE_DNN:
        return None, None, None
    try:
        model = hbm_runtime.HB_HBMRuntime(YOLO_MODEL_PATH)
        mn = model.model_names[0]
        in_shapes = model.input_shapes[mn]
        input_name = model.input_names[mn][0]
        input_H = in_shapes[input_name][2]
        input_W = in_shapes[input_name][3]
        print(f"[YOLO] Model loaded: {mn} {input_W}x{input_H}")
        return model, input_W, input_H
    except Exception as e:
        print(f"[YOLO] Load failed: {e}")
        return None, None, None


def mipi_capture_loop(model, input_W, input_H):
    global latest_frame, latest_detections, cam, running
    import utils.postprocess_utils as post_utils
    global vr_buzzer
    if vr_buzzer is None:
        vr_buzzer = Buzzer(BUZZER_GPIO)
    try:
        cam = srcampy.Camera()
        cam.open_cam(0, -1, 30, [input_W, 1920, 1920], [input_H, 1072, 1080])
        print(f"[MIPI] Camera opened: {input_W}x{input_H} infer + 1920x1072 display")
        encoder = srcampy.Encoder()
        encoder.encode(0, 3, 1920, 1072)
        classes = []
        if os.path.exists(LABEL_PATH):
            with open(LABEL_PATH) as f:
                classes = [l.strip() for l in f.readlines()]
        if not classes:
            classes = [str(i) for i in range(80)]
        mn = model.model_names[0]
        in_name = model.input_names[mn][0]
        out_names = model.output_names[mn]
        out_quants = model.output_quants[mn]
        frame_count = 0
        fps_timer = time.time()
        resize_type = 0
        DISP_W, DISP_H = 1920, 1072
        _fm = init_face_mesh()
        _fm_frame = 0
        while running:
            try:
                t_frame = time.time()
                raw_infer = cam.get_img(2, input_W, input_H)
                if raw_infer is None or len(raw_infer) == 0:
                    time.sleep(0.01); continue
                nv12 = np.frombuffer(raw_infer, dtype=np.uint8).reshape(1, input_H * 3 // 2, input_W, 1)
                input_tensor = {mn: {in_name: nv12}}
                outputs_raw = model.run(input_tensor)
                outputs = outputs_raw[mn]
                fp32 = post_utils.dequantize_outputs(outputs, out_quants)
                pred = post_utils.decode_outputs(out_names, fp32, STRIDES, ANCHORS, 80)
                xyxy, scores, cls_ids = post_utils.filter_predictions(pred, 0.25)
                keep = post_utils.NMS(xyxy, scores, cls_ids, 0.45)
                xyxy = post_utils.scale_coords_back(xyxy[keep], DISP_W, DISP_H, input_W, input_H, resize_type)
                img_disp = cam.get_img(1, DISP_W, DISP_H)
                if img_disp is None or len(img_disp) == 0:
                    time.sleep(0.01); continue
                objects_list = []
                if len(xyxy) > 0:
                    for i in range(len(xyxy)):
                        cls_id = int(cls_ids[i])
                        label = classes[cls_id] if cls_id < len(classes) else str(cls_id)
                        bbox_h = float(xyxy[i][3]) - float(xyxy[i][1])
                        dist_m = None
                        vr_alarm_obj = False
                        if cls_id == 0 and bbox_h > 5:
                            dist_m = (REAL_HEIGHT_M * load_focal_y()) / bbox_h
                            vr_alarm_obj = bool(dist_m < DANGER_DISTANCE_M)
                        objects_list.append({
                            'id': cls_id,
                            'cls': classify_coco(cls_id),
                            'label': label,
                            'score': round(float(scores[i]), 2),
                            'bbox': [float(x) for x in xyxy[i]],
                            'distance_m': round(dist_m, 2) if dist_m is not None else None,
                            'vr_alarm': vr_alarm_obj,
                        })
                # ── OpenCV 画彩色框(替换 encoder) ──
                nv12_arr = np.frombuffer(img_disp, dtype=np.uint8).reshape(-1, DISP_W)
                bgr = cv2.cvtColor(nv12_arr, cv2.COLOR_YUV2BGR_NV12)
                dw, dh = 640, 360
                bgr_s = cv2.resize(bgr, (dw, dh))
                sx, sy = dw / DISP_W, dh / DISP_H
                if len(xyxy) > 0:
                    for j in range(len(xyxy)):
                        x1, y1, x2, y2 = map(int, xyxy[j])
                        cat = classify_coco(int(cls_ids[j]))
                        col = CAT_COLORS.get(cat, (255, 255, 255))
                        eng = classes[int(cls_ids[j])] if int(cls_ids[j]) < len(classes) else str(int(cls_ids[j]))
                        cls_j = int(cls_ids[j])
                        if cls_j == 0 and j < len(objects_list) and objects_list[j].get('distance_m') is not None:
                            dist = objects_list[j]['distance_m']
                            label_str = f"person {dist}m"
                            if objects_list[j].get('vr_alarm'):
                                col = (0, 0, 255); thick = 3; font_scale = 0.6
                            else:
                                col = (0, 255, 255); thick = 3; font_scale = 0.6
                        else:
                            label_str = f'{eng} {float(scores[j]):.2f}'
                            thick = 2; font_scale = 0.45
                        cv2.rectangle(bgr_s, (int(x1*sx), int(y1*sy)), (int(x2*sx), int(y2*sy)), col, thick)
                        cv2.putText(bgr_s, label_str, (int(x1*sx), max(int(y1*sy)-6, 14)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, col, thick)
                _, jpeg_buf = cv2.imencode('.jpg', bgr_s, [cv2.IMWRITE_JPEG_QUALITY, 70])
                jpeg_data = jpeg_buf.tobytes()
                frame_count += 1
                if frame_count % 30 == 0:
                    elapsed = time.time() - fps_timer
                    fps = 30 / elapsed if elapsed > 0 else 0
                    fps_timer = time.time(); frame_count = 0
                    infer_ms = int((time.time() - t_frame) * 1000)
                    print(f"[MIPI+YOLO] {len(objects_list)} obj, {fps:.1f} fps, {infer_ms}ms/frame")
                with frame_lock:
                    latest_frame = jpeg_data
                # 红外避障读取(每帧, 轻量级 sysfs I/O, 依赖 LM393 硬件迟滞去抖)
                car_data['infrared']['left_rear_raw'] = ir_left_rear.read_raw()
                car_data['infrared']['right_rear_raw'] = ir_right_rear.read_raw()
                car_data['infrared']['left_rear'] = 0 if car_data['infrared']['left_rear_raw'] == 0 else 1
                car_data['infrared']['right_rear'] = 0 if car_data['infrared']['right_rear_raw'] == 0 else 1
                # 倒车影像与红外避障联动:触发时更新报警文本(前端 revWarn banner 自动渲染)
                ir_l = car_data['infrared']['left_rear']
                ir_r = car_data['infrared']['right_rear']
                if not ir_l and not ir_r:
                    car_data['rear_alarm'] = True
                    car_data['rear_alarm_text'] = '后方两侧有障碍'
                elif not ir_l:
                    car_data['rear_alarm'] = True
                    car_data['rear_alarm_text'] = '左后方有障碍'
                elif not ir_r:
                    car_data['rear_alarm'] = True
                    car_data['rear_alarm_text'] = '右后方有障碍'
                else:
                    car_data['rear_alarm'] = False
                    car_data['rear_alarm_text'] = ''
                # 疲劳检测:每 N 帧跑一次,降 CPU 占用
                _fm_frame += 1
                if HAVE_FACE_MESH and _fm is not None and not fatigue_paused and (_fm_frame % FATIGUE_FRAME_SKIP == 0):
                    try:
                        small = cv2.resize(bgr_s, (FATIGUE_DETECT_W, FATIGUE_DETECT_H))
                        process_fatigue(small, int(time.time() * 1000))
                    except Exception as _e:
                        if _fm_frame % 90 == 0:
                            print(f'[FATIGUE] err: {_e}', flush=True)
                person_detected = any(o['cls'] == '\u4eba' and o['score'] > 0.3 for o in objects_list)
                persons = [o for o in objects_list if o['cls'] == '\u4eba' and o['score'] > VR_PERSON_CONF]
                min_dist = None
                vr_alarm_global = False
                for o in persons:
                    d = o.get('distance_m')
                    if d is not None:
                        if min_dist is None or d < min_dist:
                            min_dist = d
                        if d < DANGER_DISTANCE_M:
                            vr_alarm_global = True
                sentinel_alarm = bool(sentinel_mode and person_detected)
                fatigue_alarm_global = bool(fatigue_state.get('alarm', False)) and not fatigue_paused
                rear_alarm_global = bool(car_data.get('rear_alarm', False))
                if vr_buzzer is not None:
                    if vr_alarm_global or (sentinel_alarm and sentinel_buzzer_on) or fatigue_alarm_global or rear_alarm_global:
                        vr_buzzer.on()
                    elif not _user_buzzer_hold:
                        vr_buzzer.off()
                with det_lock:
                    latest_detections = {
                        'count': len(objects_list),
                        'objects': objects_list,
                        'fps': round(fps, 1) if frame_count == 0 else 0,
                        'person_alarm': person_detected,
                        'person_count': sum(1 for d in objects_list if d.get('cls') == '\u4eba'),
                        'person_distance_m': round(min_dist, 2) if min_dist is not None else None,
                        'vr_alarm': vr_alarm_global,
                        'sentinel_mode': sentinel_mode,
                        'sentinel_alarm': sentinel_alarm,
                        'focal_y_px': load_focal_y(),
                        'car_alarm': any(o['cls'] == '\u8f66\u8f86' for o in objects_list),
                        'obstacle_alarm': any(o['cls'] == '\u969c\u788d\u7269' for o in objects_list),
                        'fatigue': {
                            'face': fatigue_state.get('face', False),
                            'ear': fatigue_state.get('ear', 0.0),
                            'mar': fatigue_state.get('mar', 0.0),
                            'eye_closed': fatigue_state.get('eye_closed', False),
                            'mouth_open': fatigue_state.get('mouth_open', False),
                            'eye_boxes': fatigue_state.get('eye_boxes'),
                            'mouth_box': fatigue_state.get('mouth_box'),
                            'blinks': fatigue_state.get('blinks', 0),
                            'yawns': fatigue_state.get('yawns', 0),
                            'perclos': fatigue_state.get('perclos', 0.0),
                            'state': fatigue_state.get('state', 'normal'),
                            'alarm': fatigue_state.get('alarm', False),
                            'drowsy_hold_ms': fatigue_state.get('drowsy_hold_ms', 0),
                        },
                    }
            except Exception as e:
                print(f"[Loop] {e}"); time.sleep(0.1)
    except Exception as e:
        print(f"[MIPI] Init error: {e}")

HMI_HTML = '''<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,user-scalable=no">
<title>RDK X5 车载系统</title>
<style>
:root{
  --bg-primary:#0b0b14;
  --bg-secondary:#12121f;
  --bg-card:rgba(255,255,255,0.04);
  --bg-card-hover:rgba(255,255,255,0.08);
  --accent:#00d4aa;
  --accent-glow:rgba(0,212,170,0.3);
  --text-primary:#f0f0f5;
  --text-secondary:rgba(255,255,255,0.55);
  --text-tertiary:rgba(255,255,255,0.3);
  --glass-border:rgba(255,255,255,0.06);
  --glass-bg:rgba(255,255,255,0.03);
  --radius-sm:10px;
  --radius-md:16px;
  --radius-lg:24px;
}
*{margin:0;padding:0;box-sizing:border-box;-webkit-tap-highlight-color:transparent}
body{background:var(--bg-primary);color:var(--text-primary);font-family:-apple-system,'PingFang SC','SF Pro Display','Segoe UI',sans-serif;overflow:hidden;height:100vh;width:100vw;user-select:none;-webkit-font-smoothing:antialiased}

/* ── Dynamic Ambient Background ── */
@keyframes ambientShift{
  0%{background-position:0% 50%}
  50%{background-position:100% 50%}
  100%{background-position:0% 50%}
}
.bg-ambient{position:fixed;top:0;left:0;right:0;bottom:0;z-index:-1;
  background:radial-gradient(ellipse 80% 60% at 20% 30%,rgba(0,212,170,0.04) 0%,transparent 60%),
             radial-gradient(ellipse 60% 50% at 80% 70%,rgba(0,100,255,0.03) 0%,transparent 50%),
             var(--bg-primary);
  background-size:200% 200%;
  animation:ambientShift 12s ease-in-out infinite;
}

/* ── Cursor-following glow overlay ── */
.cursor-glow{position:fixed;top:0;left:0;right:0;bottom:0;z-index:0;
  pointer-events:none;
  background:radial-gradient(400px circle at var(--mx,50%) var(--my,50%),
    rgba(0,212,170,0.06) 0%,transparent 60%);
  transition:background .08s;
}

/* ── Ripple Effect ── */
.ripple{position:absolute;border-radius:50%;pointer-events:none;
  background:rgba(0,212,170,0.25);
  transform:scale(0);
  animation:rippleAnim .55s ease-out forwards;
}
@keyframes rippleAnim{
  to{transform:scale(4);opacity:0}
}
.ripple-container{position:relative;overflow:hidden;}

/* ── Card glow on hover/tap ── */
.m-btn::after{content:'';position:absolute;top:-1px;left:-1px;right:-1px;bottom:-1px;
  border-radius:inherit;
  background:linear-gradient(135deg,rgba(0,212,170,0.15),transparent 40%,transparent 60%,rgba(0,212,170,0.05));
  opacity:0;
  transition:opacity .25s ease;
  pointer-events:none;
  z-index:-1;
}
.m-btn:hover::after{opacity:1}
.m-btn:active::after{opacity:1.2}

/* ── Pulse glow on highlighted cards ── */
@keyframes cardPulse{
  0%,100%{box-shadow:0 0 8px rgba(0,212,170,0.06),inset 0 0 8px rgba(0,212,170,0.02)}
  50%{box-shadow:0 0 20px rgba(0,212,170,0.12),inset 0 0 12px rgba(0,212,170,0.04)}
}
.m-btn.hl{animation:cardPulse 3s ease-in-out infinite;}

/* ── Back button glow ── */
.back-btn:hover{box-shadow:0 0 14px rgba(0,212,170,0.12);border-color:rgba(0,212,170,0.15);}
.back-btn:active{box-shadow:0 0 20px rgba(0,212,170,0.2);border-color:rgba(0,212,170,0.25);}
/* ── Vinyl Record ── */
@keyframes spin{from{transform:rotate(0deg)}to{transform:rotate(360deg)}}
.vinyl-wrapper{width:200px;height:200px;border-radius:50%;position:relative;
  margin-bottom:20px;flex-shrink:0;
  box-shadow:0 8px 40px rgba(0,0,0,.4),0 0 60px rgba(0,212,170,.06);
  transition:box-shadow .4s ease}
.vinyl-wrapper.playing{box-shadow:0 8px 40px rgba(0,0,0,.4),0 0 80px rgba(0,212,170,.12)}
.vinyl-disc{width:100%;height:100%;border-radius:50%;
  background:radial-gradient(circle at 50% 50%,
    #2a2a2a 0%,#222 12%,#1c1c1c 25%,#181818 40%,#141414 55%,#0f0f0f 70%,#0a0a0a 100%),
    repeating-radial-gradient(circle at 50% 50%,
      transparent 0,transparent 3px,rgba(255,255,255,0.008) 3px,rgba(255,255,255,0.008) 4px);
  position:relative;animation:spin 2s linear infinite;animation-play-state:paused;
  border:1px solid rgba(255,255,255,.06)}
.vinyl-wrapper.playing .vinyl-disc{animation-play-state:running}
.vinyl-label{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
  width:70px;height:70px;border-radius:50%;
  background:linear-gradient(135deg,rgba(0,212,170,.2),rgba(0,100,255,.12));
  border:2px solid rgba(0,212,170,.2);
  display:flex;align-items:center;justify-content:center;
  flex-direction:column;gap:2px;pointer-events:none}
.vinyl-label-ring{width:64px;height:64px;border-radius:50%;
  border:1px dashed rgba(0,212,170,.15);position:absolute}
.vinyl-hole{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
  width:8px;height:8px;border-radius:50%;background:rgba(0,0,0,.6);
  border:1px solid rgba(255,255,255,.08);z-index:2}
.vinyl-arc{position:absolute;top:4px;left:4px;right:4px;bottom:4px;border-radius:50%;
  border:1px solid rgba(255,255,255,.02);pointer-events:none}
.vinyl-shine{position:absolute;top:0;left:0;right:0;bottom:0;border-radius:50%;
  background:linear-gradient(135deg,rgba(255,255,255,.04) 0%,transparent 40%,transparent 60%,rgba(255,255,255,.02) 100%);
  pointer-events:none}
/* ── MP3 两栏布局:左 黑胶 + 控件,右 列表 ── */
.mp3-layout{display:flex;flex-direction:row;gap:16px;flex:1;min-height:0;padding:12px 14px;align-items:stretch}
.mp3-left{display:flex;flex-direction:column;align-items:center;gap:10px;flex:0 0 240px;min-width:0;padding:6px 4px}
.mp3-left .vinyl-wrapper{width:170px;height:170px;margin-bottom:4px}
.mp3-now{width:100%;text-align:center;margin-top:2px}
.mp3-now .title{font-size:15px;font-weight:700;color:var(--text-primary);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:220px;margin:0 auto 2px}
.mp3-now .artist{font-size:11px;color:var(--text-tertiary);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:220px;margin:0 auto}
.mp3-progress{width:100%;max-width:220px;margin-top:4px}
.mp3-controls{display:flex;align-items:center;gap:18px;margin-top:8px}
.mp3-controls .ctrl{width:40px;height:40px;display:flex;align-items:center;justify-content:center;border-radius:50%;background:var(--glass-bg);border:1px solid var(--glass-border);cursor:pointer;transition:all .15s;font-size:18px}
.mp3-controls .ctrl.play{width:54px;height:54px;background:var(--accent);border:none;font-size:24px;box-shadow:0 0 22px var(--accent-glow)}
.mp3-volume{display:flex;align-items:center;gap:8px;width:100%;max-width:220px;margin-top:6px}
.mp3-volume input[type=range]{flex:1;-webkit-appearance:none;appearance:none;height:3px;border-radius:3px;background:rgba(255,255,255,.08);outline:none}
.mp3-right{flex:1;display:flex;flex-direction:column;border-left:1px solid var(--glass-border);padding-left:14px;min-height:0}
.mp3-pl-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;flex:0 0 auto}
.mp3-pl-head .lbl{font-size:11px;color:var(--text-tertiary);letter-spacing:2px}
.mp3-pl-head .cnt{font-size:10px;color:var(--text-tertiary)}
.mp3-pl-list{flex:1;overflow-y:auto;min-height:0;padding-right:4px}


/* ── Slider thumb glow on drag ── */
.setting-slider input[type=range]::-webkit-slider-thumb:active{
  box-shadow:0 0 20px var(--accent-glow);
}
.page{display:none;flex-direction:column;height:100vh;width:100vw;opacity:0;transform:translateY(12px);transition:opacity .3s ease,transform .35s cubic-bezier(.22,1,.36,1)}
.page.active{display:flex;opacity:1;transform:translateY(0)}
#pageMenu{position:relative}
.fullscreen-btn{position:absolute;right:16px;bottom:16px;width:44px;height:44px;border-radius:50%;background:rgba(0,212,170,.15);border:1px solid rgba(0,212,170,.4);color:#00d4aa;font-size:20px;cursor:pointer;display:flex;align-items:center;justify-content:center;z-index:50;transition:all .2s;backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px);box-shadow:0 4px 12px rgba(0,0,0,.3)}
.fullscreen-btn:hover{background:rgba(0,212,170,.3);border-color:rgba(0,212,170,.7);transform:scale(1.08)}
.fullscreen-btn:active{transform:scale(.95)}
.status-bar{display:flex;justify-content:space-between;align-items:center;padding:0 20px;height:36px;font-size:11px;color:var(--text-secondary);background:rgba(0,0,0,.3);backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);letter-spacing:.3px;flex-shrink:0}
.status-left{display:flex;gap:12px;align-items:center}
.status-right{display:flex;gap:10px;align-items:center}
.status-icon{font-size:13px;opacity:.8}
.status-signal{display:flex;gap:2px;align-items:flex-end}
.status-signal span{display:block;width:3px;border-radius:1px;background:var(--text-secondary)}
.status-signal span:nth-child(1){height:6px}
.status-signal span:nth-child(2){height:9px}
.status-signal span:nth-child(3){height:12px}
.status-signal span:nth-child(4){height:15px;background:var(--accent)}
.status-battery{display:flex;align-items:center;gap:4px}
.battery-icon{width:20px;height:10px;border:1px solid var(--text-secondary);border-radius:2px;padding:1px;position:relative}
.battery-icon::after{content:'';position:absolute;right:-3px;top:2px;width:2px;height:4px;background:var(--text-secondary);border-radius:0 1px 1px 0}
.battery-fill{height:100%;width:85%;background:var(--accent);border-radius:1px}
.clock-hero{display:flex;flex-direction:column;align-items:center;justify-content:center;padding:28px 0 10px;position:relative;background:linear-gradient(180deg,rgba(0,212,170,0.04) 0%,transparent 60%)}
.clock-display{font-size:76px;font-weight:700;color:#fff;letter-spacing:2px;line-height:1;font-variant-numeric:tabular-nums;text-shadow:0 0 60px rgba(0,212,170,.12)}
.clock-display .seconds{font-size:28px;font-weight:400;color:var(--text-secondary);vertical-align:super;margin-left:4px}
.clock-date{font-size:12px;color:var(--text-secondary);margin-top:8px;letter-spacing:4px;font-weight:400}
.clock-meta{display:flex;gap:16px;margin-top:6px;align-items:center}
.clock-meta-item{display:flex;align-items:center;gap:4px;font-size:12px;color:var(--text-tertiary)}
.gear-chip{padding:2px 10px;border-radius:6px;font-size:11px;font-weight:600;letter-spacing:1px;backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px)}
.gear-D{background:rgba(68,138,255,.15);color:#448aff;border:1px solid rgba(68,138,255,.2)}
.gear-P{background:rgba(0,212,170,.12);color:var(--accent);border:1px solid rgba(0,212,170,.2)}
.gear-R{background:rgba(255,82,82,.15);color:#ff5252;border:1px solid rgba(255,82,82,.2)}
.gear-N{background:rgba(255,193,7,.12);color:#ffc107;border:1px solid rgba(255,193,7,.2)}
.menu-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;padding:10px 16px 6px;flex:1;align-content:start;max-width:500px;margin:0 auto;width:100%}
.m-btn{display:flex;flex-direction:row;align-items:center;gap:12px;background:var(--glass-bg);border:1px solid var(--glass-border);border-radius:var(--radius-md);padding:14px 16px;cursor:pointer;transition:all .2s cubic-bezier(.22,1,.36,1);backdrop-filter:blur(24px);-webkit-backdrop-filter:blur(24px);position:relative;overflow:hidden;min-height:64px}
.m-btn::before{content:'';position:absolute;top:0;left:0;right:0;bottom:0;background:linear-gradient(135deg,rgba(0,212,170,0.03) 0%,transparent 50%);opacity:0;transition:opacity .3s ease;border-radius:inherit;pointer-events:none}
.m-btn:hover::before{opacity:1}
.m-btn:active{transform:scale(.97);background:var(--bg-card-hover);border-color:rgba(0,212,170,.15)}
.m-btn .mi{font-size:24px;width:40px;height:40px;display:flex;align-items:center;justify-content:center;border-radius:12px;background:rgba(0,212,170,.08);flex-shrink:0}
.m-btn .mi-cont{display:flex;flex-direction:column;flex:1;min-width:0}
.m-btn .ml{font-size:14px;font-weight:600;color:var(--text-primary);line-height:1.3}
.m-btn .mb{font-size:11px;color:var(--text-tertiary);margin-top:2px}
.m-btn.hl{border-color:rgba(0,212,170,.12);background:rgba(0,212,170,.04)}
.m-btn.hl .mi{background:rgba(0,212,170,.14);color:var(--accent)}
.m-btn.dg{border-color:rgba(255,82,82,.1);background:rgba(255,82,82,.03)}
.m-btn.dg .mi{background:rgba(255,82,82,.1);color:#ff5252}
.info-bar{display:flex;justify-content:space-around;padding:6px 16px 12px;font-size:11px;color:var(--text-tertiary);background:transparent;flex-shrink:0;margin:0 16px;border-top:1px solid rgba(255,255,255,.04)}
.info-bar span{transition:color .2s,opacity .2s}
.info-bar span{display:flex;align-items:center;gap:4px}
.func-header{display:flex;align-items:center;padding:10px 16px;background:transparent;gap:10px;flex-shrink:0}
.back-btn{font-size:22px;cursor:pointer;width:36px;height:36px;display:flex;align-items:center;justify-content:center;border-radius:10px;background:var(--glass-bg);border:1px solid var(--glass-border);transition:all .15s;backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px)}
.back-btn:active{transform:scale(.9);background:var(--bg-card-hover)}
.func-title{font-size:15px;font-weight:600;flex:1}.func-status{font-size:10px;color:var(--text-tertiary);padding:4px 10px;border-radius:20px;background:var(--glass-bg);border:1px solid var(--glass-border)}
.func-body{flex:1;position:relative;overflow:hidden}
.sentinel-bar{position:absolute;top:12px;left:12px;right:12px;z-index:996;display:flex;justify-content:space-between;align-items:center;padding:10px 14px;background:rgba(0,0,0,.5);backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);border-radius:12px;border:1px solid var(--glass-border)}
.sentinel-toggle{display:flex;align-items:center;gap:6px;border:none;padding:7px 14px;border-radius:9px;font-size:12px;font-weight:600;font-family:inherit;cursor:pointer;transition:all .15s;background:rgba(0,212,170,.18);color:var(--accent);border:1px solid rgba(0,212,170,.35)}
.sentinel-toggle.off{background:rgba(255,255,255,.06);color:var(--text-tertiary);border-color:rgba(255,255,255,.1)}
.sentinel-toggle:active{transform:scale(.97)}
.func-body img{width:100%;height:100%;object-fit:contain;background:#000}
.radar-overlay{position:absolute;bottom:0;left:0;right:0;padding:10px 20px 15px;background:linear-gradient(transparent,rgba(0,0,0,.7))}
.radar-row{display:flex;gap:8px;justify-content:space-between;max-width:320px;margin:0 auto}
.radar-blk{width:64px;border-radius:6px 6px 0 0;text-align:center;font-size:11px;padding-top:6px;color:#fff;transition:all .2s;font-weight:600}
.radar-blk.s{background:linear-gradient(0deg,#1b5e20,#2e7d32);height:22px;opacity:.7}
.radar-blk.c{background:linear-gradient(0deg,#e65100,#ef6c00);height:32px}
.radar-blk.d{background:linear-gradient(0deg,#b71c1c,#d32f2f);height:42px}
.warning-banner{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);padding:16px 32px;border-radius:var(--radius-md);font-size:18px;font-weight:600;display:none;text-align:center;z-index:100;backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px)}
.warning-banner.show{display:block;animation:warnPulse 1s ease-in-out infinite}
.warning-banner.crit{background:rgba(180,0,0,.75);color:#fff;border:1px solid rgba(255,23,68,.4)}

/* ── Red flash overlay ── */
@keyframes redFlash{
  0%,100%{opacity:0.08}
  30%{opacity:0.55}
  60%{opacity:0.15}
  85%{opacity:0.65}
}
@keyframes redBorderPulse{
  0%,100%{border-color:rgba(255,0,0,0.3);box-shadow:inset 0 0 80px rgba(255,0,0,0.15)}
  25%{border-color:rgba(255,0,0,0.9);box-shadow:inset 0 0 180px rgba(255,0,0,0.6)}
  50%{border-color:rgba(255,0,0,0.4);box-shadow:inset 0 0 100px rgba(255,0,0,0.2)}
  75%{border-color:rgba(255,0,0,1);box-shadow:inset 0 0 240px rgba(255,0,0,0.7)}
}
@keyframes alarmTextPulse{
  0%,100%{opacity:0.3;transform:scale(1)}
  50%{opacity:1;transform:scale(1.08)}
}
.person-alarm-overlay{position:fixed;top:0;left:0;right:0;bottom:0;z-index:999;pointer-events:none;
  display:none;
  background:radial-gradient(ellipse at 50% 50%,transparent 40%,rgba(255,0,0,0.12) 70%,rgba(255,0,0,0.25) 100%);
  animation:redFlash 0.8s ease-in-out infinite}
.person-alarm-overlay.active{display:block}
.person-alarm-border{position:fixed;top:0;left:0;right:0;bottom:0;z-index:1000;pointer-events:none;
  display:none;
  border:18px solid rgba(255,0,0,0.5);
  box-shadow:inset 0 0 200px rgba(255,0,0,0.3),0 0 200px rgba(255,0,0,0.15);
  animation:redBorderPulse 1s ease-in-out infinite}
.person-alarm-border.active{display:block}
.person-alarm-text{position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);z-index:1001;pointer-events:none;
  display:none;
  font-size:28px;font-weight:900;color:#ff1744;
  text-shadow:0 0 40px rgba(255,0,0,0.8),0 0 80px rgba(255,0,0,0.4);
  letter-spacing:6px;white-space:nowrap;
  animation:alarmTextPulse 0.8s ease-in-out infinite}
.person-alarm-text.active{display:block}

/* exit button (车机最右边中间) */
@keyframes warnPulse{0%,100%{transform:translate(-50%,-50%) scale(1);opacity:1}50%{transform:translate(-50%,-50%) scale(1.04);opacity:.9}}
.tag{display:inline-block;padding:3px 10px;border-radius:20px;font-size:10px;font-weight:600;background:rgba(0,0,0,.6);color:var(--accent);border:1px solid rgba(0,212,170,.3);margin:2px;backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px)}

/* ── 右上角警告图标 ── */
.top-right-warn{position:absolute;top:8px;right:50px;padding:6px 14px;border-radius:8px;font-size:14px;font-weight:700;z-index:100;backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px);pointer-events:none}
.warn-car{background:rgba(0,100,255,.85);color:#fff;border:1px solid rgba(100,180,255,.5)}
.warn-obstacle{background:rgba(255,180,0,.85);color:#000;border:1px solid rgba(255,220,80,.5)}

/* ── 红色箭头地标图标 ── */
.red-arrow-marker{background:transparent !important;border:none !important}
.red-arrow{position:relative;width:30px;height:42px;display:flex;flex-direction:column;align-items:center}
.red-arrow .arrow-body{width:0;height:0;border-left:12px solid transparent;border-right:12px solid transparent;border-top:28px solid #e53935;filter:drop-shadow(0 2px 4px rgba(0,0,0,.4))}
.red-arrow .arrow-circle{width:14px;height:14px;border-radius:50%;background:#fff;border:3px solid #e53935;position:absolute;bottom:12px;left:50%;transform:translateX(-50%);box-shadow:0 1px 3px rgba(0,0,0,.3)}
.card-top-view{display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%;background:transparent;gap:12px}
.settings-list{padding:16px;display:flex;flex-direction:column;gap:10px;max-width:420px;margin:0 auto;width:100%}
.setting-item{display:flex;justify-content:space-between;align-items:center;padding:14px 16px;background:var(--glass-bg);border:1px solid var(--glass-border);border-radius:var(--radius-md);font-size:14px;backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px)}
.setting-item-row{display:flex;flex-direction:column;padding:14px 16px;background:var(--glass-bg);border:1px solid var(--glass-border);border-radius:var(--radius-md);gap:10px;backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px)}
.setting-label{display:flex;justify-content:space-between;align-items:center;font-size:14px}
.setting-slider{display:flex;align-items:center;gap:12px}
.setting-slider input[type=range]{-webkit-appearance:none;appearance:none;flex:1;height:4px;border-radius:4px;background:rgba(255,255,255,.08);outline:none;transition:all .15s}
.setting-slider input[type=range]::-webkit-slider-track{height:4px;border-radius:4px}
.setting-slider input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;appearance:none;width:18px;height:18px;border-radius:50%;background:var(--accent);cursor:pointer;border:none;box-shadow:0 0 12px var(--accent-glow);transition:all .15s}
.setting-slider input[type=range]::-webkit-slider-thumb:active{transform:scale(1.15)}
.setting-slider input[type=range]::-moz-range-thumb{width:18px;height:18px;border-radius:50%;background:var(--accent);cursor:pointer;border:none}
.setting-val{font-size:16px;font-weight:700;color:var(--accent);min-width:36px;text-align:right;font-family:monospace;font-variant-numeric:tabular-nums}
.scroll{overflow-y:auto;flex:1;-webkit-overflow-scrolling:touch;scrollbar-width:thin;scrollbar-color:rgba(255,255,255,.06) transparent}
.scroll::-webkit-scrollbar{width:3px}
.scroll::-webkit-scrollbar-thumb{background:rgba(255,255,255,.08);border-radius:2px}
.sensor-dot{width:10px;height:10px;border-radius:50%;display:inline-block;margin:0 4px}
.sensor-dot.g{background:#4caf50;box-shadow:0 0 10px rgba(76,175,80,.5)}
.sensor-dot.o{background:#ff9800;box-shadow:0 0 10px rgba(255,152,0,.5)}
.sensor-dot.r{background:#ff1744;box-shadow:0 0 10px rgba(255,23,68,.5)}
/* ── 疲劳检测面板 ── */
.fat-stat{background:rgba(11,11,20,.78);border:1px solid rgba(0,212,170,.18);border-radius:10px;padding:8px 10px;backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px)}
.fat-stat-l{font-size:10px;color:var(--text-tertiary);letter-spacing:1px;margin-bottom:2px}
.fat-stat-v{font-size:18px;font-weight:700;color:#00d4aa;font-family:monospace;line-height:1.1}
.fat-stat-v.warn{color:#ffcc00}
.fat-stat-v.crit{color:#ff5252;animation:warnPulse 1s ease-in-out infinite}
#map{background:#0b0b14 !important}
.leaflet-control-zoom a{background:var(--glass-bg) !important;color:var(--text-primary) !important;border-color:var(--glass-border) !important}
@keyframes fadeSlideIn{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:translateY(0)}}
@media(min-width:600px){.menu-grid{grid-template-columns:1fr 1fr 1fr;grid-template-rows:repeat(4,auto);gap:12px;padding:12px 20px 8px}.menu-grid>.m-btn:nth-child(1){grid-column:1;grid-row:1}.menu-grid>.m-btn:nth-child(2){grid-column:1;grid-row:2}.menu-grid>.m-btn:nth-child(3){grid-column:1;grid-row:3}.menu-grid>.m-btn:nth-child(4){grid-column:2;grid-row:1}.menu-grid>.m-btn:nth-child(5){grid-column:2;grid-row:2}.menu-grid>.m-btn:nth-child(6){grid-column:2;grid-row:3}.menu-grid>.m-btn:nth-child(7){grid-column:2;grid-row:4}.menu-grid>.m-btn:nth-child(8){grid-column:3;grid-row:1}.menu-grid>.m-btn:nth-child(9){grid-column:3;grid-row:2}.menu-grid>.m-btn:nth-child(10){grid-column:3;grid-row:3}}
</style>  <link rel="stylesheet" href="/leaflet/leaflet.css"/>
</head><body>
<div class="bg-ambient"></div>
<div class="cursor-glow" id="cursorGlow"></div>

<div class="page active" id="pageMenu">
<div class="status-bar">
<div class="status-left"><span class="status-icon">🚗</span><span id="mspeed">0 km/h</span><span id="mgear"><span class="gear-chip gear-P">P</span></span></div>
<div class="status-right">
<span class="status-signal"><span></span><span></span><span></span><span></span></span>
<span class="status-battery"><span class="battery-icon"><span class="battery-fill"></span></span><span id="mbat">85%</span></span>
<span id="clockDots" style="font-variant-numeric:tabular-nums;color:var(--text-primary)">00:00</span>
</div>
</div>
<div class="clock-hero"><div class="clock-display"><span id="clockD">00</span><span class="seconds" id="clockSec">00</span></div><div class="clock-date" id="clockDate">----</div><div class="clock-meta"><span class="clock-meta-item" id="clockTemp">🌡 26.5°C</span><span class="clock-meta-item" id="clockWeather">☀ --</span><span class="clock-meta-item" id="clockSpeed">🚗 0 km/h</span></div></div>
<div class="menu-grid">
<div class="m-btn hl ripple-container" onclick="nav('reverse')"><span class="mi">📹</span><div class="mi-cont"><span class="ml">倒车影像</span><span class="mb">MIPI+雷达</span></div></div>
<div class="m-btn hl ripple-container" onclick="nav('front')"><span class="mi">🛣️</span><div class="mi-cont"><span class="ml">前视检测</span><span class="mb">YOLOv5s BPU</span></div></div>
<div class="m-btn hl ripple-container" id="headlightBtn" onclick="toggleHeadlight()"><span class="mi">💡</span><div class="mi-cont"><span class="ml">车灯开关</span><span class="mb" id="headlightStatus">前+后 关闭</span></div></div>
<div class="m-btn hl ripple-container" onclick="nav('parking')"><span class="mi">🎵</span><div class="mi-cont"><span class="ml">车载MP3</span><span class="mb">本地音乐</span></div></div>
<div class="m-btn ripple-container" onclick="nav('navi')"><span class="mi">🗺️</span><div class="mi-cont"><span class="ml">导航</span><span class="mb">GPS路线</span></div></div>
<div class="m-btn ripple-container" onclick="nav('settings')"><span class="mi">⚙️</span><div class="mi-cont"><span class="ml">系统设置</span><span class="mb">音量/亮度</span></div></div>
<div class="m-btn hl ripple-container" onclick="nav('fatigue')"><span class="mi">😴</span><div class="mi-cont"><span class="ml">疲劳检测</span><span class="mb">眼睛+嘴巴</span></div></div>
<div class="m-btn hl ripple-container" onclick="nav('sentinel')"><span class="mi">🛡️</span><div class="mi-cont"><span class="ml">哨兵模式</span><span class="mb">人员侦测</span></div></div><div class="m-btn ripple-container" onclick="nav('info')"><span class="mi">ℹ️</span><div class="mi-cont"><span class="ml">关于本机</span><span class="mb">RDK X5</span></div></div><div class="m-btn ripple-container" onclick="nav('mpark')"><span class="mi">🅿</span><div class="mi-cont"><span class="ml">手动泊车</span><span class="mb">辅助泊车</span></div></div></div>
<div class="info-bar"><span id="iFuel">⛽ --</span><span id="iOdo">📊 --</span><span id="iObj">🎯 0</span><span id="iFps">📶 0</span></div>
<button id="fullscreenBtn" class="fullscreen-btn" onclick="toggleFullscreen()" title="进入全屏">⤢</button>
</div>

<div class="page" id="pageReverse"><div class="func-header"><span class="back-btn" onclick="gb()">‹</span><span class="func-title">📹 倒车影像</span><span class="func-status" id="revInfo">MIPI</span></div>
 <div class="func-body"><img id="liveCamReverse" alt="MIPI Cam"><div id="revTags" style="position:absolute;top:8px;left:50%;transform:translateX(-50%);display:flex;gap:4px;flex-wrap:wrap"></div><div class="warning-banner" id="revWarn"></div>
<div class="radar-overlay"><div class="radar-row"><div class="radar-blk s" id="r0">—</div><div class="radar-blk s" id="r3">—</div></div></div></div></div>

<div class="page" id="pageFront"><div class="func-header"><span class="back-btn" onclick="gb()">‹</span><span class="func-title">🛣️ 前视检测 · MIPI</span><span class="func-status">YOLOv5s BPU</span></div>
<div class="func-body"><img id="liveCamFront" alt="MIPI Cam"><div id="frontTags" style="position:absolute;top:8px;left:50%;transform:translateX(-50%);display:flex;gap:4px;flex-wrap:wrap"></div><div class="warning-banner" id="frontWarn"></div><div class="person-alarm-overlay" id="personAlarm"></div><div class="person-alarm-border" id="personBorder"></div><div class="person-alarm-text" id="personText">⚠️ 行人预警</div></div></div></div>

<div class="page" id="pageFatigue"><div class="func-header"><span class="back-btn" onclick="gb()">‹</span><span class="func-title">😴 疲劳检测</span><span class="func-status" id="fatSt">监控中</span></div>
<div class="func-body" style="position:relative;background:#000">
<img id="liveCamFatigue" alt="MIPI Cam" style="width:100%;height:100%;object-fit:contain;background:#000">
<canvas id="fatCanvas" style="position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:5"></canvas>
<div class="warning-banner" id="fatWarn"></div>
<div style="position:absolute;top:12px;left:12px;display:grid;grid-template-columns:1fr 1fr;gap:8px;z-index:10;width:280px">
<div class="fat-stat"><div class="fat-stat-l">EAR 眼睛</div><div class="fat-stat-v" id="fatEar">--</div></div>
<div class="fat-stat"><div class="fat-stat-l">MAR 嘴巴</div><div class="fat-stat-v" id="fatMar">--</div></div>
<div class="fat-stat" style="grid-column:1/3"><div class="fat-stat-l">眼睛状态 (阈值 <span id="earThreshT" style="color:#00d4aa;font-family:monospace">0.21</span>)</div><div class="fat-stat-v" id="fatEyeSt">--</div></div>
<div class="fat-stat" style="grid-column:1/3"><div class="fat-stat-l">嘴巴状态 (阈值 > <span id="marThreshT" style="color:#00d4aa;font-family:monospace">0.60</span>)</div><div class="fat-stat-v" id="fatMouthSt">--</div></div>
</div>
<div style="position:absolute;bottom:0;left:0;right:0;text-align:center;font-size:11px;color:#cfd6df;z-index:50;background:rgba(0,0,0,.7);padding:8px 10px;border-top:1px solid rgba(255,255,255,.08)">
  <div style="display:flex;align-items:center;gap:14px;justify-content:center;flex-wrap:wrap">
    <label style="display:flex;align-items:center;gap:6px">EAR 阈值 <input type="range" id="earThresh" min="0.10" max="0.40" step="0.005" value="0.21" style="width:90px"><span id="earThreshV" style="font-family:monospace;color:#00d4aa;min-width:42px">0.210</span></label>
    <label style="display:flex;align-items:center;gap:6px">MAR 阈值 <input type="range" id="marThresh" min="0.30" max="0.90" step="0.01" value="0.60" style="width:90px"><span id="marThreshV" style="font-family:monospace;color:#00d4aa;min-width:42px">0.60</span></label>
    <button id="btnCancelFatigueAlarm" onclick="cancelFatigueAlarm()" style="padding:6px 14px;border:1px solid rgba(0,212,170,.4);background:rgba(0,212,170,.15);color:#00d4aa;border-radius:8px;cursor:pointer;font-size:12px;font-weight:600;letter-spacing:1px;transition:all .2s;margin-left:6px" onmouseover="this.style.background='rgba(0,212,170,.3)'" onmouseout="this.style.background='rgba(0,212,170,.15)'">🔕 取消警报</button>
  </div>
  <div style="margin-top:4px;color:#888">基于 MediaPipe FaceMesh 468 关键点 · EAR 眼睛纵横比 · MAR 嘴巴纵横比</div>
</div>

</div>
<div id="carWarn" class="top-right-warn warn-car" style="display:none">⚠ 车辆</div>
<div id="obsWarn" class="top-right-warn warn-obstacle" style="display:none">⚠ 障碍物</div>

<div class="page" id="pageRadar"><div class="func-header"><span class="back-btn" onclick="gb()">‹</span><span class="func-title">📡 超声波雷达</span><span class="func-status">4 传感器</span></div>
<div class="func-body scroll"><div class="card-top-view"><div style="font-size:60px">🚗</div><div style="font-size:28px;font-weight:700;color:#00e5ff;font-family:monospace" id="radarDist">--</div><div style="font-size:12px;color:#888">最近障碍物</div><div style="display:flex;gap:20px;margin-top:15px"><div><div class="sensor-dot g" id="s0"></div><div style="font-size:10px;color:#888;margin-top:4px">左</div></div><div><div class="sensor-dot g" id="s1"></div><div style="font-size:10px;color:#888;margin-top:4px">中左</div></div><div><div class="sensor-dot g" id="s2"></div><div style="font-size:10px;color:#888;margin-top:4px">中右</div></div><div><div class="sensor-dot g" id="s3"></div><div style="font-size:10px;color:#888;margin-top:4px">右</div></div></div></div></div></div></div>

<div class="page" id="pageNavi"><div class="func-header"><span class="back-btn" onclick="gb()">‹</span><span class="func-title">🗺️ 导航 - 石家庄</span><span class="func-status" id="navSt">加载中</span></div>
<div class="func-body" style="display:flex;flex-direction:column"><div style="flex:1;position:relative;height:100%" id="mapBox">

<div id="map" style="width:100%;height:100%"></div>
<div id="mapLoad" style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);color:#888;font-size:13px;text-align:center;z-index:1000">🗺️ 加载地图中...<br><span style="font-size:11px">已离线缓存 Leaflet 库</span></div>
<!-- search bar -->
<div id="searchBar" style="position:absolute;top:10px;left:50%;transform:translateX(-50%);z-index:1001;background:rgba(11,11,20,.85);border:1px solid rgba(255,255,255,.12);border-radius:10px;padding:5px 10px;display:flex;gap:6px;backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);width:80%;max-width:340px;box-shadow:0 4px 24px rgba(0,0,0,.5)">
<input id="searchInput" type="text" placeholder="🔍 搜索地点..." style="flex:1;background:transparent;border:none;color:#f0f0f5;font-size:13px;outline:none;padding:4px 0"/>
<button onclick="doSearch()" style="background:rgba(0,212,170,.15);border:none;border-radius:6px;padding:4px 12px;color:#00d4aa;font-weight:600;cursor:pointer;font-size:12px">搜索</button>
</div>
</div>
<script src="/leaflet/leaflet.js"></script><script>
// 导航地图初始化（带重试保护）
/* ── 红色箭头地标图标(纯 CSS 绘制) ── */
var redArrowIcon = L.divIcon({
  className: 'red-arrow-marker',
  html: '<div class="red-arrow"><div class="arrow-body"></div><div class="arrow-circle"></div></div>',
  iconSize: [30, 42],
  iconAnchor: [15, 42],
  popupAnchor: [0, -42]
});

var mapInitialized = false, mapInstance = null, mapRetries = 0

function initMap() {
  try {
    if (mapInitialized) {
      if (mapInstance) setTimeout(function(){ mapInstance.invalidateSize() }, 150)
      document.getElementById('mapLoad').style.display='none'
      return
    }
    if (typeof L === 'undefined') {
      mapRetries++;
      if (mapRetries < 30) { setTimeout(initMap, 500); return }
      document.getElementById('mapLoad').innerHTML = '🗺️ 地图库加载失败<br><span style="font-size:11px">请检查网络或刷新页面</span>';
      return
    }
    var mapEl = document.getElementById('map');
    if (!mapEl || mapEl.offsetWidth === 0 || mapEl.offsetHeight === 0) {
      mapRetries++;
      if (mapRetries < 20) { setTimeout(initMap, 400); return }
      document.getElementById('mapLoad').innerHTML = '🗺️ 地图容器不可见<br><span style="font-size:11px">请切换到导航页后重试</span>';
      return
    }
    
    L.Icon.Default.imagePath = '/leaflet'
    mapInstance = L.map('map', {
      zoomControl: true,
      attributionControl: false
    }).setView([38.04, 114.48], 13)
    
    L.tileLayer('/tiles/{z}/{x}/{y}.png', {
      maxZoom: 18, minZoom: 8, detectRetina: false
    }).addTo(mapInstance)
    
    L.control.scale({position:'bottomleft', metric:true, imperial:false}).addTo(mapInstance)

    // 石家庄市区兴趣点
    var grayDotIcon = L.divIcon({className:'gray-dot-marker', html:'<div class="gray-dot"></div>', iconSize:[10,10], iconAnchor:[5,5]}); var poiIcon = grayDotIcon
    var pois = [
  {name:'石家庄站',lat:38.008,lng:114.485},
  {name:'石家庄北站',lat:38.075,lng:114.460},
  {name:'正定国际机场',lat:38.280,lng:114.697},
  {name:'河北博物院',lat:38.044,lng:114.517},
  {name:'长安公园',lat:38.056,lng:114.510},
  {name:'裕彤体育中心',lat:38.033,lng:114.533},
  {name:'万象城',lat:38.035,lng:114.465},
  {name:'万达广场',lat:38.019,lng:114.533}
]
pois.forEach(function(p){
  L.marker([p.lat,p.lng],{icon:poiIcon}).addTo(mapInstance).bindPopup('<b>'+p.name+'</b>')
})
L.circle([38.0428,114.5149],{radius:3000,color:'#00d4aa',fillColor:'#00d4aa',fillOpacity:0.05,weight:1,dashArray:'5,10'}).addTo(mapInstance)

document.getElementById('mapLoad').style.display='none'
document.getElementById('navSt').textContent='📍 石家庄'
// 延迟刷新确保 flex 布局尺寸正确
setTimeout(function(){ if(mapInstance) mapInstance.invalidateSize() }, 350)
mapInstance.on('moveend',function(){var c=mapInstance.getCenter();document.getElementById('navCoords').textContent='📍 '+c.lat.toFixed(4)+','+c.lng.toFixed(4)})
mapInitialized = true

  } catch(e) {
    mapRetries++;
    console.error('[Map] init error:', e)
    if (mapRetries < 10) {
      setTimeout(initMap, 800)
    } else {
      document.getElementById('mapLoad').innerHTML = '🗺️ 地图加载出错<br><span style="font-size:11px">' + (e.message || '未知错误') + '</span>'
    }
  }
}

// 瓦片缓存状态
function refreshTileStats(){
  fetch('/api/tile_stats').then(function(r){return r.json()}).then(function(s){
    document.getElementById('tileStats').textContent='🗂️ '+s.total_tiles+' 张 / '+s.size_mb+'MB'
    var zs='';for(var z in s.by_zoom){zs+='z'+z+':'+s.by_zoom[z]+' '}
    document.getElementById('tileDetail').textContent=zs
  }).catch(function(){})
}
setTimeout(refreshTileStats,1000)

/* -- loc search: Photon geocode + Leaflet flyTo -- */
var searchMarker=null;
function doSearch(){
  var q=document.getElementById('searchInput').value.trim();
  if(!q)return;
  fetch('/api/geocode?q='+encodeURIComponent(q))
    .then(function(r){return r.json()})
    .then(function(data){
      if(data.results&&data.results.length>0){
        var r=data.results[0];
        if(mapInstance)mapInstance.flyTo([r.lat,r.lon],15);
        if(searchMarker)mapInstance.removeLayer(searchMarker);
        searchMarker=L.marker([r.lat,r.lon],{icon:redArrowIcon}).addTo(mapInstance)
          .bindPopup('<b>'+r.name+'</b><br>'+r.display).openPopup();
        document.getElementById('navSt').textContent='📍 '+r.name
      }else{alert('未找到该地点')}
    }).catch(function(){alert('搜索失败，请检查网络')})
}
document.addEventListener('keydown',function(e){
  if(e.key==='Enter'&&document.activeElement===document.getElementById('searchInput')){doSearch()}
})

// 离线预载带进度
var preloadTimer=null
function startPreload(zoom,scope){
  var btn=document.getElementById('preloadBtn')
  btn.textContent='⏳ 下载中...'
  btn.style.opacity='0.5'
  btn.style.pointerEvents='none'
  document.getElementById('navSt').textContent='📥 准备下载...'
  document.getElementById('preloadProg').style.display='block'
  document.getElementById('progText').textContent='准备中...'
  document.getElementById('progBar').style.width='0%'
  fetch('/api/preload_tiles?zoom='+zoom+'&scope='+scope).then(function(r){return r.json()}).then(function(d){
    document.getElementById('navSt').textContent='📥 '+d.message
    if(d.ok){
      if(preloadTimer)clearInterval(preloadTimer)
      preloadTimer=setInterval(function(){
        fetch('/api/preload_progress').then(function(r){return r.json()}).then(function(p){
          if(!p.running){
            clearInterval(preloadTimer);preloadTimer=null
            btn.textContent='📥 离线预载'
            btn.style.opacity='1'
            btn.style.pointerEvents='auto'
            document.getElementById('navSt').textContent='✅ '+p.message
            document.getElementById('progText').textContent='✅ 完成! 新下载 '+p.downloaded+' 张, 已有 '+p.skipped+' 张'
            document.getElementById('progBar').style.width='100%'
            refreshTileStats()
            return
          }
          var pct=Math.min(100,(p.downloaded+p.skipped)/Math.max(p.total,1)*100)
          document.getElementById('progBar').style.width=pct+'%'
          document.getElementById('progText').textContent='Z'+p.current_zoom+': 新'+p.downloaded+' / 已有'+p.skipped+' / 失败'+p.errors
        })
      },500)
    }else{
      btn.textContent='📥 离线预载'
      btn.style.opacity='1'
      btn.style.pointerEvents='auto'
      document.getElementById('navSt').textContent='⚠️ '+d.message
    }
  })
}
</script>
</div>
<div style="display:flex;flex-direction:column;background:rgba(255,255,255,0.02);border-top:1px solid var(--glass-border);font-size:12px;color:var(--text-secondary)">
<div style="display:flex;align-items:center;gap:8px;padding:8px 12px 4px">
<span id="navCoords">📍 38.0428,114.5149</span>
<span style="flex:1"></span>
<span id="tileStats" style="font-size:11px;color:var(--text-tertiary)">🗂️ 加载中...</span>
</div>
<div id="preloadProg" style="display:none;padding:0 12px;margin-bottom:2px">
<div style="height:4px;background:rgba(255,255,255,.08);border-radius:4px;overflow:hidden">
<div id="progBar" style="width:0%;height:100%;background:linear-gradient(90deg,#00d4aa,#00e5ff);border-radius:4px;transition:width .3s"></div></div>
<div id="progText" style="font-size:10px;padding:2px 0;color:var(--text-tertiary)"></div>
</div>
<div style="display:flex;gap:6px;padding:4px 12px 8px;flex-wrap:wrap">
<span onclick="startPreload('quick','city')" id="preloadBtn" style="cursor:pointer;padding:4px 10px;border-radius:6px;background:#151530;color:#00e5ff;font-size:11px">📥 快速预载</span>
<span onclick="startPreload('full','city')" style="cursor:pointer;padding:4px 10px;border-radius:6px;background:#151530;color:#00d4aa;font-size:11px">📦 完整预载 (z8-16)</span>
<span onclick="startPreload('street','core')" style="cursor:pointer;padding:4px 10px;border-radius:6px;background:#151530;color:#aaa;font-size:11px">🏙️ 核心街道</span>
<span onclick="refreshTileStats()" style="cursor:pointer;padding:4px 10px;border-radius:6px;background:rgba(255,255,255,0.04);color:var(--text-tertiary);font-size:11px">🔄</span>
</div>
<div id="tileDetail" style="padding:0 12px 6px;font-size:10px;color:var(--text-tertiary);word-break:break-all"></div>
</div></div></div>

<div class="page" id="pageSettings"><div class="func-header"><span class="back-btn" onclick="gb()">‹</span><span class="func-title">⚙️ 系统设置</span><span class="func-status">RDK X5</span></div>
<div class="func-body scroll"><div class="settings-list">
<div class="setting-item-row">
<div class="setting-label"><span>🔊 音量</span><span id="volLabel" class="setting-val">70%</span></div>
<div class="setting-slider"><input type="range" id="volSlider" min="0" max="100" value="70" oninput="setVolume(this.value)"></div>
</div>
<div class="setting-item-row">
<div class="setting-label"><span>💡 亮度</span><span id="briLabel" class="setting-val">80%</span></div>
<div class="setting-slider"><input type="range" id="briSlider" min="0" max="100" value="80" oninput="setBrightness(this.value)"></div>
</div>
<div class="setting-item"><span>🌐 WiFi</span><span style="color:#888">已连接</span></div>
<div class="setting-item"><span>📷 摄像头</span><span style="color:#888" id="setCam">MIPI IMX219</span></div>
<div class="setting-item"><span>🧠 AI 模型</span><span style="color:#888">YOLOv5s (BPU 10T)</span></div>
</div></div></div>

<div class="page" id="pageInfo"><div class="func-header"><span class="back-btn" onclick="gb()">‹</span><span class="func-title">ℹ️ 关于本机</span><span class="func-status">v2.0</span></div>
<div class="func-body scroll"><div class="card-top-view" style="gap:8px;padding:20px"><div style="font-size:50px">🚗</div><div style="font-size:18px;font-weight:700;color:#00e5ff">RDK X5 车载系统 · MIPI 版</div><div style="font-size:12px;color:#888;line-height:1.8;text-align:center">处理器: 8×Cortex-A55<br>AI: 10 TOPS BPU (Bayes)<br>视觉: MIPI IMX219 + YOLOv5s<br>系统: RDK OS 3.x<br>摄像头: 1920×1080 @30fps</div></div></div></div></div>

<div class="page" id="pageParking"><div class="func-header"><span class="back-btn" onclick="gb()">‹</span><span class="func-title">🎵 车载MP3</span><span class="func-status">本地播放</span></div>
<div class="func-body scroll" style="display:flex;flex-direction:column">
<div class="mp3-layout">
  <div class="mp3-left">
    <!-- Vinyl Record -->
    <div class="vinyl-wrapper" id="vinylWrapper">
      <div class="vinyl-disc">
        <div class="vinyl-shine"></div>
        <div class="vinyl-arc"></div>
        <div class="vinyl-label-ring"></div>
        <div class="vinyl-label">
          <span id="vinylIcon" style="font-size:22px;opacity:.9">🎵</span>
          <span id="vinylLetter" style="font-size:9px;color:rgba(255,255,255,.5);letter-spacing:1px;font-weight:300">A</span>
        </div>
        <div class="vinyl-hole"></div>
      </div>
    </div>
    <!-- Song Info -->
    <div class="mp3-now">
      <div class="title" id="songTitle">未播放</div>
      <div class="artist" id="songArtist">点击播放开始</div>
    </div>
    <!-- Progress Bar -->
    <div class="mp3-progress">
      <div style="display:flex;justify-content:space-between;font-size:10px;color:var(--text-tertiary);margin-bottom:4px">
        <span id="progCurrent">0:00</span>
        <span id="progTotal">0:00</span>
      </div>
      <div style="position:relative;height:4px;background:rgba(255,255,255,.08);border-radius:4px;cursor:pointer" id="progBar" onclick="seekProg(event)">
        <div id="progFill" style="height:100%;width:0%;border-radius:4px;background:linear-gradient(90deg,var(--accent),rgba(0,212,170,.6));transition:width .3s linear"></div>
      </div>
    </div>
    <!-- Controls -->
    <div class="mp3-controls">
      <div class="ctrl" onclick="prevTrack()">⏮</div>
      <div class="ctrl play" onclick="togglePlay()" id="playBtn">▶</div>
      <div class="ctrl" onclick="nextTrack()">⏭</div>
    </div>
    <!-- Volume -->
    <div class="mp3-volume">
      <span style="font-size:13px;color:var(--text-tertiary)">🔊</span>
      <input type="range" id="mp3Vol" min="0" max="100" value="70" oninput="setMp3Vol(this.value)">
      <span id="mp3VolVal" style="font-size:11px;color:var(--text-tertiary);min-width:26px;text-align:right;font-family:monospace">70</span>
    </div>
  </div>
  <div class="mp3-right">
    <div class="mp3-pl-head">
      <span class="lbl">播放列表</span>
      <span class="cnt" id="plCount">0 首</span>
    </div>
    <div class="mp3-pl-list">
      <div id="playlist"></div>
      <div id="plEmpty" style="text-align:center;padding:24px 0;color:var(--text-tertiary);font-size:12px">
        📂 暂无音乐<br><span style="font-size:10px">请用 SCP 将 MP3 文件传至 /music 目录</span>
      </div>
    </div>
  </div>
</div>
</div>
<script>
// MP3 Player State
var mp3={playlist:[],current:0,playing:false,audio:null,volume:70};
var audio=null;

function initPlayer(){
  fetch('/api/music_list').then(function(r){return r.json()}).then(function(list){
    if(!list||list.length===0){
      mp3.playlist=[];document.getElementById('plEmpty').style.display='block';
      document.getElementById('plCount').textContent='0 首';
      document.getElementById('playlist').innerHTML='';
      document.getElementById('songTitle').textContent='暂无音乐';
      document.getElementById('songArtist').textContent='请通过 SCP 传歌';
      document.getElementById('vinylWrapper').classList.remove('playing');
      document.getElementById('vinylIcon').textContent='💿';
      document.getElementById('vinylLetter').textContent='?';
      return;
    }
    document.getElementById('plEmpty').style.display='none';
    // Build playlist with artist parsed from filename (artist - title.mp3) or fallback
    mp3.playlist=list.map(function(f,i){
      var name=f.title;
      var artist='未知',title=name;
      if(name.indexOf('-')>0){
        var p=name.split('-');
        artist=p[0].trim();title=p[1].trim();
      }
      return {title:title,artist:artist,src:f.path,file:f.file};
    });
    document.getElementById('plCount').textContent=mp3.playlist.length+' 首';
    renderPlaylist();
    if(mp3.playlist.length>0){
      mp3.current=0;
      document.getElementById('songTitle').textContent=mp3.playlist[0].title;
      document.getElementById('songArtist').textContent=mp3.playlist[0].artist;
    }
  }).catch(function(err){
    document.getElementById('songTitle').textContent='加载失败';
    document.getElementById('songArtist').textContent=err.message;
  });
}

function renderPlaylist(){
  var pl=document.getElementById('playlist');pl.innerHTML='';
  mp3.playlist.forEach(function(s,i){
    var d=document.createElement('div');
    d.style.cssText='display:flex;align-items:center;padding:8px 10px;border-radius:10px;cursor:pointer;transition:all .15s;gap:10px;margin-bottom:2px';
    d.innerHTML='<span style="font-size:16px;width:24px;text-align:center;color:'+(i===mp3.current?'var(--accent)':'var(--text-tertiary)')+'">'+(i===mp3.current?'▶':'🎵')+'</span><div style="flex:1;min-width:0"><div style="font-size:13px;font-weight:500;color:'+(i===mp3.current?'var(--accent)':'var(--text-primary)')+';overflow:hidden;text-overflow:ellipsis;white-space:nowrap">'+s.title+'</div><div style="font-size:10px;color:var(--text-tertiary)">'+s.artist+'</div></div>';
    d.onclick=function(){playTrack(i)};
    d.onmouseenter=function(){this.style.background='var(--bg-card)'};
    d.onmouseleave=function(){this.style.background='transparent'};
    pl.appendChild(d);
  });
  updateMeta();
}

function playTrack(i){
  if(audio){audio.pause();audio=null}
  mp3.current=i;mp3.playing=true;
  // Try to play actual audio file, fallback to simulated play
  try{
    audio=new Audio(mp3.playlist[i].src);
    audio.volume=mp3.volume/100;
    audio.play().catch(function(){simulatePlay(i)});
    audio.ontimeupdate=function(){updateProg()};
    audio.onended=function(){nextTrack()};
  }catch(e){simulatePlay(i)}
  document.getElementById('playBtn').textContent='⏸';
  document.getElementById('songTitle').textContent=mp3.playlist[i].title;
  document.getElementById('songArtist').textContent=mp3.playlist[i].artist;
  document.getElementById('vinylWrapper').classList.add('playing');
  document.getElementById('vinylIcon').textContent='🎵';
  document.getElementById('vinylLetter').textContent=mp3.playlist[i].title.charAt(0).toUpperCase();
  updatePlaylistUI();
}

// Simulated playback for demo (when audio files don't exist)
var simTimer=null;var simProg=0;
function simulatePlay(i){
  mp3.playing=true;simProg=0;
  document.getElementById('vinylWrapper').classList.add('playing');
  document.getElementById('progTotal').textContent='3:30';
  if(simTimer)clearInterval(simTimer);
  simTimer=setInterval(function(){
    simProg+=1;
    var min=Math.floor(simProg/60),sec=simProg%60;
    document.getElementById('progCurrent').textContent=min+':'+('0'+sec).slice(-2);
    var pct=Math.min(100,simProg/210*100);
    document.getElementById('progFill').style.width=pct+'%';
    if(simProg>=210){clearInterval(simTimer);nextTrack()}
  },1000);
}

function togglePlay(){
  if(!mp3.playing&&!audio&&mp3.playlist.length>0){playTrack(mp3.current);return}
  if(audio){
    if(mp3.playing){
      audio.pause();clearInterval(simTimer);
      document.getElementById('playBtn').textContent='▶';
      document.getElementById('vinylWrapper').classList.remove('playing');
    }else{
      audio.play();
      document.getElementById('playBtn').textContent='⏸';
      document.getElementById('vinylWrapper').classList.add('playing');
    }
    mp3.playing=!mp3.playing;
  }else if(simTimer){
    if(mp3.playing){
      clearInterval(simTimer);
      document.getElementById('playBtn').textContent='▶';
      document.getElementById('vinylWrapper').classList.remove('playing');
      mp3.playing=false;
    }else{
      simulatePlay(mp3.current);
      document.getElementById('playBtn').textContent='⏸';
      document.getElementById('vinylWrapper').classList.add('playing');
    }
  }
}

function nextTrack(){playTrack((mp3.current+1)%mp3.playlist.length)}
function prevTrack(){playTrack((mp3.current-1+mp3.playlist.length)%mp3.playlist.length)}

function seekProg(e){
  var bar=e.currentTarget;
  var pct=(e.clientX-bar.getBoundingClientRect().left)/bar.offsetWidth;
  if(audio&&audio.duration){audio.currentTime=pct*audio.duration}
  else if(simTimer){simProg=Math.floor(pct*210);document.getElementById('progFill').style.width=(pct*100)+'%'}
}

function updateProg(){
  if(!audio||!audio.duration)return;
  var pct=audio.currentTime/audio.duration*100;
  document.getElementById('progFill').style.width=pct+'%';
  var m=Math.floor(audio.currentTime/60),s=Math.floor(audio.currentTime%60);
  document.getElementById('progCurrent').textContent=m+':'+('0'+s).slice(-2);
  var tm=Math.floor(audio.duration/60),ts=Math.floor(audio.duration%60);
  document.getElementById('progTotal').textContent=tm+':'+('0'+ts).slice(-2);
}

function updateMeta(){
  if(!mp3.playlist.length)return;
  document.getElementById('songTitle').textContent=mp3.playlist[mp3.current].title;
  document.getElementById('songArtist').textContent=mp3.playlist[mp3.current].artist;
}

function updatePlaylistUI(){
  var items=document.getElementById('playlist').children;
  for(var i=0;i<items.length;i++){
    var c=items[i].children;
    c[0].textContent=(i===mp3.current?'▶':'🎵');
    c[0].style.color=(i===mp3.current?'var(--accent)':'var(--text-tertiary)');
    c[1].children[0].style.color=(i===mp3.current?'var(--accent)':'var(--text-primary)');
  }
}

function setMp3Vol(v){
  mp3.volume=v;
  document.getElementById('mp3Vol').value=v;
  document.getElementById('mp3VolVal').textContent=v;
  if(audio)audio.volume=v/100;
}
initPlayer();
</script></div>

<!-- ═══ 哨兵模式 页面 ═══ -->
<!-- 手动泊车 页面 -->
<!-- 手动泊车 页面 -->
<div class="page" id="pageMpark"><div class="func-header"><span class="back-btn" onclick="gb()">‹</span><span class="func-title">🅿 手动泊车</span><span class="func-status" id="mparkSt">未连接</span></div>
<div class="func-body" style="display:flex;flex-direction:column;align-items:center;justify-content:center;gap:28px;padding:24px;color:var(--text-secondary);">
<div style="text-align:center">
<div style="font-size:13px;font-weight:600;letter-spacing:2px;color:var(--text-secondary)">按住方向键控制小车</div>
<div style="font-size:11px;color:var(--text-tertiary);margin-top:4px">松开自动停止 · 默认速度 60%</div>
</div>
<div class="dpad">
<button class="dpad-btn dpad-up" data-action="forward">▲</button>
<button class="dpad-btn dpad-left" data-action="turn_left">◀</button>
<button class="dpad-btn dpad-center dpad-stop" data-action="stop">停</button>
<button class="dpad-btn dpad-right" data-action="turn_right">▶</button>
<button class="dpad-btn dpad-down" data-action="backward">▼</button>
</div>
<button id="mparkHorn" class="horn-btn" type="button">🔊 按住喇叭</button>
<div style="display:flex;gap:10px;align-items:center;font-size:12px">
<span>速度</span>
<input id="mparkSpeed" type="range" min="10" max="100" step="10" value="60" style="width:160px;accent-color:var(--accent)">
<span id="mparkSpeedVal" style="min-width:34px">60%</span>
</div>
<div id="mparkLog" style="font-size:11px;color:var(--text-tertiary);min-height:14px;max-width:340px;text-align:center">就绪</div>
</div></div>

<div class="page" id="pageSentinel"><div class="func-header"><span class="back-btn" onclick="gb()">‹</span><span class="func-title">🛡 哨兵模式</span><span class="func-status" id="sentinelSt">监控中</span></div>
<div class="func-body" style="position:relative;background:#000">
  <img id="liveCamSentinel" alt="MIPI Cam" style="width:100%;height:100%;object-fit:contain">
  <div class="sentinel-bar">
    <span style="font-size:11px;color:var(--text-tertiary);letter-spacing:1px">检测到人时</span>
    <button id="sentinelBuzzerBtn" class="sentinel-toggle on" onclick="toggleSentinelBuzzer()">
      <span style="font-size:14px">🔔</span>
      <span>蜂鸣器</span>
      <span id="sentinelBuzzerStatus">开</span>
    </button>
  </div>
  <div id="sentinelWarnBar" style="position:absolute;top:0;left:0;right:0;padding:20px 0;display:flex;justify-content:center;pointer-events:none;z-index:998;opacity:0;transition:opacity .2s ease">
    <div style="background:rgba(229,57,53,.95);color:#fff;padding:18px 32px;border-radius:14px;display:flex;align-items:center;gap:16px;box-shadow:0 8px 32px rgba(229,57,53,.6);border:2px solid #fff">
      <svg width="56" height="56" viewBox="0 0 24 24" fill="#fff"><path d="M12 1 L23 21 L1 21 Z" stroke="#fff" stroke-width="1.5" stroke-linejoin="round"/><rect x="11" y="8" width="2" height="7" fill="#fff"/><circle cx="12" cy="18" r="1.2" fill="#fff"/></svg>
      <div>
        <div style="font-size:24px;font-weight:900;letter-spacing:6px;text-shadow:0 2px 8px rgba(0,0,0,.5)">检 测 到 人</div>
        <div style="font-size:13px;opacity:.95;margin-top:2px">PERSON DETECTED</div>
      </div>
    </div>
  </div>
  <div id="sentinelOverlay" style="position:absolute;inset:0;pointer-events:none;z-index:997;border:6px solid rgba(229,57,53,.7);box-sizing:border-box;opacity:0;transition:opacity .15s ease;animation:sentinelBorderPulse .8s ease-in-out infinite"></div>
  <div style="position:absolute;left:12px;bottom:12px;background:rgba(0,0,0,.55);padding:6px 12px;border-radius:8px;font-size:12px;color:#fff;backdrop-filter:blur(8px);z-index:996">
    <span id="sentinelCount" style="color:#00d4aa;font-weight:700;font-size:14px">0</span> 人 · 阈值 0.30
  </div>
  <div style="position:absolute;right:12px;bottom:12px;background:rgba(0,0,0,.55);padding:6px 12px;border-radius:8px;font-size:11px;color:#aaa;backdrop-filter:blur(8px);z-index:996">
    YOLOv5s BPU · 640×360
  </div>
</div></div>

<style>
/* ── 手动泊车 喇叭按钮 ── */
.horn-btn{border:none;border-radius:14px;background:rgba(255,180,40,0.16);color:#ffb84a;font-size:15px;font-weight:700;letter-spacing:3px;cursor:pointer;user-select:none;-webkit-tap-highlight-color:transparent;transition:all .12s;padding:14px 36px;font-family:inherit;min-width:220px;display:inline-flex;align-items:center;justify-content:center;gap:8px}
.horn-btn:active,.horn-btn.active{background:#ffb84a;color:#1a1500;box-shadow:0 0 28px rgba(255,184,74,.65);transform:scale(0.97)}

/* ── 手动泊车 D-pad ── */
.dpad{display:grid;grid-template-columns:64px 64px 64px;grid-template-rows:64px 64px 64px;gap:8px}
.dpad-btn{border:none;border-radius:12px;background:rgba(255,255,255,0.08);color:var(--text-primary);font-size:22px;font-weight:700;cursor:pointer;user-select:none;-webkit-tap-highlight-color:transparent;transition:all .1s;font-family:inherit}
.dpad-btn:active,.dpad-btn.active{background:var(--accent);color:var(--bg-primary);box-shadow:0 0 20px var(--accent-glow);transform:scale(0.96)}
.dpad-up{grid-column:2;grid-row:1}
.dpad-left{grid-column:1;grid-row:2}
.dpad-center{grid-column:2;grid-row:2}
.dpad-right{grid-column:3;grid-row:2}
.dpad-down{grid-column:2;grid-row:3}
.dpad-stop{background:rgba(255,80,80,0.18);color:#ff8080}
.dpad-stop:active,.dpad-stop.active{background:#ff5252;color:#fff}
@keyframes sentinelBorderPulse{0%,100%{box-shadow:inset 0 0 30px rgba(229,57,53,.5),0 0 30px rgba(229,57,53,.4)}50%{box-shadow:inset 0 0 80px rgba(229,57,53,.8),0 0 80px rgba(229,57,53,.7)}}
#sentinelWarnBar.active{opacity:1;animation:sentinelBarShake .35s ease-in-out infinite}
@keyframes sentinelBarShake{0%,100%{transform:translateX(0)}25%{transform:translateX(-3px)}75%{transform:translateX(3px)}}
</style>

<script>
/* ── 手动泊车 D-pad 交互 ── */
(function(){
 var log=document.getElementById('mparkLog');
 var speedInp=document.getElementById('mparkSpeed');
 var speedVal=document.getElementById('mparkSpeedVal');
 var statusEl=document.getElementById('mparkSt');
 function send(a,sp){
   return fetch('/api/car/control?action='+a+'&speed='+sp).then(function(r){return r.json();}).then(function(d){
     if(log){
       if(d.ok){var m={forward:'前进',backward:'后退',turn_left:'左转',turn_right:'右转',stop:'停止'}[a]||a;log.textContent=m+' · '+sp+'%';}
       else{log.textContent='⚠ '+(d.msg||'失败');}
     }
     return d;
   }).catch(function(e){if(log)log.textContent='⚠ 网络错误';return {ok:false,msg:'net'};});
 }
 if(speedInp){speedInp.addEventListener('input',function(){speedVal.textContent=speedInp.value+'%';});}
 function getSp(){return speedInp?speedInp.value:60;}
 // 启动时查询一次后端状态
 fetch('/api/car/control?action=stop').then(function(r){return r.json();}).then(function(d){
   if(statusEl){statusEl.textContent=d.ok?'就绪':'未连接';statusEl.style.color=d.ok?'var(--accent)':'#ff8080';}
   if(!d.ok && log){log.textContent='⚠ '+(d.msg||'小车未就绪');}
 });
 document.querySelectorAll('#pageMpark .dpad-btn').forEach(function(btn){
   var a=btn.dataset.action;var timer=null;var pressed=false;
   function start(e){if(e)e.preventDefault();if(pressed)return;pressed=true;btn.classList.add('active');
     if(a==='stop'){send('stop',getSp());return;}
     send(a,getSp());timer=setInterval(function(){send(a,getSp());},120);document.addEventListener('mouseup',end);document.addEventListener('touchend',end);}
   function end(e){if(e)e.preventDefault();if(!pressed)return;pressed=false;btn.classList.remove('active');
     if(timer){clearInterval(timer);timer=null;}
     if(a!=='stop')send('stop',getSp());document.removeEventListener('mouseup',end);document.removeEventListener('touchend',end);}
   btn.addEventListener('mousedown',start);btn.addEventListener('mouseup',end);
   btn.addEventListener('touchstart',start,{passive:false});btn.addEventListener('touchend',end,{passive:false});  btn.addEventListener('touchcancel',end,{passive:false});
 });
 /* ── 喇叭：按下响 松开停 ── */
 var hornBtn=document.getElementById('mparkHorn');
 if(hornBtn){
  var hornPressed=false;
  function hornStart(e){if(e)e.preventDefault();if(hornPressed)return;hornPressed=true;hornBtn.classList.add('active');
   fetch('/api/_debug_buzzer?on=1').then(function(r){return r.json();}).then(function(d){
    if(d.ok){if(log)log.textContent='🔊 喇叭响';}
    else{if(log)log.textContent='⚠ '+(d.msg||'蜂鸣器未就绪');}
   }).catch(function(){if(log)log.textContent='⚠ 网络错误';});document.addEventListener('mouseup',hornEnd);document.addEventListener('touchend',hornEnd);
  }
  function hornEnd(e){if(e)e.preventDefault();if(!hornPressed)return;hornPressed=false;hornBtn.classList.remove('active');
   fetch('/api/_debug_buzzer?on=0').then(function(r){return r.json();}).then(function(d){
    if(d.ok){if(log)log.textContent='🔇 喇叭静音';}
   }).catch(function(){});document.removeEventListener('mouseup',hornEnd);document.removeEventListener('touchend',hornEnd);
  }
  hornBtn.addEventListener('mousedown',hornStart);hornBtn.addEventListener('mouseup',hornEnd);
  hornBtn.addEventListener('touchstart',hornStart,{passive:false});hornBtn.addEventListener('touchend',hornEnd,{passive:false});hornBtn.addEventListener('touchcancel',hornEnd,{passive:false});
 }
})();
</script>

<script>
var sentinelLastAlarm=0,sentinelCooldown=1500;
function updateSentinel(){
  fetch('/api/detections').then(function(r){return r.json()}).then(function(d){
    var persons=(d.objects||[]).filter(function(o){return o.cls==='人'});
    var cnt=persons.length;
    var now=Date.now();
    var el=document.getElementById('sentinelWarnBar');
    var ov=document.getElementById('sentinelOverlay');
    document.getElementById('sentinelCount').textContent=cnt;
    if(cnt>0){
      el.classList.add('active');ov.style.opacity='1';
      document.getElementById('sentinelSt').textContent='⚠ 检测到 ' + cnt + ' 人';
      document.getElementById('sentinelSt').style.color='#ff5252';
      sentinelLastAlarm=now;
    } else if(now - sentinelLastAlarm > sentinelCooldown){
      el.classList.remove('active');ov.style.opacity='0';
      document.getElementById('sentinelSt').textContent='监控中';
      document.getElementById('sentinelSt').style.color='';
    }
  }).catch(function(){});
}
setInterval(updateSentinel,400);
</script>

<script>
var cD={speed:0,gear:'P',radar:[0,0,0,0],battery:85,temp:26.5,fuel:68,odo:12543,warning:false,warningText:''}
var dm=false,dt={count:0,objects:[],fps:0},cp='menu'
var vol=70,bri=80

function nav(p){document.querySelectorAll('.page').forEach(x=>x.classList.remove('active'));var e=document.getElementById('page'+p.charAt(0).toUpperCase()+p.slice(1));if(e)e.classList.add('active');cp=p;fetch('/api/navigate?page='+p);if(p=='navi')setTimeout(initMap,200);if(p=='settings')loadSettings()}
function toggleHeadlight(){fetch('/api/headlight?on=toggle').then(function(r){return r.json()}).then(function(d){if(d&&d.ok)updateHeadlightUI(d.on)}).catch(function(){})}
function toggleSentinelBuzzer(){fetch('/api/sentinel_buzzer?on=toggle').then(function(r){return r.json()}).then(function(d){if(d&&d.ok)updateSentinelBuzzerUI(d.on)}).catch(function(){})}
function updateSentinelBuzzerUI(on){var st=document.getElementById('sentinelBuzzerStatus');var btn=document.getElementById('sentinelBuzzerBtn');if(!st||!btn)return;st.textContent=on?'开':'关';btn.classList.toggle('off',!on)}
fetch('/api/sentinel_buzzer').then(function(r){return r.json()}).then(function(d){if(d&&d.ok)updateSentinelBuzzerUI(d.on)}).catch(function(){})
function updateHeadlightUI(on){var st=document.getElementById('headlightStatus');var btn=document.getElementById('headlightBtn');if(!st||!btn)return;st.textContent=on?'前+后 开启':'前+后 关闭';btn.classList.toggle('dg',!!on)}
fetch('/api/headlight').then(function(r){return r.json()}).then(function(d){if(d&&d.ok)updateHeadlightUI(d.on)}).catch(function(){})
function gb(){document.querySelectorAll('.page').forEach(x=>x.classList.remove('active'));document.getElementById('pageMenu').classList.add('active');cp='menu';fetch('/api/navigate?page=menu')}
function td(){dm=!dm;fetch('/api/demo?on='+(dm?'1':'0'));document.querySelector('[onclick*="td()"]').classList.toggle('dg')}

function setVolume(v){
 vol=v;
 document.getElementById('volSlider').value=v;
 document.getElementById('volLabel').textContent=v+'%';
 fetch('/api/set_volume?val='+v)
}
function setBrightness(v){
 bri=v;
 document.getElementById('briSlider').value=v;
 document.getElementById('briLabel').textContent=v+'%';
 fetch('/api/set_brightness?val='+v)
}
function loadSettings(){
 fetch('/api/settings').then(r=>r.json()).then(function(d){
  vol=d.volume||70;bri=d.brightness||80;
  document.getElementById('volSlider').value=vol;
  document.getElementById('volLabel').textContent=vol+'%';
  document.getElementById('briSlider').value=bri;
  document.getElementById('briLabel').textContent=bri+'%';
 }).catch(function(){})
}

function uc(){
 document.getElementById('clockD').textContent=('0'+new Date().getHours()).slice(-2)+':'+('0'+new Date().getMinutes()).slice(-2);document.getElementById('clockSec').textContent=('0'+new Date().getSeconds()).slice(-2);document.getElementById('clockDots').textContent=('0'+new Date().getHours()).slice(-2)+':'+('0'+new Date().getMinutes()).slice(-2)
 var n=new Date();var ds=['日','一','二','三','四','五','六'];document.getElementById('clockDate').textContent=n.getFullYear()+'.'+('0'+(n.getMonth()+1)).slice(-2)+'.'+('0'+n.getDate()).slice(-2)+' · 星期'+ds[n.getDay()]
 document.getElementById('clockTemp').innerHTML='🌡 '+(cD.temp||26.5).toFixed(1)+'°C';document.getElementById('clockWeather').textContent=cD.weather||'☀ --';document.getElementById('clockSpeed').innerHTML='🚗 '+Math.round(cD.speed)+' km/h'
 document.getElementById('mspeed').textContent=Math.round(cD.speed)+' km/h'
 document.getElementById('mbat').textContent=(cD.battery||'--')+'%'
 document.getElementById('mgear').innerHTML='<span class="gear-chip gear-'+(cD.gear||'P')+'">'+(cD.gear||'P')+'</span>'
 document.getElementById('iFuel').textContent='⛽ '+(cD.fuel||'--')+'%'
 document.getElementById('iOdo').textContent='📊 '+(cD.odo||'--')+'km'
 document.getElementById('iObj').textContent='🎯 '+(dt.count||0)+'obj'
 document.getElementById('iFps').textContent='📶 '+(dt.fps||0)+'fps'
 if(cp=='reverse'){
  // 4 元 radar 块 -> 实际只显示 r0 (左后红外) / r3 (右后红外);r1/r2 已删,getElementById 返 null 时跳过
for(var i=0;i<4;i++){var el=document.getElementById('r'+i);if(!el)continue;var d=cD.radar[i]||0;if(i===0&&cD.infrared&&cD.infrared.left_rear===0){d=0.05;el.textContent='\u26a0\u5de6'}else if(i===3&&cD.infrared&&cD.infrared.right_rear===0){d=0.05;el.textContent='\u26a0\u53f3'}else{el.textContent=d>0?d.toFixed(1)+'m':'\u2014'}el.className='radar-blk '+(d>0?(d<.5?'d':d<1?'c':'s'):'s')}
  var w=document.getElementById('revWarn');if(cD.rear_alarm&&cD.rear_alarm_text){w.textContent='⚠️ '+cD.rear_alarm_text;w.className='warning-banner show crit'}else{w.className='warning-banner'}
  var tg=document.getElementById('revTags');tg.innerHTML='';(dt.objects||[]).slice(0,4).forEach(function(o){var t=document.createElement('span');t.className='tag';t.textContent=o.cls+' '+(o.score*100).toFixed(0)+'%';tg.appendChild(t)})
  document.getElementById('revInfo').textContent=dt.count+' obj'
 }
 if(cp=='front'){
  var tg=document.getElementById('frontTags');tg.innerHTML='';(dt.objects||[]).slice(0,4).forEach(function(o){var t=document.createElement('span');t.className='tag';t.textContent=o.cls+' '+(o.score*100).toFixed(0)+'%';tg.appendChild(t)})
  var w=document.getElementById('frontWarn');if(cD.warning&&cD.warningText){w.textContent='⚠️ '+cD.warningText;w.className='warning-banner show crit'}else{w.className='warning-banner'}
  var pa=document.getElementById('personAlarm');var pb=document.getElementById('personBorder');var pt=document.getElementById('personText');
  if(dt.person_alarm){pa.classList.add('active');pb.classList.add('active');pt.classList.add('active')}else{pa.classList.remove('active');pb.classList.remove('active');pt.classList.remove('active')}
  var hc=(dt.objects||[]).some(function(o){return o.cls==='\u8f66\u8f86'});var ho=(dt.objects||[]).some(function(o){return o.cls==='\u969c\u788d\u7269'});document.getElementById('carWarn').style.display=hc?'block':'none';document.getElementById('obsWarn').style.display=ho?'block':'none'
 }
 if(cp=='visual_range'){
  var d=dt.person_distance_m;
  var alarm=dt.vr_alarm;
  var cnt=dt.person_count||(dt.objects||[]).filter(function(o){return o.cls==='\u4eba'}).length;
  var foc=dt.focal_y_px||0;
  var dEl=document.getElementById('vrDist'),cEl=document.getElementById('vrCount'),sEl=document.getElementById('vrState'),aEl=document.getElementById('vrAlarm2'),fEl=document.getElementById('vrFocal');
  if(dEl)dEl.textContent=(d===null||d===undefined)?'- m':d.toFixed(2)+' m';
  if(cEl)cEl.textContent=cnt;
  if(fEl)fEl.textContent=foc?foc.toFixed(0):'-';
  if(sEl){if(alarm){sEl.textContent='\u26a0\ufe0f \u62a5\u8b66';sEl.style.color='#ff5252';}else if(cnt>0){sEl.textContent='\ud83d\udfe1 \u68c0\u6d4b';sEl.style.color='#ffcc00';}else{sEl.textContent='\ud83d\udfe2 \u9759\u97f3';sEl.style.color='#00d4aa';}}
  if(aEl)aEl.textContent=alarm?'\u5f00':'\u5173';
 }
 if(cp=='fatigue'){
  var f=dt.fatigue||{};
  var earEl=document.getElementById('fatEar'),marEl=document.getElementById('fatMar');
  var eyeStEl=document.getElementById('fatEyeSt'),mouthStEl=document.getElementById('fatMouthSt');
  var earTh=parseFloat((document.getElementById('earThresh')||{value:0.21}).value);
  var marTh=parseFloat((document.getElementById('marThresh')||{value:0.60}).value);
  if(earEl)earEl.textContent=f.ear?(+f.ear).toFixed(3):'--';
  if(marEl)marEl.textContent=f.mar?(+f.mar).toFixed(3):'--';
  if(eyeStEl){var ec=!!(f.face&&f.ear&&f.ear<earTh);eyeStEl.textContent=!f.face?'--':(ec?'闭眼':'张眼');eyeStEl.className='fat-stat-v '+(ec?'crit':(f.face?'':'warn'));}
  if(mouthStEl){var mo=!!(f.face&&f.mar&&f.mar>marTh);mouthStEl.textContent=!f.face?'--':(mo?'哈欠':'正常');mouthStEl.className='fat-stat-v '+(mo?'warn':'');}
  drawFatigueBoxes(f);
  var st=document.getElementById('fatSt');
  if(st){st.textContent=f.face?'监控中':'等待人脸';st.style.color=f.face?'#00d4aa':'#888';}
  var wn=document.getElementById('fatWarn');
  if(wn)wn.className='warning-banner';
  var vrAlarm=!!(dt&&dt.vr_alarm);
  var ov=document.getElementById('vrAlarm'),bd=document.getElementById('vrAlarmBorder'),tx=document.getElementById('vrAlarmText'),vrwn=document.getElementById('vrWarn');
  if(ov&&bd&&tx){if(vrAlarm){ov.classList.add('active');bd.classList.add('active');tx.classList.add('active');}else{ov.classList.remove('active');bd.classList.remove('active');tx.classList.remove('active');}}
  if(vrwn){vrwn.textContent=vrAlarm?'⚠ 距离 < 0.5m 蜂鸣报警':'';vrwn.className=vrAlarm?'warning-banner show crit':'warning-banner';}
 }
 if(cp=='radar'){for(var i=0;i<4;i++){var d=cD.radar[i]||0;var e=document.getElementById('s'+i);if(e){e.className='sensor-dot '+(d>0?(d<.5?'r':d<1?'o':'g'):'g')}}
  if(cp=='radar'){var mr=Math.min.apply(null,cD.radar.filter(function(x){return x>0}));document.getElementById('radarDist').textContent=mr<999?mr.toFixed(2)+'m':'--'}}
 }
setInterval(function(){
 fetch('/api/car_data').then(function(r){return r.json()}).then(function(d){cD=d;uc()}).catch(function(){})
 fetch('/api/detections').then(function(r){return r.json()}).then(function(d){dt=d}).catch(function(){})
},500)
function startLiveCam(id,fps){
  var img=document.getElementById(id);if(!img)return;
  var interval=Math.max(80,Math.floor(1000/fps));
  setInterval(function(){
    fetch('/snapshot?_t='+Date.now()).then(function(r){return r.blob()}).then(function(b){
      if(img._lastUrl)URL.revokeObjectURL(img._lastUrl);
      var u=URL.createObjectURL(b);img._lastUrl=u;img.src=u;
    }).catch(function(){});
  },interval);
}
startLiveCam('liveCamReverse',5);
startLiveCam('liveCamFront',5);
startLiveCam('liveCamFatigue',5);
startLiveCam('liveCamSentinel',5);
/* === Fatigue: 画眼睛/嘴巴框 === */
function drawFatigueBoxes(f){
 var cv=document.getElementById('fatCanvas');
 var img=document.getElementById('liveCamFatigue');
 if(!cv||!img)return;
 var cw=cv.clientWidth,ch=cv.clientHeight;
 if(!cw||!ch)return;
 if(cv.width!==cw||cv.height!==ch){cv.width=cw;cv.height=ch;}
 var ctx=cv.getContext('2d');
 ctx.clearRect(0,0,cw,ch);
 if(!f||!f.face||!f.eye_boxes||!f.mouth_box)return;
 var scale=Math.min(cw/640,ch/360);
 var dw=640*scale,dh=360*scale;
 var ox=(cw-dw)/2,oy=(ch-dh)/2;
 function map(b){return [ox+b[0]*scale,oy+b[1]*scale,ox+b[2]*scale,oy+b[3]*scale];}
 var earTh=parseFloat((document.getElementById('earThresh')||{value:0.21}).value);
 var marTh=parseFloat((document.getElementById('marThresh')||{value:0.60}).value);
 var eyeClosed=!!(f.ear&&f.ear<earTh);
 var mouthOpen=!!(f.mar&&f.mar>marTh);
 ctx.lineWidth=2.5;ctx.font='bold 13px -apple-system,sans-serif';
 ctx.strokeStyle=eyeClosed?'#ff5252':'#00ff88';ctx.fillStyle=ctx.strokeStyle;
 var l=map(f.eye_boxes[0]);ctx.strokeRect(l[0],l[1],l[2]-l[0],l[3]-l[1]);ctx.fillText(eyeClosed?'闭眼':'眼睛',l[0],Math.max(14,l[1]-4));
 var r=map(f.eye_boxes[1]);ctx.strokeRect(r[0],r[1],r[2]-r[0],r[3]-r[1]);ctx.fillText(eyeClosed?'闭眼':'眼睛',r[0],Math.max(14,r[1]-4));
 ctx.strokeStyle=mouthOpen?'#ffcc00':'#00d4aa';ctx.fillStyle=ctx.strokeStyle;
 var m=map(f.mouth_box);ctx.strokeRect(m[0],m[1],m[2]-m[0],m[3]-m[1]);ctx.fillText(mouthOpen?'哈欠':'嘴巴',m[0],Math.max(14,m[1]-4));
}
/* === 取消疲劳警报按钮 === */
function cancelFatigueAlarm(){
 var btn=document.getElementById('btnCancelFatigueAlarm');
 if(btn){btn.disabled=true;btn.textContent='\u2713 已静默';btn.style.background='rgba(120,120,120,.3)';btn.style.color='#aaa';btn.style.borderColor='rgba(120,120,120,.4)';}
 fetch('/api/fatigue/ack').then(function(r){return r.json()}).then(function(d){
  console.log('[fatigue] ack:', d);
 }).catch(function(e){console.error('[fatigue] ack err:', e);});
 setTimeout(function(){
  if(btn){btn.disabled=false;btn.textContent='\ud83d\udd14 取消警报';btn.style.background='rgba(0,212,170,.15)';btn.style.color='#00d4aa';btn.style.borderColor='rgba(0,212,170,.4)';}
 },3000);
}
/* === 阈值 slider 联动 === */
(function(){
 var earInp=document.getElementById('earThresh'),marInp=document.getElementById('marThresh');
 var earT=document.getElementById('earThreshT'),marT=document.getElementById('marThreshT');
 function sync(inp,prec){inp.nextElementSibling.textContent=parseFloat(inp.value).toFixed(prec);}
 if(earInp){earInp.addEventListener('input',function(){sync(earInp,3);if(earT)earT.textContent=parseFloat(earInp.value).toFixed(2);});}
 if(marInp){marInp.addEventListener('input',function(){sync(marInp,2);if(marT)marT.textContent=parseFloat(marInp.value).toFixed(2);});}
})();
function toggleFullscreen(){if(!document.fullscreenElement){(document.documentElement.requestFullscreen||function(){}).call(document.documentElement)}else{(document.exitFullscreen||function(){}).call(document)}}
document.addEventListener("fullscreenchange",function(){var b=document.getElementById("fullscreenBtn");if(!b)return;b.textContent=document.fullscreenElement?"⤡":"⤢";b.title=document.fullscreenElement?"退出全屏":"进入全屏"});
setInterval(uc,1000);uc()


/* ── Dynamic Light & Ripple ── */

// Cursor/touch ambient glow
var cg=document.getElementById('cursorGlow');
document.addEventListener('mousemove',function(e){
  cg.style.setProperty('--mx',e.clientX+'px');
  cg.style.setProperty('--my',e.clientY+'px');
});
document.addEventListener('touchmove',function(e){
  var t=e.touches[0];
  cg.style.setProperty('--mx',t.clientX+'px');
  cg.style.setProperty('--my',t.clientY+'px');
},{passive:true});

// Ripple effect on .ripple-container
document.addEventListener('click',function(e){
  var btn=e.target.closest('.ripple-container');
  if(!btn)return;
  var rect=btn.getBoundingClientRect();
  var x=e.clientX-rect.left;
  var y=e.clientY-rect.top;
  var size=Math.max(rect.width,rect.height)*1.2;
  var r=document.createElement('span');
  r.className='ripple';
  r.style.width=r.style.height=size+'px';
  r.style.left=(x-size/2)+'px';
  r.style.top=(y-size/2)+'px';
  btn.appendChild(r);
  setTimeout(function(){r.remove()},600);
});

// Long press flash effect
var longTimer=null;
document.addEventListener('mousedown',function(e){
  var btn=e.target.closest('.m-btn');
  if(!btn)return;
  longTimer=setTimeout(function(){
    btn.style.transition='box-shadow .1s';
    btn.style.boxShadow='0 0 40px rgba(0,212,170,0.35), inset 0 0 30px rgba(0,212,170,0.08)';
    setTimeout(function(){btn.style.boxShadow='';},300);
  },400);
});
document.addEventListener('mouseup',function(){clearTimeout(longTimer);longTimer=null;});
document.addEventListener('mouseleave',function(){clearTimeout(longTimer);longTimer=null;});

</script></body></html>'''

# Flask and main
@app.route('/')
def index():
    return Response(HMI_HTML.encode('utf-8', errors='replace'), mimetype='text/html; charset=utf-8')

@app.route('/api/car_data')
def api_car_data():
    return jsonify(car_data)

@app.route('/api/detections')
def api_detections():
    with det_lock:
        return jsonify(latest_detections)

@app.route('/api/fatigue')
def api_fatigue():
    with det_lock:
        d = latest_detections.get('fatigue', {}) if isinstance(latest_detections, dict) else {}
    return jsonify({
        'face': d.get('face', False),
        'ear': d.get('ear', 0.0),
        'mar': d.get('mar', 0.0),
        'blinks': d.get('blinks', 0),
        'yawns': d.get('yawns', 0),
        'perclos': d.get('perclos', 0.0),
        'state': d.get('state', 'normal'),
        'alarm': d.get('alarm', False),
        'eye_closed': d.get('eye_closed', False),
        'mouth_open': d.get('mouth_open', False),
        'drowsy_hold_ms': d.get('drowsy_hold_ms', 0),
        'have_face_mesh': HAVE_FACE_MESH,
    })

@app.route('/api/fatigue/ack')
def api_fatigue_ack():
    # 疲劳面板的'取消警报'按钮:立即关蜂鸣器 + 清疲劳 alarm 标志 + 标记本轮事件已确认。
    # 事件真正退出(驾驶员恢复)后,下次事件再发生时仍会正常报警。
    if vr_buzzer is not None:
        vr_buzzer.off()
    fatigue_state['alarm'] = False
    fatigue_state['_event_acked'] = True
    return jsonify({'ok': True, 'silenced': True, 'acked': True})

@app.route('/api/infrared')
def api_infrared():
    return jsonify(car_data.get('infrared', {}))

@app.route('/api/rear')
def api_rear():
    return jsonify({
        'rear_alarm': car_data.get('rear_alarm', False),
        'rear_alarm_text': car_data.get('rear_alarm_text', ''),
        'left_rear': car_data.get('infrared', {}).get('left_rear', 0),
        'right_rear': car_data.get('infrared', {}).get('right_rear', 0),
    })

@app.route('/api/settings')
def api_get_settings():
    return jsonify(settings)

@app.route('/api/set_volume')
def api_set_volume():
    val = request.args.get('val', '70')
    try:
        v = max(0, min(100, int(val)))
        settings['volume'] = v
        subprocess.run(['amixer', 'set', 'DAC', f'{v}%'], capture_output=True, timeout=2)
        return jsonify({'ok': True, 'volume': v})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})

@app.route('/api/set_brightness')
def api_set_brightness():
    val = request.args.get('val', '80')
    try:
        v = max(0, min(100, int(val)))
        settings['brightness'] = v
        bl = '/sys/class/backlight'
        if os.path.exists(bl):
            for d in os.listdir(bl):
                bp = f'{bl}/{d}/brightness'
                if os.path.exists(bp):
                    with open(f'{bl}/{d}/max_brightness') as f:
                        mx = int(f.read().strip())
                    with open(bp, 'w') as f:
                        f.write(str(int(mx * v / 100)))
        return jsonify({'ok': True, 'brightness': v})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})

@app.route('/api/demo')
def api_demo():
    global demo_mode
    demo_mode = request.args.get('on', '0') == '1'
    return jsonify({'demo': demo_mode})

@app.route('/api/sentinel')
def api_sentinel():
    global sentinel_mode
    sentinel_mode = request.args.get('on', '0') == '1'
    fatigue_paused = sentinel_mode  # 哨兵模式下暂停疲劳检测
    return jsonify({'sentinel_mode': sentinel_mode, 'fatigue_paused': fatigue_paused})

@app.route('/api/sentinel_buzzer')
def api_sentinel_buzzer():
    global sentinel_buzzer_on
    act = request.args.get('on', '').lower()
    if act in ('1', 'true', 'on'):
        sentinel_buzzer_on = True
    elif act in ('0', 'false', 'off'):
        sentinel_buzzer_on = False
    elif act == 'toggle':
        sentinel_buzzer_on = not sentinel_buzzer_on
        # 关闭时立即消音
        if not sentinel_buzzer_on and vr_buzzer is not None:
            vr_buzzer.off()
    return jsonify({'ok': True, 'on': sentinel_buzzer_on})

@app.route('/api/headlight')
def api_headlight():
    global headlight
    if headlight is None:
        try:
            headlight = Headlight(HEADLIGHT_GPIOS)
        except Exception as e:
            return jsonify({'ok': False, 'error': str(e), 'gpios': HEADLIGHT_GPIOS})
    act = request.args.get('on', '').lower()
    if act in ('1', 'true', 'on'):
        headlight.on()
    elif act in ('0', 'false', 'off'):
        headlight.off()
    elif act == 'toggle':
        headlight.toggle()
    return jsonify({'ok': True, 'on': headlight.state, 'gpios': HEADLIGHT_GPIOS})

@app.route('/api/car/control')
def api_car_control():
    if not _car_available:
        return jsonify({'ok': False, 'msg': 'car control not available (Hobot.GPIO / PWM 未就绪)'})
    action = request.args.get('action', 'stop')
    sp_arg = request.args.get('speed')
    try:
        sp = int(sp_arg) if sp_arg is not None else None
        if   action == 'forward':    _car.forward(sp)
        elif action == 'backward':   _car.backward(sp)
        elif action == 'turn_left':  _car.turn_left(sp)
        elif action == 'turn_right': _car.turn_right(sp)
        elif action == 'stop':       _car.stop()
        else:
            return jsonify({'ok': False, 'msg': f'unknown action: {action}'})
        return jsonify({'ok': True, 'action': action, 'speed': sp})
    except Exception as e:
        return jsonify({'ok': False, 'msg': str(e)})

@app.route('/api/_debug_buzzer_status')
def api_debug_buzzer_status():
    if vr_buzzer is None:
        return jsonify({'ok': False, 'msg': 'buzzer not init yet (no MIPI loop)'})
    return jsonify({
        'ok': True,
        'alarm_set': vr_buzzer._alarm.is_set(),
        'thread_alive': vr_buzzer._thread.is_alive(),
        'thread_ident': vr_buzzer._thread.ident,
        'gpio_value': int(open(f"{vr_buzzer.base}/value").read().strip()),
    })

@app.route('/api/_debug_buzzer')
def api_debug_buzzer():
    # 调试用:强制驱动蜂鸣器验证 GPIO 接线
    if vr_buzzer is None:
        return jsonify({'ok': False, 'msg': 'buzzer not init yet (no MIPI loop)'})
    global _user_buzzer_hold
    if request.args.get('on', '0') == '1':
        vr_buzzer.on()
        _user_buzzer_hold = True
    else:
        vr_buzzer.off()
        _user_buzzer_hold = False
    return jsonify({'ok': True, 'gpio': vr_buzzer.gpio, 'value': int(open(f"{vr_buzzer.base}/value").read().strip())})

@app.route('/api/navigate')
def api_navigate():
    car_data['screen'] = request.args.get('page', 'menu')
    return jsonify({'ok': True})
@app.route('/api/calibrate_focal', methods=['POST'])
def api_calibrate_focal():
    global vr_focal_y_px
    try:
        d = float(request.args.get('distance_m', '2.0'))
        h = float(request.args.get('real_h_m', str(REAL_HEIGHT_M)))
    except Exception:
        return jsonify({'ok': False, 'msg': 'params err'})
    with det_lock:
        dets = latest_detections.get('objects', [])
        person_bbox_h = 0.0
        for o in dets:
            if o.get('cls') == '\u4eba' and o.get('score', 0) > VR_PERSON_CONF:
                bb = o.get('bbox', [0, 0, 0, 0])
                hh = bb[3] - bb[1]
                if hh > person_bbox_h:
                    person_bbox_h = hh
        if person_bbox_h < 10:
            return jsonify({'ok': False, 'msg': 'no person detected'})
        focal = (person_bbox_h * d) / h
    try:
        os.makedirs(os.path.dirname(VR_FOCAL_FILE), exist_ok=True)
        with open(VR_FOCAL_FILE, 'w') as fp:
            json.dump({'focal_y_px': focal, 'real_h_m': h, 'saved_at': time.strftime('%Y-%m-%d %H:%M:%S')}, fp, indent=2)
        vr_focal_y_px = focal
        return jsonify({'ok': True, 'focal_y_px': focal, 'distance_m': d, 'real_h_m': h, 'bbox_h_px': person_bbox_h})
    except Exception as e:
        return jsonify({'ok': False, 'msg': f'save fail: {e}'})


@app.route('/tiles/<int:z>/<int:x>/<int:y>.png')
def serve_tile(z, x, y):
    tp = f'{TILE_DIR}/{z}/{x}/{y}.png'
    if os.path.exists(tp):
        return send_file(tp, mimetype='image/png')
    # 多源下载瓦片 —— 按优先级尝试
    TILE_SOURCES = [
        'https://tile.openstreetmap.de/{z}/{x}/{y}.png',
        'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
        'http://tile.openstreetmap.org/{z}/{x}/{y}.png',
    ]
    os.makedirs(f'{TILE_DIR}/{z}/{x}', exist_ok=True)
    for src in TILE_SOURCES:
        url = src.replace('{z}', str(z)).replace('{x}', str(x)).replace('{y}', str(y))
        try:
            r = requests.get(url, headers={'User-Agent': 'RDK-CAR/2.0'}, timeout=8)
            if r.status_code == 200:
                with open(tp, 'wb') as f: f.write(r.content)
                return send_file(tp, mimetype='image/png')
        except: pass
    return Response('', status=204)


# ── 本地 Leaflet 静态资源服务 ──
@app.route('/leaflet/<path:filename>')
def serve_leaflet(filename):
    fp = os.path.join(LEAFLET_DIR, filename)
    if os.path.exists(fp):
        return send_file(fp)
    # 回退到 CDN
    return redirect(f'https://unpkg.com/leaflet@1.9.4/dist/{filename}')

# ── 瓦片缓存统计 ──
@app.route('/api/tile_stats')
def api_tile_stats():
    total = 0; size = 0; by_zoom = {}
    for root, dirs, files in os.walk(TILE_DIR):
        for f in files:
            if f.endswith('.png'):
                fp = os.path.join(root, f)
                s = os.path.getsize(fp)
                size += s
                total += 1
                z = os.path.basename(os.path.dirname(os.path.dirname(fp)))
                by_zoom[z] = by_zoom.get(z, 0) + 1
    return jsonify({
        'total_tiles': total,
        'size_bytes': size,
        'size_mb': round(size / 1048576, 1),
        'by_zoom': {k: by_zoom[k] for k in sorted(by_zoom, key=int)}
    })

# ── 预载进度查询 ──
@app.route('/api/preload_progress')
def api_preload_progress():
    return jsonify(preload_state)

# ── 停止预载 ──
@app.route('/api/stop_preload')
def api_stop_preload():
    if preload_state['running']:
        preload_state['running'] = False
        preload_state['message'] = '用户已暂停'
        return jsonify({'ok': True, 'message': '预载已暂停'})
    return jsonify({'ok': True, 'message': '当前无运行中的预载任务'})

# ── 改进版离线瓦片预载 ──
# 石家庄市区及周边范围
SJZ_BOUNDS = {'lat_min': 37.85, 'lat_max': 38.25, 'lon_min': 114.20, 'lon_max': 114.80}
# 放大石家庄中心城区
SJZ_CORE = {'lat_min': 37.99, 'lat_max': 38.10, 'lon_min': 114.42, 'lon_max': 114.62}

def tile_xy(lat, lon, z):
    """经纬度转瓦片坐标"""
    import math as m
    n = 2.0 ** z
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - m.log(m.tan(m.radians(lat)) + 1.0 / m.cos(m.radians(lat))) / m.pi) / 2.0 * n)
    return x, y

def download_tile(z, x, y):
    """下载单张瓦片，返回成功与否"""
    tp = f'{TILE_DIR}/{z}/{x}/{y}.png'
    if os.path.exists(tp):
        return 'skipped'
    TILE_SOURCES = [
        'https://tile.openstreetmap.de/{z}/{x}/{y}.png',
        'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
        'http://tile.openstreetmap.org/{z}/{x}/{y}.png',
    ]
    os.makedirs(f'{TILE_DIR}/{z}/{x}', exist_ok=True)
    for src in TILE_SOURCES:
        url = src.replace('{z}', str(z)).replace('{x}', str(x)).replace('{y}', str(y))
        try:
            r = requests.get(url, headers={'User-Agent': 'RDK-CAR/2.0'}, timeout=8)
            if r.status_code == 200:
                with open(tp, 'wb') as f:
                    f.write(r.content)
                return 'ok'
        except: pass
    return 'error'

def run_preload(zoom_levels, bounds):
    """在后台线程中执行预载"""
    global preload_state
    preload_state['running'] = True
    preload_state['start_time'] = time.time()
    preload_state['downloaded'] = 0
    preload_state['skipped'] = 0
    preload_state['errors'] = 0
    preload_state['message'] = ''

    total_est = 0
    for z in zoom_levels:
        x1, y1 = tile_xy(bounds['lat_max'], bounds['lon_min'], z)
        x2, y2 = tile_xy(bounds['lat_min'], bounds['lon_max'], z)
        total_est += (abs(x2 - x1) + 1) * (abs(y2 - y1) + 1)
    preload_state['total'] = total_est

    for z in zoom_levels:
        preload_state['current_zoom'] = z
        x1, y1 = tile_xy(bounds['lat_max'], bounds['lon_min'], z)
        x2, y2 = tile_xy(bounds['lat_min'], bounds['lon_max'], z)
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        for x in range(x1, x2 + 1):
            for y in range(y1, y2 + 1):
                if not preload_state['running']:
                    preload_state['message'] = '已中止'
                    return
                result = download_tile(z, x, y)
                if result == 'ok':
                    preload_state['downloaded'] += 1
                elif result == 'skipped':
                    preload_state['skipped'] += 1
                else:
                    preload_state['errors'] += 1
                preload_state['message'] = f'Z{z}: ({x},{y})'

    elapsed = time.time() - preload_state['start_time']
    preload_state['message'] = f'完成！耗时 {elapsed:.1f}s，下载 {preload_state["downloaded"]}，已存在 {preload_state["skipped"]}，失败 {preload_state["errors"]}'
    preload_state['running'] = False

@app.route('/api/preload_tiles')
def api_preload_tiles():
    if preload_state['running']:
        return jsonify({'ok': False, 'message': '已在下载中，请等待完成'})

    zoom_param = request.args.get('zoom', 'full')
    scope_param = request.args.get('scope', 'city')

    if scope_param == 'core':
        bounds = SJZ_CORE
    else:
        bounds = SJZ_BOUNDS

    if zoom_param == 'full':
        # z8=总览, z9-z10=区域, z11-z13=城市, z14-z15=街道, z16=详细
        zoom_levels = [8, 9, 10, 11, 12, 13, 14, 15, 16]
    elif zoom_param == 'quick':
        zoom_levels = [10, 11, 12, 13, 14]
    elif zoom_param == 'street':
        zoom_levels = [14, 15, 16]
    else:
        try:
            zoom_levels = [int(zoom_param)]
        except:
            zoom_levels = [10, 11, 12, 13, 14, 15, 16]

    t = threading.Thread(target=run_preload, args=(zoom_levels, bounds), daemon=True)
    t.start()
    est = preload_state['total']
    return jsonify({
        'ok': True,
        'message': f'开始预载 {len(zoom_levels)} 个层级 (z{zoom_levels[0]}-z{zoom_levels[-1]})，预计 {est} 张瓦片',
        'zoom_levels': zoom_levels,
        'estimated_tiles': est
    })


@app.route('/api/geocode')
def api_geocode():
    q = request.args.get('q', '').strip()
    if not q:
        return jsonify({'results': []})
    import requests as req
    try:
        url = 'https://photon.komoot.io/api/?q=' + q + '&limit=5'
        r = req.get(url, headers={'User-Agent': 'RDK-CAR/2.0'}, timeout=10)
        if r.status_code == 200:
            data = r.json()
            results = []
            for f in data.get('features', []):
                c = f['geometry']['coordinates']
                p = f['properties']
                results.append({
                    'lat': c[1], 'lon': c[0],
                    'name': p.get('name', ''),
                    'city': p.get('city', ''),
                    'state': p.get('state', ''),
                    'country': p.get('country', ''),
                    'display': f"{p.get('name','')}, {p.get('city','')}, {p.get('country','')}"
                })
            return jsonify({'results': results})
        return jsonify({'error': 'no results', 'results': []})
    except Exception as e:
        return jsonify({'error': str(e), 'results': []})


@app.route('/api/music_list')
def api_music_list():
    import glob, os
    mp3s = []
    music_dir = '/music'
    if os.path.exists(music_dir):
        for f in sorted(os.listdir(music_dir)):
            if f.lower().endswith(('.mp3','.wav','.flac','.ogg','.aac')):
                mp3s.append({'title': os.path.splitext(f)[0], 'file': f, 'path': f'/music/{f}'})
    return jsonify(mp3s)

@app.route('/video_feed')
def video_feed():
    def gen():
        global running
        while running:
            with frame_lock:
                if latest_frame is not None:
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+latest_frame+b'\r\n')
                else:
                    f=np.zeros((480,640,3),dtype=np.uint8)
                    cv2.putText(f,"MIPI Camera Starting...",(120,240),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,229,255),2)
                    _,b=cv2.imencode('.jpg',f,[cv2.IMWRITE_JPEG_QUALITY,70])
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+b.tobytes()+b'\r\n')
            time.sleep(0.03)
    return Response(gen(),mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/snapshot')
def snapshot():
    with frame_lock:
        if latest_frame is not None:
            return Response(latest_frame, mimetype='image/jpeg')
    f = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.putText(f, "MIPI Camera Starting...", (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 229, 255), 2)
    _, b = cv2.imencode('.jpg', f, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return Response(b.tobytes(), mimetype='image/jpeg')

def cleanup(sig=None,frame=None):
    global running,cam; running=False
    if cam:
        try: cam.close_cam()
        except: pass
    if _car_available:
        try: _car.cleanup()
        except: pass
    print("[HMI] Shutdown"); sys.exit(0)

# ── 石家庄实时温度(Open-Meteo 免 key) ──
WEATHER_LAT = 38.0428
WEATHER_LON = 114.5149
WEATHER_UPDATE_S = 600
WEATHER_URL = f'https://api.open-meteo.com/v1/forecast?latitude={WEATHER_LAT}&longitude={WEATHER_LON}&current_weather=true'

# WMO weathercode → (emoji, 中文描述)
WMO_WEATHER = {
    0:('☀','晴'),1:('🌤','少云'),2:('⛅','多云'),3:('☁','阴'),
    45:('🌫','雾'),48:('🌫','雾凇'),
    51:('🌦','小毛毛雨'),53:('🌦','毛毛雨'),55:('🌧','大毛毛雨'),
    56:('🌧','冻毛毛雨'),57:('🌧','强冻毛毛雨'),
    61:('🌧','小雨'),63:('🌧','中雨'),65:('🌧','大雨'),
    66:('🌧','冻雨'),67:('🌧','强冻雨'),
    71:('🌨','小雪'),73:('🌨','中雪'),75:('❄','大雪'),77:('🌨','米雪'),
    80:('🌦','小阵雨'),81:('🌧','阵雨'),82:('⛈','强阵雨'),
    85:('🌨','小阵雪'),86:('❄','大阵雪'),
    95:('⛈','雷暴'),96:('⛈','雷雹'),99:('⛈','强雷雹'),
}

def fetch_weather_loop():
    while True:
        try:
            r = requests.get(WEATHER_URL, timeout=10)
            r.raise_for_status()
            cw = r.json().get('current_weather', {})
            t = cw.get('temperature')
            wc = cw.get('weathercode')
            if isinstance(t, (int, float)):
                car_data['temp'] = float(t)
            if isinstance(wc, int) and wc in WMO_WEATHER:
                em, txt = WMO_WEATHER[wc]
                car_data['weather'] = f'{em} {txt}'
            print(f'[WEATHER] 石家庄: {t}°C code={wc} → {car_data.get("weather")}')
        except Exception as _we:
            print(f'[WEATHER] 拉取失败: {_we!r}')
        time.sleep(WEATHER_UPDATE_S)

if __name__=='__main__':
    signal.signal(signal.SIGINT,cleanup); signal.signal(signal.SIGTERM,cleanup)
    # 车灯 GPIO 402 — 立即初始化,不依赖摄像头
    try:
        headlight = Headlight(HEADLIGHT_GPIOS)
    except Exception as _e:
        print(f'[HMI] Headlight init failed: {_e}')
    model,input_W,input_H=load_model()
    if HAVE_MIPI and model is not None:
        threading.Thread(target=mipi_capture_loop,args=(model,input_W,input_H),daemon=True).start()
        print("[HMI] MIPI + YOLOv5s started")
    else:
        print(f"[HMI] MIPI={HAVE_MIPI}, DNN={HAVE_DNN and model is not None} — demo mode")
    threading.Thread(target=fetch_weather_loop, daemon=True).start()
    print("[HMI] weather updater started (Open-Meteo)")
    print(f"[HMI] http://0.0.0.0:{HTTP_PORT}")
    app.run(host='0.0.0.0',port=HTTP_PORT,debug=False,threaded=True)
