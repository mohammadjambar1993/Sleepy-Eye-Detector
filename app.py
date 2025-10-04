import os
import cv2
import time
import wave
import shutil
import subprocess
import warnings
import numpy as np
import mediapipe as mp

# ——— Hilangkan warning protobuf yang muncul di log ———
warnings.filterwarnings("ignore", message="SymbolDatabase.GetPrototype")

# ====== KONFIGURASI ======
ALARM_PATH = os.path.abspath("alarm.wav")
EAR_THRESHOLD = 0.20        # Ambang EAR: < threshold dianggap mata tertutup
CLOSED_TIME = 0.8           # Detik mata tertutup sebelum alarm aktif
RETRIGGER_INTERVAL = 6.0    # Detik, jeda picu alarm berikutnya saat masih tertutup

CAM_INDEX = 0               # Index kamera (0 = default)
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
TARGET_FPS = 30
ENABLE_AUDIO = True         # Set False jika ingin visual-only

# ====== MUAT AUDIO ALARM ======
AUDIO_BYTES = b""
SR = 0
CHANNELS = 0
SAMPLE_WIDTH = 0

if os.path.exists(ALARM_PATH):
    with wave.open(ALARM_PATH, "rb") as wf:
        SR = wf.getframerate()
        CHANNELS = wf.getnchannels()
        SAMPLE_WIDTH = wf.getsampwidth()
        AUDIO_BYTES = wf.readframes(wf.getnframes())
else:
    print(f"[PERINGATAN] File alarm tidak ditemukan: {ALARM_PATH}. Alarm audio dinonaktifkan.")
    ENABLE_AUDIO = False

# Untuk backend sounddevice (butuh ndarray int16, shape (N, C) bila stereo)
AUDIO_INT16 = np.frombuffer(AUDIO_BYTES, dtype=np.int16) if AUDIO_BYTES else np.array([], dtype=np.int16)
if AUDIO_INT16.size and CHANNELS > 1:
    # reshape interleaved ke (frames, channels)
    AUDIO_INT16 = AUDIO_INT16.reshape(-1, CHANNELS)

# ====== BACKEND AUDIO: simpleaudio -> sounddevice -> aplay ======
_AUDIO_BACKEND = None
play_obj = None     # handle untuk simpleaudio
_aplay_proc = None  # handle proses aplay

# Coba import simpleaudio dulu
try:
    import simpleaudio as sa
    _AUDIO_BACKEND = "simpleaudio"
except Exception:
    _AUDIO_BACKEND = None

def _fallback_start_alarm():
    """Coba backend alternatif: sounddevice lalu aplay. Return True jika sukses."""
    global _AUDIO_BACKEND, _aplay_proc

    # 1) sounddevice (PortAudio)
    try:
        import sounddevice as sd
        _AUDIO_BACKEND = "sounddevice"
        sd.stop()
        if AUDIO_INT16.size:
            sd.play(AUDIO_INT16, SR, blocking=False)
            return True
        else:
            print("[WARN] AUDIO_INT16 kosong; lewati sounddevice.")
    except Exception as e:
        print("[WARN] sounddevice gagal:", e)

    # 2) aplay (ALSA CLI)
    if shutil.which("aplay") and os.path.exists(ALARM_PATH):
        try:
            _AUDIO_BACKEND = "aplay"
            proc = subprocess.Popen(["aplay", "-q", ALARM_PATH])
            _aplay_proc = proc
            return True
        except Exception as e:
            print("[WARN] aplay gagal:", e)

    print("[WARN] Tidak ada backend audio yang berfungsi. Alarm audio dimatikan.")
    return False

alarm_playing = False

def start_alarm():
    """Mulai bunyikan alarm dengan backend terbaik yang tersedia."""
    global play_obj, alarm_playing, _AUDIO_BACKEND
    if not ENABLE_AUDIO:
        return
    if not AUDIO_BYTES:
        print("[WARN] alarm.wav tidak tersedia; lewati alarm.")
        return

    if _AUDIO_BACKEND == "simpleaudio":
        try:
            if play_obj is None or not play_obj.is_playing():
                play_obj = sa.play_buffer(AUDIO_BYTES, CHANNELS, SAMPLE_WIDTH, SR)
                alarm_playing = True
                return
        except Exception as e:
            print("[WARN] simpleaudio gagal:", e)

    # Fallback otomatis
    if _fallback_start_alarm():
        alarm_playing = True
    else:
        alarm_playing = False

def stop_alarm():
    """Hentikan alarm dari backend manapun."""
    global play_obj, _aplay_proc, alarm_playing

    # simpleaudio
    try:
        if play_obj is not None and hasattr(play_obj, "stop"):
            play_obj.stop()
    except Exception:
        pass
    play_obj = None

    # sounddevice
    try:
        import sounddevice as sd
        sd.stop()
    except Exception:
        pass

    # aplay
    try:
        if _aplay_proc is not None and _aplay_proc.poll() is None:
            _aplay_proc.terminate()
            _aplay_proc.wait(timeout=1)
    except Exception:
        pass
    _aplay_proc = None

    alarm_playing = False

# ====== INISIALISASI MEDIAPIPE FACEMESH ======
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Landmark kunci untuk perhitungan EAR (MediaPipe indices)
LEFT_EYE_IDX = {"left": 33, "right": 133, "top": 159, "bottom": 145}
RIGHT_EYE_IDX = {"left": 362, "right": 263, "top": 386, "bottom": 374}

# ====== STATE DETEKSI ======
ttl_closed = None           # timestamp saat mata mulai tertutup
last_alarm_time = None      # timestamp alarm terakhir dipicu

def calculate_ear(landmarks, idxs, w, h):
    """Eye Aspect Ratio sederhana: (jarak vertikal) / (jarak horizontal)."""
    L = np.array([landmarks[idxs["left"]].x * w,
                  landmarks[idxs["left"]].y * h])
    R = np.array([landmarks[idxs["right"]].x * w,
                  landmarks[idxs["right"]].y * h])
    T = np.array([landmarks[idxs["top"]].x * w,
                  landmarks[idxs["top"]].y * h])
    B = np.array([landmarks[idxs["bottom"]].x * w,
                  landmarks[idxs["bottom"]].y * h])
    denom = np.linalg.norm(L - R)
    if denom == 0:
        return 1.0
    return np.linalg.norm(T - B) / denom

def process_frame(frame_bgr):
    """Proses 1 frame: hitung EAR, kelola alarm, dan overlay anotasi."""
    global ttl_closed, last_alarm_time, alarm_playing

    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    ear = 1.0

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark
        ear_left = calculate_ear(lm, LEFT_EYE_IDX, w, h)
        ear_right = calculate_ear(lm, RIGHT_EYE_IDX, w, h)
        ear = (ear_left + ear_right) / 2.0

        # Gambar "kontur" sederhana (empat titik per mata)
        for eye in (LEFT_EYE_IDX, RIGHT_EYE_IDX):
            pts = np.array([
                (int(lm[eye["left"]].x * w),   int(lm[eye["left"]].y * h)),
                (int(lm[eye["top"]].x * w),    int(lm[eye["top"]].y * h)),
                (int(lm[eye["right"]].x * w),  int(lm[eye["right"]].y * h)),
                (int(lm[eye["bottom"]].x * w), int(lm[eye["bottom"]].y * h)),
            ], dtype=np.int32)
            cv2.polylines(frame_bgr, [pts], isClosed=True, color=(0, 255, 0), thickness=1)

    # Logika deteksi kantuk
    if ear < EAR_THRESHOLD:
        if ttl_closed is None:
            ttl_closed = time.time()

        elapsed = time.time() - ttl_closed
        cv2.putText(frame_bgr, "EYES CLOSED", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame_bgr, f"Closed: {elapsed:.1f}s", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Jika sudah melewati durasi, picu alarm tiap RETRIGGER_INTERVAL
        if elapsed >= CLOSED_TIME:
            now = time.time()
            need_trigger = (last_alarm_time is None) or ((now - last_alarm_time) >= RETRIGGER_INTERVAL)
            if need_trigger and not alarm_playing:
                start_alarm()
                last_alarm_time = now

            # Overlay tanda ALARM besar agar tetap terlihat di mode visual-only
            cv2.putText(frame_bgr, "ALARM!", (30, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
    else:
        # Mata terbuka -> reset timer & hentikan alarm jika masih bunyi
        if alarm_playing:
            stop_alarm()
        ttl_closed = None
        last_alarm_time = None


    cv2.putText(frame_bgr, f"EAR: {ear:.2f}", (30, 170),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return frame_bgr

def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    if not cap.isOpened():
        print("Tidak bisa membuka kamera. Coba ganti CAM_INDEX.")
        return

    print("Tekan 'q' untuk keluar.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Gagal membaca frame dari kamera.")
                break

            annotated = process_frame(frame)
            cv2.imshow("Deteksi Ngantuk", annotated)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        try:
            stop_alarm()
        except Exception:
            pass
        cap.release()
        cv2.destroyAllWindows()
        try:
            face_mesh.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()

