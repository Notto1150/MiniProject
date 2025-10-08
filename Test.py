from flask import Flask, render_template, Response, request, url_for, redirect, send_from_directory, abort, flash, jsonify
import cv2
import os
from werkzeug.utils import secure_filename
import threading
import time
import datetime
import glob
import numpy as np
from scipy.spatial.distance import cosine 

# --- Import สำหรับ DeepFace ---
from deepface import DeepFace
# ------------------------------

# --- Import สำหรับการส่งอีเมล (SMTP) ---
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
# ---------------------------------------

app = Flask(__name__)
app.secret_key = 'your_super_secure_secret_key' 

# --- การตั้งค่าโฟลเดอร์ ---
UPLOAD_FOLDER = 'E:/AppServ/www/image'
VIDEO_FOLDER = 'E:/AppServ/www/Cam'
UNKNOWN_FOLDER = 'E:/AppServ/www/Unkown'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VIDEO_FOLDER, exist_ok=True)
os.makedirs(UNKNOWN_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['VIDEO_FOLDER'] = VIDEO_FOLDER
app.config['UNKNOWN_FOLDER'] = UNKNOWN_FOLDER
# -----------------------------

# --- การตั้งค่าอีเมล ---
SENDER_EMAIL = 'notsirapop1150@gmail.com'
SENDER_PASSWORD = 'jpvi xlib xfgh yryp' 
RECEIVER_EMAIL = 'sirapophandsomeboy@gmail.com'
SMTP_SERVER = 'smtp.gmail.com' 
SMTP_PORT = 465 
# ----------------------------------------------------

# --- การตั้งค่า DeepFace ---
templates_encodings = [] 
FACE_RECOGNITION_THRESHOLD = 0.45 
MODEL_NAME = "OpenFace" # ใช้ OpenFace เพื่อความเร็ว
DETECTOR_BACKEND = "opencv" 
# ------------------------------------

def load_encodings(folder: str):
    """อ่านภาพทั้งหมด, ตรวจจับใบหน้า, และสร้าง Face Embeddings ด้วย DeepFace"""
    exts = ("*.png", "*.jpg", "*.jpeg")
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(folder, e)))
        
    encodings = []
    
    for p in sorted(paths):
        try:
            results = DeepFace.represent(
                img_path=p, 
                model_name=MODEL_NAME, 
                enforce_detection=True, 
                detector_backend=DETECTOR_BACKEND
            )
        except Exception as e:
            print(f"⚠️ Error or No face detected in template {os.path.basename(p)}: {e}")
            continue
            
        if len(results) != 1:
            print(f"⚠️ Found {len(results)} faces in template: {os.path.basename(p)}. Skipping.")
            continue
            
        face_encoding = np.array(results[0]['embedding'])
        name_only = os.path.splitext(os.path.basename(p))[0]
        
        encodings.append((name_only, face_encoding)) 
        print(f"✅ Encoded template: {name_only} (DeepFace/{MODEL_NAME})")
        
    return encodings

templates_encodings = load_encodings(UPLOAD_FOLDER)

# ⚠️ แก้ไข: ตัวแปรนี้ควบคุมการบันทึกภาพ Unknown ไม่ใช่การส่งอีเมล
last_save_time = {'time': 0, 'lock': threading.Lock()} 

# 🟢 เพิ่ม: ตัวแปรนี้ควบคุมการส่งอีเมล
last_email_time = {'time': 0, 'lock': threading.Lock()}
EMAIL_INTERVAL_SECONDS = 60 # 60 วินาที = 1 นาที

def detect_and_recognize_faces(frame, threshold=FACE_RECOGNITION_THRESHOLD, save_unknown=True, save_interval=5):
    """
    ตรวจจับใบหน้าด้วย DeepFace.extract_faces, 
    คำนวณ encoding, และเปรียบเทียบกับเทมเพลตที่เก็บไว้
    """
    
    current_time = time.time()
    detections = []
    
    known_encodings = [enc for label, enc in templates_encodings]
    known_names = [label for label, enc in templates_encodings]

    # 1. ตรวจจับใบหน้าทั้งหมดในเฟรม
    try:
        extracted_faces = DeepFace.extract_faces(
            img_path=frame, 
            detector_backend=DETECTOR_BACKEND, 
            enforce_detection=False 
        )
    except Exception as e:
        return []

    for face_info in extracted_faces:
        x, y, w, h = face_info['facial_area']['x'], face_info['facial_area']['y'], face_info['facial_area']['w'], face_info['facial_area']['h']
        face_crop = frame[y:y+h, x:x+w]
        
        best_label = "Unknown"
        best_distance = 1.0 
        face_encoding = None

        if face_crop.size > 0:
            try:
                # 2. สร้าง Embedding ของใบหน้าในเฟรม
                results = DeepFace.represent(
                    img_path=face_crop, 
                    model_name=MODEL_NAME, 
                    enforce_detection=False 
                )
                
                if results and len(results) >= 1:
                    face_encoding = np.array(results[0]['embedding'])
                    
            except Exception as e:
                pass 

        if face_encoding is not None and known_encodings:
            # 3. เปรียบเทียบระยะห่าง (Cosine Distance)
            distances = [cosine(known_enc, face_encoding) for known_enc in known_encodings]

            best_match_index = np.argmin(distances)
            best_distance = distances[best_match_index]
            
            # 4. ตรวจสอบ Threshold
            if best_distance < threshold:
                best_label = known_names[best_match_index]
        
        # 5. บันทึก Unknown Face (ควบคุมโดย last_save_time)
        if best_label == "Unknown" and save_unknown:
            with last_save_time['lock']:
                # บันทึกภาพใหม่ทุกๆ 5 วินาที
                if current_time - last_save_time['time'] >= save_interval:
                    
                    unknown_files = [f for f in os.listdir(UNKNOWN_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    if len(unknown_files) >= 10:
                        print(f"--- พบใบหน้าไม่รู้จักครบ {len(unknown_files)} ภาพแล้ว ทำการลบภาพทั้งหมดและเริ่มบันทึกใหม่ ---")
                        for f in unknown_files:
                            try: os.remove(os.path.join(UNKNOWN_FOLDER, f))
                            except Exception as e: print(f"Error deleting old unknown file {f}: {e}")
                    
                    face_bgr_crop = frame[y:y+h, x:x+w] 
                    
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = os.path.join(UNKNOWN_FOLDER, f"unknown_{timestamp}.jpg")
                    cv2.imwrite(filename, face_bgr_crop) 
                    
                    print(f"Saved unknown face: {filename}")
                    last_save_time['time'] = current_time
        
        # 6. ส่งพิกัดใบหน้าที่ตรวจจับได้และ Label ออกไป
        detections.append((x, y, w, h, best_label))
        
    return detections

# ฟังก์ชันวาดข้อความที่พลิกภาพ (ไม่เปลี่ยนแปลง)
def put_flipped_text(frame, text, x, y, font=cv2.FONT_HERSHEY_SIMPLEX,
                     font_scale=1.0, color=(0, 255, 0), thickness=3):
    ((text_w, text_h), _) = cv2.getTextSize(text, font, font_scale, thickness)
    text_img = np.zeros((text_h + 20, text_w + 20, 3), dtype=np.uint8)

    cv2.putText(text_img, text, (10, text_h + 10), font, font_scale, (0, 0, 0), thickness + 2, lineType=cv2.LINE_AA)
    cv2.putText(text_img, text, (10, text_h + 10), font, font_scale, color, thickness, lineType=cv2.LINE_AA)

    flipped_text_img = cv2.flip(text_img, 1)

    h, w, _ = flipped_text_img.shape
    x = max(0, x - w)
    y = max(h, y)

    if y - h < 0 or x + w > frame.shape[1]:
        return

    roi = frame[y - h:y, x:x + w]
    mask = np.any(flipped_text_img != [0, 0, 0], axis=2)
    roi[mask] = flipped_text_img[mask]

    frame[y - h:y, x:x + w] = roi


def draw_detections(frame, detections):
    """วาดกล่องและป้ายชื่อบนเฟรมที่พลิกแล้ว"""
    for x, y, w, h, label in detections:
        color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        put_flipped_text(frame, label, x, y - 10, color=color)

# เปิดกล้อง (ไม่เปลี่ยนแปลง)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ ERROR: Failed to open camera with index 0. Trying index 1...")
    cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("❌ FATAL ERROR: Cannot open any camera.")

# ... (ส่วนอื่น ๆ ของ Flask routes และ functions ที่ไม่เกี่ยวข้องกับการประมวลผลวิดีโอ/อีเมล) ...

def get_unique_filename(folder, filename):
    """สร้างชื่อไฟล์ที่ไม่ซ้ำกัน"""
    name, ext = os.path.splitext(filename)
    counter = 1
    new_filename = filename
    while os.path.exists(os.path.join(folder, new_filename)):
        new_filename = f"{name}_{counter}{ext}"
        counter += 1
    return new_filename

# ตัวแปรสำหรับการบันทึกวิดีโอ
recording = False
out = None
record_thread = None

def record_video():
    """ฟังก์ชันสำหรับบันทึกวิดีโอในเธรดแยก"""
    global recording, out, cap
    
    fourcc = cv2.VideoWriter_fourcc(*'avc1') 
    
    if not cap.isOpened():
        print("Error: Camera not open for recording.")
        recording = False 
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(VIDEO_FOLDER, f"record_{timestamp}.mp4")
    
    try:
        out = cv2.VideoWriter(filename, fourcc, 20.0, (width, height))
    except Exception as e:
        print(f"Error creating VideoWriter: {e}")
        recording = False
        return
        
    print(f'Start recording: {filename}')
    
    while recording:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame for recording.")
            break
        frame = cv2.flip(frame, 1) 
        out.write(frame)
        time.sleep(0.05) 
        
    if out:
        out.release()
    print('Stopped recording.')

# ------------------------------------
# --- Routes สำหรับ API ควบคุมวิดีโอ ---
# ------------------------------------
@app.route('/start_recording', methods=['POST'])
def start_recording():
    global recording, record_thread
    if recording:
        return jsonify({'status': 'already recording'}), 400
    if not cap.isOpened():
        return jsonify({'status': 'camera not open'}), 500
        
    recording = True
    record_thread = threading.Thread(target=record_video)
    record_thread.start()
    return jsonify({'status': 'recording started'})

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    global recording, record_thread
    if not recording:
        return jsonify({'status': 'not recording'}), 400
    recording = False
    if record_thread and record_thread.is_alive():
        record_thread.join() 
    return jsonify({'status': 'recording stopped'})
# ------------------------------------

# ------------------------------------
# --- Routes สำหรับหน้าเว็บ ---
# ------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/camera')
def camera():
    is_camera_open = cap.isOpened()
    return render_template('camera.html', is_camera_open=is_camera_open)

@app.route('/upload', methods=['GET'])
def upload():
    images = os.listdir(app.config['UPLOAD_FOLDER'])
    images = [img for img in images if img.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    success = request.args.get('success')
    filename = request.args.get('filename')
    return render_template('upload.html', images=images, success=success, filename=filename)

@app.route('/upload', methods=['POST'])
def upload_image():
    global templates_encodings 
    
    if 'image_file' not in request.files:
        flash('ไม่พบไฟล์ที่อัปโหลด', 'error')
        return redirect(url_for('upload'))
        
    file = request.files['image_file']
    
    if file.filename == '':
        flash('ไม่ได้เลือกไฟล์', 'error')
        return redirect(url_for('upload'))
        
    filename = secure_filename(file.filename)
    filename = get_unique_filename(app.config['UPLOAD_FOLDER'], filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path) 
    
    templates_encodings = load_encodings(app.config['UPLOAD_FOLDER'])
    
    name_only = os.path.splitext(filename)[0]
    is_template_created = any(t[0] == name_only for t in templates_encodings)

    if is_template_created:
        flash(f'✅ อัปโหลดและประมวลผลไฟล์ **{filename}** เป็นเทมเพลต (Face Encoding) เรียบร้อยแล้ว', 'success')
    else:
        flash(f'⚠️ อัปโหลดไฟล์ **{filename}** แล้ว แต่ **ไม่พบใบหน้าเดียว** ในภาพ ภาพนี้จะไม่ถูกใช้เป็นเทมเพลต', 'warning')
        
    return redirect(url_for('upload', success='true', filename=filename))

@app.route('/delete/<filename>', methods=['POST'])
def delete_image(filename):
    global templates_encodings
    filename = secure_filename(filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if os.path.exists(file_path):
        os.remove(file_path)
        templates_encodings = load_encodings(app.config['UPLOAD_FOLDER']) 
        flash(f'🗑️ ลบรูป **{filename}** เรียบร้อยแล้ว', 'info')
        return redirect(url_for('upload'))
    else:
        abort(404)

@app.route('/delete_all', methods=['POST'])
def delete_all_images():
    global templates_encodings
    images = os.listdir(app.config['UPLOAD_FOLDER'])
    count = 0
    for img in images:
        if img.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            try:
                os.remove(os.path.join(app.config['UPLOAD_FOLDER'], img))
                count += 1
            except Exception as e:
                print(f'Error deleting file {img}: {e}')
                
    templates_encodings = load_encodings(app.config['UPLOAD_FOLDER'])
    
    flash(f'🗑️ ลบรูปทั้งหมด {count} ไฟล์เรียบร้อยแล้ว', 'info')
    return redirect(url_for('upload'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/videos')
def videos():
    files = os.listdir(app.config['VIDEO_FOLDER'])
    videos = [v for v in files if v.lower().endswith(('.mp4', '.avi', '.mov'))]
    videos_with_time = []
    for v in videos:
        path = os.path.join(app.config['VIDEO_FOLDER'], v)
        try:
            timestamp = os.path.getmtime(path)
            dt = datetime.datetime.fromtimestamp(timestamp)
            formatted_time = dt.strftime('%Y-%m-%d %H:%M:%S')
        except OSError:
            formatted_time = 'N/A' 
        videos_with_time.append({
            'filename': v,
            'time': formatted_time
        })
    return render_template('videos.html', videos=videos_with_time)

@app.route('/videos/<filename>')
def video_file(filename):
    filename = secure_filename(filename)
    return send_from_directory(app.config['VIDEO_FOLDER'], filename)

@app.route('/delete_video/<filename>', methods=['POST'])
def delete_video(filename):
    filename = secure_filename(filename)
    file_path = os.path.join(app.config['VIDEO_FOLDER'], filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        flash(f'🗑️ ลบวิดีโอ **{filename}** เรียบร้อยแล้ว', 'info')
        return redirect(url_for('videos'))
    else:
        abort(404)

@app.route('/video_feed')
def video_feed():
    if not cap.isOpened():
        return Response("Camera Not Available", status=500, mimetype='text/plain')
        
    try:
        threshold = float(request.args.get('threshold', FACE_RECOGNITION_THRESHOLD)) 
    except:
        threshold = FACE_RECOGNITION_THRESHOLD

    def generate():
        
        process_this_frame = 0
        SKIP_FRAMES = 5
        detections = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_width = frame.shape[1]
            
            process_this_frame += 1
            
            if process_this_frame % SKIP_FRAMES == 0:
                process_this_frame = 0
                
                frame_resized = cv2.resize(frame, (480, 270)) 
                
                detections_resized = detect_and_recognize_faces(frame_resized, threshold=threshold)
                
                scale_x = frame.shape[1] / frame_resized.shape[1]
                scale_y = frame.shape[0] / frame_resized.shape[0]
                
                detections = []
                for x, y, w, h, label in detections_resized:
                    x_full = int(x * scale_x)
                    y_full = int(y * scale_y)
                    w_full = int(w * scale_x)
                    h_full = int(h * scale_y)
                    detections.append((x_full, y_full, w_full, h_full, label))
            
            frame = cv2.flip(frame, 1)

            flipped_detections = []
            for x, y, w, h, label in detections: 
                flipped_x = frame_width - (x + w) 
                flipped_detections.append((flipped_x, y, w, h, label))

            draw_detections(frame, flipped_detections)

            ret2, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
# ----------------------------------------------------------------------
# --- โค้ดสำหรับการแจ้งเตือนด้วย Email (SMTP) ---
# ----------------------------------------------------------------------

def send_notification_email(filename, image_path):
    """ฟังก์ชันส่งอีเมลแจ้งเตือนพร้อมแนบรูปภาพ Unknown Face"""
    print(f"--- กำลังเตรียมส่งอีเมลแจ้งเตือนสำหรับ Unknown Face: {filename} ---")
    
    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECEIVER_EMAIL
    msg['Subject'] = f"🚨 แจ้งเตือน: พบใบหน้าไม่รู้จัก ({filename})"

    body = f"เรียน ผู้ดูแลระบบ,\n\nตรวจพบใบหน้าไม่รู้จักในเวลา {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\nโปรดตรวจสอบไฟล์ที่แนบมา\n"
    msg.attach(MIMEText(body, 'plain'))

    try:
        with open(image_path, 'rb') as fp:
            img = MIMEImage(fp.read(), _subtype="jpeg")
            img.add_header('Content-Disposition', 'attachment', filename=filename)
            msg.attach(img)

        server = smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT)
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
        server.quit()
        
        print("✅ ส่งอีเมลแจ้งเตือนสำเร็จ")
        
    except smtplib.SMTPAuthenticationError:
        print("❌ ข้อผิดพลาดในการยืนยันตัวตน SMTP: ตรวจสอบ App Password ว่าถูกต้องหรือไม่")
    except Exception as e:
        print(f"❌ Error ในการส่งอีเมล: {e}")


def monitor_unknown_folder():
    """ฟังก์ชันที่รันในเธรดแยกเพื่อเฝ้าดูโฟลเดอร์ Unknown และเรียกฟังก์ชันแจ้งเตือนด้วยอีเมลเมื่อมีไฟล์ใหม่ถูกสร้าง"""
    global last_email_time
    print("--- เริ่มต้นการมอนิเตอร์โฟลเดอร์ Unknown ---")
    
    last_files = set(os.listdir(app.config['UNKNOWN_FOLDER']))
    
    while True:
        try:
            current_files = set(os.listdir(app.config['UNKNOWN_FOLDER']))
            new_files = current_files - last_files
            
            # 🟢 แก้ไข: ตรวจสอบเงื่อนไขการส่งอีเมล
            if new_files:
                current_time = time.time()
                
                with last_email_time['lock']:
                    # ตรวจสอบว่าเกิน 1 นาที (60 วินาที) นับจากอีเมลล่าสุดหรือไม่
                    if current_time - last_email_time['time'] >= EMAIL_INTERVAL_SECONDS:
                        
                        # เลือกไฟล์ที่ใหม่ที่สุดในชุด new_files สำหรับแนบอีเมล
                        latest_filename = sorted(list(new_files))[-1] 
                        file_path = os.path.join(app.config['UNKNOWN_FOLDER'], latest_filename)
                        
                        # รอให้ไฟล์ถูกเขียนเสร็จสมบูรณ์ก่อนส่ง
                        time.sleep(1) 
                        send_notification_email(latest_filename, file_path)
                        
                        # อัปเดตเวลาส่งอีเมลล่าสุด
                        last_email_time['time'] = current_time
                        
                    else:
                        remaining_time = EMAIL_INTERVAL_SECONDS - (current_time - last_email_time['time'])
                        # print(f"ℹ️ พบไฟล์ใหม่ แต่ต้องรออีก {remaining_time:.1f} วินาที ก่อนส่งอีเมลใหม่")

            # ⚠️ อัปเดตรายการไฟล์ล่าสุดเสมอเพื่อป้องกันการแจ้งเตือนซ้ำ
            last_files = current_files
        
        except Exception as e:
            print(f"Error ใน Unknown Folder Monitor: {e}")
        
        time.sleep(1) 

# ----------------------------------------------------------------------

monitor_thread = threading.Thread(target=monitor_unknown_folder, daemon=True)
monitor_thread.start()

if __name__ == '__main__':
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=True, threaded=True, use_reloader=False) 
    except KeyboardInterrupt:
        print("Server shutting down...")
    finally:
        if cap.isOpened():
            cap.release()
            print("Camera released.")