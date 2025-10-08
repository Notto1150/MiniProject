from flask import Flask, render_template, Response, request, url_for, redirect, send_from_directory, abort, flash, jsonify
import cv2
import os
from werkzeug.utils import secure_filename
import threading
import time
import datetime
import glob
import numpy as np

# --- Import สำหรับการส่งอีเมล (SMTP) ---
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
# ---------------------------------------

app = Flask(__name__)
# แนะนำให้ใช้คีย์ที่ซับซ้อนขึ้นใน production
app.secret_key = 'your_secret_key' 

# --- การตั้งค่าโฟลเดอร์ ---
# ⚠️ เปลี่ยนเส้นทางเหล่านี้ตามโครงสร้างเครื่องของคุณ
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

# --- การตั้งค่าอีเมล (⚠️ ต้องเปลี่ยนเป็นข้อมูลของคุณ) ---
SENDER_EMAIL = 'notsirapop1150@gmail.com'
SENDER_PASSWORD = 'jpvi xlib xfgh yryp' # 👈 App Password ของคุณ
RECEIVER_EMAIL = 'sirapophandsomeboy@gmail.com'
SMTP_SERVER = 'smtp.gmail.com' 
SMTP_PORT = 465 
# ----------------------------------------------------

# --- การตั้งค่าเฉพาะสำหรับการประมวลผลภาพที่อัปโหลด ---
TARGET_SIZE = (150, 150) # ขนาดเป้าหมายสำหรับ Resize ในฟังก์ชัน upload_image
# ----------------------------------------------------

# --- การตั้งค่าสำหรับการบันทึก Unknown Face และการแจ้งเตือน ---
# ⚠️ ควบคุมความถี่การแจ้งเตือนอีเมล (หน่วยเป็นวินาที)
EMAIL_NOTIFICATION_INTERVAL = 10 
# ตัวแปรบันทึกเวลาบันทึกภาพ unknown เพื่อควบคุมความถี่ในการบันทึกภาพ
last_save_time = [0] 
# ตัวแปรบันทึกเวลาส่งอีเมลล่าสุด เพื่อควบคุมความถี่ในการส่งอีเมล
last_email_time = [0]
# -------------------------------------------------------------

# โหลดเทมเพลตหน้า (แปลงเป็น Grayscale และครอบตัด)
def load_templates(folder: str):
    """โหลดภาพเทมเพลตทั้งหมด, แปลงเป็น Grayscale, ตรวจจับใบหน้า, และครอบตัด"""
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(folder, e)))
    templates = []
    face_cascade_local = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    for p in sorted(paths):
        img = cv2.imread(p)
        if img is None: 
            continue
            
        # 1. แปลงเป็น Grayscale 
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # 2. ตรวจจับใบหน้า
        faces = face_cascade_local.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) == 0:
            print(f"No face detected in template: {os.path.basename(p)}")
            continue
            
        # เลือกใบหน้าแรกที่ตรวจพบ
        x, y, w, h = faces[0]
        
        # 3. ครอบตัดเฉพาะภาพใบหน้า (เป็น Grayscale)
        # ปรับขนาดขยายเล็กน้อยเพื่อเพิ่มขอบ
        margin = 10 
        y1, y2 = max(0, y - margin), min(gray.shape[0], y + h + margin)
        x1, x2 = max(0, x - margin), min(gray.shape[1], x + w + margin)
        
        face_crop = gray[y1:y2, x1:x2]
        
        templates.append((os.path.basename(p), face_crop))
        print(f"Loaded template: {os.path.basename(p)}, face size: {face_crop.shape}")
        
    return templates

# โหลดเทมเพลตและ Cascade Classifier สำหรับใช้ทั่วทั้งแอป
templates = load_templates(UPLOAD_FOLDER)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ฟังก์ชัน matching แบบ multi scale
def multi_scale_match(face_gray: np.ndarray, template_gray: np.ndarray, scales: list):
    """ทำการ Template Matching โดยการปรับขนาดเทมเพลตหลายขนาดเพื่อหาคะแนนที่ดีที่สุด"""
    best_score = 0
    h_face, w_face = face_gray.shape[:2]
    
    for s in scales:
        th, tw = max(1, int(template_gray.shape[0] * s)), max(1, int(template_gray.shape[1] * s))
        
        if th > h_face or tw > w_face: 
            continue # ขนาดเทมเพลตใหญ่เกินกว่าใบหน้า
        
        t_resized = cv2.resize(template_gray, (tw, th))
        
        # ใช้ TM_CCOEFF_NORMED สำหรับเปรียบเทียบความคล้ายคลึง
        res = cv2.matchTemplate(face_gray, t_resized, cv2.TM_CCOEFF_NORMED) 
        _, max_val, _, _ = cv2.minMaxLoc(res)
        
        if max_val > best_score: 
            best_score = max_val
            
    return best_score

# --- ล็อกสำหรับป้องกัน Race Condition ในการเข้าถึงโฟลเดอร์ Unknown/Email ---
# ใช้สำหรับควบคุมการบันทึกเมื่อมีหลายใบหน้า Unknown ในเฟรมเดียวกัน
unknown_lock = threading.Lock()
# --------------------------------------------------------------------------

def detect_face_templates(frame, threshold=0.5, scales=[0.8,0.9,1.0,1.1,1.2], min_size=50, save_unknown=True, save_interval=3):
    """ตรวจจับใบหน้าและเปรียบเทียบกับเทมเพลต"""
    global last_save_time
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_size, min_size))
    detections = []
    current_time = time.time()

    for i, (x,y,w,h) in enumerate(faces):
        
        # ปรับขนาดขยายเล็กน้อยเพื่อเพิ่มขอบให้กับ ROI ที่จะใช้ Template Matching
        margin = 10
        y1, y2 = max(0, y - margin), min(gray.shape[0], y + h + margin)
        x1, x2 = max(0, x - margin), min(gray.shape[1], x + w + margin)
        face_roi = gray[y1:y2, x1:x2]
        
        # ปรับพิกัด x, y, w, h ให้เป็นพิกัดเดิมของใบหน้า (ไม่รวม margin) สำหรับวาดกล่อง
        x_orig, y_orig, w_orig, h_orig = x, y, w, h
        
        best_label = "Unknown"
        best_score = 0
        
        # Template Matching
        for label, tmpl in templates:
            score = multi_scale_match(face_roi, tmpl, scales)
            
            if score > best_score:
                best_score = score
                if score >= threshold:
                    # ถ้าคะแนนสูงกว่า Threshold ให้กำหนดป้ายชื่อ
                    name = os.path.splitext(label)[0]
                    best_label = name
                    
        # --- ✅ การแก้ไข: บันทึก Unknown Face ทันทีเมื่อตรวจพบ (Real-Time Saving) ---
        if best_label == "Unknown" and save_unknown:
            # ใช้ last_save_time เพื่อควบคุมความถี่ **การบันทึกภาพ** if last_save_time[0] == 0 or (current_time - last_save_time[0] >= save_interval):
                with unknown_lock:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = os.path.join(UNKNOWN_FOLDER, f"unknown_{timestamp}.jpg")
                    
                    # บันทึกเฉพาะส่วนใบหน้า Unknown (ที่ครอบตัดด้วย margin)
                    cv2.imwrite(filename, face_roi) 
                    
                    print(f"Saved unknown face: {filename}")
                    last_save_time[0] = current_time # อัปเดตเวลาบันทึกภาพล่าสุด
        # ----------------------------------------------------------------------
                    
        # ใช้พิกัดเดิมสำหรับส่งออก
        detections.append((x_orig, y_orig, w_orig, h_orig, best_label))
        
    return detections

# ฟังก์ชันวาดข้อความที่พลิกภาพ (ไม่มีการเปลี่ยนแปลง)
def put_flipped_text(frame, text, x, y, font=cv2.FONT_HERSHEY_SIMPLEX,
                     font_scale=1.0, color=(0, 255, 0), thickness=3):
    ((text_w, text_h), _) = cv2.getTextSize(text, font, font_scale, thickness)
    text_img = np.zeros((text_h + 20, text_w + 20, 3), dtype=np.uint8)

    # วาดขอบดำหนา (outline)
    cv2.putText(text_img, text, (10, text_h + 10), font, font_scale, (0, 0, 0), thickness + 2, lineType=cv2.LINE_AA)
    # วาดข้อความจริงสีที่ต้องการทับบนขอบ
    cv2.putText(text_img, text, (10, text_h + 10), font, font_scale, color, thickness, lineType=cv2.LINE_AA)

    # พลิกข้อความในแนวนอนเพื่อให้ดูปกติในวิดีโอที่ถูกพลิก
    flipped_text_img = cv2.flip(text_img, 1)

    h, w, _ = flipped_text_img.shape
    x = max(0, x - w)
    y = max(h, y)

    if y - h < 0 or x + w > frame.shape[1]:
        return

    roi = frame[y - h:y, x:x + w]

    # แปะแบบ non-black pixel (ข้อความและขอบ)
    mask = np.any(flipped_text_img != [0, 0, 0], axis=2)
    roi[mask] = flipped_text_img[mask]

    frame[y - h:y, x:x + w] = roi


def draw_detections(frame, detections):
    """วาดกล่องและป้ายชื่อบนเฟรมที่พลิกแล้ว"""
    for x, y, w, h, label in detections:
        color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        put_flipped_text(frame, label, x, y - 10, color=color)

# เปิดกล้อง
cap = cv2.VideoCapture(0)

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
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    if not cap.isOpened():
        print("Error: Camera not open for recording.")
        recording = False # หยุดสถานะการบันทึกหากกล้องไม่มี
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
    
    # ดึงเฟรมจากกล้องเพื่อบันทึก
    while recording:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame for recording.")
            break
        frame = cv2.flip(frame, 1) # พลิกภาพให้ตรงกับหน้าจอ
        out.write(frame)
        time.sleep(0.05) # หน่วงเวลาเล็กน้อย
        
    if out:
        out.release()
    print('Stopped recording.')

# --- Routes สำหรับ API ควบคุมวิดีโอ (ไม่มีการเปลี่ยนแปลง) ---
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
    # รอให้เธรดบันทึกจบ
    if record_thread and record_thread.is_alive():
        record_thread.join() 
    return jsonify({'status': 'recording stopped'})
# ------------------------------------

# --- Routes สำหรับหน้าเว็บ (ไม่มีการเปลี่ยนแปลงหลัก) ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/camera')
def camera():
    # ตรวจสอบว่ากล้องพร้อมหรือไม่
    is_camera_open = cap.isOpened()
    return render_template('camera.html', is_camera_open=is_camera_open)

@app.route('/upload', methods=['GET'])
def upload():
    images = os.listdir(app.config['UPLOAD_FOLDER'])
    # กรองเฉพาะไฟล์ภาพที่รองรับ
    images = [img for img in images if img.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    success = request.args.get('success')
    filename = request.args.get('filename')
    return render_template('upload.html', images=images, success=success, filename=filename)

@app.route('/upload', methods=['POST'])
def upload_image():
    global templates 
    
    if 'image_file' not in request.files:
        flash('ไม่พบไฟล์ที่อัปโหลด', 'error')
        return redirect(url_for('upload'))
        
    file = request.files['image_file']
    
    if file.filename == '':
        flash('ไม่ได้เลือกไฟล์', 'error')
        return redirect(url_for('upload'))
        
    # 1. บันทึกไฟล์ต้นฉบับ
    filename = secure_filename(file.filename)
    filename = get_unique_filename(app.config['UPLOAD_FOLDER'], filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path) 
    
    # --- ส่วนที่แก้ไข: การประมวลผลภาพหลังอัปโหลด (Grayscale + Resize) ---
    try:
        # โหลดภาพที่เพิ่งบันทึก
        img = cv2.imread(save_path)
        if img is None:
            flash(f'❌ ไม่สามารถโหลดไฟล์ภาพ **{filename}** ได้', 'error')
            os.remove(save_path) # ลบไฟล์ที่ผิดพลาดออก
            return redirect(url_for('upload'))

        # 1. แปลงเป็น Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. ปรับขนาด (Resize) 
        # ใช้ TARGET_SIZE ที่กำหนดไว้ด้านบน
        resized_gray = cv2.resize(gray, TARGET_SIZE, interpolation=cv2.INTER_AREA)

        # 3. บันทึกภาพที่ประมวลผลแล้วทับไฟล์ต้นฉบับ
        cv2.imwrite(save_path, resized_gray) 
        print(f"Processed image saved: {save_path} (Grayscale, {TARGET_SIZE})")

    except Exception as e:
        print(f"Error processing image {filename}: {e}")
    # ----------------------------------------------------------------------
    
    # 2. โหลดเทมเพลตใหม่ทั้งหมดทันที
    templates_before = len(templates)
    templates = load_templates(app.config['UPLOAD_FOLDER'])
    templates_after = len(templates)
    
    # 3. ตรวจสอบว่าไฟล์ที่เพิ่งอัปโหลดมีการประมวลผลเป็นเทมเพลตหรือไม่ (ตรวจพบใบหน้าหรือไม่)
    is_template_created = any(t[0] == filename for t in templates)

    if is_template_created:
        flash(f'✅ อัปโหลด, ประมวลผล (Grayscale, {TARGET_SIZE[0]}x{TARGET_SIZE[1]}), และใช้ไฟล์ **{filename}** เป็นเทมเพลตเรียบร้อยแล้ว', 'success')
    else:
        # หากไฟล์ถูกบันทึกแต่ไม่เป็นเทมเพลต (เพราะไม่พบใบหน้า)
        flash(f'⚠️ อัปโหลดและประมวลผลไฟล์ **{filename}** แล้ว แต่ **ไม่พบใบหน้า** ในภาพ ภาพนี้จะไม่ถูกใช้เป็นเทมเพลต', 'warning')
        
    return redirect(url_for('upload', success='true', filename=filename))

@app.route('/delete/<filename>', methods=['POST'])
def delete_image(filename):
    global templates 
    filename = secure_filename(filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if os.path.exists(file_path):
        os.remove(file_path)
        
        # อัปเดตเทมเพลตทันทีหลังจากลบ
        templates = load_templates(app.config['UPLOAD_FOLDER']) 
        
        flash(f'🗑️ ลบรูป **{filename}** เรียบร้อยแล้ว', 'info')
        return redirect(url_for('upload'))
    else:
        abort(404)

@app.route('/delete_all', methods=['POST'])
def delete_all_images():
    global templates 
    images = os.listdir(app.config['UPLOAD_FOLDER'])
    count = 0
    for img in images:
        if img.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            try:
                os.remove(os.path.join(app.config['UPLOAD_FOLDER'], img))
                count += 1
            except Exception as e:
                print(f'Error deleting file {img}: {e}')
                
    # อัปเดตเทมเพลตทันทีหลังจากลบทั้งหมด
    templates = load_templates(app.config['UPLOAD_FOLDER'])
    
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
        # หากกล้องไม่พร้อม อาจจะ redirect หรือแสดงข้อความแจ้งเตือน
        return Response("Camera Not Available", status=500, mimetype='text/plain')
        
    try:
        threshold = float(request.args.get('threshold', 0.5))
    except:
        threshold = 0.5
    try:
        min_size = int(request.args.get('min_size', 50))
    except:
        min_size = 50

    scales = [0.8, 0.9, 1.0, 1.1, 1.2]

    def generate():
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 1. ตรวจจับใบหน้าก่อนพลิกภาพ (ใช้ภาพสี)
            detections = detect_face_templates(frame, threshold=threshold, scales=scales, min_size=min_size)

            # 2. พลิกภาพสำหรับแสดงผลในเว็บแคม
            frame = cv2.flip(frame, 1)

            # 3. ปรับตำแหน่งกล่องหลังพลิก
            flipped_detections = []
            for x, y, w, h, label in detections:
                # คำนวณตำแหน่ง x ใหม่หลังการพลิก
                flipped_x = frame_width - (x + w) 
                flipped_detections.append((flipped_x, y, w, h, label))

            # 4. วาดกล่องและข้อความที่พลิกกลับมาแล้ว
            draw_detections(frame, flipped_detections)

            ret2, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ----------------------------------------------------------------------
# --- โค้ดสำหรับการแจ้งเตือนด้วย Email (SMTP) ---
# ----------------------------------------------------------------------

def send_notification_email(filename, image_path):
    """
    ฟังก์ชันส่งอีเมลแจ้งเตือนพร้อมแนบรูปภาพ Unknown Face
    คืนค่าเป็น True หากส่งสำเร็จและลบไฟล์, False หากมีข้อผิดพลาด
    """
    print(f"--- กำลังเตรียมส่งอีเมลแจ้งเตือนสำหรับ Unknown Face: {filename} ---")
    
    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECEIVER_EMAIL
    msg['Subject'] = f"🚨 แจ้งเตือน: พบใบหน้าไม่รู้จัก ({filename})"

    body = f"เรียน ผู้ดูแลระบบ,\n\nตรวจพบใบหน้าไม่รู้จักในเวลา {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\nโปรดตรวจสอบไฟล์ที่แนบมา\n"
    msg.attach(MIMEText(body, 'plain'))
    
    email_sent_successfully = False

    try:
        # แนบรูปภาพ
        # ⚠️ ต้องแน่ใจว่าไฟล์ถูกเขียนเสร็จสมบูรณ์ก่อนเปิด
        if not os.path.exists(image_path):
             print(f"❌ Error: File not found at {image_path}")
             return False
             
        with open(image_path, 'rb') as fp:
            img = MIMEImage(fp.read(), _subtype="jpeg")
            img.add_header('Content-Disposition', 'attachment', filename=filename)
            msg.attach(img)

        # 🎯 ใช้ smtplib.SMTP_SSL สำหรับ Port 465 
        server = smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT)
        
        # ล็อกอิน
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        
        # ส่งอีเมล
        server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
        server.quit()
        
        print("✅ ส่งอีเมลแจ้งเตือนสำเร็จ")
        email_sent_successfully = True
        
    except smtplib.SMTPAuthenticationError:
        print("❌ ข้อผิดพลาดในการยืนยันตัวตน SMTP: ตรวจสอบ App Password ว่าถูกต้องหรือไม่")
    except Exception as e:
        print(f"❌ Error ในการส่งอีเมล: {e}")
        
    # --- ✅ การแก้ไข: ลบไฟล์หลังจากส่งอีเมลสำเร็จ ---
    if email_sent_successfully:
        try:
            os.remove(image_path)
            print(f"🗑️ ลบไฟล์ {filename} หลังจากส่งอีเมลสำเร็จ")
        except Exception as e:
            print(f"❌ Error ในการลบไฟล์ {filename}: {e}")
            
    return email_sent_successfully


def monitor_unknown_folder():
    """
    ฟังก์ชันที่รันในเธรดแยกเพื่อเฝ้าดูโฟลเดอร์ Unknown
    และเรียกฟังก์ชันแจ้งเตือนด้วยอีเมลเมื่อมีไฟล์ใหม่ถูกสร้าง (ควบคุมความถี่การส่งอีเมล)
    """
    global last_email_time
    print("--- เริ่มต้นการมอนิเตอร์โฟลเดอร์ Unknown ---")
    
    while True:
        try:
            current_time = time.time()
            
            # 💡 ควบคุมความถี่การส่งอีเมลรวม: ส่งได้ก็ต่อเมื่อครบตามช่วงเวลาที่กำหนด
            if current_time - last_email_time[0] >= EMAIL_NOTIFICATION_INTERVAL:
                
                # ลิสต์ไฟล์ทั้งหมดในโฟลเดอร์ Unknown
                # กรองเฉพาะไฟล์ภาพที่ต้องการประมวลผล
                files = os.listdir(app.config['UNKNOWN_FOLDER'])
                files_to_process = sorted([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

                if files_to_process:
                    # เลือกไฟล์แรกที่พบ (ไฟล์ที่เก่าที่สุด) เพื่อประมวลผล
                    filename = files_to_process[0]
                    file_path = os.path.join(app.config['UNKNOWN_FOLDER'], filename)
                    
                    # รอสักครู่เพื่อให้แน่ใจว่าไฟล์ถูกเขียนเสร็จสมบูรณ์
                    time.sleep(0.5) 
                    
                    # เรียกฟังก์ชันแจ้งเตือนด้วยอีเมล (ฟังก์ชันนี้จะลบไฟล์เองหากส่งสำเร็จ)
                    if send_notification_email(filename, file_path):
                        # อัปเดตเวลาการส่งอีเมลสำเร็จล่าสุด เพื่อเริ่มนับช่วงเวลาใหม่
                        last_email_time[0] = current_time
                        
        
        except Exception as e:
            print(f"Error ใน Unknown Folder Monitor: {e}")
        
        time.sleep(1) # ตรวจสอบทุกๆ 1 วินาที

# ----------------------------------------------------------------------

# รันเธรดมอนิเตอร์เมื่อเริ่มต้นแอป (Daemon=True เพื่อให้เธรดหยุดเมื่อโปรแกรมหลักหยุด)
monitor_thread = threading.Thread(target=monitor_unknown_folder, daemon=True)
monitor_thread.start()

if __name__ == '__main__':
    # โหลดเทมเพลตอีกครั้งใน Main Thread เพื่อความมั่นใจ
    templates = load_templates(UPLOAD_FOLDER) 
    
    try:
        # ⚠️ ใช้ use_reloader=False เพื่อป้องกันการรันซ้ำซ้อนในเธรดหลายครั้ง
        app.run(host='0.0.0.0', port=5000, debug=True, threaded=True, use_reloader=False) 
    except KeyboardInterrupt:
        print("Server shutting down...")
    finally:
        # ปล่อยกล้องเมื่อแอปฯ หยุดทำงาน
        if cap.isOpened():
            cap.release()
            print("Camera released.")