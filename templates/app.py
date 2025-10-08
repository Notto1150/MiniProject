from flask import Flask, render_template, Response, request, url_for, redirect, send_from_directory, abort, flash, jsonify
import cv2
import os
from werkzeug.utils import secure_filename
import threading
import time
import datetime

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # สำหรับ flash message

UPLOAD_FOLDER = 'C:/AppServ/www/image'
VIDEO_FOLDER = 'C:/AppServ/www/Cam'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VIDEO_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['VIDEO_FOLDER'] = VIDEO_FOLDER

cap = cv2.VideoCapture(0)

# ฟังก์ชันสำหรับ stream กล้อง
def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)  # สลับซ้ายขวา
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ตัวแปร global สำหรับบันทึกวิดีโอ
recording = False
out = None
record_thread = None

def record_video():
    global recording, out, cap
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # codec mp4
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    filename = os.path.join(VIDEO_FOLDER, f"record_{int(time.time())}.mp4")
    out = cv2.VideoWriter(filename, fourcc, 20.0, (width, height))
    print(f'Start recording: {filename}')
    while recording:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        out.write(frame)
        time.sleep(0.05)  # ลดโหลด CPU
    out.release()
    print('Stopped recording.')

@app.route('/start_recording', methods=['POST'])
def start_recording():
    global recording, record_thread
    if recording:
        return jsonify({'status': 'already recording'}), 400
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
    record_thread.join()
    return jsonify({'status': 'recording stopped'})

# --- หน้าแสดงรายการวิดีโอ พร้อมเวลาบันทึก ---
@app.route('/videos')
def videos():
    files = os.listdir(app.config['VIDEO_FOLDER'])
    videos = [v for v in files if v.lower().endswith(('.mp4', '.avi', '.mov'))]
    videos_with_time = []
    for v in videos:
        path = os.path.join(app.config['VIDEO_FOLDER'], v)
        timestamp = os.path.getmtime(path)
        dt = datetime.datetime.fromtimestamp(timestamp)
        formatted_time = dt.strftime('%Y-%m-%d %H:%M:%S')
        videos_with_time.append({
            'filename': v,
            'time': formatted_time
        })
    return render_template('videos.html', videos=videos_with_time)

# ส่งไฟล์วิดีโอ (แก้ url ให้สอดคล้องกับ template)
@app.route('/videos/<filename>')
def video_file(filename):
    filename = secure_filename(filename)
    return send_from_directory(app.config['VIDEO_FOLDER'], filename)

# ลบวิดีโอทีละไฟล์
@app.route('/delete_video/<filename>', methods=['POST'])
def delete_video(filename):
    filename = secure_filename(filename)
    file_path = os.path.join(app.config['VIDEO_FOLDER'], filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        flash(f'ลบวิดีโอ {filename} เรียบร้อยแล้ว')
        return redirect(url_for('videos'))
    else:
        abort(404)

# --- routes อื่นๆ ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/camera')
def camera():
    return render_template('camera.html')

@app.route('/upload', methods=['GET'])
def upload():
    images = os.listdir(app.config['UPLOAD_FOLDER'])
    images = [img for img in images if img.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    
    success = request.args.get('success')
    filename = request.args.get('filename')
    
    return render_template('upload.html', images=images, success=success, filename=filename)

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image_file' not in request.files:
        return 'ไม่มีไฟล์', 400
    file = request.files['image_file']
    if file.filename == '':
        return 'ไม่ได้เลือกไฟล์', 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)
    return redirect(url_for('upload', success='true', filename=filename))

@app.route('/delete/<filename>', methods=['POST'])
def delete_image(filename):
    filename = secure_filename(filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        flash(f'ลบรูป {filename} เรียบร้อยแล้ว')
        return redirect(url_for('upload'))
    else:
        abort(404)

@app.route('/delete_all', methods=['POST'])
def delete_all_images():
    images = os.listdir(app.config['UPLOAD_FOLDER'])
    for img in images:
        if img.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            try:
                os.remove(os.path.join(app.config['UPLOAD_FOLDER'], img))
            except Exception as e:
                print(f'Error deleting file {img}: {e}')
    flash('ลบรูปทั้งหมดเรียบร้อยแล้ว')
    return redirect(url_for('upload'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
