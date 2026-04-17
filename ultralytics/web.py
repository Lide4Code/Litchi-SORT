import io
import os
from flask import Flask, request, send_file, Response
import numpy as np
import warnings
from PIL import Image
import tempfile
import cv2
from ultralytics import YOLO

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)
app = Flask(__name__)
model = YOLO('runs/train/model/weights/best.pt') 

@app.route('/detect', methods=['POST'])
def detect_image():
    if 'image' not in request.files:
        return Response("No image provided", status=400)
    
    file = request.files['image']
    if file.filename == '':
        return Response("Empty filename", status=400)
    try:
        img = Image.open(io.BytesIO(file.read()))
        img_array = np.array(img)
        results = model.predict(
            source=img_array,
            imgsz=640,
            device='cpu',
            verbose=False  
        )
        processed_array = results[0].plot() 
        img_byte_arr = io.BytesIO()
        Image.fromarray(processed_array).save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        return send_file(
            img_byte_arr,
            mimetype='image/jpeg',
            as_attachment=True,
            download_name='processed.jpg'
        )
    
    except Exception as e:
        print(e)
        return Response(f"Error processing image: {str(e)}", status=500)
    
@app.route('/track', methods=['POST'])
def track_video():
    if 'video' not in request.files:
        return Response("No video provided", status=400)
    
    file = request.files['video']
    if file.filename == '':
        return Response("Empty filename", status=400)
    
    try:
        # 将上传的视频保存为临时文件
        tmp_in = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        file.save(tmp_in.name)
        tmp_in.close()
        
        # 通过 OpenCV 获取视频基本信息
        cap = cv2.VideoCapture(tmp_in.name)
        if not cap.isOpened():
            return Response("Error opening video file", status=400)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30  # 若获取失败，默认为 30
        cap.release()
        
        # 创建一个临时输出文件保存处理后的视频
        tmp_out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp_out.close()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(tmp_out.name, fourcc, fps, (width, height))
        
        # 执行目标跟踪处理
        results = model.track(
            source=tmp_in.name,
            tracker='ultralytics/cfg/trackers/botsort.yaml',
            imgsz=1024,
            save=False,
            show=False
        )
        
        # 遍历每一帧的跟踪结果，并写入输出视频
        for frame_idx, result in enumerate(results):
            # 使用 result.plot() 获取绘制检测框后的帧图像
            frame = result.plot()
            out_video.write(frame)
        
        out_video.release()
        
        # 返回处理后的视频文件
        return send_file(
            tmp_out.name,
            mimetype='video/mp4',
            as_attachment=True,
            download_name='processed.mp4'
        )
    
    except Exception as e:
        print(e)
        return Response(f"Error processing video: {str(e)}", status=500)
    
# 新增接口：点击链接即可下载一张图片
@app.route('/download_image', methods=['GET'])
def download_image():
    image_path = 'image/litchi.jpg'
    if not os.path.exists(image_path):
        return Response("Image not found", status=404)
    return send_file(
        image_path,
        mimetype='image/jpeg',
        as_attachment=True,
        download_name='sample.jpg'
    )

# 新增接口：点击链接即可下载一个视频
@app.route('/download_video', methods=['GET'])
def download_video():
    video_path = 'video/litchi1.mp4'
    if not os.path.exists(video_path):
        return Response("Video not found", status=404)
    return send_file(
        video_path,
        mimetype='video/mp4',
        as_attachment=True,
        download_name='sample.mp4'
    )    

#  curl -X POST -F "image=@image/litchi.jpg" http://localhost:5000/detect --output result.jpg
#  curl -X POST -F "video=@video/litchi2.mp4" http://localhost:5000/track --output processed.mp4

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, threaded=True)