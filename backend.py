import json
import numpy as np
import potrace
import os
import sys
import traceback
import webbrowser
from threading import Timer
from flask import Flask, request, render_template
from flask_cors import CORS

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    from skimage import feature, color
    from PIL import Image
    import io as _io


def get_resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

app = Flask(__name__, template_folder=get_resource_path('frontend'))
CORS(app)
PORT = 5000

UPLOADED_IMAGE = None

DEFAULT_CANNY_LOW = 30
DEFAULT_CANNY_HIGH = 200
DEFAULT_TURDSIZE = 2
DEFAULT_ALPHAMAX = 1.0
DEFAULT_OPTTOLERANCE = 0.2

COLOUR = '#2464b4'


def canny_edge_detect(image, low, high):
    if HAS_CV2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Canny(gray, low, high)
    gray = color.rgb2gray(image)
    edged = feature.canny(gray, low_threshold=low / 255.0, high_threshold=high / 255.0)
    return (edged * 255).astype(np.uint8)


def decode_image(file_bytes):
    if HAS_CV2:
        image = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image")
        return image
    pil_img = Image.open(_io.BytesIO(file_bytes))
    return np.array(pil_img.convert("RGB"))


def process_image_to_latex(image, turdsize, alphamax, opttolerance, canny_low, canny_high):
    edged = canny_edge_detect(image, canny_low, canny_high)

    data = edged[::-1]
    data[data > 1] = 1
    bmp = potrace.Bitmap(data)

    # 4. Potrace 曲线追踪拟合
    # turdsize: 噪点抑制大小
    # alphamax: 转折平滑度 (0.0 表示折线，1.0~1.3 表示平滑曲线)
    # opttolerance: 数值优化容差 (越小拟合越贴合原边缘)
    path = bmp.trace(
        turdsize=turdsize,
        turnpolicy=potrace.TURNPOLICY_MINORITY,
        alphamax=alphamax,
        opticurve=1,
        opttolerance=opttolerance
    )

    # 5. 生成公式列表
    latex = []
    exprid = 0

    for curve in path.curves:
        segments = curve.segments
        start = curve.start_point
        for segment in segments:
            x0, y0 = start
            if segment.is_corner:
                x1, y1 = segment.c
                x2, y2 = segment.end_point
                # 折角处生成两条直线公式
                latex.append({
                    'id': f'expr-{exprid + 1}',
                    'latex': f'((1-t)*{x0:.3f}+t*{x1:.3f},(1-t)*{y0:.3f}+t*{y1:.3f})',
                    'color': COLOUR
                })
                exprid += 1
                latex.append({
                    'id': f'expr-{exprid + 1}',
                    'latex': f'((1-t)*{x1:.3f}+t*{x2:.3f},(1-t)*{y1:.3f}+t*{y2:.3f})',
                    'color': COLOUR
                })
                exprid += 1
            else:
                x1, y1 = segment.c1
                x2, y2 = segment.c2
                x3, y3 = segment.end_point
                # 贝塞尔三次曲线公式
                formula = (
                    f'((1-t)*((1-t)*((1-t)*{x0:.3f}+t*{x1:.3f})+t*((1-t)*{x1:.3f}+t*{x2:.3f}))+t*((1-t)*((1-t)*{x1:.3f}+t*{x2:.3f})+t*((1-t)*{x2:.3f}+t*{x3:.3f})),'
                    f'(1-t)*((1-t)*((1-t)*{y0:.3f}+t*{y1:.3f})+t*((1-t)*{y1:.3f}+t*{y2:.3f}))+t*((1-t)*((1-t)*{y1:.3f}+t*{y2:.3f})+t*((1-t)*{y2:.3f}+t*{y3:.3f})))'
                )
                latex.append({
                    'id': f'expr-{exprid + 1}',
                    'latex': formula,
                    'color': COLOUR
                })
                exprid += 1
            start = segment.end_point

    height, width = image.shape[0], image.shape[1]
    return latex, width, height


@app.route('/upload', methods=['POST'])
def upload():
    """
    接收用户上传的图片文件并缓存在内存中。使用默认参数进行首次提取。
    """
    global UPLOADED_IMAGE
    file = request.files.get('image')
    if not file:
        return {'error': 'No file uploaded'}, 400

    try:
        file_bytes = file.read()
        image = decode_image(file_bytes)

        UPLOADED_IMAGE = image
        
        # 使用默认参数计算贝塞尔曲线
        latex_list, width, height = process_image_to_latex(
            image,
            turdsize=DEFAULT_TURDSIZE,
            alphamax=DEFAULT_ALPHAMAX,
            opttolerance=DEFAULT_OPTTOLERANCE,
            canny_low=DEFAULT_CANNY_LOW,
            canny_high=DEFAULT_CANNY_HIGH
        )

        return {
            'result': latex_list,
            'width': width,
            'height': height
        }
    except Exception as e:
        traceback.print_exc()
        return {'error': str(e)}, 500


@app.route('/process', methods=['POST'])
def process():
    """
    接收最新滑块参数，对已上传的图片重新做边缘检测和曲线追踪。
    """
    global UPLOADED_IMAGE
    if UPLOADED_IMAGE is None:
        return {'error': 'No image uploaded yet. Please upload an image first.'}, 400

    try:
        data = request.json or {}
        turdsize = int(data.get('turdsize', DEFAULT_TURDSIZE))
        alphamax = float(data.get('alphamax', DEFAULT_ALPHAMAX))
        opttolerance = float(data.get('opttolerance', DEFAULT_OPTTOLERANCE))
        canny_low = int(data.get('canny_low', DEFAULT_CANNY_LOW))
        canny_high = int(data.get('canny_high', DEFAULT_CANNY_HIGH))

        latex_list, width, height = process_image_to_latex(
            UPLOADED_IMAGE,
            turdsize=turdsize,
            alphamax=alphamax,
            opttolerance=opttolerance,
            canny_low=canny_low,
            canny_high=canny_high
        )

        return {
            'result': latex_list,
            'width': width,
            'height': height
        }
    except Exception as e:
        traceback.print_exc()
        return {'error': str(e)}, 500


@app.route("/calculator")
def client():
    """
    渲染前端界面，传递 Desmos 开发者 API key。
    """
    return render_template('index.html', api_key='dcb31709b452b1cf9dc26972add0fda6')


if __name__ == '__main__':
    # 自动在浏览器中打开主页
    def open_browser():
        webbrowser.open(f'http://127.0.0.1:{PORT}/calculator')
    Timer(1, open_browser).start()

    app.run(host='127.0.0.1', port=PORT)
