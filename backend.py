import json
import numpy as np
import os
import sys
import re
import subprocess
import tempfile
import traceback
import webbrowser
from threading import Timer
from flask import Flask, request, render_template
from flask_cors import CORS
import xml.etree.ElementTree as ET

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
    height, width = image.shape[0], image.shape[1]

    data = (edged[::-1] > 1).astype(np.uint8)

    tmp = tempfile.NamedTemporaryFile(suffix='.pbm', delete=False)
    try:
        with open(tmp.name, 'w') as f:
            f.write(f'P1\n{width} {height}\n')
            for y in range(height):
                row = ''
                for x in range(width):
                    row += '1' if data[y, x] else '0'
                f.write(row + '\n')

        svg_file = tmp.name + '.svg'
        subprocess.run(
            ['potrace', '-s', '-b', 'svg',
             '-t', str(turdsize),
             '-a', f'{alphamax:.2f}',
             '-O', f'{opttolerance:.2f}',
             '-o', svg_file, tmp.name],
            check=True, capture_output=True, timeout=30
        )

        tree = ET.parse(svg_file)
        ns = {'svg': 'http://www.w3.org/2000/svg'}
        path_elems = tree.findall('.//svg:path', ns)

        latex = []
        exprid = 0
        for path_elem in path_elems:
            d = path_elem.get('d', '')
            cmds = re.findall(r'[MCL]\s*[\d\.\-e\s]+', d)
            i = 0
            while i < len(cmds):
                parts = cmds[i].strip().split()
                cmd = parts[0]
                coords = list(map(float, parts[1:]))
                if cmd == 'M':
                    start = (coords[0], coords[1])
                    i += 1
                elif cmd == 'C' and len(coords) >= 6:
                    x0, y0 = start
                    x1, y1 = coords[0], coords[1]
                    x2, y2 = coords[2], coords[3]
                    x3, y3 = coords[4], coords[5]
                    formula = (
                        f'((1-t)*((1-t)*((1-t)*{x0:.3f}+t*{x1:.3f})+t*((1-t)*{x1:.3f}+t*{x2:.3f}))+t*((1-t)*((1-t)*{x1:.3f}+t*{x2:.3f})+t*((1-t)*{x2:.3f}+t*{x3:.3f})),'
                        f'(1-t)*((1-t)*((1-t)*{y0:.3f}+t*{y1:.3f})+t*((1-t)*{y1:.3f}+t*{y2:.3f}))+t*((1-t)*((1-t)*{y1:.3f}+t*{y2:.3f})+t*((1-t)*{y2:.3f}+t*{y3:.3f})))'
                    )
                    latex.append({'id': f'expr-{exprid + 1}', 'latex': formula, 'color': COLOUR})
                    exprid += 1
                    start = (x3, y3)
                    i += 1
                elif cmd == 'L' and len(coords) >= 2:
                    x0, y0 = start
                    x1, y1 = coords[0], coords[1]
                    latex.append({'id': f'expr-{exprid + 1}', 'latex': f'((1-t)*{x0:.3f}+t*{x1:.3f},(1-t)*{y0:.3f}+t*{y1:.3f})', 'color': COLOUR})
                    exprid += 1
                    start = (x1, y1)
                    i += 1
                else:
                    i += 1
    finally:
        os.unlink(tmp.name)
        if os.path.exists(svg_file):
            os.unlink(svg_file)

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
