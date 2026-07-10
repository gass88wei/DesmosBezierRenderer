import json
import logging
import numpy as np
import os
import sys
import re
import subprocess
import tempfile
import traceback
import webbrowser
from threading import Timer
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import xml.etree.ElementTree as ET

log_dir = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.abspath('.')
LOG_FILE = os.path.join(log_dir, 'desmos-debug.log')
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    force=True
)
logging.info('=== Backend started ===')
logging.info('sys.executable=%s', sys.executable)
logging.info('frozen=%s', getattr(sys, 'frozen', False))
logging.info('cwd=%s', os.path.abspath('.'))

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
        ret = os.path.join(sys._MEIPASS, relative_path)
        logging.info('get_resource_path(%s) via _MEIPASS -> %s (exists=%s)', relative_path, ret, os.path.exists(ret))
        return ret
    exe_dir = os.path.dirname(sys.executable)
    exe_path = os.path.join(exe_dir, relative_path)
    if os.path.exists(exe_path):
        logging.info('get_resource_path(%s) via exe_dir -> %s', relative_path, exe_path)
        return exe_path
    cwd_path = os.path.join(os.path.abspath("."), relative_path)
    logging.info('get_resource_path(%s) via cwd -> %s (exists=%s)', relative_path, cwd_path, os.path.exists(cwd_path))
    return cwd_path


def get_potrace_path():
    # Try 1: bundled next to EXE (--onedir mode)
    exe_dir = os.path.dirname(sys.executable)
    bundled = os.path.join(exe_dir, 'potrace_bundle', 'potrace.exe')
    if os.path.exists(bundled):
        logging.info('get_potrace_path -> bundled exe dir: %s', bundled)
        return bundled
    # Try 2: bundled relative to resource path (fallback)
    res_dir = os.path.dirname(get_resource_path('frontend'))
    bundled2 = os.path.join(res_dir, 'potrace_bundle', 'potrace.exe')
    if os.path.exists(bundled2):
        logging.info('get_potrace_path -> bundled resource dir: %s', bundled2)
        return bundled2
    # Try 3: in CWD
    cwd_bundle = os.path.join(os.path.abspath('.'), 'potrace_bundle', 'potrace.exe')
    if os.path.exists(cwd_bundle):
        logging.info('get_potrace_path -> bundled cwd: %s', cwd_bundle)
        return cwd_bundle
    # Try 4: system PATH
    logging.warning('get_potrace_path -> falling back to PATH')
    return 'potrace'

def get_subprocess_kwargs():
    kwargs = {'capture_output': True, 'timeout': 30}
    if sys.platform == 'win32':
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        kwargs['startupinfo'] = si
    return kwargs

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
    tmp.close()
    try:
        with open(tmp.name, 'w', encoding='utf-8') as f:
            f.write(f'P1\n{width} {height}\n')
            for y in range(height):
                row = ''
                for x in range(width):
                    row += '1' if data[y, x] else '0'
                f.write(row + '\n')

        svg_file = tmp.name + '.svg'
        potrace_bin = get_potrace_path()
        logging.info('Running potrace: %s', potrace_bin)
        subprocess.run(
            [potrace_bin, '-s', '-b', 'svg',
             '-t', str(turdsize),
             '-a', f'{alphamax:.2f}',
             '-O', f'{opttolerance:.2f}',
             '-o', svg_file, tmp.name],
            check=True, **get_subprocess_kwargs()
        )

        tree = ET.parse(svg_file)
        ns = {'svg': 'http://www.w3.org/2000/svg'}
        path_elems = tree.findall('.//svg:path', ns)

        latex = []
        exprid = 0
        for path_elem in path_elems:
            d = path_elem.get('d', '')
            # Match any SVG path command (uppercase or lowercase) with its coordinates
            cmds = re.findall(r'[MmCcLl]\s*[\d\.\-e\s]+', d)
            for cmd_str in cmds:
                parts = cmd_str.strip().split()
                raw = parts[0]
                cmd = raw[0].upper()        # first char, uppercased
                # separate command letter from leading digits
                extra = raw[1:]
                if extra:
                    coords = [float(extra)] + list(map(float, parts[1:]))
                else:
                    coords = list(map(float, parts[1:]))

                if cmd == 'M':
                    if len(coords) >= 2:
                        start = (coords[0], coords[1])
                elif cmd == 'C':
                    # split into groups of 6 (cubic bezier)
                    for k in range(0, len(coords) - 5, 6):
                        x0, y0 = start
                        x1, y1 = coords[k], coords[k+1]
                        x2, y2 = coords[k+2], coords[k+3]
                        x3, y3 = coords[k+4], coords[k+5]
                        if raw[0].islower():
                            x1 += x0; y1 += y0
                            x2 += x0; y2 += y0
                            x3 += x0; y3 += y0
                        formula = (
                            f'((1-t)*((1-t)*((1-t)*{x0:.3f}+t*{x1:.3f})+t*((1-t)*{x1:.3f}+t*{x2:.3f}))+t*((1-t)*((1-t)*{x1:.3f}+t*{x2:.3f})+t*((1-t)*{x2:.3f}+t*{x3:.3f})),'
                            f'(1-t)*((1-t)*((1-t)*{y0:.3f}+t*{y1:.3f})+t*((1-t)*{y1:.3f}+t*{y2:.3f}))+t*((1-t)*((1-t)*{y1:.3f}+t*{y2:.3f})+t*((1-t)*{y2:.3f}+t*{y3:.3f})))'
                        )
                        latex.append({'id': f'expr-{exprid + 1}', 'latex': formula, 'color': COLOUR})
                        exprid += 1
                        start = (x3, y3)
                elif cmd == 'L':
                    # split into groups of 2 (line segments)
                    for k in range(0, len(coords) - 1, 2):
                        x0, y0 = start
                        x1, y1 = coords[k], coords[k+1]
                        if raw[0].islower():
                            x1 += x0; y1 += y0
                        latex.append({'id': f'expr-{exprid + 1}', 'latex': f'((1-t)*{x0:.3f}+t*{x1:.3f},(1-t)*{y0:.3f}+t*{y1:.3f})', 'color': COLOUR})
                        exprid += 1
                        start = (x1, y1)
    finally:
        os.unlink(tmp.name)
        if os.path.exists(svg_file):
            os.unlink(svg_file)

    return latex, width, height


@app.route('/upload', methods=['POST'])
def upload():
    global UPLOADED_IMAGE
    file = request.files.get('image')
    if not file:
        return {'error': 'No file uploaded'}, 400

    try:
        file_bytes = file.read()
        image = decode_image(file_bytes)

        UPLOADED_IMAGE = image

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
    except subprocess.CalledProcessError as e:
        err_msg = e.stderr.decode() if e.stderr else str(e)
        logging.error('Upload failed (subprocess): %s', err_msg)
        logging.error('stdout: %s', e.stdout.decode() if e.stdout else '')
        return {'error': f'potrace failed: {err_msg}'}, 500
    except FileNotFoundError as e:
        logging.error('Upload failed (file not found): %s', e)
        return {'error': f'potrace not found: {e}'}, 500
    except Exception as e:
        tb = traceback.format_exc()
        logging.error('Upload failed:\n%s', tb)
        return {'error': str(e)}, 500


@app.route('/process', methods=['POST'])
def process():
    global UPLOADED_IMAGE
    if UPLOADED_IMAGE is None:
        return {'error': 'No image uploaded yet'}, 400

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
    except subprocess.CalledProcessError as e:
        err_msg = e.stderr.decode() if e.stderr else str(e)
        logging.error('Re-process failed (subprocess): %s', err_msg)
        return {'error': f'potrace failed: {err_msg}'}, 500
    except FileNotFoundError as e:
        logging.error('Re-process failed (file not found): %s', e)
        return {'error': f'potrace not found: {e}'}, 500
    except Exception as e:
        tb = traceback.format_exc()
        logging.error('Re-process failed:\n%s', tb)
        return {'error': str(e)}, 500


@app.route("/diag")
def diag():
    info = {
        'sys.executable': sys.executable,
        'sys.frozen': getattr(sys, 'frozen', False),
        'sys._MEIPASS': getattr(sys, '_MEIPASS', None),
        'cwd': os.path.abspath('.'),
        'HAS_CV2': HAS_CV2,
        'LOG_FILE': LOG_FILE,
        'template_folder': app.template_folder,
    }
    for label, path in [
        ('frontend', get_resource_path('frontend')),
        ('frontend/index.html', os.path.join(get_resource_path('frontend'), 'index.html')),
    ]:
        info[f'exists:{label}'] = os.path.exists(path)
    for label, path in [
        ('exe_dir', os.path.dirname(sys.executable)),
        ('potrace_bundle', os.path.join(os.path.dirname(sys.executable), 'potrace_bundle')),
        ('potrace.exe', get_potrace_path()),
    ]:
        info[f'exists:{label}'] = os.path.exists(path)
    if getattr(sys, 'frozen', False):
        exe_dir = os.path.dirname(sys.executable)
        try:
            info['exe_dir_contents'] = os.listdir(exe_dir)
        except Exception as e:
            info['exe_dir_contents'] = f'Error: {e}'
        pb_dir = os.path.join(exe_dir, 'potrace_bundle')
        if os.path.isdir(pb_dir):
            info['potrace_bundle_contents'] = os.listdir(pb_dir)
    # Try running potrace --version
    try:
        r = subprocess.run([get_potrace_path(), '--version'], capture_output=True, text=True, timeout=10)
        info['potrace_version'] = r.stdout.strip() or r.stderr.strip()
    except Exception as e:
        info['potrace_version_error'] = str(e)
    return jsonify(info)


@app.route("/calculator")
def client():
    return render_template('index.html', api_key='dcb31709b452b1cf9dc26972add0fda6')


if __name__ == '__main__':
    # 自动在浏览器中打开主页
    def open_browser():
        webbrowser.open(f'http://127.0.0.1:{PORT}/calculator')
    Timer(1, open_browser).start()

    app.run(host='127.0.0.1', port=PORT)
