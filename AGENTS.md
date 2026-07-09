# AGENTS.md — Desmos Bezier Renderer

## Quick start

```powershell
python -m venv env
.\env\Scripts\Activate.ps1
pip install -r requirements.txt
python backend.py
# Open http://127.0.0.1:5000/calculator
```

Requires system dep `potrace` (`brew install potrace` on macOS, `apt install libpotrace-dev` on Linux/WSL). On Windows you need WSL or to install `potrace` manually.

## Project structure

- `backend.py` — Flask server (port 5000), routes: `/upload` POST, `/process` POST, `/calculator` GET
- `frontend/index.html` — single-file SPA, Desmos API v1.8, all logic inline
- `calculator.html` — redirect stub to `localhost:5000/calculator`
- `run.sh` — macOS/Linux auto-setup script (ignores Windows)

## Architecture

- Image upload → OpenCV (Canny edge detection) → pypotrace (Bézier tracing) → Desmos LaTeX formula list → frontend renders via Desmos Calculator API
- Uploaded image cached in memory (`UPLOADED_IMAGE` global); slider changes POST to `/process` to retrace with new params
- Desmos API key (`dcb31709b452b1cf9dc26972add0fda6`) is hardcoded in `backend.py:190`

## Key conventions

- All formulas use IDs prefixed `expr-` (e.g. `expr-1`, `expr-42`)
- Parameter ranges: `canny_high` 50–255, `canny_low` 5–150, `turdsize` 0–30, `opttolerance` 0.01–1.00, `alphamax` 0.0–1.3
- Color hardcoded as `#2464b4` in `COLOUR` constant
- Image Y-axis is flipped (`edged[::-1]`) to match Desmos coordinate system

## Edge detection backends

`backend.py` tries OpenCV first; if unavailable falls back to scikit-image (`skimage.feature.canny`). This is necessary for MSYS2 builds where opencv has no compatible wheel.

## CI (GitHub Actions)

`.github/workflows/build.yml` — triggered on every push/PR and on `v*` tags.

- Builds **Windows .exe** via PyInstaller on `windows-latest` runner
- Uses **MSYS2 MINGW64** environment to compile `pypotrace` from source (pkg-config finds MSYS2's `mingw-w64-x86_64-potrace`)
- OpenCV is **not** available in MSYS2; the skimage fallback is used instead
- All Python deps installed via MSYS2 packages + pip
- Smoke test: launches `backend.exe`, hits `/calculator` route, expects HTTP 200
- Tag pushes (`v*`) create a GitHub Release and upload the zip

## PyInstaller packaging

**macOS** (local):
```sh
pyinstaller --noconfirm --onedir --windowed --name "DesmosBezierRenderer" --hidden-import "potrace.bezier" --hidden-import "potrace.agg" --hidden-import "potrace.agg.curves" --add-data "frontend:frontend" backend.py
```

**Windows** (CI or MSYS2):
```sh
pyinstaller --noconfirm --onedir --noconsole --name "DesmosBezierRenderer" --add-data "frontend;frontend" --hidden-import "potrace.bezier" --hidden-import "potrace.agg" --hidden-import "potrace.agg.curves" backend.py
```

## .gitignore

Ignores `env/`, `dist/`, `build/`, `__pycache__/`, image uploads (`*.png`, `*.jpg`, etc.) except `github/*.png` and `frames/*`.
