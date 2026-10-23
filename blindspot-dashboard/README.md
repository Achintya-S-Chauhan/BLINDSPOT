# BLINDSPOT — Activity Intelligence & Live Desktop Dashboard

A context-aware desktop productivity and telemetry system integrating BLINDSPOT's real Python perception engine (Win32 foreground window hooks, in-memory Tesseract OCR, `DesktopContext`, and `ContextHistory`) with a modern React web dashboard and FastAPI REST backend.

---

## Architecture Overview

```
Windows Desktop (Win32 / Tesseract OCR)
                │
                ▼
BLINDSPOT Core Engine (`main.py` / `vision.py`)
  ├── Background Perception Loop (300ms poll)
  ├── DesktopContext & Ring-buffer ContextHistory (50 items)
  └── activity_intelligence.py (Session aggregation & health diagnostics)
                │
                ▼
FastAPI Service (`api.py` on http://127.0.0.1:8000)
  ├── GET /api/health
  ├── GET /api/context/live
  ├── GET /api/context/screen
  ├── GET /api/activity/history
  ├── GET /api/overview
  ├── GET /api/activity/intelligence
  └── POST /api/engine/toggle
                │
                ▼
React Dashboard (`blindspot-dashboard` on http://localhost:5173)
  ├── Pure JavaScript (ES2022+ / JSX, 0 TypeScript)
  ├── Truthful Connection State (LIVE ENGINE / DISCONNECTED / DEMO MODE)
  └── Pages: Overview, Live Activity, Screen Context, History, Intelligence, Settings
```

---

## Quickstart: Starting the API and Dashboard

### Step 1: Start the Backend API

Make sure dependencies are installed:
```bash
pip install fastapi uvicorn pydantic psutil
```

#### Option A: Standalone Engine & API (Recommended for Dashboard)
Starts the BLINDSPOT perception engine in a background daemon thread and serves the REST API:
```bash
python api.py
```
*API will run at `http://127.0.0.1:8000`.*

#### Option B: BLINDSPOT Main with Dashboard Flag
Runs the full BLINDSPOT CLI engine and simultaneously launches the API in the background:
```bash
python main.py --api
```

---

### Step 2: Start the React Dashboard

In a separate terminal:
```bash
cd blindspot-dashboard
npm install
npm run dev
```
*Dashboard will run at `http://localhost:5173`.*

---

## Dashboard Pages & Data Flow

1. **Overview**: Displays live active application, Win32 PID, tracked focus time today, total distinct applications observed, category distribution breakdown, and recent sequential focus transitions.
2. **Live Activity**: Displays the active foreground window title, resolved canonical application name, process ID, live ticking focus duration, Win32 DPI bounding box coordinates, and real application switches from `ContextHistory`.
3. **Screen Context**: Displays the current `DesktopContext`, raw in-memory Tesseract OCR buffer, word/character counts, detected keywords, confidence score, and DPI window dimensions.
4. **Activity History**: Searchable and filterable log of completed desktop episodes stitched with the active episode, featuring search by keyword, category filter, date filter, summary KPIs, and direct CSV export.
5. **Activity Intelligence**: Powered by `activity_intelligence.py`, transforms raw desktop events into structured session timelines, total active duration aggregates, and diagnostic warnings (empty OCR buffers, timestamp retrograde, missing identifiers).
6. **Settings**: Reflects actual runtime parameters enforced by the BLINDSPOT backend (300ms poll, 1500ms stability threshold, 3.0s periodic OCR, 50-episode history limit, in-memory RAM-only privacy).

---

## Truthful Connection Status

The dashboard enforces strict truthful status indicators:
- **`LIVE ENGINE` (Emerald)**: Active when the local FastAPI server (`http://127.0.0.1:8000`) is online and connected to the running BLINDSPOT engine.
- **`DISCONNECTED` (Rose)**: Active when the backend server is stopped or unreachable. Live controls are disabled and no fake data is presented.
- **`DEMO MODE` (Amber)**: Clearly labeled fallback mode using isolated mock datasets for testing and UI demonstration.

---

## Running Verification Tests

### Python Backend Tests
Run the Activity Intelligence and API test suite:
```bash
python -m pytest -q tests/test_activity_intelligence.py
```

Run the full pytest suite:
```bash
python -m pytest tests/
```

### Frontend Lint & Production Build
In `blindspot-dashboard`:
```bash
cd blindspot-dashboard
npm run lint
npm run build
```
*(All frontend application code is written strictly in JavaScript/JSX; zero TypeScript or TSX files).*
