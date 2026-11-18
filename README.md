# BLINDSPOT

> **Autonomous Context-Aware Desktop Perception, Companion, & Telemetry System**

[![Live Dashboard](https://img.shields.io/badge/Vercel-Live%20Dashboard-000000?style=for-the-badge&logo=vercel&logoColor=white)](https://blindspot-dashboard-six.vercel.app)
[![FastAPI Backend](https://img.shields.io/badge/FastAPI-127.0.0.1%3A8000-009688?style=for-the-badge&logo=fastapi&logoColor=white)](http://127.0.0.1:8000)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev)

---

### 🚀 Live Dashboard Deployment
The BLINDSPOT web dashboard is live in production on Vercel:
👉 **[https://blindspot-dashboard-six.vercel.app](https://blindspot-dashboard-six.vercel.app)**

When opened in your browser:
- **Live Engine Mode:** Streams real-time telemetry from your local BLINDSPOT engine (`http://127.0.0.1:8000`).
- **Interactive Demo Mode:** Automatically activates realistic telemetry simulation when disconnected from a local engine.

---

## Overview

**BLINDSPOT** is an intelligent desktop telemetry and autonomous companion system that continuously perceives foreground applications, extracts in-memory optical text (OCR), structures productivity sessions, and provides assistive task execution and AI interaction.

### Core Highlights
- **Real-Time Perception Engine:** Win32 foreground hooks, 300ms polling, and in-memory Tesseract OCR without writing images to disk.
- **Activity Intelligence (`activity_intelligence.py`):** Transforms raw desktop switches into structured work sessions, calculates focus aggregates, and diagnoses telemetry health (clock retrograde, OCR dropouts).
- **FastAPI REST Service (`api.py`):** High-performance backend exposing live desktop state, active window geometry, OCR text buffers, and telemetry analytics.
- **Modern Web Dashboard (`blindspot-dashboard`):** React dashboard with overview KPIs, live focus timers, screen context inspector, searchable activity history, and visual intelligence analytics.
- **Autonomous Execution & Safe Desktop Actions:** Task planning with dependency graphs, strict human-in-the-loop confirmation, context verification, and automatic recovery.

---

## System Architecture

```
Windows Desktop (Win32 API / In-Memory Tesseract OCR)
                      │
                      ▼
        BLINDSPOT Core Perception Engine
           (main.py / context.py / desktop_io.py)
          ├── 300ms Background Polling Loop
          ├── 50-Item Ring Buffer Context History
          └── activity_intelligence.py (Session Aggregation & Diagnostics)
                      │
        ┌─────────────┴─────────────┐
        ▼                           ▼
Desktop UI Companion        FastAPI REST API
   (ui.py / Tkinter)       (api.py on port 8000)
  ├── AI Chat & Prompts            │
  ├── Safe Action Approval         ▼
  └── Verification Overlays  React Dashboard
                             (blindspot-dashboard)
                             ├── Local: http://localhost:5173
                             └── Hosted: https://blindspot-dashboard-six.vercel.app
```

---

## Quickstart Guide

### 1. Prerequisites & Environment Setup

Clone the repository and set up a Python virtual environment:
```bash
git clone https://github.com/Achintya-S-Chauhan/BLINDSPOT.git
cd BLINDSPOT
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install fastapi uvicorn pydantic psutil
```

### 2. Start BLINDSPOT & Backend Services

#### Option A: Desktop UI Companion (Primary Interface - Default)
```bash
python main.py
```
*Launches the native desktop UI companion with real-time perception. To also start the FastAPI background server for the web dashboard:*
```bash
python main.py --api
```

#### Option B: Terminal CLI Monitoring Mode
```bash
python main.py --cli
```
*Runs the continuous perception loop and task executor directly in your terminal.*

#### Option C: Standalone Engine & REST API
```bash
python api.py
```
*Runs the headless engine and FastAPI service on `http://127.0.0.1:8000` (docs at `/docs`).*

---

### 3. Access the Dashboard

- **Option A (Instant - No installation):** Open the hosted deployment in your browser:
  **[https://blindspot-dashboard-six.vercel.app](https://blindspot-dashboard-six.vercel.app)**
- **Option B (Run locally):**
  ```bash
  cd blindspot-dashboard
  npm install
  npm run dev
  ```
  *Local dev server will run at `http://localhost:5173`.*

---

## Verification & Testing

Run the automated test suite:
```bash
# Test Activity Intelligence analytics & diagnostics
python -m pytest -q tests/test_activity_intelligence.py

# Run all test suites
python -m pytest tests/
```

Frontend lint & build check:
```bash
cd blindspot-dashboard
npm run lint
npm run build
```

---

## Project Structure

```
BLINDSPOT/
├── activity_intelligence.py   # Telemetry sessionizer, aggregates, and diagnostic rules
├── ai.py                      # Gemini LLM orchestration, function calling, prompt pipeline
├── api.py                     # FastAPI REST server exposing live desktop state & telemetry
├── assistant.py               # Conversational assistant & companion handler
├── context.py                 # DesktopContext & foreground window models
├── desktop_io.py              # Win32 screen capture, window rects, DPI scaling, and OCR
├── history.py                 # Ring-buffered episodic context history
├── main.py                    # Core CLI entry point & continuous perception loop
├── recovery.py                # Action execution recovery & fault-tolerance mechanisms
├── task.py                    # Multi-step task planner and dependency execution graph
├── tools.py                   # Controlled desktop actions & tool dispatch
├── ui.py                      # Desktop Tkinter companion UI with approval modal
├── verification.py            # Context-aware pre/post execution verification
├── blindspot-dashboard/       # React + Vite web dashboard (Deployed on Vercel)
│   ├── README.md              # Dashboard-specific documentation
│   ├── src/pages/             # Overview, Live, Screen Context, History, Intelligence, Settings
│   └── src/context/           # Truthful connection state & mock fallback
└── tests/                     # Unit and integration pytest suite
```

---

## License

This project is licensed under the MIT License.
