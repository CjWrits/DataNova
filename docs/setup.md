# Development Setup & Environment Guide

## Prerequisites

- **Python**: 3.11 - 3.13 (Verified on Python 3.13.14)
- **Node.js**: v20+ LTS or v24+ (Verified on Node.js v24.11.0)
- **npm**: v10+ or v11+ (Verified on npm 11.6.1)
- **Git**: 2.40+ (Verified on Git 2.51.1)
- **Operating System**: Windows, Linux, or macOS

---

## 1. Initial Repository Setup

Clone the repository and inspect the branch:
```bash
git clone <repository-url>
cd DataNova
git status
```

---

## 2. Python Virtual Environment Configuration

### On Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### On Linux / macOS (Bash):
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Environment Variables Configuration

Copy the sample environment file and review local configurations:
```bash
cp .env.example .env
```

---

## 4. Verifying Python Setup & Running Tests

Verify package installations and run initial test fixtures:
```bash
pytest tests/ -v
```

---

## 5. Frontend Setup (Phase 5)

When transitioning to Phase 5, initialize and start the Next.js development server:
```bash
cd frontend
npm install
npm run dev
```
The frontend will be accessible at `http://localhost:3000`.

---

## 6. Backend API Server Execution (Phase 4)

When transitioning to Phase 4, start the FastAPI ASGI server:
```bash
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
Interactive OpenAPI documentation will be accessible at:
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`
