# Audio Grammar Analyzer (Backend + Frontend)

Full-stack project for grammatical fluency scoring from audio:
- Backend: FastAPI + Wav2Vec2 + RoBERTa + XGBoost
- Frontend: React + Vite (audio upload/record + score visualization)

## Architecture

1. Audio is processed at `16 kHz`.
2. Acoustic features come from `facebook/wav2vec2-base-960h`.
3. Semantic features come from `roberta-base` over decoded transcript.
4. Feature fusion (`1536` dims) is scored with `xgb_fusion_model.json`.

## API

### `GET /`
Health check:
```json
{"status":"ok"}
```

### `POST /score-audio/`
Accepts `multipart/form-data` with field `file`.

Response:
```json
{
  "filename": "sample.wav",
  "transcription": "she go to school yesterday",
  "grammar_score": 4.27
}
```

## Manual Local Setup (Without Docker)

### Prerequisites
- Python 3.10+
- Node.js 20+
- `xgb_fusion_model.json` in project root

### 1) Backend setup
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Start backend
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 3) Frontend setup
```bash
cd frontend
npm install
```

### 4) Start frontend (new terminal)
```bash
cd frontend
npm run dev
```

### 5) Open app
- Frontend: `http://localhost:5173`
- Backend health: `http://localhost:8000/`

## Containerization and Deployment (Docker Compose)

This repo now includes:
- `Dockerfile` for backend
- `frontend/Dockerfile` for production frontend build + Nginx
- `frontend/nginx.conf` to proxy `/score-audio/` to backend service
- `docker-compose.yml` for full-stack deployment

### 1) Build and start containers
```bash
docker compose up --build -d
```

### 2) Verify services
```bash
docker compose ps
curl http://localhost:8000/
```

### 3) Access application
- Frontend (containerized): `http://localhost:5173`
- Backend API: `http://localhost:8000`

### 4) Stop deployment
```bash
docker compose down
```

## Project Structure

- `main.py` - FastAPI inference API
- `requirements.txt` - Python dependencies
- `Dockerfile` - Backend container image
- `docker-compose.yml` - Full-stack orchestration
- `frontend/` - React application
- `frontend/Dockerfile` - Frontend container image
- `frontend/nginx.conf` - Frontend runtime + API proxy config
- `xgb_fusion_model.json` - Trained XGBoost regressor

## Notes

- Model weights are downloaded by Hugging Face on first startup.
- Backend startup can take time due to model loading.
- Dockerized frontend proxies `/score-audio/` to backend internally.
