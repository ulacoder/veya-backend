# VEya Web Backend

FastAPI backend for VEya eye disease detection model.

## Setup

```bash
cd backend
pip install -r requirements.txt
```

## Run

```bash
python app/main.py
```

Server runs on http://localhost:8000

## Model

- **Location:** `C:\Users\Ulagat\veya-dataset\trained_model\veya_model_final.keras`
- **Classes:** Cataract, Conjunctivitis, Normal, Pterygium
- **Accuracy:** 81.22%
- **Input:** 224×224 RGB images

## API Endpoints

### `GET /health`
Health check and model status

### `POST /api/analyze`
Analyze eye image

**Request:**
```bash
curl -X POST http://localhost:8000/api/analyze \
  -F "file=@eye.jpg"
```

**Response:**
```json
{
  "prediction": "Normal",
  "confidence": 0.95,
  "probabilities": {
    "Cataract": 0.02,
    "Conjunctivitis": 0.01,
    "Normal": 0.95,
    "Pterygium": 0.02
  },
  "model_version": "1.0.0"
}
```

## Environment Variables

- `MODEL_API_URL` - Set in Next.js `.env.local` (default: http://localhost:8000)

## Development

1. Start backend: `python app/main.py`
2. Start frontend: `cd .. && npm run dev`
3. Open http://localhost:3000

## Deployment

For production, deploy backend separately (Railway, Render, etc.) and set `MODEL_API_URL` in Vercel environment variables.
