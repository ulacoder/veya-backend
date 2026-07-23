# VEya AI Backend

Eye disease detection API powered by MobileNetV2.

## 🚀 Quick Deploy

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template/veya-backend?referralCode=vfHgR3)

**Or manually:**

1. Fork this repo
2. Go to [Railway](https://railway.app/new)
3. Deploy from GitHub → Select `veya-backend`
4. Wait 3-5 minutes
5. Generate domain in Settings

## 🧠 Model Info

- **Architecture:** MobileNetV2
- **Classes:** Cataract, Conjunctivitis, Normal, Pterygium
- **Accuracy:** 81.22%
- **Model Size:** 26MB

## 📡 API Endpoints

### `GET /health`
Health check
```bash
curl https://your-backend.railway.app/health
```

### `POST /api/analyze`
Analyze eye image
```bash
curl -X POST https://your-backend.railway.app/api/analyze \
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
  }
}
```

## 🛠️ Tech Stack

- FastAPI
- TensorFlow 2.15
- Python 3.11
- Uvicorn

## 📦 Local Development

```bash
pip install -r requirements.txt
python app/main.py
```

Server runs on http://localhost:8000

## 🔗 Frontend

Connect to [VEya Web](https://veya-web-zeta.vercel.app):

Add environment variable in Vercel:
```
MODEL_API_URL=https://your-backend.railway.app
```

## 📄 License

MIT
