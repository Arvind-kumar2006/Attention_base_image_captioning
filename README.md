# Lumina — Attention-Based Image Captioning with AI Visual Understanding

An image captioning system built on a CNN Encoder + Soft Attention + LSTM Decoder
(the "Show, Attend and Tell" architecture), extended into three guided analysis
modes: **Museum AI**, **Prescription AI**, and **Product Label AI**.

## What it does

1. Upload an image (JPEG/PNG).
2. The model generates a natural-language caption, with a per-word attention
   heatmap showing what it focused on while generating each word.
3. Pick an analysis mode for a deeper, plain-language explanation:
   - **Museum AI** — artwork/artifact history, grounded with RAG (FAISS)
   - **Prescription AI** — medicine, dosage, and frequency from a handwritten prescription (OCR)
   - **Product Label AI** — ingredients, allergens, and a health score (OCR)

## Tech stack

- **Model**: PyTorch, Torchvision (ResNet encoder)
- **Backend**: FastAPI, PostgreSQL, SQLAlchemy
- **OCR**: PaddleOCR / EasyOCR
- **RAG**: FAISS + LangChain
- **LLM**: OpenAI / Llama / Gemma (swappable)
- **Frontend**: React + Tailwind CSS

## Project structure

```
backend/     FastAPI service — captioning, OCR, RAG, LLM, analyzers
ml/          Model architecture, training script, training notebook
frontend/    React + Tailwind UI
```

## Setup

```bash
# 1. Environment
cp .env.example .env   # add your LLM API key + DB URL

# 2. Backend
pip install -r backend/requirements.txt

# 3. Train (or drop a checkpoint into ml/weights/)
python -m ml.train

# 4. Run
docker-compose up          # backend + Postgres
# or
uvicorn backend.main:app --reload

# 5. Frontend
cd frontend
npm install
npm run dev
```

## Evaluation

| Module | Metric | Target |
|---|---|---|
| Captioning | BLEU-4, CIDEr, METEOR | CIDEr ≥ 1.0, BLEU-4 ≥ 0.30 |
| Museum AI | Human-in-the-Loop grounding review | 90% grounded |
| Prescription AI | OCR CER/WER + field accuracy | 80% correct extraction |
| Product Label AI | OCR CER/WER + coverage | 90% readable labels processed |

## Status

🚧 In progress — core captioning model training, analysis modes being built out.

## Based on

"Show, Attend and Tell: Neural Image Caption Generation with Visual Attention"
(Xu et al., ICML 2015)
