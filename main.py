from typing import Optional
import tempfile
import os
import av
import librosa
import numpy as np
import torch
import xgboost as xgb
from fastapi import FastAPI, File, HTTPException, UploadFile
from transformers import AutoModel, AutoTokenizer, Wav2Vec2ForCTC, Wav2Vec2Processor
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Audio Grammar Scoring API", version="1.0.0")

# Allow your frontend to communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins during development. (Change to specific URLs in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (POST, GET, OPTIONS, etc.)
    allow_headers=["*"],  # Allows all headers
)

DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
W2V_PROCESSOR: Optional[Wav2Vec2Processor] = None
W2V_MODEL: Optional[Wav2Vec2ForCTC] = None
ROBERTA_TOKENIZER: Optional[AutoTokenizer] = None
ROBERTA_MODEL: Optional[AutoModel] = None
FUSION_MODEL: Optional[xgb.XGBRegressor] = None


@app.on_event("startup")
def load_models() -> None:
    global W2V_PROCESSOR, W2V_MODEL, ROBERTA_TOKENIZER, ROBERTA_MODEL, FUSION_MODEL

    W2V_PROCESSOR = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    W2V_MODEL = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(DEVICE)
    W2V_MODEL.eval()

    ROBERTA_TOKENIZER = AutoTokenizer.from_pretrained("roberta-base")
    ROBERTA_MODEL = AutoModel.from_pretrained("roberta-base").to(DEVICE)
    ROBERTA_MODEL.eval()

    FUSION_MODEL = xgb.XGBRegressor()
    FUSION_MODEL.load_model("xgb_fusion_model.json")


@app.get("/")
def health_check() -> dict:
    return {"status": "ok"}


def _require_loaded() -> None:
    if any(
        obj is None
        for obj in [W2V_PROCESSOR, W2V_MODEL, ROBERTA_TOKENIZER, ROBERTA_MODEL, FUSION_MODEL]
    ):
        raise HTTPException(status_code=503, detail="Models are not loaded yet.")


def _extract_features(audio: np.ndarray) -> tuple[str, np.ndarray]:
    if audio.size == 0:
        raise HTTPException(status_code=400, detail="Audio is empty.")

    assert W2V_PROCESSOR is not None
    assert W2V_MODEL is not None
    assert ROBERTA_TOKENIZER is not None
    assert ROBERTA_MODEL is not None

    with torch.inference_mode():
        # Acoustic branch: transcript + 768-d pooled wav2vec2 embedding.
        w2v_inputs = W2V_PROCESSOR(
            audio, sampling_rate=16000, return_tensors="pt", padding=True
        )
        w2v_inputs = {k: v.to(DEVICE) for k, v in w2v_inputs.items()}
        
        # FIX 1: Set output_hidden_states=True
        w2v_outputs = W2V_MODEL(**w2v_inputs, output_hidden_states=True, return_dict=True)

        predicted_ids = torch.argmax(w2v_outputs.logits, dim=-1)
        transcription = W2V_PROCESSOR.batch_decode(predicted_ids)[0].strip()

        # FIX 2: Access the last layer of the hidden_states tuple
        last_hidden_state = w2v_outputs.hidden_states[-1]
        
        acoustic_embedding = (
            last_hidden_state.mean(dim=1).squeeze(0).detach().cpu().numpy()
        )

        # Semantic branch: [CLS] token embedding from roberta.
        if transcription:
            rb_inputs = ROBERTA_TOKENIZER(
                transcription, return_tensors="pt", truncation=True, padding=True
            )
            rb_inputs = {k: v.to(DEVICE) for k, v in rb_inputs.items()}
            rb_outputs = ROBERTA_MODEL(**rb_inputs, return_dict=True)
            semantic_embedding = (
                rb_outputs.last_hidden_state[:, 0, :].squeeze(0).detach().cpu().numpy()
            )
        else:
            semantic_embedding = np.zeros(768, dtype=np.float32)

    if acoustic_embedding.shape[0] != 768:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected acoustic embedding shape: {acoustic_embedding.shape}",
        )
    if semantic_embedding.shape[0] != 768:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected semantic embedding shape: {semantic_embedding.shape}",
        )

    fused = np.concatenate([acoustic_embedding, semantic_embedding], axis=0).reshape(1, 1536)
    return transcription, fused


def _load_audio_av(file_path: str, sample_rate: int = 16000) -> np.ndarray:
    """Decode any audio format (WebM, Opus, MP4, OGG, WAV …) using PyAV.

    Returns a mono float32 numpy array resampled to `sample_rate` Hz.
    PyAV ships its own ffmpeg binaries so no system ffmpeg is required.
    """
    container = av.open(file_path)
    audio_stream = next((s for s in container.streams if s.type == "audio"), None)
    if audio_stream is None:
        raise ValueError("No audio stream found in the uploaded file.")

    resampler = av.AudioResampler(
        format="fltp",   # float32 planar
        layout="mono",
        rate=sample_rate,
    )

    chunks: list[np.ndarray] = []
    for frame in container.decode(audio_stream):
        for resampled in resampler.resample(frame):
            chunks.append(resampled.to_ndarray()[0])

    # Flush remaining samples
    for resampled in resampler.resample(None):
        chunks.append(resampled.to_ndarray()[0])

    container.close()

    if not chunks:
        return np.array([], dtype=np.float32)
    return np.concatenate(chunks, axis=0).astype(np.float32)


@app.post("/score-audio/")
async def score_audio(file: UploadFile = File(...)) -> dict:
    _require_loaded()
    assert FUSION_MODEL is not None

    try:
        content = await file.read()
        
        # 1. Grab the extension, with a safe fallback to .webm
        file_ext = os.path.splitext(file.filename or "")[1]
        if not file_ext:
            file_ext = ".webm"
            
        # 2. Create a secure temporary file on disk
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        # 3. GUARANTEE CLEANUP: Use try/finally so the file is always deleted
        try:
            # Use PyAV to decode — handles WebM/Opus, MP4, OGG, WAV without
            # requiring a system-level ffmpeg installation.
            audio = _load_audio_av(tmp_path, sample_rate=16000)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        # 4. Process through your pipeline as normal
        transcription, fused_features = _extract_features(audio)
        prediction = float(FUSION_MODEL.predict(fused_features)[0])

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to process audio: {exc}") from exc

    return {
        "filename": file.filename,
        "transcription": transcription,
        "grammar_score": round(prediction, 2),
    }