import { useMemo, useRef, useState } from "react";

// VITE_API_URL is set in Vercel's environment variables (points to HF Spaces backend).
// Falls back to localhost for local development.
const API_URL = (import.meta.env.VITE_API_URL ?? "http://localhost:8000") + "/score-audio/";
// Dynamically check what audio format the user's browser actually supports
function getSupportedMimeType() {
  const possibleTypes = [
    "audio/webm",
    "audio/mp4",
    "audio/ogg",
    "audio/wav"
  ];
  for (const type of possibleTypes) {
    if (MediaRecorder.isTypeSupported(type)) {
      return type;
    }
  }
  return ""; // Fallback to browser default
}

function normalizeAudioFile(file) {
  if (!file) return file;
  if (/\.weba$/i.test(file.name)) {
    const normalizedName = file.name.replace(/\.weba$/i, ".webm");
    return new File([file], normalizedName, { type: file.type || "audio/webm" });
  }
  return file;
}

function getScoreMeta(score) {
  if (score >= 8) {
    return { label: "Strong", badge: "bg-emerald-500", bar: "bg-emerald-500" };
  }
  if (score >= 5) {
    return { label: "Moderate", badge: "bg-amber-500", bar: "bg-amber-500" };
  }
  return { label: "Needs Improvement", badge: "bg-rose-500", bar: "bg-rose-500" };
}

export default function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [audioPreviewUrl, setAudioPreviewUrl] = useState("");
  const [isDragging, setIsDragging] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  const recorderRef = useRef(null);
  const streamRef = useRef(null);
  const chunksRef = useRef([]);

  const score = Number(result?.grammar_score ?? 0);
  const scorePercent = useMemo(() => Math.max(0, Math.min(100, (score / 10) * 100)), [score]);
  const scoreMeta = useMemo(() => getScoreMeta(score), [score]);

  const clearStateForNewInput = () => {
    setError("");
    setResult(null);
  };

  const applyFile = (file) => {
    if (!file) return;
    clearStateForNewInput();
    const normalizedFile = normalizeAudioFile(file);

    // We expanded the allowed extensions to cover Safari/iOS defaults
    const isAudioType = normalizedFile.type.startsWith("audio/");
    const isAllowedExt = /\.(wav|webm|weba|mp4|m4a|ogg)$/i.test(normalizedFile.name);
    
    if (!isAudioType && !isAllowedExt) {
      setError("Please provide an audio file (.wav, .webm, .mp4, or .ogg).");
      return;
    }

    setSelectedFile(normalizedFile);

    if (audioPreviewUrl) {
      URL.revokeObjectURL(audioPreviewUrl);
    }
    setAudioPreviewUrl(URL.createObjectURL(normalizedFile));
  };

  const onFileChange = (event) => {
    const file = event.target.files?.[0];
    applyFile(file);
  };

  const onDrop = (event) => {
    event.preventDefault();
    setIsDragging(false);
    const file = event.dataTransfer.files?.[0];
    applyFile(file);
  };

  const startRecording = async () => {
    clearStateForNewInput();
    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        setError("Your browser does not support voice recording.");
        return;
      }

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      chunksRef.current = [];

      // Find the best supported format for the current browser
      const mimeType = getSupportedMimeType();
      
      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
      recorderRef.current = recorder;

      recorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      recorder.onstop = () => {
        // Extract the correct extension (e.g., 'webm' or 'mp4') from the mimeType
        const actualMimeType = recorder.mimeType || "audio/webm";
        const extension = actualMimeType.split('/')[1].split(';')[0]; 
        
        const blob = new Blob(chunksRef.current, { type: actualMimeType });
        const file = new File([blob], `recorded_${Date.now()}.${extension}`, { type: actualMimeType });
        applyFile(file);

        if (streamRef.current) {
          streamRef.current.getTracks().forEach((track) => track.stop());
          streamRef.current = null;
        }
      };

      recorder.start();
      setIsRecording(true);
    } catch (recordError) {
      setError(`Microphone error: ${recordError.message}`);
    }
  };

  const stopRecording = () => {
    if (!recorderRef.current) return;
    recorderRef.current.stop();
    setIsRecording(false);
  };

  const analyzeAudio = async () => {
    if (!selectedFile) {
      setError("Upload or record an audio sample first.");
      return;
    }

    setIsLoading(true);
    setError("");
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const response = await fetch(API_URL, {
        method: "POST",
        body: formData
      });

      if (!response.ok) {
        const errorPayload = await response.text();
        throw new Error(`Backend returned ${response.status}: ${errorPayload}`);
      }

      const payload = await response.json();
      setResult(payload);
    } catch (requestError) {
      setError(requestError.message || "Unable to process audio right now.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-[radial-gradient(circle_at_20%_20%,#cffafe,transparent_40%),radial-gradient(circle_at_85%_10%,#bbf7d0,transparent_35%),linear-gradient(140deg,#f8fafc,#e2e8f0)] px-4 py-8 sm:px-8">
      <div className="mx-auto max-w-4xl">
        <section className="rounded-3xl border border-slate-200/70 bg-white/85 p-6 shadow-2xl backdrop-blur-md sm:p-8">
          <header>
            <h1 className="text-3xl font-extrabold tracking-tight sm:text-4xl">
              Audio Grammar Analyzer
            </h1>
            <p className="mt-2 text-slate-600">
              Upload or record voice and score grammatical fluency using your multimodal backend.
            </p>
          </header>

          <div
            className={`mt-8 rounded-2xl border-2 border-dashed p-6 text-center transition ${
              isDragging ? "border-cyan-500 bg-cyan-50" : "border-slate-300 bg-slate-50/80"
            }`}
            onDragOver={(event) => {
              event.preventDefault();
              setIsDragging(true);
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={onDrop}
          >
            <p className="text-lg font-semibold">Drag and drop an audio file</p>
            <p className="mt-1 text-sm text-slate-500">Supported: .wav, .webm, .mp4, or .ogg</p>

            <label className="mt-4 inline-flex cursor-pointer items-center rounded-xl bg-slate-900 px-4 py-2 text-sm font-semibold text-white hover:bg-slate-800">
              Choose Audio File
              <input type="file" accept="audio/*" className="hidden" onChange={onFileChange} />
            </label>
          </div>

          <div className="mt-4 flex flex-wrap gap-3">
            {!isRecording ? (
              <button
                onClick={startRecording}
                className="rounded-xl bg-emerald-600 px-4 py-2 font-semibold text-white hover:bg-emerald-500"
              >
                Record Voice
              </button>
            ) : (
              <button
                onClick={stopRecording}
                className="rounded-xl bg-rose-600 px-4 py-2 font-semibold text-white hover:bg-rose-500"
              >
                Stop Recording
              </button>
            )}

            <button
              onClick={analyzeAudio}
              disabled={isLoading}
              className="rounded-xl bg-cyan-600 px-4 py-2 font-semibold text-white hover:bg-cyan-500 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {isLoading ? "Processing AI Models..." : "Analyze Grammar"}
            </button>
          </div>

          {selectedFile && (
            <div className="mt-5 rounded-xl border border-slate-200 bg-slate-50 p-4">
              <p className="text-sm text-slate-500">Selected file</p>
              <p className="break-all font-semibold">{selectedFile.name}</p>
              {audioPreviewUrl && <audio src={audioPreviewUrl} controls className="mt-3 w-full" />}
            </div>
          )}

          {error && (
            <div className="mt-5 rounded-xl border border-rose-200 bg-rose-50 p-4 text-rose-700">
              {error}
            </div>
          )}

          {result && (
            <div className="mt-8 space-y-5">
              <div className="rounded-2xl border border-slate-200 bg-white p-5">
                <p className="text-sm text-slate-500">Transcription</p>
                <p className="mt-2 text-lg leading-relaxed">
                  {result.transcription || "(No transcript generated)"}
                </p>
              </div>

              <div className="rounded-2xl border border-slate-200 bg-white p-5">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <p className="text-sm text-slate-500">Grammar Score (out of 10)</p>
                  <span className={`rounded-full px-3 py-1 text-xs font-bold text-white ${scoreMeta.badge}`}>
                    {scoreMeta.label}
                  </span>
                </div>

                <p className="mt-2 text-4xl font-extrabold tracking-tight">{score.toFixed(2)}</p>

                <div className="mt-4 h-3 w-full overflow-hidden rounded-full bg-slate-200">
                  <div
                    className={`h-full ${scoreMeta.bar} transition-all duration-500`}
                    style={{ width: `${scorePercent}%` }}
                  />
                </div>
              </div>
            </div>
          )}
        </section>
      </div>
    </main>
  );
}