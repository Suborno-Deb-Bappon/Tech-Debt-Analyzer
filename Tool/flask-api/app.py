# app.py
# DebNetX+RAG:
# 1) Detect Technical Debt via saved DeBERTa (text) + CodeBERT (code) combined model
# 2) If TD exists, classify TD Type using GOOD RAG:
#       - OpenAI "text-embedding-3-large" embeddings
#       - Chroma persistent vector store
#       - Chunking + ANN retrieval
#       - gpt-5-mini for final JSON classification (temperature=0)
#
# Required env vars:
#   OPENAI_API_KEY=...           # OpenAI API key for embeddings + gpt-5-mini
#
# Optional env vars:
#   TD_RAG_CSV=td_types.csv      # path to CSV with columns: Cleaned Body, Final TD Type
#   CHROMA_DIR=./chroma          # path for Chroma persistence
#   COLLECTION_NAME=td_types_v1  # Chroma collection name
#   EMBEDDING_MODEL=text-embedding-3-large
#   RAG_TOP_K=6
#   CHUNK_CHAR_LEN=1800          # ~300-500 tokens depending on language
#   CHUNK_CHAR_OVERLAP=200
#
#   MODEL_STATE_PATH=deberta_codebert.pt
#   TOKENIZER_TEXT_DIR=sm_deberta_codebert/deberta_codebert/tokenizer_text
#   TOKENIZER_CODE_DIR=sm_deberta_codebert/deberta_codebert/tokenizer_code
#
# Endpoints:
#   GET  /health
#   POST /predict   { text, code }

import os, json, hashlib
from typing import List, Dict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from dotenv import load_dotenv
load_dotenv()

# --------------------------------------------------
# CUDA allocator hint 
# --------------------------------------------------
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# --------------------------------------------------
# Paths / Caches 
# --------------------------------------------------
BASE_OUT_DIR = "output"
HF_LOCAL_CACHE = os.path.join(BASE_OUT_DIR, "hf_cache")
os.makedirs(HF_LOCAL_CACHE, exist_ok=True)
os.environ["TRANSFORMERS_CACHE"] = HF_LOCAL_CACHE
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --------------------------------------------------
# Config (RAG)
# --------------------------------------------------
TD_RAG_CSV         = os.environ.get("TD_RAG_CSV", "td_types.csv")
CHROMA_DIR         = os.environ.get("CHROMA_DIR", "./chroma")
COLLECTION_NAME    = os.environ.get("COLLECTION_NAME", "td_types_v1")
EMBEDDING_MODEL    = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-large")
RAG_TOP_K          = int(os.environ.get("RAG_TOP_K", "3"))
CHUNK_CHAR_LEN     = int(os.environ.get("CHUNK_CHAR_LEN", "1800"))
CHUNK_CHAR_OVERLAP = int(os.environ.get("CHUNK_CHAR_OVERLAP", "200"))

TD_LABELS = [
    "Architecture","Build","Code","Design","Documentation",
    "Infrastructure","Test","Requirements","Versioning"
]

# --------------------------------------------------
# Model + label config 
# --------------------------------------------------
id2label = {0: "No Technical Debt", 1: "Technical Debt"}
label2id = {v: k for k, v in id2label.items()}
num_labels = len(id2label)
MAX_LEN_TEXT = 512
MAX_LEN_CODE = 512

# --------------------------------------------------
# Combined Model (DeBERTa text + CodeBERT code), loaded from saved state_dict
# --------------------------------------------------
from transformers import DebertaV2Model, DebertaV2Tokenizer, RobertaModel, RobertaTokenizer

class CombinedModel(nn.Module):
    def __init__(self, num_labels: int = 2):
        super().__init__()
        self.text_model = DebertaV2Model.from_pretrained('microsoft/deberta-v3-base', cache_dir=HF_LOCAL_CACHE)
        self.code_model = RobertaModel.from_pretrained('microsoft/codebert-base', cache_dir=HF_LOCAL_CACHE)

        text_hidden_size = self.text_model.config.hidden_size
        code_hidden_size = self.code_model.config.hidden_size

        self.adjust_code_hidden_size = nn.Linear(code_hidden_size, text_hidden_size)
        combined_hidden_size = text_hidden_size + text_hidden_size

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(combined_hidden_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512)
        )
        self.skip_linear = nn.Linear(combined_hidden_size, 512)
        self.final_classifier = nn.Linear(1024, num_labels)

    def forward(self, input_ids_text, attention_mask_text, input_ids_code, attention_mask_code=None):
        outputs_text = self.text_model(input_ids=input_ids_text, attention_mask=attention_mask_text)
        outputs_code = self.code_model(input_ids=input_ids_code, attention_mask=attention_mask_code)

        pooled_output_text = outputs_text.last_hidden_state[:, 0]
        pooled_output_code = self.adjust_code_hidden_size(outputs_code.last_hidden_state[:, 0])

        combined_output = torch.cat((pooled_output_text, pooled_output_code), dim=1)
        combined_output = self.dropout(combined_output)

        classifier_output = self.classifier(combined_output)
        skip_output = self.skip_linear(combined_output)
        final_input = torch.cat((classifier_output, skip_output), dim=1)

        logits = self.final_classifier(final_input)
        return logits

def ensure_pad_token(tok):
    if tok.pad_token is None:
        if tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        else:
            tok.add_special_tokens({"pad_token": "[PAD]"})
    tok.padding_side = "right"
    return tok

# Load tokenizers from saved dirs 
TOKENIZER_TEXT_DIR = os.environ.get("TOKENIZER_TEXT_DIR", "sm_deberta_codebert/deberta_codebert/tokenizer_text")
TOKENIZER_CODE_DIR = os.environ.get("TOKENIZER_CODE_DIR", "sm_deberta_codebert/deberta_codebert/tokenizer_code")
tok_text = DebertaV2Tokenizer.from_pretrained(TOKENIZER_TEXT_DIR, use_fast=True, cache_dir=HF_LOCAL_CACHE)
tok_code = RobertaTokenizer.from_pretrained(TOKENIZER_CODE_DIR, use_fast=True, cache_dir=HF_LOCAL_CACHE)
tok_text = ensure_pad_token(tok_text)
tok_code = ensure_pad_token(tok_code)

# Device + model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_STATE_PATH = os.environ.get("MODEL_STATE_PATH", "deberta_codebert.pt")
model = CombinedModel(num_labels=num_labels)
state = torch.load(MODEL_STATE_PATH, map_location=device)
model.load_state_dict(state, strict=True)
model.to(device)
model.eval()

@torch.inference_mode()
def logits_for_combined(text: str, code: str, max_len_text: int, max_len_code: int) -> np.ndarray:
    # Tokenize inputs 
    enc_text = tok_text(
        text or "",
        truncation=True,
        max_length=min(max_len_text, 512),
        padding=True,
        return_tensors="pt"
    )
    enc_code = tok_code(
        code or "",
        truncation=True,
        max_length=min(max_len_code, 512),
        padding=True,
        return_tensors="pt"
    )
    batch = {
        "input_ids_text": enc_text["input_ids"].to(device, non_blocking=True),
        "attention_mask_text": enc_text["attention_mask"].to(device, non_blocking=True),
        "input_ids_code": enc_code["input_ids"].to(device, non_blocking=True),
        "attention_mask_code": enc_code["attention_mask"].to(device, non_blocking=True),
    }
    logits = model(**batch).detach().cpu().numpy()
    return logits[0]

def softmax(x: np.ndarray) -> np.ndarray:
    m = x.max()
    e = np.exp(x - m)
    return e / e.sum()

# --------------------------------------------------
# OpenAI Client (embeddings + gpt-5-mini)
# --------------------------------------------------
from openai import OpenAI
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY for OpenAI embeddings and gpt-5-mini.")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

def openai_embed_texts(texts: List[str]) -> List[List[float]]:
    resp = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )
    return [d.embedding for d in resp.data]

# --------------------------------------------------
# Chroma vector store (persistent)
# --------------------------------------------------
import chromadb
from chromadb.config import Settings

os.makedirs(CHROMA_DIR, exist_ok=True)
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(allow_reset=True))

try:
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
except Exception:
    collection = chroma_client.create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

# --------------------------------------------------
# CSV → chunks → embeddings → Chroma
# --------------------------------------------------
def hash_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def chunk_text(s: str, size: int = CHUNK_CHAR_LEN, overlap: int = CHUNK_CHAR_OVERLAP) -> List[str]:
    s = (s or "").strip()
    if not s:
        return []
    chunks = []
    i = 0
    while i < len(s):
        chunk = s[i:i+size]
        chunks.append(chunk)
        if i + size >= len(s):
            break
        i += size - overlap
    return chunks

def ensure_corpus_indexed():
    if not os.path.exists(TD_RAG_CSV):
        raise RuntimeError(f"RAG CSV not found at {TD_RAG_CSV} (needs columns: Cleaned Body, Final TD Type).")

    df = pd.read_csv(TD_RAG_CSV, encoding="latin-1")

    if not {"Cleaned Body","Final TD Type"}.issubset(df.columns):
        raise RuntimeError("RAG CSV must have columns: Cleaned Body, Final TD Type")

    df["Cleaned Body"] = df["Cleaned Body"].astype(str).fillna("").str.strip()
    df["Final TD Type"] = df["Final TD Type"].astype(str).stripped() if hasattr(str, "stripped") else df["Final TD Type"].astype(str).str.strip()

    existing_count = collection.count()
    print(f"[RAG] Existing Chroma docs: {existing_count}")

    add_ids, add_docs, add_metas = [], [], []
    batch_limit = 128

    for _, row in df.iterrows():
        body = row["Cleaned Body"]
        td_type = row["Final TD Type"]

        doc_uid = hash_text(body)[:16]
        chunks = chunk_text(body)
        if not chunks:
            continue

        for c_idx, chunk in enumerate(chunks):
            cid = f"{doc_uid}_{c_idx}"
            try:
                res = collection.get(ids=[cid])
                if res and len(res.get("ids", [])) > 0:
                    continue
            except Exception:
                pass

            add_ids.append(cid)
            add_docs.append(chunk)
            add_metas.append({
                "doc_uid": doc_uid,
                "chunk_index": c_idx,
                "td_type": td_type
            })

            if len(add_ids) >= batch_limit:
                embs = openai_embed_texts(add_docs)
                collection.add(
                    ids=add_ids,
                    documents=add_docs,
                    metadatas=add_metas,
                    embeddings=embs
                )
                add_ids, add_docs, add_metas = [], [], []

    if add_ids:
        embs = openai_embed_texts(add_docs)
        collection.add(
            ids=add_ids,
            documents=add_docs,
            metadatas=add_metas,
            embeddings=embs
        )

    print(f"[RAG] Chroma now contains {collection.count()} chunks.")

def retrieve_contexts(query: str, top_k: int = RAG_TOP_K) -> List[Dict]:
    if not query.strip():
        return []
    q_emb = openai_embed_texts([query])[0]
    res = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k
    )
    outs = []
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    for doc, meta, dist in zip(docs, metas, dists):
        sim = 1.0 - float(dist) if dist is not None else None
        outs.append({
            "body": doc,
            "label": meta.get("td_type", ""),
            "similarity": sim
        })
    return outs

# Build / load index at startup
ensure_corpus_indexed()

# --------------------------------------------------
# GPT-5-mini prompt + call
# --------------------------------------------------
def build_td_type_prompt(query_text: str, retrieved: List[Dict], allowed_labels: List[str]) -> List[Dict]:
    td_definitions = {
        "Code": "Issues arising from suboptimal or poorly written code that needs refactoring.",
        "Design": "Flaws in the software design that hinder future development and scalability.",
        "Build": "Complications in the build process that slow down development or introduce errors.",
        "Architecture": "Deficiencies in the overall system architecture affecting its robustness and flexibility.",
        "Documentation": "Incomplete or outdated documentation that hampers understanding and maintenance.",
        "Infrastructure": "Shortcomings in the underlying infrastructure affecting deployment and operations.",
        "Requirements": "Unmet or poorly defined requirements that hinder future development and maintenance.",
        "Test": "Insufficient or inadequate testing that reduces the reliability of the software.",
        "Versioning": "Problems related to version control that complicate code management and integration."
    }

    system = (
        "You are a precise classifier for software technical debt categories.\n"
        "You will be given a Stack Overflow post and a list of retrieved similar examples.\n"
        "Each retrieved example includes a known technical debt label and similarity score.\n\n"
        "Use these steps:\n"
        "1. Carefully read the definitions of each technical debt type.\n"
        "2. Compare the query to the retrieved examples.\n"
        "3. Prefer choosing a label that appears frequently and with high similarity among the retrieved examples.\n"
        "4. Only pick a label if the post fits its definition. Do NOT cite unrelated evidence as supporting it.\n\n"
        "Output strictly as JSON with keys: type, rationale.\n"
        "In your rationale, reference specific retrieved example numbers that have the SAME label as your chosen type."
    )

    ctx_lines = []
    for i, r in enumerate(retrieved, 1):
        snippet = (r["body"] or "")[:900].replace("\n", " ").strip()
        sim_txt = f"{r['similarity']:.3f}" if r.get("similarity") is not None else "NA"
        ctx_lines.append(f"[{i}] label={r['label']} sim={sim_txt} : {snippet}")

    user = f"""
ALLOWED_LABELS = {allowed_labels}

QUERY (Stack Overflow post):
{query_text}

RETRIEVED_EXAMPLES (each has a known label and similarity score):
{os.linesep.join(ctx_lines)}

Make sure your rationale mentions only retrieved examples that match the chosen label.

Respond ONLY as valid JSON:
{{
  "type": "<one of {allowed_labels}>",
  "rationale": "<brief reason referencing retrieved evidence numbers that have the SAME label>"
}}
""".strip()

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]

def token_overlap(a: str, b: str) -> int:
    sa = set(a.replace("/"," ").replace("-"," ").split())
    sb = set(b.replace("/"," ").replace("-"," ").split())
    return len(sa & sb)

def best_label_fallback(pred: str, labels: List[str]) -> str:
    p = (pred or "").lower()
    scores = [(lab, token_overlap(p, lab.lower())) for lab in labels]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[0][0]

def call_gpt5mini_for_type(query_text: str, retrieved: List[Dict]) -> Dict:
    messages = build_td_type_prompt(query_text, retrieved, TD_LABELS)

    try:
        print("📨 Sending GPT-5-mini request...")
        resp = openai_client.chat.completions.create(
            model="gpt-5-mini",
            messages=messages,
            max_completion_tokens=1000
        )

        choice = resp.choices[0]
        if choice.finish_reason == "length":
            raise ValueError("GPT stopped due to token limit. Increase max_completion_tokens or shorten prompt.")

        content = (choice.message.content or "").strip()
        if not content:
            raise ValueError("Empty response from GPT (no content returned)")

        try:
            out = json.loads(content)
        except json.JSONDecodeError as je:
            raise ValueError(f"Invalid JSON returned: {je}")

        td_type = (out.get("type") or "").strip()
        rationale = (out.get("rationale") or "").strip()

        if td_type not in TD_LABELS:
            td_type = best_label_fallback(td_type, TD_LABELS)

        return {
            "type": td_type,
            "rationale": rationale or "(no rationale given)"
        }

    except Exception as e:
        print(f"⚠ GPT error: {e}")
        return {
            "type": "Null",
            "rationale": f"Could not parse GPT output or API failed: {e}"
        }

# --------------------------------------------------
# Flask App
# --------------------------------------------------
app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "td_detector_labels": id2label,
        "td_types": TD_LABELS,
        "rag_collection": COLLECTION_NAME,
        "rag_chunks": int(collection.count()),
        "embedding_model": EMBEDDING_MODEL
    })

@app.after_request
def add_cors_headers(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    resp.headers["Access-Control-Allow-Private-Network"] = "true"
    return resp

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    """
    Input JSON:
      { "text": "...", "code": "..." }

    Output JSON:
      {
        "prediction": "Technical Debt" | "No Technical Debt",
        "confidence": float,
        "probabilities": {label: prob, ...},
        "td_type": "<one of TD_LABELS>" | null,
        "td_type_rationale": str | null,
        "rag_used": [ {label, similarity, body_snippet}, ... ]
      }
    """
    if request.method == "OPTIONS":
        return ("", 204)

    try:
        data = request.get_json(force=True)
        text = (data.get("text") or "").strip()
        code = (data.get("code") or "").strip()
        if not text and not code:
            return jsonify({"error": "Provide at least 'text' or 'code'."}), 400

        # --- TD Detection via saved DeBERTa+CodeBERT combined model
        fused_logits = logits_for_combined(text, code, MAX_LEN_TEXT, MAX_LEN_CODE)
        probs = softmax(fused_logits)
        pred_id = int(probs.argmax())
        pred_label = id2label[pred_id]
        confidence = float(probs[pred_id])

        result = {
            "prediction": pred_label,
            "confidence": confidence,
            "probabilities": {id2label[i]: float(p) for i, p in enumerate(probs)},
            "td_type": None,
            "td_type_rationale": None,
            "rag_used": []
        }

        # --- If TD exists → RAG + GPT classification
        is_td = pred_label.lower() in {"technical debt","td","debt"} or \
                ("Technical Debt" in id2label.values() and pred_id == label2id.get("Technical Debt", -999))
        if is_td:
            query_text = (text + "\n\n" + code).strip()
            retrieved = retrieve_contexts(query_text, top_k=RAG_TOP_K)
            result["rag_used"] = [
                {
                    "label": r["label"],
                    "similarity": float(r["similarity"]) if r.get("similarity") is not None else None,
                    "body_snippet": (r["body"] or "")[:300]
                }
                for r in retrieved
            ]
            td_json = call_gpt5mini_for_type(query_text, retrieved)
            result["td_type"] = td_json.get("type")
            result["td_type_rationale"] = td_json.get("rationale")

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Bind 0.0.0.0 for extension
    app.run(host="0.0.0.0", port=5000)