# 📘 Tech-Debt-Analyzer (DebtNetX)

Detect **technical debt** in Stack Overflow questions in **real time**, right in the browser — powered by a fine-tuned **DeBERTa + CodeBERT** classifier and an OpenAI-backed **RAG** pipeline for Technical Debt Type classification.

---

## 📑 Overview

Tech-Debt-Analyzer includes:

- 🧩 A **Chrome extension** that injects a “Detect Technical Debt” button into Stack Overflow question pages  
- 🔍 A **Flask API backend** that performs:  
  - Technical Debt Detection (binary classifier)  
  - Technical Debt **Type** classification using **RAG + GPT**  
- 🧠 A **training notebook** (`source_code.ipynb`) for building the combined DeBERTa + CodeBERT model

**Supported TD Types:**

- Architecture  
- Build  
- Code  
- Design  
- Documentation  
- Infrastructure  
- Requirements  
- Test  
- Versioning  

---

## 📂 Repository Structure

    Tech-Debt-Analyzer/
    ├─ Tool/
    │ ├─ chrome-extension/
    │ │ ├─ manifest.json
    │ │ ├─ content.js
    │ │ └─ background.js
    │ └─ flask-api/
    │ ├─ app.py
    │ ├─ requirements.txt
    │ ├─ td_types.csv
    │ └─ output/
    │ └─ chroma/
    ├─ assets/ 
    ├─ source_code.ipynb
    └─ .git/


> ⚠️ **Must be provided locally:**  
> - Fine-tuned model weights (`deberta_codebert.pt`)  
> - DeBERTa and CodeBERT tokenizer directories  

---

## ⚙️ Backend (Flask API)

### Requirements

- Python **3.10+**
- OpenAI API Key  
- Optional GPU (PyTorch will use CUDA if available)

Install dependencies:

```bash
pip install -r Tool/flask-api/requirements.txt
```
---
## Environment Variables
Required:
```bash
OPENAI_API_KEY=
```
Optional (defaults shown):
```bash
MODEL_STATE_PATH=deberta_codebert.pt
TOKENIZER_TEXT_DIR=sm_deberta_codebert/deberta_codebert/tokenizer_text
TOKENIZER_CODE_DIR=sm_deberta_codebert/deberta_codebert/tokenizer_code

TD_RAG_CSV=td_types.csv
CHROMA_DIR=./chroma
COLLECTION_NAME=td_types_v1
EMBEDDING_MODEL=text-embedding-3-large

RAG_TOP_K=3
CHUNK_CHAR_LEN=1800
CHUNK_CHAR_OVERLAP=200
```
To reuse the included local vector store:
```bash
export CHROMA_DIR=output/chroma
```
---
## Running the API
```bash
cd Tool/flask-api
python app.py
```
Runs on:
```
http://localhost:5000
```
---
## API Endpoints
### GET /health

Returns basic status + configuration metadata.

### POST /predict

Request Body:
```json
{
  "text": "question body text...",
  "code": "extracted code blocks..."
}
```
Example Response:
```json
{
  "prediction": "Technical Debt",
  "confidence": 0.92,
  "probabilities": {
    "No Technical Debt": 0.08,
    "Technical Debt": 0.92
  },
  "td_type": "Code",
  "td_type_rationale": "Explanation...",
  "rag_used": [
    {
      "label": "Code",
      "similarity": 0.87,
      "body_snippet": "..."
    }
  ]
}
```
### 📸 API Output Snapshot

A visual snapshot of the tool’s prediction and Technical Debt Type classification:

![API Output Screenshot](./assets/api_output_snapshot.png)
---
## 🧩 Chrome Extension
- What It Does
- Detects Stack Overflow question pages
- Adds a “Detect Technical Debt” button
- Extracts:
    - Text
    - Code blocks
- Sends them to the Flask API
- Displays:
    - TD / No TD with confidence bar
    - Technical Debt Type
    - GPT-generated explanation
    - RAG evidence examples
---
### 📸 Chrome Extension Snapshot

Here is how the **Detect Technical Debt** button look inside Stack Overflow:

![Chrome Extension Screenshot](./assets/chrome_extension_snapshot.png)
---
## Installing the Extension

- Open Chrome → chrome://extensions
- Enable Developer Mode
- Select Load Unpacked
- Choose:
```swift
Tech-Debt-Analyzer/Tool/chrome-extension
```
Visit any Stack Overflow question → click Detect Technical Debt.
---
## 🧠 Model & RAG System

### **Model Architecture**

- **Text Encoder:** DeBERTa-v3-base  
- **Code Encoder:** CodeBERT  
- Code CLS projection → concatenated → MLP classifier  

**Outputs:**
- "Technical Debt"  
- "No Technical Debt"  

---

### **RAG Pipeline**

- Builds/loads a **ChromaDB** vector store using `td_types.csv`  
- Embeds text chunks with **text-embedding-3-large**  
- Retrieves **top-K similar chunks**  
- Sends them to **GPT-5-mini** for:
  - Technical Debt Type classification  
  - Explanation generation  

---

## 🧪 Training

Training is provided in:

`source_code.ipynb`

**Includes:**

- Data preprocessing  
- Tokenization  
- Model construction  
- Training loop  
- Saving:
  - `deberta_codebert.pt`
  - Tokenizer folders  

These must match the paths used by the Flask API.

---

## 🛠️ Troubleshooting

**Extension error:**  
“Failed to get prediction from the local API.”  
→ Ensure Flask is running at **http://localhost:5000**.

**API error:**  
“Set OPENAI_API_KEY.”  
→ Add environment variable or `.env` file.

**Model load issues:**  
→ Check tokenizer and model weight paths.

**RAG has zero chunks:**  
→ Ensure `td_types.csv` is valid  
→ Or use provided vector store:

```bash
export CHROMA_DIR=output/chroma
```
---