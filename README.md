# Support Copilot – Agentic RAG with Validation & Fine-Tuning

A production-style **Agentic Retrieval-Augmented Generation (RAG)** system designed to reduce hallucinations and enforce grounded, reliable responses in support workflows.

This project demonstrates how to build a **trustworthy AI agent** using deterministic guardrails, validation loops, structured outputs, and behavior-aligned fine-tuning — moving beyond basic RAG demos toward production-ready reliability.

---

## 🚀 Key Results

- **25% reduction** in unsupported responses through validation and retry logic  
- Retry rate reduced from **0.8 → 0.6** after behavior fine-tuning  
- **100% citation validation coverage** via deterministic grounding checks  
- Eliminated hallucinated tool actions using structured JSON outputs  
- Improved confidence calibration under strict evaluation  

This system focuses on reliability and measurable behavior improvement rather than raw generation quality.

---

## ❗ Problem This Solves

Standard RAG systems frequently fail in support scenarios by:

- Inventing policy thresholds  
- Recommending unsupported escalation actions  
- Producing confident but ungrounded answers  
- Mixing domains when retrieval is weak  

Support Copilot addresses this by combining:

- deterministic guardrails  
- LLM validation  
- structured outputs  
- agentic retry logic  

to ensure responses remain grounded in retrieved evidence.

---

## ⚠️ Example Failure → Recovery

**User Query**

> Can I escalate refund requests above $500?

### Initial Generation

The model attempts escalation guidance without sufficient KB support.

### Hard Check Result

**FAIL:** Escalation not supported by retrieved evidence

### System Action

RETRY_WITH_MORE_CONTEXT
→ retrieval depth increased
→ response regenerated
→ validation re-run


### Final Output

Grounded response generated without unsupported escalation instructions.

This demonstrates how the system handles common RAG failure modes instead of relying on ideal inputs.

---


# 🏗️ System Architecture

```
User Question
     ↓
Retrieval Layer
  - Knowledge Base (MSMARCO / Amazon QA)
  - Historical Tickets (Twitter Support)
     ↓
Generator (Structured JSON Output)
     ↓
Hard Checks
  - Citation validation
  - Escalation grounding
  - Domain match
     ↓
LLM Validator (LLM-as-judge)
     ↓
Decision:
  PASS | RETRY_WITH_MORE_CONTEXT | REFUSE
     ↓
Final Answer
```

---
# 🧠 Design Highlights

## 1️⃣ Structured Outputs Only

All answers must match:

```json
{
  "answer": "...",
  "citations": [{"source": "...", "chunk_id": 123}],
  "confidence": "low | medium | high",
  "missing_info": "...",
  "next_steps": [],
  "similar_cases": []
}
```

This eliminates vague, unverifiable responses.

---

## 2️⃣ Deterministic Guardrails Before LLM Validation

- Citation must match retrieved CITE_KEY
- Escalation blocked unless KB mentions escalation
- Domain mismatch rejected

Fast, cheap, and reliable hallucination control.

---

## 3️⃣ LLM-as-Judge Validation

A second model verifies:

- Every actionable instruction is supported
- No invented policy thresholds
- Answer remains in-domain

If validation fails → system retries with higher retrieval depth.

---

## 4️⃣ Agentic Retry Loop

```
RETRY_WITH_MORE_CONTEXT
→ Increase retrieval k
→ Regenerate
→ Revalidate
```

Mimics production fallback logic.

---

## 5️⃣ Behavior Fine-Tuning

Fine-tuned model trained on:

- RAG-style question + KB evidence
- Strict JSON schema outputs
- Low-confidence training examples
- Guardrail-aligned responses

Result:
- Reduced unsupported escalation
- Improved validation pass rate
- Better confidence calibration

---

# 📊 Evaluation

Run evaluation:

```bash
python -m eval.run_eval
```

Compares:
- Baseline model
- Fine-tuned model

Metrics:
- Retry rate
- Pass count
- Citation count
- Confidence distribution

---

# 📂 Repository Structure

```
agent/
  graph.py              # Agent state machine
  hard_checks.py        # Deterministic guardrails
  retrieval.py          # Multi-source retrieval
  state.py              # Typed agent state
  validator.py          # LLM-as-judge validation

rag/
  embeddings.py         # Embedding wrapper
  generator.py          # Structured JSON generator
  schema.py             # Pydantic output schema
  validate.py           # Schema validation
  vectorstore.py        # FAISS index loader

ingestion/
  kb_passages.py        # KB preprocessing
  tickets.py            # Ticket preprocessing

finetune/
  build_dataset.py
  split_dataset.py
  inspect_dataset.py
  data/

eval/
  run_eval.py           # Quantitative evaluation

scripts/
  download_*.py         # Dataset ingestion
  build_*_index.py      # FAISS index builders
  hello_*.py            # Demo scripts
  openai_*.py           # Fine-tuning utilities
```

---

# ⚙️ Setup

## 1️⃣ Clone

```bash
git clone <repo>
cd support-copilot-agentic-rag
```

## 2️⃣ Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3️⃣ Configure `.env`

```
OPENAI_API_KEY=...

BASE_MODEL=gpt-3.5-turbo-0125
GEN_MODEL=ft:gpt-3.5-turbo-0125:personal::D8YYyJwR
TUNED_MODEL=ft:gpt-3.5-turbo-0125:personal::D8YYyJwR

DEBUG_VALIDATOR=0
```

---

# 📦 Data Ingestion

Download datasets:

```bash
python -m scripts.download_twitter_support_subset
python -m scripts.download_kb_msmarco_subset
```

Build indexes:

```bash
python -m scripts.build_tickets_index
python -m scripts.build_kb_index
```

---

# ▶️ Run Demo

```bash
python -m scripts.hello_agentic
```

---

# 🔍 Debug Mode

Enable detailed validator logs:

```bash
export DEBUG_VALIDATOR=1
python -m eval.run_eval
```

---

# 🔁 Fine-Tuning Workflow

1. Build dataset:
```bash
python -m finetune.build_dataset
```

2. Split:
```bash
python -m finetune.split_dataset
```

3. Upload:
```bash
python -m scripts.openai_upload_finetune_files
```

4. Create job:
```bash
python -m scripts.openai_create_finetune_job
```

5. Monitor:
```bash
python -m scripts.openai_check_finetune_job
```

---

# 🛡️ Why This Is Not a Basic RAG Demo

| Feature | Basic RAG | This System |
|----------|------------|-------------|
| Citation grounding | ❌ | ✅ |
| Deterministic guardrails | ❌ | ✅ |
| Escalation enforcement | ❌ | ✅ |
| Domain mismatch detection | ❌ | ✅ |
| Retry loop | ❌ | ✅ |
| Quantitative evaluation | ❌ | ✅ |
| Behavior fine-tuning | ❌ | ✅ |

---

# 🎯 What This Project Demonstrates

- Production-style RAG architecture
- Deterministic + LLM hybrid validation
- Agentic orchestration with retries
- Strict grounding enforcement
- Behavior fine-tuning for reliability
- Measurable performance improvements

---

# 👤 Author

Built to explore advanced RAG + validation + fine-tuning pipelines with production-aligned safety mechanisms.
