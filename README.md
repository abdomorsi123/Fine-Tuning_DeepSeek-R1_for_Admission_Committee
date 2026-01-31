# Fine-Tuning DeepSeek-R1 for MSc AI Admissions Decision Support

This repository contains a Kaggle notebook that fine-tunes **DeepSeek-R1-Distill-Llama-8B** using **Unsloth** to support decision-making for MSc Artificial Intelligence admissions at THWS.

The model is trained to analyze applicant transcripts and produce:
- A structured reasoning (Chain-of-Thought)
- A final decision: **ACCEPT**, **REJECT**, or **UNCLEAR**

The goal is **decision support**, not automated admission.

---

## Project Overview

University admissions often involve:
- Hard eligibility rules (ECTS, grades, subject requirements)
- Soft, human judgment for borderline or unclear cases

This project explores whether a fine-tuned reasoning LLM can:
- Apply formal admission regulations
- Explain its reasoning step by step
- Abstain (`UNCLEAR`) when information is missing or borderline

---

## Model & Tools

- **Base Model**: `unsloth/DeepSeek-R1-Distill-Llama-8B`
- **Fine-tuning**: LoRA via Unsloth
- **Frameworks**:  
  - Hugging Face Transformers  
  - TRL `SFTTrainer`  
  - PEFT  
- **Training Environment**: Kaggle (GPU)
- **Tracking**: Weights & Biases

---

## Dataset

### Training Data
- JSON format
- Each sample contains:
  - `question`: admission case prompt
  - `answer`: structured Chain-of-Thought + final decision

### Test Data
- Applicant transcripts only
- Ground-truth labels used **only for evaluation**

---

## Prompting Strategy

The model is instructed to:
1. Carefully analyze the transcript
2. Reason step-by-step (`<think>...</think>`)
3. Output one of:
   - `ACCEPT`
   - `REJECT`
   - `UNCLEAR`

The prompt explicitly enforces strict handling of:
- Grades
- ECTS requirements
- Non-German degree conversions
- Missing or incomplete transcripts

---

## Training Setup

- Context length: **8192 tokens**
- LoRA rank: **16**
- Batch size: **1** (with gradient accumulation)
- Epochs: **3**
- Optimizer: `adamw_8bit`
- Mixed precision: FP16 / BF16 (hardware dependent)

---

## Evaluation

The notebook compares **before vs after fine-tuning** using:
- Label distribution analysis
- Confusion matrices (raw & normalized)
- Abstention (`UNCLEAR`) rate
- Classification reports (excluding `UNCLEAR`)
- Schema section coverage (reasoning structure)
- Outcome comparison:
  - Correct
  - Wrong
  - Abstained

---

## Output

- Fine-tuned model uploaded to Hugging Face:
  - `AbdoMorsi/deepseek-admission-cot`
- Test predictions saved as JSON:
  - Before fine-tuning
  - After fine-tuning

---

## Disclaimer

This system is intended **only as a research prototype and decision-support tool**.  
Final admission decisions must always be made by qualified human committee members.

---
MSc Artificial Intelligence — THWS  
Junior Machine Learning Engineer / Data Scientist
