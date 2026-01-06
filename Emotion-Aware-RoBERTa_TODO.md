# To‑Do List — Reproducere *Emotion‑Aware RoBERTa* (ESA + TF‑IDF gating)

> Scop: implementare end‑to‑end a pipeline‑ului descris în articol (preprocesare → oversampling → TF‑IDF gating → RoBERTa → ESA → head → evaluare → optimizări inferență).  

---

## 1) Date: colectare, split, bazelines

### 1.1 Import dataset + split
- [ ] Încărcați datasetul (Kaggle / HF) și normalizați schema: `text`, `label`.
- [ ] Split 80/10/10 (train/val/test) cu stratificare după label.
- [ ] EDA rapid:
  - [ ] Distribuția claselor
  - [ ] Lungimi text (tokens) înainte/după curățare

**Deliverable:** `data/README.md` + script `prepare_data.py` care salvează spliturile.

### 1.2 Baseline RoBERTa (fără ESA, fără TF‑IDF gating)
- [ ] Fine‑tuning `roberta-base` cu head standard.
- [ ] Metrici: accuracy, precision/recall/F1 (macro + weighted), MCC, confusion matrix.
- [ ] Păstrați un checkpoint “baseline” pentru comparații.

**Deliverable:** `reports/baseline_results.md` + graf training (loss/acc).

---

## 2) Preprocesare avansată (înainte de TF‑IDF)

### 2.1 Curățare + normalizare text
- [ ] Implementați `clean_text(text)`:
  - [ ] eliminați URL-uri, user mentions, whitespace excesiv
  - [ ] eliminați simboluri non‑alfanumerice **dar păstrați punctuația relevantă emoțional** (ex. `!`, `?`)
  - [ ] decideți politica pentru emojis (ștergere sau mapare în tokens; dacă ștergeți, notați explicit)
- [ ] Teste unitare pe 20–30 exemple (în special social media).

**Deliverable:** `src/data/preprocess.py` + teste minime.

### 2.2 Tokenizare “slang‑aware”
- [ ] Folosiți `RobertaTokenizerFast`.
- [ ] Verificați:
  - [ ] handling pentru slang/abrevieri (“gonna”, “kinda” etc.)
  - [ ] max length (ex. 128/256) + truncation policy
  - [ ] dynamic padding cu `DataCollatorWithPadding`

**Deliverable:** `src/data/tokenize.py` + notebook scurt cu exemple.

---

## 3) Random Oversampling (înainte de TF‑IDF)

- [ ] Aplicați **Random Oversampling** pe **train** (nu pe val/test).
- [ ] Verificați distribuția claselor după oversampling.
- [ ] Salvați mapping-ul și statisticile (pentru raport).

**Deliverable:** `reports/oversampling.md` + funcție `oversample_train()`.

---

## 4) TF‑IDF gating (componenta 1)

### 4.1 Fit TF‑IDF pe train (după oversampling)
- [ ] Alegeți granularitatea:
  - [ ] word‑level (simplu) sau subword‑level (mai fidel tokenizării)
- [ ] `TfidfVectorizer` fit pe corpusul oversampled (train).

**Deliverable:** `src/features/tfidf.py` + fișier salvat `artifacts/tfidf_vectorizer.joblib`.

### 4.2 Thresholding + “min 4 tokens”
- [ ] Implementați filtrarea la prag **3.5**:
  - [ ] tokenii/cuvintele cu TF‑IDF < 3.5 sunt candidați de eliminare / reducere
- [ ] **Regulă de siguranță:** păstrați minimum **4 tokeni** per sample.
- [ ] Log: câți tokeni se păstrează în medie după gating.

**Deliverable:** raport `reports/tfidf_gating_stats.md`.

### 4.3 Două moduri de gating (alegeți 1; păstrați și 2 ca ablație)
- [ ] **Hard gating**: eliminați tokenii sub prag (reduce lungimea secvenței).
- [ ] **Soft gating**: păstrați secvența, dar **modulați embedding‑urile** cu o greutate derivată din TF‑IDF.

**Deliverable:** implementare clară + flag în config: `gating_mode: hard|soft`.

> Notă: articolul menționează explicit că gating-ul **modulează embeddings** înainte de RoBERTa, dar și că reduce overhead prin eliminarea tokenilor low‑info; e util să aveți ambele variante ca ablație.

---

## 5) Model: RoBERTa backbone + ESA layer + head (componenta 2)

### 5.1 Backbone
- [ ] Instanțiați `RobertaModel` (ex. `roberta-base`) cu output `last_hidden_state`.

**Deliverable:** `src/model/backbone.py`.

### 5.2 ESA layer (implementare conform ecuațiilor din articol)
- [ ] Implementați ESA ca modul custom:
  - [ ] **Adăugare positional encodings**: `E_input = E + P`
  - [ ] Proiecții Q, K, V: `Q=E_input Wq`, `K=E_input Wk`, `V=E_input Wv`
  - [ ] Atenție standard: `softmax(QK^T / sqrt(H))`
  - [ ] Output contextualizat `Z`
  - [ ] **Vector învățabil de scalare S** (dim H) aplicat element‑wise: `Z_final = (Attention(Q,K,V) ⊙ S) + P`
- [ ] Atenție la shapes:
  - [ ] `E: (B, L, H)`, `P: (L, H)` sau `(1, L, H)` cu broadcast
  - [ ] `S: (H,)` cu broadcast pe (B,L,H)
- [ ] Aplicați attention mask ca padding-ul să nu influențeze.

**Deliverable:** `src/model/esa.py` + teste de shape + forward pass pe batch dummy.

### 5.3 Pooling + head de clasificare
- [ ] Decideți pooling:
  - [ ] CLS pooling, mean pooling, sau attention pooling (consistent cu raportare)
- [ ] Head:
  - [ ] Dense + dropout (0.3) + `Softmax` pentru 6 clase
- [ ] Loss: `CrossEntropyLoss` (single‑label).

**Deliverable:** `src/model/emotion_aware_roberta.py` (model complet).

---

## 6) Training (hyperparams + buclă + logging)

### 6.1 Config standard (din articol)
- [ ] Optimizer: AdamW
- [ ] LR: `1e-5`
- [ ] Batch size: `16`
- [ ] Epochs: `10`
- [ ] Dropout: `0.3`
- [ ] Scheduler: linear / cosine (documentați alegerea)

**Deliverable:** `configs/train.yaml`.

### 6.2 Buclă de training
- [ ] Implementare cu HuggingFace `Trainer` **sau** custom loop PyTorch:
  - [ ] eval pe val la fiecare epocă
  - [ ] early stopping (opțional, dar util)
  - [ ] salvare best checkpoint by weighted F1 / accuracy

**Deliverable:** `scripts/train.py` + `reports/training_curves.md`.

---

## 7) Evaluare + ablații (calitate științifică)

### 7.1 Metrici
- [ ] accuracy
- [ ] macro F1 + weighted F1
- [ ] MCC
- [ ] confusion matrix
- [ ] raport per‑clasă (precision/recall/F1)

**Deliverable:** `reports/final_metrics.md` + `reports/confusion_matrix.png`.

### 7.2 Ablation study (minim 3 rulari)
- [ ] Baseline RoBERTa (fără ESA, fără TF‑IDF)
- [ ] RoBERTa + ESA (fără TF‑IDF)
- [ ] Full: RoBERTa + TF‑IDF gating + ESA

**Deliverable:** tabel comparativ `reports/ablation_table.md`.

### 7.3 Interpretabilitate: heatmaps / saliency
- [ ] Extrageți atenția/importance pentru exemple (ex. “Love … happiness”).
- [ ] Gerați heatmaps comparativ baseline vs emotion‑aware.

**Deliverable:** `reports/attention_heatmaps/` + descriere în `reports/xai_notes.md`.

---

## 8) Optimizare inferență (FP16 + AMP)

- [ ] Încărcați best checkpoint.
- [ ] Convertiți la FP16: `model.to(torch.float16)`
- [ ] Inferență în `torch.amp.autocast("cuda")`
- [ ] Benchmark:
  - [ ] timp mediu per sample
  - [ ] peak GPU memory

**Deliverable:** `reports/inference_benchmark.md` + tabel before/after.
