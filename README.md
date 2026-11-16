# Day 7 — Agentic RAG (Multimodal WDR 2024)

**Bài cá nhân · Python 3.11+ · MSSV: `2A202603949` · Lê Tuấn Đạt**

Repo thực hiện đầy đủ phần cơ bản Day 7 (Agentic RAG với 4 chiến lược retrieval: `bm25`, `dense`, `hybrid_rrf`, `hybrid_rerank`, kèm mở rộng chunk lân cận và kiểm thử với model thật) và phần Thử thách nâng cao trên tài liệu **World Development Report 2024: The Middle-Income Trap** (World Bank, 276 trang vật lý, giấy phép **CC BY 3.0 IGO**).

---

## 1. Cấu trúc chính

- [configs/experiment.json](configs/experiment.json): cấu hình corpus, embedding model (`Xenova/all-MiniLM-L6-v2`, 384 chiều, ONNX quantized), local Qdrant store (`cosine` / `exact`), chunking và tiêu chí chọn cấu hình trên `dev`.
- [data/manifest.json](data/manifest.json): nguồn tải gốc `WDR_2024.pdf`, SHA-256 (`dc69d858e364a1534a57c49fbb72237f69d88165a63933a98f01f3c20262aded`), giấy phép `CC BY 3.0 IGO`, thống kê phân loại trang (`228` trang nội dung được index / `276` trang vật lý).
- [src/](src/): pipeline ingestion hai cột + trích bảng (`pdfplumber`) + render ảnh trang (`pypdfium2`), chunking, local ONNX MiniLM embedding + Qdrant store, 4 chiến lược retrieval + neighbor expansion, và Multimodal Tool-Calling Agent (`search`, `inspect_image`, `answer`, `abstain`).
- [benchmark/questions.json](benchmark/questions.json) & [benchmark/freeze.json](benchmark/freeze.json): `48` câu hỏi đã khóa hash trước khi chạy (`18` dev = `15` có đáp án + `3` thiếu bằng chứng; `30` test = `6` bảng + `5` ảnh/sơ đồ + `5` biểu đồ + `4` multi-hop + `4` text + `6` thiếu bằng chứng).
- [results/](results/): trace retrieval (`retrieval_dev/`, `retrieval_test/`), lựa chọn chốt trên dev (`dev_selection.json`), trace/nhãn audit/metrics của agent (`agent_dev_improved/`, `agent_test_improved/`, `agent_dev_baseline_dense/`, `agent_test_baseline_dense/`).
- [report/individual/2A202603949_LeTuanDat.md](report/individual/2A202603949_LeTuanDat.md): báo cáo kết quả, kiểm chứng đối chiếu bản gốc và phân tích lỗi.

---

## 2. Cài đặt và tái tạo dữ liệu

### 2.1. Tạo môi trường ảo và cài thư viện

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2.2. Chuẩn bị tài liệu gốc (không commit PDF gốc / `.env` / vector store)

Tải **World Development Report 2024: The Middle-Income Trap** về `data/raw/WDR_2024.pdf` (kiểm tra đúng SHA-256 trong [data/manifest.json](data/manifest.json)):

- URL tải trực tiếp: `https://openknowledge.worldbank.org/bitstreams/862729db-69a7-48a8-ba91-df8fe8d8745b/download`
- Trang metadata World Bank Open Knowledge: `https://openknowledge.worldbank.org/handle/10986/41919`

Tạo file `.env` từ [.env.example](.env.example) và điền API key (`VERTEX_API_KEY` cho Vertex AI Gemini hoặc `OPENAI_API_KEY` cho OpenAI).

---

## 3. Lệnh chạy tái lập

### 3.1. Chạy unit / integration test (không tốn API call)

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_day7_pipeline.py -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

### 3.2. Ingestion, chunking, render ảnh trang & build Qdrant index

```powershell
.\.venv\Scripts\python.exe -m src.build_index --force-render
```

Sinh ra `data/processed/` (`pages.jsonl`, `visuals.json`, `chunks.json`), `data/crops/page_*.png` (`scale=2.2`), `storage/qdrant-wdr2024/`, `results/preparation.json` và `results/index_summary.json`.

### 3.3. Đánh giá 5 cấu hình retrieval trên `dev` và `test`

```powershell
.\.venv\Scripts\python.exe -m src.eval_retrieval --split dev
.\.venv\Scripts\python.exe -m src.eval_retrieval --split test
```

### 3.4. Chạy Multimodal Agent với model thật và tổng hợp metrics

```powershell
# Chạy baseline và improved trên dev / test
.\.venv\Scripts\python.exe -m src.run_agent_eval --split dev --mode improved
.\.venv\Scripts\python.exe -m src.run_agent_eval --split test --mode improved
.\.venv\Scripts\python.exe -m src.run_agent_eval --split dev --mode baseline_dense
.\.venv\Scripts\python.exe -m src.run_agent_eval --split test --mode baseline_dense

# Tổng hợp metrics với nhãn audit thủ công
.\.venv\Scripts\python.exe starter/agent_metrics.py --cases benchmark/questions.json --traces results/agent_dev_improved/traces.json --reviews results/agent_dev_improved/reviews.json --split dev --output results/agent_dev_improved/metrics.json
.\.venv\Scripts\python.exe starter/agent_metrics.py --cases benchmark/questions.json --traces results/agent_test_improved/traces.json --reviews results/agent_test_improved/reviews.json --split test --output results/agent_test_improved/metrics.json
```

---

## 4. Hướng dẫn gốc của đề bài

- [GUIDE.md](GUIDE.md), [BENCHMARK.md](BENCHMARK.md), [ADVANCED.md](ADVANCED.md), [RUBRIC.md](RUBRIC.md), [RULES.md](RULES.md), [SUBMISSION.md](SUBMISSION.md).
- Bản gốc starter được giữ tại [starter/](starter/).
