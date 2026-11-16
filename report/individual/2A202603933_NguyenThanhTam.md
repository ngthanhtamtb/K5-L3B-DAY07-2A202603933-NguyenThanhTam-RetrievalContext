# Báo cáo Thực nghiệm Day 7 — Agentic RAG
**Học viên:** Nguyễn Thành Tâm  
**MSSV:** 2A202603933  
**Lớp:** L3B  
**Chủ đề & Corpus:** Lịch sử Đế chế La Mã (*The History of the Decline and Fall of the Roman Empire*, Edward Gibbon - Tập I, Chương I–IV)  
**Môi trường triển khai:** 100% Local
---

## 1. Dữ liệu và benchmark

- **Nguồn, quyền sử dụng, số trang và số chunk:**
  - Văn bản: *The History of the Decline and Fall of the Roman Empire*, Volume I (Chapters I–IV) của Edward Gibbon (Project Gutenberg eBook #25717, Public Domain).
  - Quy mô: 62 trang hợp lệ (327.417 ký tự chuẩn hóa).
  - Tổng số chunk: **884 chunks** (kế thừa trọn vẹn từ Chiến lược B - Paragraph Chunking của Day 6, lưu trong `data/chunks.json`).
- **Vector store, embedding model, phiên bản và số chiều:**
  - Vector Store: **PostgreSQL 16 + pgvector 0.8.7** chạy Docker Compose local tại `127.0.0.1:55433` (database `day6_lab`, bảng `chunks`).
  - Embedding Model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (dimension = 384), chạy local qua `sentence-transformers==6.3.0`.
  - Metric tìm kiếm: Cosine similarity (`score = 1 - (embedding <=> query_vec)`).
- **Agent model, cách chạy và phân chia dev/test:**
  - Agent Controller: `LocalAgentController` cài đặt giao thức `Controller Protocol` theo thiết kế Grounding-first Reasoning (100% offline, suy luận nội bộ dựa trên entity & lexical-semantic overlap).
  - Phân chia benchmark (24 câu theo chuẩn `BENCHMARK.md`):
    - **Dev (17 câu):** 15 câu có đáp án (q01–q15) + 2 câu thiếu bằng chứng (q21_unans, q22_unans).
    - **Test (7 câu):** 5 câu có đáp án (q16–q20) + 2 câu thiếu bằng chứng (q23_unans, q24_unans).
- **Cách xác định câu có/thiếu evidence và tiêu chí chấm:**
  - Câu có đáp án (`answerable: true`): Chấm điểm Grounded Answer theo thang 0 (sai/không trích dẫn), 1 (đúng một phần), 2 (đúng và đủ bằng chứng). Citation bắt buộc có dạng `[chunk_id]` và nằm trong context thực tế.
  - Câu thiếu bằng chứng (`answerable: false`): Các câu hỏi in-domain distractor ngoài phạm vi Chương I–IV (như sự kiện Valerian, Constantine, Alaric, Belisarius). Yêu cầu agent kích hoạt trạng thái **Abstain** (từ chối khẳng định vô căn cứ).

---

## 2. Retrieval và thử nghiệm

### Bảng so sánh các phương pháp Retrieval và Cấu hình Agent trên Dev & Test

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|:---:|:---:|:---:|:---:|:---:|
| **Baseline (BM25)** | Okapi BM25, top-k=3, budget=800 chars | Truy xuất từ khóa chính xác | dev | 9/15 (60.0%) | 13/15 (86.7%) | 9/15 (60.0%) | 1.97 ms / $0.00 |
| **Phương pháp 2 (Dense)** | `pgvector` Cosine, top-k=3, budget=800 chars | Bắt ngữ nghĩa vector Day 6 | dev | 10/15 (66.7%) | 11/15 (73.3%) | 10/15 (66.7%) | 65.76 ms / $0.00 |
| **Phương pháp 3 (Hybrid RRF)** | Dense + BM25 RRF fusion ($k=60$), top-k=3, budget=800 chars | Kết hợp điểm mạnh từ khóa và ngữ nghĩa | dev | 11/15 (73.3%) | 14/15 (93.3%) | 11/15 (73.3%) | 53.93 ms / $0.00 |
| **Cải thiện 1 (Agent Dense)** | Dense Retriever + Agent Controller, budget=800 chars | Agent điều phối và kiểm soát trích dẫn | dev | 11/15 (73.3%) | 11/15 (73.3%) | 11/15 (73.3%) | 47.0 ms / $0.00 |
| **Cấu hình tối ưu (Hybrid + Budget 1200)** | Hybrid RRF + Entity Grounding Controller, budget=1200 chars | Mở rộng budget tránh rớt context, lọc entity kiểm soát hallucination | dev | 14/15 (93.3%) | 14/15 (93.3%) | 14/15 (93.3%) | 53.6 ms / $0.00 |
| **Baseline trên Test** | Dense Retriever + Agent Controller, budget=800 chars | Đánh giá độc lập baseline trên hold-out test | test | 3/5 (60.0%) | 3/5 (60.0%) | 3/5 (60.0%) | 70.1 ms / $0.00 |
| **Cấu hình tối ưu trên Test** | Hybrid RRF + Entity Grounding Controller, budget=1200 chars | Kiểm chứng cấu hình đã freeze trên test | test | **4/5 (80.0%)** | **4/5 (80.0%)** | **4/5 (80.0%)** | 63.8 ms / $0.00 |

### Phân tích lỗi trên tập Dev và Lý do chọn Cấu hình

1. **Lỗi 1 (Câu q11 - False Abstention do context budget thấp):**
   - *Mô tả:* Câu hỏi về 2 chức vụ trọn đời của Augustus (*consular* và *tribunitian*). Ở Baseline (budget=800 chars), đoạn văn `p041_para_001` bị cắt khỏi context do dung lượng không đủ sau khi nạp đoạn `p039`.
   - *Khắc phục:* Mở rộng context budget lên 1200 chars kết hợp xếp hạng Hybrid RRF giúp chunk `p041` lọt trọn vẹn vào context $\to$ Agent trả lời chính xác điểm 2/2.
2. **Lỗi 2 (Câu q21_unans - Nguy cơ False Positive khi gặp tên riêng trong footnote):**
   - *Mô tả:* Câu hỏi về hoàng đế Constantine dời đô (ngoài phạm vi). Đoạn footnote `p019_para_011` có nhắc thoáng qua chữ "Rome, founded by Constantine" khiến retriever lấy về và agent suýt trả lời nhầm.
   - *Khắc phục:* Bổ sung cơ chế **Entity Grounding Verification** trong `LocalAgentController`: kiểm tra sự hiện diện đầy đủ của các thực thể chính của câu hỏi trong đoạn trích dẫn. Nếu không đủ căn cứ, agent từ chối trả lời (Abstain) đạt chuẩn 100%.

👉 **Quyết định chốt cấu hình (Freeze):** Chọn **Hybrid RRF + Entity Grounding Controller (Budget 1200 chars)** vì đạt chất lượng tổng thể cao nhất (Grounded Answer 1.87/2.0 trên Dev, giảm 75% lỗi từ chối nhầm).

---

## 3. Agent và đánh giá

- **Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi:**
  - Tools: Cung cấp `search` (gọi Hybrid / Dense / BM25 Retriever).
  - Giới hạn: `max_searches = 4`, `top_k = 3`, `budget = 1200` ký tự.
  - Chọn context: Loại bỏ trùng lặp (`deduplication`), gói văn bản theo định dạng `[chunk_id | source | page] text`.

### Bảng chỉ số Agent trên Dev và Test (Tính bằng `starter/agent_metrics.py`)

| Cấu hình / split | Tool use | Số lượt gọi | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm (False Abstention) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline Dense (dev)** | 17/17 (100%) | 1–2 lượt | 0: 4 \| 1: 0 \| 2: 11 (TB: 1.47/2) | 11/11 (100%) | 11/11 (100%) | 2/2 (100%) | 4/15 (26.7%) |
| **Cấu hình tối ưu (dev)** | 17/17 (100%) | 1–2 lượt | 0: 1 \| 1: 0 \| 2: 14 (**TB: 1.87/2**) | 14/14 (100%) | 14/14 (100%) | **2/2 (100%)** | **1/15 (6.7%)** |
| **Baseline Dense (test)** | 7/7 (100%) | 1–2 lượt | 0: 2 \| 1: 0 \| 2: 3 (TB: 1.20/2) | 3/3 (100%) | 3/3 (100%) | 2/2 (100%) | 2/5 (40.0%) |
| **Cấu hình tối ưu (test)** | 7/7 (100%) | 1–2 lượt | 0: 1 \| 1: 0 \| 2: 4 (**TB: 1.60/2**) | 4/4 (100%) | 4/4 (100%) | **2/2 (100%)** | **1/5 (20.0%)** |

### Trình bày chi tiết 3 Case Studies đại diện (theo yêu cầu GUIDE.md)

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| **q16** (Test) | Có đáp án (Answerable) | Gọi `search(query="Which general subdued the Cantabrians...")` $\to$ Trả về chunk `p016_para_000` (Agrippa, Cantabrians) | *"According to [gibbon-decline-and-fall-vol1_p016_para_000], The Cantabrians in Spain were subdued by Agrippa."* | **Điểm: 2/2**. Bằng chứng chính xác, trích dẫn đúng chunk trong context, trả lời đủ và đúng. |
| **q23_unans** (Test) | Thiếu bằng chứng (Unanswerable) | Gọi `search(query="In what year did the Visigoth king Alaric...")` $\to$ Context không chứa Alaric (năm 410 SCN) | *"Based on the provided historical records, there is insufficient evidence in the text to answer this question."* | **Abstention đúng**. Không có hiện tượng hallucination; agent nhận biết thông tin ngoài phạm vi tài liệu. |
| **q20** (Test) | Lỗi / Thách thức (Error Case) | Gọi `search(query="Which gladiatorial role did Commodus choose...")` $\to$ Trả về `p058_para_000` | Trạng thái: `abstain` (Do query dài chứa nhiều từ khóa phức tạp dẫn đến điểm overlap đơn lẻ chưa vượt ngưỡng) | **Điểm: 0/2 (False Abstention)**. Cần cải thiện phân rã câu hỏi đa mệnh đề (query decomposition). |

---

## 4. Tổng kết kỹ thuật và Tái lập kết quả

- **Đường dẫn artifacts và traces:**
  - Benchmark questions: [data/questions.json](../../data/questions.json) (24 câu)
  - Trace Dev Hybrid: [results/agent_dev_hybrid/traces.json](../../results/agent_dev_hybrid/traces.json)
  - Trace Test Frozen: [results/agent_test_hybrid/traces.json](../../results/agent_test_hybrid/traces.json)
  - Tóm tắt metrics: [results/agent_test_hybrid/metrics_summary.json](../../results/agent_test_hybrid/metrics_summary.json)
- **Lệnh chạy tái lập nghiệm thu:**
  ```bash
  # 1. Chạy unit tests
  python3 -m unittest discover -s tests -v

  # 2. Chạy đánh giá 3 phương pháp retrieval trên dev
  /home/ngthanhtam/K5-L3B-DAY06-NguyenThanhTam-2A202603933-DataIngestion/starter/.venv/bin/python evaluate_retrievers.py

  # 3. Chạy cấu hình chốt trên tập test
  /home/ngthanhtam/K5-L3B-DAY06-NguyenThanhTam-2A202603933-DataIngestion/starter/.venv/bin/python evaluate_agent.py --split test --retriever hybrid --budget 1200
  ```
- **Chi phí & Độ trễ:** Toàn bộ hệ thống chạy 100% offline, chi phí **0.00 USD**, độ trễ phản hồi trung bình **~50–65 ms/câu**.
