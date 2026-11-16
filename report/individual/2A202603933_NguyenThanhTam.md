# Báo cáo Thực nghiệm Day 7 — Agentic RAG
**Học viên:** Nguyễn Thành Tâm  
**MSSV:** 2A202603933  
**Lớp:** L3B  
**Chủ đề & Corpus cơ bản:** Lịch sử Đế chế La Mã (*The History of the Decline and Fall of the Roman Empire*, Edward Gibbon - Tập I, Chương I–IV)  
**Chủ đề & Corpus nâng cao (Bonus):** Báo cáo thường niên 2024 Vinamilk (*Luôn Cầu Tiến*, 190 trang logic, đa phương thức Text - Bảng - Sơ đồ - Biểu đồ)  
**Môi trường triển khai:** 100% Local (Không tốn chi phí API, không phụ thuộc mạng ngoài)

---

## 1. Dữ liệu và benchmark (Phần cơ bản)

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

## 2. Retrieval và thử nghiệm (Phần cơ bản)

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

## 3. Agent và đánh giá (Phần cơ bản)

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

## 4. Thử thách nâng cao (+15 điểm Bonus)

### 4.1. Tài liệu và Ingestion đa phương thức (100+ trang)
- **Tài liệu:** Báo cáo thường niên 2024 của Công ty Cổ phần Sữa Việt Nam (Vinamilk) — *"Luôn Cầu Tiến"*.
  - Link tải chính thức: [Vinamilk Annual Report 2024](https://www.vinamilk.com.vn/bao-cao-thuong-nien/bao-cao/2024/doc/vi/full-bao-cao.pdf).
  - Bản lưu trữ trong repo: `data/advanced/vinamilk-annual-report-2024.pdf`.
  - Quy mô tài liệu: 96 trang vật lý (được dàn trang dạng trang đôi - two-page spread, tương đương **190 trang logic nội dung**), tổng dung lượng **381.821 ký tự**.
- **Pipeline xử lý đa phương thức (`src/advanced_ingest.py`):**
  - Quét toàn bộ 190 trang logic và tự động phân loại cấu trúc nội dung:
    - **`table`** (46 chunks): Các bảng cân đối kế toán, báo cáo kết quả hoạt động kinh doanh, chỉ số tài chính, thù lao HĐQT.
    - **`diagram`** (101 chunks): Danh mục thuật ngữ viết tắt (HMO, Horeca, FSSC, UBKT, UBCL, TGĐ, HĐQT), chuỗi giá trị và cơ cấu quản trị.
    - **`chart`** (4 chunks): Các chỉ số tăng trưởng doanh thu (+2,2%), lợi nhuận (+4,8%), đóng góp ngân sách (+11,3%), thị trường xuất khẩu mới (+3 quốc gia).
    - **`text`** (308 chunks): Thông điệp lãnh đạo, tầm nhìn, sứ mệnh, cam kết phát triển bền vững và thông tin liên hệ.
  - Tổng số chunk trích xuất: **459 chunks** lưu tại `data/advanced/chunks.json`.

### 4.2. Bộ Benchmark Đa phương thức (48 câu)
Được tạo và kiểm chứng nghiêm ngặt trong `data/advanced/questions.json`:
- **Tập Dev (18 câu):** 15 câu có đáp án + 3 câu unanswerable.
- **Tập Test (30 câu):** Đảm bảo cơ cấu tối thiểu theo đúng [ADVANCED.md](../../ADVANCED.md):
  - 6 câu **Bảng (Tables)**: Tra cứu doanh thu (61.824 tỷ), lợi nhuận (9.453 tỷ), thuế, nộp ngân sách, doanh thu thuần kiểm toán (52.576.991 triệu VND).
  - 5 câu **Ảnh/Sơ đồ (Diagrams)**: Giải nghĩa từ viết tắt HMO, Horeca, FSSC, UBKT, UBCL.
  - 5 câu **Biểu đồ (Charts)**: Tính toán chênh lệch tốc độ tăng trưởng nộp ngân sách vs doanh thu (+9,1%), lợi nhuận vs doanh thu (+2,6%), cột mốc 48 năm.
  - 4 câu **Multi-hop**: Kết hợp thông tin giữa các trang (Trang 2 & 3: Xếp hạng Fortune 500 Đông Nam Á; Trang 4 & 92: Báo cáo tài chính kiểm toán; Trang 92: So sánh tăng trưởng doanh thu 2 năm liên tiếp).
  - 4 câu **Văn bản (Text)**: Tầm nhìn, sứ mệnh cộng đồng, trụ sở chính (10 Tân Trào, Q7, TP.HCM), website.
  - 6 câu **Thiếu bằng chứng (Unanswerable)**: Kiểm tra khả năng từ chối trả lời trước các câu hỏi về kế hoạch quý 3/2025, thị phần đối thủ TH True Milk, giá cổ phiếu năm 2026, bí mật công nghệ, thâu tóm Highlands Coffee, đầu tư vệ tinh viễn thông.

### 4.3. Kết quả Benchmark Nâng cao trên Hold-out Test (30 câu)

Chạy thực nghiệm độc lập bằng script `evaluate_advanced.py --split test`:

| Nhóm bài toán kiểm tra | Số câu đạt / Tổng số câu | Tỷ lệ đạt (%) | Nhận xét chi tiết |
| :--- | :---: | :---: | :--- |
| **1. Bảng số liệu (Tables)** | **6 / 6** | **100.0%** | Tra cứu chính xác số liệu doanh thu, lợi nhuận, nộp ngân sách |
| **2. Ảnh, Sơ đồ (Diagrams)** | **5 / 5** | **100.0%** | Nhận diện đúng 100% định nghĩa thuật ngữ và chức danh viết tắt |
| **3. Biểu đồ (Charts)** | **5 / 5** | **100.0%** | Đọc đúng giá trị tỷ lệ tăng trưởng và tính toán chênh lệch phần trăm |
| **4. Đa bước (Multi-hop)** | **4 / 4** | **100.0%** | Kết hợp hoàn hảo bằng chứng từ 2 trang khác nhau |
| **5. Văn bản (Text)** | **4 / 4** | **100.0%** | Trích dẫn đầy đủ và nguyên vẹn tuyên ngôn tầm nhìn, sứ mệnh |
| **6. Thiếu bằng chứng (Unanswerable)** | **6 / 6** | **100.0%** | 100% nhận diện đúng câu hỏi ngoài phạm vi và kích hoạt Abstain |
| **TOÀN BỘ TEST BENCHMARK** | **30 / 30** | **100.0%** 🏆 | **Vượt xa ngưỡng tối thiểu 90% (>= 27/30 câu) của ADVANCED.md** |

- **Thời gian phản hồi:** Trung bình **< 1.0 ms / câu** (cực kỳ tối ưu nhờ Inverted Index trên bộ chunk đa phương thức).
- **Chi phí phát sinh:** **0.00 USD** (100% Local).
- **Lỗi hệ thống / Timeout:** **0**.

---

## 5. Tổng kết kỹ thuật và Lệnh tái lập kết quả

- **Đường dẫn artifacts:**
  - Corpus cơ bản: [data/chunks.json](../../data/chunks.json) (884 chunks)
  - Benchmark cơ bản: [data/questions.json](../../data/questions.json) (24 câu)
  - Corpus nâng cao (Vinamilk): [data/advanced/chunks.json](../../data/advanced/chunks.json) (459 chunks đa phương thức)
  - Benchmark nâng cao: [data/advanced/questions.json](../../data/advanced/questions.json) (48 câu)
  - Traces đánh giá nâng cao: [results/advanced_test/traces.json](../../results/advanced_test/traces.json)
  - Tóm tắt kết quả nâng cao: [results/advanced_test/summary.json](../../results/advanced_test/summary.json)
- **Lệnh chạy tái lập toàn bộ:**
  ```bash
  # 1. Chạy bộ Unit Tests cơ bản (15/15 PASS)
  python3 -m unittest discover -s tests -v

  # 2. Chạy đánh giá Agent phần Cơ bản trên Test (Đạt 80% answerable, 100% unanswerable)
  /home/ngthanhtam/K5-L3B-DAY06-NguyenThanhTam-2A202603933-DataIngestion/starter/.venv/bin/python evaluate_agent.py --split test --retriever hybrid --budget 1200

  # 3. Chạy đánh giá Thử thách Nâng cao trên Test (Đạt 100% = 30/30 câu)
  /home/ngthanhtam/K5-L3B-DAY06-NguyenThanhTam-2A202603933-DataIngestion/starter/.venv/bin/python evaluate_advanced.py --split test
  ```
