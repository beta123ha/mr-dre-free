# MR DRE baseline — Kaggle Free

Baseline **trước ReviFence**, tái lập các bước của *Beyond Single-shot Writing: Deep Research Agents are Unreliable at Multi-turn Report Revision* (ACL 2026). Mặc định chỉ dùng **ResearcherBench (65 câu, nhỏ nhất)** và **LangChain Open Deep Research**.

## Chạy bằng một notebook

1. Trên [Kaggle Code](https://www.kaggle.com/code), tạo Notebook, chọn **File → Import Notebook** và tải lên `MRDRE_Baseline_Kaggle_Complete.ipynb` trong `notebooks/` (cũng được phát hành thành file riêng).
2. Trong **Settings** chọn **Accelerator = GPU T4 x2**, **Internet = On**. Không cần nhập API key. Kaggle yêu cầu xác minh tài khoản để dùng GPU trên một số tài khoản.
3. Chạy các ô từ trên xuống. Notebook tự giải nén mã/dữ liệu, cài Ollama và graph ODR gốc, chạy **một câu thử thật**, rồi tiếp tục 65 câu. Để chạy không cần giữ phiên tương tác, dùng **Save Version → Save & Run All**.
4. Lấy `/kaggle/working/baseline-results.zip` trong tab **Output** sau khi phiên kết thúc. Nếu chưa xong, tải ZIP đó lên Kaggle thành **Dataset**, rồi **Add Input** Dataset vào notebook và chạy lại nguyên notebook. Ô đầu tự phục hồi checkpoint, bỏ qua các câu/điểm chấm đã xong.
5. Chỉ xem dòng `complete=True` trong `output/metrics/*_summary.csv` là xong trọn bộ. Các ô trống không phải điểm 0, mà là phần đánh giá chưa thực hiện.

**Mặc định:** `N_QUESTIONS=65`, `FEEDBACK='all'`, `EXTRA_EVAL='all'`. `all` gồm **Content1, Format, Reflect** (bảng kết quả chính của paper, riêng ResearcherBench); phản hồi tự suy xét không có Incorporation. Nếu cần **hoàn thành một thí nghiệm đủ sáu chỉ số nhanh hơn**, đổi `FEEDBACK='content'` rồi mới chạy thêm `format` và `reflect` ở các phiên sau, **không thay mô hình hoặc tham số**. Turn 1 được dùng chung.

## Sáu chỉ số

| Cột CSV | Định nghĩa gốc |
|---|---|
| `coverage_t1/t2`, `coverage_delta` | Coverage (Cov.): chấm checklist 0 / 0,5 / 1, trung bình có trọng số, kể cả trọng số âm |
| `faithfulness_t1/t2`, `faithfulness_delta` | Citation Faithfulness (Fa.): tỷ lệ khẳng định có dẫn nguồn được nguồn ủng hộ |
| `groundedness_t1/t2`, `groundedness_delta` | Claim Groundedness (Gr.): tỷ lệ mọi khẳng định được chứng cứ ủng hộ |
| `presentation_t1/t2`, `presentation_delta` | Presentation (Pre.): 10 tiêu chí định dạng, bỏ những tiêu chí không áp dụng |
| `incorporation` | Inc.: đạt đầy đủ mục tiêu Content1 hoặc chấm cặp báo cáo cho Format (N/A với Reflect) |
| `break_rate` | Brk.: số tiêu chí đã đạt nhưng bị giảm / số tiêu chí đã đạt, **theo Phụ lục B.4 paper**, tính cả tiêu chí nằm trong phản hồi |

Tệp đầu ra: `output/metrics/{content,format,reflect}_summary.csv` và `*_per_question.csv`. Cột `n_*` cho biết có bao nhiêu câu đã thực sự được chấm ở từng chỉ số. **Không tính 0 giả cho câu lỗi**, không tuyên bố kết quả hoàn chỉnh nếu thiếu bất kỳ đánh giá bắt buộc nào.

## Tối ưu chi phí thời gian

- T4 x2: GPU0 tạo báo cáo; GPU1 chấm; bài đã có kết quả được bỏ qua, điểm chấm và phản hồi được tái sử dụng.
- Mô hình Ollama `qwen3:4b-instruct` (nhỏ, hỗ trợ gọi công cụ và cấu trúc dữ liệu). Chỉ tải mô hình một lần rồi hai server GPU dùng chung tệp mô hình.
- Câu hỏi và feedback seed cố định; lưu JSONL nguyên tử, checkpoint ZIP theo từng nhóm 5 câu. Chạy lại không mất tiến độ đã chấm xong.
- Đọc web qua DuckDuckGo + thư viện HTML/PDF miễn phí; không chuyển dữ liệu qua API LLM trả phí.
- Bật `EXTRA_EVAL='all'` để **đủ sáu chỉ số**. FA/GR là phần tốn nhiều nhất do bóc tách từng khẳng định, tải nguồn, tóm tắt và đối chiếu; bỏ phần này sẽ không còn là bảng sáu chỉ số.

**Khác với số liệu bài báo:** paper chạy 5 DRA, dùng GPT-4.1-mini/GPT-4.1-nano để chấm, còn bản Kaggle này chỉ chạy 1 DRA và thay mô hình bằng Qwen3-4B, thay dịch vụ web. Cấu hình nhanh giới hạn số vòng nghiên cứu (3) và lượt công cụ (6), thấp hơn mặc định ODR (6 và 10). Đây là **tái lập quy trình và thước đo với hạn chế tài nguyên**, không phải tái lập được con số trong Table 2; dùng **cùng cấu hình** cho ReviFence để so sánh công bằng. Không sử dụng hai phương án PE/Reviser trong §6 của paper và không áp dụng ReviFence.

## Cấu trúc

```text
MRDRE-Baseline-Kaggle-Complete/
├── notebooks/MRDRE_Baseline_Kaggle_Complete.ipynb
├── kaggle/
│   ├── bootstrap.py               # Cài Ollama / LangGraph / dịch vụ tìm kiếm
│   ├── run.py                     # Pipeline sinh, sửa, chấm, checkpoint
│   └── metrics.py                 # Sáu phép đo của paper, gộp theo ID
├── src/mrdre_baseline/
│   ├── agents/                    # Adapter LC Open Deep Research & model local
│   ├── workflows/                 # Sinh báo cáo, phản hồi, chấm điểm
│   ├── evaluation/                # Checklist, citation, presentation
│   ├── services/                  # MCP web tools miễn phí
│   └── core/                      # Đọc/ghi an toàn
├── data/                          # Dữ liệu gốc, giữ nguyên
├── docs/METHODOLOGY.md            # Khác biệt so với paper
├── tests/                         # Kiểm thử dữ liệu, metric, checkpoint
└── pyproject.toml
```

## Chạy dòng lệnh (trong Kaggle)

```bash
python kaggle/bootstrap.py
python kaggle/run.py --count 1 --feedback all --extra-eval all
python kaggle/run.py --count 65 --feedback all --extra-eval all --chunk-size 5 --budget-minutes 600
python -m kaggle.metrics --output-root /kaggle/working/output --feedback all --count 65
```

Mỗi phiên Kaggle miễn phí có hạn mức; **không thể đảm bảo 65 câu × 3 loại phản hồi × toàn bộ chấm trích dẫn xong trong một phiên**. Cần kết nối mạng và GPU Kaggle thật để kiểm thử đầu cuối; kiểm thử offline trong dự án không phải là bằng chứng đã chạy thực tế trên Kaggle.
