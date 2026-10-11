# Thay đổi so với Mr Dre chính thức (bản ZIP người dùng cung cấp)

Tài liệu này chỉ liệt kê các thay đổi có chủ đích, không tuyên bố tái lập số liệu chính thức.

## Các tệp vẫn nguyên vẹn

- `data/ResearchRubrics.jsonl`, `data/RigorousBench.jsonl`, `data/ResearcherBench.jsonl`: byte-identical với bản gốc.
- `feedback.py`, `feedback_refine.py`, `dr_eval/eval_prompts.py`: byte-identical với bản gốc (kể cả lời nhắc).
- `evaluate_report.py`, `engine/__init__.py`, các lớp dữ liệu và định dạng báo cáo: không sửa.
- `engine/reviser_agent` giữ nguyên vòng suy luận ReAct, chỉ thay nơi gọi mô hình và tìm kiếm.

## Các tệp thay đổi hoặc thêm mới

| Tệp | Mục đích |
|---|---|
| `engine/odr.py` (mới) | Kết nối API LangGraph của **LangChain Open Deep Research** chính thức, không dựng DRA thay thế |
| `local/free_search_mcp.py`, `local/free_web.py` (mới) | Thay dịch vụ tìm kiếm bằng DuckDuckGo và đọc nguồn trực tiếp, cung cấp công cụ MCP cho ODR |
| `engine/oai.py` | Khi `MRDRE_LLM_BACKEND=ollama`, gửi các lời nhắc LLM nguyên bản đến Ollama thay cho OpenAI |
| `dr_eval/crawler/jina.py`, `dr_eval/crawler/direct.py` (mới) | Tùy chọn đọc URL trực tiếp không dùng khóa Jina |
| `engine/reviser_agent/main.py`, `tools.py` | Thay địa chỉ mô hình và API tìm kiếm khi chọn chế độ miễn phí |
| `dr_eval/checklist_eval.py`, `citation_eval.py`, `rubric_eval.py` | Giới hạn số luồng trên GPU cục bộ; báo lỗi thay vì âm thầm bỏ qua một số kết quả khi chấm bằng Ollama |
| `evaluate_change.py` | Sửa đọc sai tên trường `organization` → `presentation`; giới hạn luồng chấm |
| `call_dra.py` | Loại bỏ thời gian chờ cố định sau lô cho `odr-local`; trả mã lỗi khi luồng chạy thất bại |
| `scripts/generate_report.sh` | Sửa `--round` thành `--turn`; giảm kích thước lô mặc định |
| `scripts/configure_odr_local.py`, `scripts/preflight_free.py` (mới) | Cấu hình và tự kiểm tra các dịch vụ cục bộ |
| `pyproject.toml`, `.env.example` | Gỡ phụ thuộc nặng vLLM và khai báo dịch vụ miễn phí |
| `README_FREE_BENCHMARK.md`, `tests/*` (mới) | Hướng dẫn Windows, kiểm thử vận hành mô phỏng |

## Giới hạn quan trọng

- **Mới cấu hình một DRA thật:** LangChain Open Deep Research, một trong các DRA mà paper đã thử nghiệm. Mã của DRA nằm ở kho chính thức riêng và cần clone/install. Bản này không tái lập nguyên 5 hệ thống.
- Đổi GPT-4.1-mini sang Qwen qua Ollama **không giữ nguyên độ tin cậy của giám khảo**. Muốn dùng làm luận văn cần hiệu chuẩn trên dữ liệu được con người gán nhãn.
- Không sử dụng API trả phí khi `.env` theo bản mẫu và ODR `search_api=none`. Việc duyệt web miễn phí phụ thuộc hạn mức/rate-limit, truy cập mạng và trang nguồn.
- **Chưa có kiểm thử thực tế** cùng lúc với LangGraph server, Ollama, trình duyệt web trên máy cá nhân. Chỉ có kiểm thử mô phỏng và kiểm tra tĩnh; không được coi là chạy thành công benchmark với số liệu thật.
- Mặc định giảm song song của *DRA* từ 5 xuống 1 để vừa GPU cục bộ. Không đổi thuật toán quyết định nghiên cứu nhưng ảnh hưởng tốc độ; nên ghi nhận cấu hình này.
- Có sai khác vốn tồn tại trong định nghĩa `Break Rate` giữa code gốc và công thức rút gọn của paper; vẫn giữ cách code gốc, không tự ý sửa định nghĩa.

## Kiểm thử trên bản ZIP này

- `python -m unittest discover -s tests -q`: **12/12 đạt**, sử dụng server/mô hình mô phỏng, không kết nối DRA hay Ollama thực.
- `python -m compileall -q .`: đạt.
- 3 dữ liệu + 3 tệp prompt/feedback nguyên bản đã được so SHA-256 trong `tests/upstream_integrity.json`.

Chi tiết cách thử **một câu thật**: đọc `README_FREE_BENCHMARK.md`.
