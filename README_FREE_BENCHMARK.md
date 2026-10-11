# Mr Dre — bản benchmark chạy cục bộ không cần API trả phí

## Mục tiêu, phạm vi, giới hạn

Đây là **bản sửa trực tiếp từ mã nguồn Mr Dre gốc trong ZIP**. Mục đích: giữ bộ câu hỏi, phản hồi và cách chấm điểm để bạn có thể (1) tạo báo cáo ban đầu bằng một tác nhân nghiên cứu thật; (2) sửa nhiều lượt; (3) sau này lắp phương pháp sửa của mình vào; (4) so sánh cùng môi trường.

DRA **chính** là **LangChain Open Deep Research (LC ODR)** — một trong 5 hệ thống được bài báo sử dụng. Mã triển khai DRA nằm ở dự án **langchain-ai/open_deep_research**, không phải tự viết lại trong Mr Dre. Kết nối `engine/odr.py` chỉ chuyển hội thoại/nhận kết quả với máy chủ LangGraph. Bộ tìm kiếm miễn phí cung cấp **công cụ** cho chính DRA đó, không thay thế việc lập kế hoạch và nghiên cứu của DRA.

**Cảnh báo trung thực:** Chuyển từ mô hình GPT-4.1 của paper sang Qwen chạy Ollama sẽ thay đổi chất lượng DRA và giám khảo. Không thể lấy điểm số này so trực tiếp với bảng số liệu tác giả. Mã máy chủ của các DRA khác trong paper không có sẵn trong ZIP gốc. DRA trả phí OpenAI/Perplexity không được tái tạo, cũng không giả mạo kết quả. Đây là **tái lập quy trình đánh giá**, *không phải tái lập nguyên các kết quả định lượng*.

## Đối chiếu với repo gốc

| Thành phần | Trạng thái |
|---|---|
| 3 tệp dữ liệu JSONL, checklist trọng số | **Giữ nguyên byte** |
| `call_dra.py`: lấy mẫu, lịch sử, tín hiệu, lưu JSONL, 4 lượt | Giữ logic gốc, sửa lỗi dừng và chờ |
| `feedback.py`: 3 loại phản hồi, lựa chọn mục chưa đạt, 21 mẫu định dạng | Giữ prompt/thuật toán gốc; thay LLM |
| `feedback_refine.py`: kế hoạch sửa tại vị trí | Giữ prompt/thuật toán gốc; thay LLM |
| `dr_eval/eval_prompts.py`: toàn bộ mẫu đánh giá | **Giữ nguyên byte** |
| Độ bao phủ, độ đúng của trích dẫn, căn cứ của nhận định, trình bày | Giữ công thức/hệ thống gốc; thay LLM + dịch vụ đọc trang |
| `evaluate_change.py`: tỷ lệ tiếp thu / làm hỏng | Giữ cách tính gốc, sửa lỗi đọc trường `presentation` |
| `engine/reviser_agent`: ReAct + tối đa 10 lần tìm kiếm | Giữ vòng tác nhân; thay máy chủ LLM + API tìm kiếm |
| 5 DRA ban đầu | Chỉ cấu hình **một** DRA mở: **LC ODR**; 2 DRA đóng không thể thay trực tiếp |

**Thông số khi chạy máy cá nhân:** Mình giảm mặc định `ODR_RESEARCH_CONCURRENCY` từ 5 của ODR xuống 1 vì GPU nhỏ không chịu được nhiều luồng sinh đồng thời. Hai giới hạn còn lại giữ mặc định upstream (6 vòng giám sát, 10 lần gọi công cụ mỗi lượt nghiên cứu). Khi so sánh, cả bản nền và bản cải tiến phải dùng cùng cấu hình.

**Sai khác cần biết trong chính repo gốc:** `evaluate_change.py` tính tỷ lệ làm hỏng chỉ trên tiêu chí không phải mục phản hồi, trong khi công thức rút gọn trong phần chính paper bao quát những mục đã được đáp ứng trước đó. Vì ưu tiên giữ cách chạy gốc, bản này **không âm thầm đổi công thức đó**. Nếu phân tích kết quả cho luận văn, phải chọn cách định nghĩa và báo rõ. Giám khảo 4B không có đảm bảo tương đương GPT-4.1-mini; cần kiểm tra độ đồng thuận trên một tập gán nhãn thủ công trước khi sử dụng kết quả khoa học.

## 1. Chuẩn bị Windows (chỉ cần làm một lần)

Yêu cầu: Python 3.12, `uv` hoặc `pip`, Git, Ollama, internet để đọc nguồn, đủ dung lượng lưu mô hình. Hướng dẫn minh họa bằng PowerShell. **Không nhập API key trả phí.** Thời gian nghiên cứu trên GPU nhỏ có thể rất lâu; tính phí API = 0 không có nghĩa là không tốn điện/mạng.

Trong thư mục Mr Dre vừa giải nén:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
Copy-Item .env.example .env
ollama pull qwen3:4b
```

Nếu máy có mô hình Ollama khác phù hợp hơn, sửa **cùng tên mô hình** ở `MRDRE_LOCAL_MODEL`, `ODR_LOCAL_MODEL`, `REVISER_MODEL` trong `.env`. ODR sử dụng *tool calling* và *structured output*: mô hình nhỏ có thể không xử lý ổn định; không có cam kết Qwen 4B sẽ hoàn thành tất cả đề.

## 2. Chạy công cụ tìm kiếm miễn phí (cửa sổ PowerShell A)

Từ thư mục Mr Dre và môi trường ảo đã kích hoạt:

```powershell
python -m local.free_search_mcp
```

Cửa sổ này phải chạy liên tục. Trong `.env`, đặt **`ODR_MCP_URL=http://127.0.0.1:8001` (không có `/mcp`)** vì mã ODR chính thức tự nối `/mcp` khi kết nối. Địa chỉ giao thức thực tế là `http://127.0.0.1:8001/mcp`. Công cụ MCP dùng DuckDuckGo (`ddgs`) và đọc trực tiếp văn bản HTML. Không cần Jina/Serper. Các trang chặn bot, JavaScript, CAPTCHA, hoặc DuckDuckGo giới hạn truy cập có thể thất bại; chương trình **không thay thế bằng nguồn tự bịa**.

## 3. Khởi chạy DRA thật LC ODR (cửa sổ PowerShell B)

Tại thư mục **khác** với Mr Dre:

```powershell
git clone https://github.com/langchain-ai/open_deep_research.git
cd open_deep_research
uv sync --python 3.11
uv pip install --python .venv\Scripts\python.exe "langgraph-cli[inmem]" langchain-ollama
```

**Quan trọng:** Repo ODR gốc có tùy chọn xác thực trực tuyến dùng Supabase. Trong thư mục Mr Dre, chạy **một lần** lệnh sau (sửa đường dẫn tới bản ODR vừa clone):

```powershell
python scripts\configure_odr_local.py "C:\duong-dan\open_deep_research"
```

Lệnh chỉ bỏ mục `auth` trong `langgraph.json` của **bản clone cục bộ** (sao lưu tệp cũ), đặt môi trường thử nghiệm không cần khóa API, **không sửa đồ thị hay thuật toán DRA**. Không dùng máy chủ đã bỏ xác thực cho mạng công cộng.

Quay lại cửa sổ B tại thư mục ODR:

```powershell
.\.venv\Scripts\langgraph.exe dev --allow-blocking --host 127.0.0.1
```

DRA cần có máy chủ ở `http://127.0.0.1:2024`. Khi lỗi 401 cần kiểm tra lại `langgraph.json` cục bộ và khởi động lại máy chủ. Khi lỗi `No tools found` cần kiểm tra MCP ở cửa sổ A và cấu hình ODR. Khi lỗi structured output/tool calling cần chọn mô hình cục bộ mạnh hơn hoặc tăng bộ nhớ/ngữ cảnh; không coi một lần chạy lỗi là kết quả benchmark.

**Lưu ý về cài đặt:** `langgraph-cli`, `langchain-ollama` và các thư viện ODR phải ở **cùng môi trường** của bản clone ODR. Lệnh trên cài trực tiếp vào `.venv` thay vì dùng một môi trường `uvx` tạm thời có thể thiếu `langchain-ollama`.

**Nguồn phần mềm:** [Open Deep Research chính thức](https://github.com/langchain-ai/open_deep_research), [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk), [Ollama](https://ollama.com/).

## 4. Thử trên 1 câu trước (cửa sổ PowerShell C)

Kích hoạt môi trường của Mr Dre, đảm bảo `.env` được nạp:

```powershell
python scripts\preflight_free.py
python call_dra.py --turn 1 --type init --dataset ResearcherBench --model odr-local --num_questions 1 --batch_size 1
```

Kiểm tra tệp `output/ResearcherBench/init/odr-local/gen_turn1.jsonl`. Phải có `full_response.report` với báo cáo **có dẫn nguồn thực**. Việc công cụ trả báo cáo không có dẫn nguồn là lỗi chất lượng phải ghi nhận, không tự sửa đáp án.

Đánh giá đúng tệp vừa sinh:

```powershell
python evaluate_report.py --report_file output/ResearcherBench/init/odr-local/gen_turn1.jsonl --dataset_file data/ResearcherBench.jsonl --eval_mode all --num_questions 1
```

**Lưu ý:** Đánh giá trích dẫn gọi mô hình nhiều lần, chi phí API = 0 nhưng có thể tốn hàng chục phút/giờ trên 4B. Nếu muốn thử nhanh trước, dùng `--eval_mode checklist` (sau này phải chạy `all` để có đủ 3 chiều).

Sau khi đã có kết quả checklist của lượt 1, tạo lượt sửa theo phản hồi nội dung:

```powershell
python call_dra.py --turn 2 --type content_feedback --dataset ResearcherBench --model odr-local --num_questions 1 --batch_size 1
python evaluate_report.py --report_file output/ResearcherBench/content_feedback/odr-local/gen_turn2.jsonl --dataset_file data/ResearcherBench.jsonl --eval_mode all --num_questions 1
python evaluate_change.py output/ResearcherBench/init/odr-local/gen_turn1.jsonl output/ResearcherBench/content_feedback/odr-local/gen_turn2.jsonl
```

Ngoài `content_feedback`, mã gốc hỗ trợ `reflection` và `format_feedback`. Để có lượt 3–4, cần chạy các lượt trước và đánh giá checklist tương ứng. Dùng `--refine` và `--use_reviser` cho phương pháp đối chứng của paper; không tự thêm phương pháp nghiên cứu mới vào bản nền.

## 5. Khi đã có ý tưởng nghiên cứu

Tạo một bộ kết nối DRA/phương pháp mới tuân thủ lớp `DeepResearchAgentBase` trong `engine/__init__.py`, **không sửa tệp `data/`, `dr_eval/eval_prompts.py` và các công thức đánh giá**. Chạy bản gốc và bản cải tiến trên **cùng câu hỏi, cùng mô hình, cùng nguồn tìm kiếm và cùng giám khảo**. Nên lưu tên mô hình, phiên bản prompt, thời điểm thu thập nguồn, độ dài báo cáo, tỉ lệ lỗi và số lần tìm kiếm; kiểm tra thủ công một tập nhỏ trước khi chạy hàng loạt.

## 6. Kiểm thử và phần chưa được xác nhận

```powershell
python -m unittest discover -s tests -v
```

Các kiểm thử có mô phỏng máy chủ bên ngoài và mô hình, không có chi phí API. **Chưa có bằng chứng chạy toàn bộ LC ODR + Ollama + DuckDuckGo + toàn bộ benchmark thật** trong môi trường tạo bản ZIP (không có Ollama/PyPI/network); phải xác minh ở bước 4 trên máy của bạn. Không gọi bản này "đã tái lập số liệu paper" cho tới khi hoàn thành kiểm thử thực và kiểm tra giám khảo.

Các phụ thuộc `mcp`, `ddgs`, `langchain-ollama` và DRA phải được tải từ internet khi cài đặt lần đầu. Điều này vẫn miễn phí về phí API. Bộ dữ liệu trong ZIP được giữ nguyên và *không* điền kết quả giả vào thư mục `output`.
