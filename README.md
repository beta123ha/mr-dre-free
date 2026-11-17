# MR DRE — single-question baseline (Local / Kaggle / Colab)

Phiên bản rework từ `mr-dre-free-free-local(5).zip`, phục vụ baseline trước ReviFence.
Giữ ba loại feedback và sáu thước đo của MR DRE ACL 2026, nhưng thay **bộ điều phối**:
**hoàn tất một câu hỏi (R1 + ba R2 + toàn bộ đánh giá) rồi mới bắt đầu câu tiếp theo**.

## 1. Chạy trên máy cá nhân (Windows 11 + WSL Ubuntu, NVIDIA 6 GB)

Trong terminal **WSL Ubuntu** (không phải PowerShell), tại thư mục repo:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
cp .env.example .env
python scripts/bootstrap.py --profile local
python -m mrdre_baseline.runtime.runner --count 1 --feedback all --extra-eval all
python -m mrdre_baseline.runtime.summary --output-root output
```

Nếu WSL không có `python3.11`, dùng một bản Python >=3.11 sẵn có (ví dụ `python3`).
Cần `git`, `curl`, Internet, driver NVIDIA hoạt động trong WSL và quyền cài Ollama.
Bootstrap dùng Ollama, LangGraph và MCP server chạy trên `127.0.0.1`; không cần API key trả phí.
Mỗi phiên chỉ khởi động dịch vụ **một lần**. Chạy tiếp bằng cùng lệnh (hoặc `--count 65`).

> RTX 4050 6 GB: mặc định 8192 token ngữ cảnh, một suy luận GPU đồng thời để hạn chế tràn VRAM.
> Hãy đo tốc độ và kiểm tra chất lượng trước khi tăng context/parallel. 8192 có thể không đủ cho
> một số báo cáo dài và làm thay đổi kết quả chấm; cùng một thực nghiệm so sánh phải giữ cấu hình thống nhất.

## 2. Kaggle và Colab

- `notebooks/MRDRE_Kaggle.ipynb`: GPU T4 × 2 (nếu có), Internet On.
- `notebooks/MRDRE_Colab.ipynb`: GPU khả dụng, Internet On.

Notebook dùng **chung repository**: cung cấp `SOURCE_ZIP` (ZIP repo đã rework, tải lên môi trường)
hoặc push repo mới lên GitHub và sử dụng `REPO_URL`.
Notebook cũ có bản sao code nhúng đã được thay, tránh lệch mã nguồn.

Để tiếp tục ở phiên khác, đưa `baseline-results.zip` làm `RESUME_ZIP`.
Với Colab có thể đặt `USE_DRIVE=True` để lưu checkpoint lên Google Drive.
Không chia nhỏ câu hỏi: chỉ những câu đầy đủ mới có trạng thái `completed`.

## 3. Pipeline một câu

```text
Qn -> Agent sinh R1
   -> Chấm Coverage R1 -> Tạo Content1 feedback --+--> Agent sửa R2-content
   -> Tạo Format feedback -------------------------+--> Agent sửa R2-format
   -> Reflect (feedback cố định) ------------------+--> Agent sửa R2-reflect
   -> Chấm Coverage, Citation, Presentation cho R1 và tất cả R2
   -> Chấm Incorporation Format; INC Content và BRK tính từ checklist
   -> Kiểm tra tất cả checkpoint -> Qn Completed -> Q(n+1)
```

Các nút có thể **chạy gối đầu trong cùng câu** khi độc lập về dữ liệu:
trên 2 GPU, Agent ở GPU0 và evaluator ở GPU1; trên 1 GPU, lần lượt chạy các
suy luận để không tranh VRAM. Không bắt buộc R1 hoàn thành mọi phép chấm trước
khi bắt đầu Reflect vì hai nhánh này không có quan hệ dữ liệu.

Mặc định `--feedback all --extra-eval all`: 4 báo cáo/câu,
Coverage 4 lượt trên toàn checklist, Citation 4 lượt, Presentation 4 lượt,
Content/Format feedback và Format Incorporation. Reflect không có INC.
**Chỉ có một vòng chỉnh sửa (Turn 1 → Turn 2).**

## 4. Các lệnh

```bash
# Xem dependency DAG, không cần GPU
python -m mrdre_baseline.runtime.runner --dry-run --count 1

# Một câu duy nhất theo ID
python -m mrdre_baseline.runtime.runner --question-id 37 --feedback all --extra-eval all

# Toàn bộ 65 câu, dừng an toàn theo ngân sách 9 giờ
python -m mrdre_baseline.runtime.runner --count 65 --budget-minutes 540 \
  --feedback all --extra-eval all --output-root output

# Tính chỉ số cuối trên CHỈ những câu đã completed
python -m mrdre_baseline.runtime.summary --output-root output

# Kiểm thử không cần mô hình
python -m pytest -q
```

Exit codes runner: `0` hoàn thành yêu cầu, `2` lỗi/gián đoạn (đã checkpoint), `3` hết ngân sách.
Khi chạy lại, công đoạn đã hoàn thành được bỏ qua. Không tự sinh kết quả giả cho phần thiếu.

## 5. Điều phối tài nguyên

- Tự phát hiện số GPU, VRAM qua `nvidia-smi`, CPU/RAM qua `psutil`.
- Hai GPU: một hàng đợi Agent và một hàng đợi chấm điểm; các việc độc lập chạy song song.
- Một GPU: dùng chung một hàng đợi suy luận, tránh tranh chấp bộ nhớ.
- Số worker chấm điểm được điều chỉnh theo VRAM khả dụng, **không vượt quá khả năng song song của Ollama**.
- Mặc định GPU <12 GB chạy một yêu cầu Ollama; GPU nhiều bộ nhớ có thể chạy hai.
- Thời gian từng công đoạn và snapshot tài nguyên lưu vào `output/run_state/<id>.json`.
- Checkpoint ZIP được cập nhật sau mỗi câu hoàn tất và khi có sự cố.
- Có hỗ trợ **thử nghiệm** vLLM cho judge trên GPU thứ hai, cần cài vLLM
  và chạy bootstrap `--judge-backend vllm`. Không bật mặc định;
  đánh giá phải dùng cùng một judge giữa baseline và ReviFence.

`Ray` không được đưa vào đường chạy mặc định vì với một câu tại một thời điểm
nó tăng độ phức tạp; bộ điều phối nhẹ có hai hàng đợi là đủ để kiểm soát tài nguyên.

## 6. Kết quả và tính tái lập

Giữ nguyên format đầu ra `output/ResearcherBench/{init,content_feedback,format_feedback,reflection}/odr-local/`.
Các file `gen_turn*.jsonl`, `checklist_turn*.jsonl`, `citation_turn*.jsonl`,
`rubric_turn*.jsonl` và `format_incorporation_turn2.jsonl` được lưu theo từng bước.

- `output/run_state/<id>.json`: trạng thái câu, danh sách công đoạn và thời gian.
- `output/run_manifest.json`: mã hash bộ dữ liệu, model và giới hạn nghiên cứu;
  từ chối trộn các kết quả khác cấu hình trong cùng thư mục.
- `output/metrics/*_per_question.csv`, `*_summary.csv`: chỉ tổng hợp các câu **completed**.
- `baseline-results.zip`: checkpoint để mang sang phiên khác, không chứa model/caches.

Giữ nguyên bộ dữ liệu và các prompt/công thức chấm của mã nguồn MR DRE.
Bản rework vẫn dùng **Qwen 4B thay GPT-4.1-mini**, web search DDG và ODR local;
đây là tái lập **logic** benchmark có hạn chế tài nguyên, **không tương đương** kết quả paper.
Tối ưu hóa chạy không bảo đảm nhanh nhất trên mọi GPU; phải đo latency thật trước khi khẳng định.

Các mô hình/dịch vụ cần tải không có trong ZIP; kiểm thử offline không chứng minh
chạy end-to-end trên GPU Kaggle, Colab hoặc laptop cá nhân.
