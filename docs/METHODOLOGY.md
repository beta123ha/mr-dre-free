# Đối chiếu phương pháp gốc

| Thành phần | MR DRE (ACL 2026) | Bản Kaggle Free |
|---|---|---|
| Bộ dữ liệu | 3 bộ: ResearchRubrics 101, RigorousBench 214, ResearcherBench 65; thí nghiệm RigorousBench lấy 100 câu | Chỉ ResearcherBench 65 câu |
| Agent được đánh giá | OpenAI DR, Sonar DR, LC ODR, Tongyi DR, DR Tulu | Chỉ graph **LC Open Deep Research** gốc |
| Backbone LC ODR | GPT-4.1-mini + GPT-4.1-nano | Qwen3:4b-instruct Ollama; miễn phí |
| Tạo phản hồi | 3 loại: Content1, Format, Reflect, GPT-4.1-mini | 3 loại cùng prompt; Qwen3:4b-instruct; seed cố định theo ID |
| Đánh giá | GPT-4.1-mini, temperature 0 | Cùng prompt/phép tính, Qwen3:4b-instruct temp=0 |
| COV | Điểm checklist có trọng số, bao gồm trọng số âm | Giữ công thức |
| FA + GR | Bóc tách claim → crawl nguồn → tóm tắt → xác minh | Giữ các giai đoạn; nguồn đọc trực tiếp thay Jina API |
| PRE | 10 tiêu chí, score nhị phân; tiêu chí không áp dụng loại khỏi trung bình | Giữ công thức |
| INC | Content: mục tiêu đạt 1 / negative weight đạt 0; Format: pairwise judge; Reflect: N/A | Giữ, với Format chấm riêng theo cặp và checkpoint |
| BRK | Theo Phụ lục B.4: tất cả mục đã đạt, cả mục có trong feedback | Đúng như paper; **không loại mục tiêu khỏi mẫu số** |
| Revision | Agent nhận toàn bộ lịch sử báo cáo/phản hồi | Cùng cấu trúc thông điệp; không PE, Reviser, ReviFence |
| Ngân sách tìm kiếm | LC ODR mặc định 6 iteration / 10 tool calls | Nhanh hơn: 3 / 6, cần công bố trong bài và giữ nguyên khi so sánh |
| Phạm vi lần chạy mặc định | Phần bảng chính 3 feedback ở Turn 2; Fig.3 còn có 4 turn | Tái lập bảng chính 3 feedback/2 turn trên 1 bộ; không tự xưng tái lập toàn bộ thí nghiệm phụ |

### Lựa chọn để có tính khoa học

Pipeline này là **baseline có kiểm soát trong điều kiện miễn phí**, không thể đối chiếu trực tiếp số điểm với tác giả dùng GPT-4.1 và 5 agent. Để đánh giá ReviFence, sử dụng cùng bộ câu hỏi, model digest, commit ODR, tham số, seed, phản hồi đã được lưu và cùng quy trình chấm. Nếu cần so số liệu trực tiếp với Table 2 của paper, phải khôi phục model/scaffold/search backend tác giả đã sử dụng.

Dữ liệu trống hoặc lỗi khi gọi agent/LLM/web không được thay bằng báo cáo giả. `complete` trong file summary chỉ True khi đủ số câu và đầy đủ các trục đánh giá bắt buộc. `Reflect` có INC không xác định (N/A), đúng theo bảng bài báo. Các file JSONL và CSV giữ nguyên để kiểm tra lại.
