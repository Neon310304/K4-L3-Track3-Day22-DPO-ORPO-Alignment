# Lưu trữ trên GitHub trước khi dọn máy

Theo yêu cầu lưu toàn bộ dữ liệu Lab22 trước khi xóa tệp tải về, các artifact đã được lưu tại [GitHub Release](https://github.com/Neon310304/K4-L3-Track3-Day22-DPO-ORPO-Alignment/releases/tag/lab22-t4-20261011). Mỗi upload được kiểm kích thước và SHA256 do GitHub trả về trước khi dọn bản local.

| Asset | Vai trò |
|---|---|
| `day22-t4-final-20261011.zip` | Bản xuất nguyên byte từ Colab: 61 tệp, đủ NB0–NB4, raw/summary, dữ liệu, metrics và lịch sử NB4 lỗi |
| `day22-training-checkpoint-20261011T010258Z.zip` | 46 tệp checkpoint SFT/DPO LoRA và tokenizer/template của lượt T4 mới; không chứa merged weights |
| `Lab22_history_artifacts_20261011.zip` | 148 tệp lịch sử local 0.6B, trọng số merged/LoRA cũ và artifact trung gian/thất bại; không dùng làm kết quả chính |
| `Lab22_submission_20261011.zip` | Bộ nguồn và bài nộp từ Git đã kiểm tra, không chứa weights |

Metadata xác nhận upload ở `evidence/release-*-receipt.json`. `private-backup-receipt.json` và `backup-source.json` ghi ý định lưu riêng ban đầu; yêu cầu lưu lên GitHub sau đó được thực hiện bằng Release, ngoài Git tree. Cache model công khai và môi trường Python được loại khỏi archive vì có thể tải/cài lại.

Bài chính vẫn có đủ năm notebook chạy thật, báo cáo, bốn biểu đồ gốc và ảnh Colab cuối. Thư mục bài làm chính được giữ. Chi tiết dọn local ghi trong `evidence/local-cleanup.json`; không xóa `.env` hoặc các tệp ngoài phạm vi Lab22. Chưa nộp LMS.
