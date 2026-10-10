# Lượt T4 bị gián đoạn ngày 2026-10-11

Notebook được tải trực tiếp từ Colab lúc 06:57 giờ Bangkok để giữ nguyên log. Log có thông báo ghi notebook đã thực thi cho NB0–NB3 và bắt đầu NB4, nhưng chưa có thông báo hoàn thành NB4 hoặc exit code cuối của pipeline.

Khi kiểm tra kết nối lại, Colab hiển thị: “Bạn hiện không thể kết nối với một GPU do hạn mức sử dụng trong Colab.” Danh sách phiên đang hoạt động trống. Đây là bằng chứng hạn mức GPU đang chặn tiếp tục; không có traceback để kết luận mã NB4 bị lỗi. Thời điểm và nguyên nhân chính xác runtime trước đó kết thúc chưa xác định được.

Hai reward model đều đạt 12/12 cặp sanity trong preflight trước huấn luyện. Kết quả preflight không thay thế sanity hoặc score của NB4.

**Lưu ý về output cũ trong notebook:** ô export (ô 8) còn giữ kết quả của lượt 2026-10-09 với Qwen sanity 0%; ô diagnostic (ô 9) là lần kiểm tra runtime trước setup mới. Hai output này không chứng minh trạng thái hoặc kết quả của lượt bị ngắt. Notebook được lưu nguyên trạng để có thể kiểm tra lịch sử, không dùng nó làm bài hoàn tất.

Các artifact huấn luyện của lượt mới chưa được tải về trước khi mất kết nối. Không ghép số liệu từ lượt khác để tự nhận lượt này đã hoàn thành. Checksum bản tải và trạng thái quan sát được nằm trong `interruption.json`.
