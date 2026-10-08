# Đặc tả hoàn thiện Day22

Hoàn thiện bài cá nhân của Trần Quốc Vương, MSSV 2A202602522, theo `README.md`, `rubric.md`, `HARDWARE-GUIDE.md` và `docs/reference.md`.

1. Cài `my_dpo_loss` bằng negative log-sigmoid của chênh lệch log-ratio, giữ gradient và kiểm chứng loss bằng log 2 khi policy bằng reference.
2. Thực hiện NB0–NB4; lưu notebook có output thực, log môi trường, cấu hình và dữ liệu/adapter fingerprint.
3. SFT dùng loss chỉ trên câu trả lời, lưu adapter và reference SFT gộp. DPO phải khởi tạo LoRA mới trên reference đó; không dùng raw base làm reference.
4. Giữ 1.000 mẫu SFT, 800/100 cặp tiếng Việt, split theo prompt không trùng, seed 42; đọc ít nhất ba cặp và đo thiên vị độ dài bằng tokenizer thật.
5. Giữ beta 0,1, LR 5e-6, một epoch DPO; ghi reward chosen/rejected và margin riêng cho train/held-out, cùng thời gian và peak VRAM đo được.
6. Sinh greedy cho cùng tám câu cố định và ít nhất 50 prompt held-out bằng hai mô hình; chấm bằng hai reward model mặc định, kiểm sanity, bootstrap CI, bias độ dài và bất đồng giám khảo.
7. Nếu GPU 4 GiB không chạy được model mặc định, dùng override model nhỏ cùng họ Qwen3, ghi rõ giới hạn và lý do; không giảm dữ liệu hay sửa tiêu chí verify để che thiếu bài.
8. Điền `submission/REFLECTION.md` bằng kết quả thật, §3 ít nhất 100 từ và §6 ít nhất 150 từ. Giải thích kết quả không có lợi cho DPO nếu xảy ra; không bịa trải nghiệm người học.
9. Lưu bốn ảnh thực từ pipeline; giữ nguyên rubric, dữ liệu gốc và các test gốc. Thêm test cho phần tự cài và hướng dẫn tái lập.
10. Chạy CPU tests, kiểm Colab đồng bộ và `make verify`. Tạo checklist theo evidence; đánh dấu bonus chưa chạy, GitHub/LMS chưa nộp nếu chưa thực hiện. Không commit/push hoặc publish HF nếu chưa có yêu cầu riêng cho Day22.

Không đưa `.env`, credential, cache hay trọng số vào Git. Người học sẽ cung cấp checklist bổ sung sau khi hoàn thành README.
