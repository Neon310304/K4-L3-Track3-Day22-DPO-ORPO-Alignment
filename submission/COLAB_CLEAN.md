# Colab và bằng chứng thực thi

Launcher tái lập: `colab/Lab22_T4_CLEAN_RUN.ipynb`, ba cell code setup/pipeline/export. Python host chỉ điều phối; job GPU chạy `/content/day22-venv/bin/python` 3.12, kernelspec IPython chuẩn, không startup extensions của Colab. Judge FP32/offload, không NF4; sanity 80% giữ nguyên.

Notebook Drive giữ các cell cũ có output lịch sử. Kết quả bài nộp chính nằm ở năm `notebooks/00…04.ipynb` đã xuất từ runtime mới. Cell setup 10 và pipeline 11 thuộc lượt 2026-10-11; cell hoàn thiện 12 ghi trạng thái phục hồi và xuất ZIP cuối. Không Run all notebook Drive cũ để tái lập: dùng launcher mới đã pin source.

Lượt này bắt đầu sạch ở NB0 và phải phục hồi NB4 sau lỗi CUDA trong kernel Unsloth. Reward inference hiện chạy trong tiến trình Transformers riêng. Script `resume_nb4.py` kiểm SHA answers và giữ log/manifest lỗi trước khi thực thi notebook mới. README/phản tư ghi đúng nguồn hỗn hợp có phục hồi; không gọi đây là lượt cuối chạy sạch từ đầu không gián đoạn.

Tệp tải về được kiểm CRC và SHA trước nhập. `export-sha256.json` mô tả byte lúc export; các báo cáo/receipt thêm sau import có hash trong bước kiểm cuối. ZIP LoRA riêng không được commit. Giữ runtime và tab kết quả để xem lại; Colab miễn phí không có bảo đảm uptime.
