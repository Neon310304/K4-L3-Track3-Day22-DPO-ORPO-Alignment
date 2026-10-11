# Checklist Day22 — Trần Quốc Vương / 2A202602522

Trạng thái 2026-10-11. Kết quả chính là Qwen3-4B trên Tesla T4; giữ riêng các lượt lịch sử và không nhận bonus chưa chạy.

- [x] NB0: loss, log(2) và gradient qua assert trong notebook.
- [x] NB1: 1.000 SFT, 125 bước, response-only loss; loss log step 10 → 120 giảm 1,884157 → 1,284026; reference SFT merged được tạo thật.
- [x] NB2: 800/100 Parquet, không trùng prompt, đọc ba mẫu; chosen dài hơn trong 527/800 cặp.
- [x] NB3: đủ một epoch/100 bước, reference SFT, reward chosen/rejected train và held-out tại 25/50/75/100.
- [x] NB4: đủ 8 câu cố định + 50 held-out, cùng greedy decode; lưu nguyên văn và SHA answers.
- [x] Lượt đầu `make pipeline` hoàn tất NB0–NB3 rồi lỗi CUDA ở NB4; giữ manifest exit 2 và log tại `history/T4_NB4_FAILED_20261011T014838Z/`.
- [x] `make eval` phục hồi NB4 bằng subprocess Transformers riêng; không train lại, không sinh lại answers. Manifest cuối ghi rõ cả hai entrypoint và đủ năm stage thành công.
- [x] Hai judge đạt 12/12 sanity mỗi model, giữ ngưỡng 80%; 232 score hữu hạn, không coi NaN là tie.
- [x] Held-out: 3 DPO / 9 SFT / 38 hòa; win rate 44%, CI95% [37%; 50%]. Chưa đủ bằng chứng DPO tốt hơn SFT.
- [x] Năm notebook giữ output và thời gian thực thi thật; đủ bốn PNG đã kiểm tra, metrics, split fingerprint, raw/summary.
- [x] `models/sft-merged/config.json` thật và receipt hash trọng số; weights giữ riêng, không nằm trong Git.
- [x] REFLECTION cập nhật số liệu mới, ví dụ hữu ích/an toàn, hạn chế thẻ công cụ và phục hồi NB4; §3/§6 có 241/304 từ.
- [x] CPU tests code hiện tại: 86 passed; receipt ghi timestamp và SHA nguồn, không thay thế sanity GPU.
- [x] `make verify`, audit và kiểm đồng bộ cả ba builder đều exit 0; kiểm riêng 13 hash nguồn và lịch sử phục hồi đều đạt.
- [x] ZIP Colab gồm 61 tệp qua CRC và toàn bộ SHA; bản sao LoRA riêng gồm 46 tệp cũng đã kiểm hash.
- [x] Bộ nguồn và bằng chứng chuẩn bị xuất bản không chứa `.env`, token, cache hoặc weights; receipt quét toàn bộ Git index lưu riêng.
- [ ] Người học đọc lại phản tư và nộp link repo vào LMS; chưa nộp LMS.

GitHub: [Neon310304/K4-L3-Track3-Day22-DPO-ORPO-Alignment](https://github.com/Neon310304/K4-L3-Track3-Day22-DPO-ORPO-Alignment). Kết quả push và commit được kiểm trực tiếp với remote sau khi các kiểm tra trên hoàn tất.

Bonus β-sweep, ORPO/RPO/LD-DPO, GGUF, benchmark, GRPO và HF Hub publication chưa thực hiện. Lịch sử 0.6B, lượt 4B có NaN ngày 2026-10-09 và lượt bị ngắt giữ trong `history/`; không dùng các log đó để chứng nhận bài chính.
