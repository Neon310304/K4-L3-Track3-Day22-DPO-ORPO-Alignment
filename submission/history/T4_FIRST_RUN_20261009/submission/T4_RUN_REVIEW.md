# Kiểm tra lượt Colab T4 / Qwen3-4B — 2026-10-09

## Kết luận

**Pipeline NB0–NB4 đã hoàn tất trên Tesla T4 với Qwen3-4B, trong workspace source-only mới. Vấn đề Qwen reward judge chưa được khắc phục.** Không coi exit code 0 là bằng chứng cả hai judge hợp lệ, hoặc DPO tốt hơn SFT.

Đây là bản kiểm tra ZIP đầu, không thay cho `REFLECTION.md` của bài nộp. Kết quả 4B đã được nhập vào `notebooks/`, `data/`, `adapters/` và phản tư đã cập nhật. Báo cáo 0.6B ở `submission/history/`, weights và log cũ ở `.cache/before-t4-import-20261009`; Git vẫn giữ lịch sử. Chưa commit/push bài 4B. Gatekeeper còn thiếu merged SFT config thật và audit chặn NaN/tie lỗi; không dùng config giả hay hạ kiểm tra để ép pass.

## Nguồn và tính toàn vẹn

- File: `day22-t4-evidence.zip`, 3.460.298 byte, 28 mục; tổng giải nén 15.519.857 byte.
- SHA-256: `272a2f3c7531a2a7652d5fecf65de2a997a7a68df55739a8c2716d2ac3521de1`.
- CRC không lỗi; cả 27 file trong `export-sha256.json` khớp hash. Không có đường dẫn vượt thư mục đích, `.env` hoặc trọng số mô hình trong ZIP.
- 13 hash nguồn core khớp blob Git tại commit `4212d5dc6eae204cbd0121c29ccbc9ef6491ba1b`. So trực tiếp với bytes file Windows sẽ lệch do CRLF; đã kiểm bằng blob Git, không kết luận nhầm là sửa nguồn.
- Hash wrapper khớp `scripts/run_clean_pipeline.py` local sau chuẩn hoá CRLF → LF.
- `pipeline.json`: `fresh_artifacts=true`, không tái dùng cache model/dataset, `exit_code=0`; cả năm stage exit code 0.
- Tổng thời gian: 5.662,96 giây, khoảng **94 phút 23 giây**; 13:29:33–15:03:56 ngày 2026-10-09, múi giờ Asia/Bangkok. Đây là thời gian pipeline, không gồm cài thư viện.

## Bằng chứng thực thi

| Notebook | Cell code đã thực thi | Cell lỗi | Mã cell khớp nguồn |
|---|---:|---:|---|
| NB0 | 9/9 | 0 | Có |
| NB1 | 7/7 | 0 | Có |
| NB2 | 5/5 | 0 | Có |
| NB3 | 6/6 | 0 | Có |
| NB4 | 5/5 | 0 | Có |

Cả bốn PNG đều mở được và đã xem trực tiếp. Biểu đồ reward có đủ chosen/rejected cho train và held-out, không chỉ có margin. Bảng side-by-side có tám câu cố định, nhưng văn bản trong ô bị cắt; nội dung đầy đủ vẫn còn trong JSONL.

## Cấu hình và dữ liệu

| Mục | Giá trị kiểm được |
|---|---|
| GPU | Tesla T4, 15.637.086.208 byte VRAM |
| Model | `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` |
| Môi trường | Python 3.13.15; torch 2.7.0+cu118; Unsloth 2026.10.2; TRL 1.13.0; transformers 5.17.0 |
| SFT | 1.000 mẫu, 1 epoch, 125 optimizer steps |
| Preference | Đọc trực tiếp Parquet: 800 train / 100 eval; không trùng prompt giữa hai phía |
| Eval | 100 prompt khác nhau; 50 prompt sinh held-out đều thuộc split eval |
| DPO | β=0,1; LR=5e-6; 1 epoch; 100 optimizer steps; max length 768; seed 42 |
| Reference | Adapter config trỏ tới `/content/day22-clean-t4/models/sft-merged` |
| Fingerprint split | Hash hai Parquet khớp `adapters/dpo/split.json` |
| NB4 | 58 ID khác nhau: 4 helpfulness + 4 safety + 50 held-out; không có câu trả lời rỗng |

## Kết quả huấn luyện

- SFT: loss log đầu **1,8842** → log cuối **1,2836**; mean train loss **1,3602**. Xu hướng giảm, không giảm đơn điệu.
- Chosen dài hơn rejected trong **65,875%** cặp train; median 94 so với 86 token.
- DPO train: chosen reward **+0,390596**, rejected **+0,296900**, margin **+0,093696**.
- DPO held-out: chosen reward **+0,409999**, rejected **+0,322729**, margin **+0,087270**, accuracy **68%**.
- Nhãn tự động là `INTENDED`, nhưng **rejected cũng tăng**. Mẫu quan sát không hoàn toàn trùng trường hợp lý tưởng chosen tăng/rejected giảm; cần diễn giải thay vì chỉ sao chép nhãn.

## Kết quả chấm và vấn đề còn lại

`judge_summary.json` và raw results có cùng SHA-256 của `side_by_side.jsonl`. Đã tính lại bốn nhóm bằng hàm `summarize(..., seed=42)`; mọi trường của overall, heldout, helpfulness và safety đều khớp.

| Judge | Sanity | Điểm hữu hạn trên 58 cặp | Tình trạng |
|---|---:|---:|---|
| Skywork Qwen3-4B | 0/12 = 0% | 0/116 | Toàn bộ score là `NaN`; bị loại khỏi panel |
| Skywork Llama-3.2-3B | 12/12 = 100% | 116/116 | Judge duy nhất trong kết quả chính |

Không diễn giải sanity 0% như bằng chứng Qwen không hiểu tiếng Việt: scorer đang lỗi số học. Code gốc so sánh `NaN` bằng `>` nên rơi vào nhánh tie; 58 tie của Qwen không phải phán quyết hợp lệ. JSON raw chứa 116 giá trị `NaN`, không tương thích parser JSON nghiêm ngặt. Win rate 50%, CI [50%, 50%] của Qwen và thống kê đồng thuận có Qwen không đáng tin. FP16 trên T4 là một giả thuyết cần đo riêng, chưa được xác nhận là nguyên nhân.

| Nhóm, theo Llama hợp lệ | DPO thắng / SFT thắng / hoà | Win rate DPO, hoà tính 0,5 | CI 95% |
|---|---|---:|---|
| Overall | 6 / 8 / 44 | 48,28% | [42,24%; 54,31%] |
| Held-out | 4 / 7 / 39 | 47% | [40%; 53%] |
| Helpfulness | 1 / 1 / 2 | 50% | [12,5%; 87,5%] |
| Safety | 1 / 0 / 3 | 62,5% | [50%; 87,5%] |

Held-out length-matched win rate **45,74%** trên 47 cặp; tỉ lệ câu dài thắng **54,55%** trên các cặp quyết định có độ dài khác nhau. Mean chars SFT/DPO lần lượt **649,28 / 622,34**. CI held-out chứa 50%: **chưa đủ bằng chứng DPO tốt hơn SFT**. 44/58 cặp có câu trả lời SFT và DPO giống hệt nhau, giải thích phần lớn số hoà.

Đầu ra còn chứa thẻ `<tool_call>` hoặc `</tool_call>` không mong muốn; đã quan sát trong cả JSONL lẫn ảnh bảng. Không xoá thẻ rồi tự dùng lại summary cũ: thay đổi câu trả lời sẽ đổi SHA và cần chấm lại.

## Việc tiếp theo

1. Giữ runtime Colab và model đang có nếu còn phiên. Chẩn đoán scorer Qwen, kiểm tra điểm hữu hạn trước khi diễn giải sanity; không hạ ngưỡng 80%.
2. Nếu cần sửa judge, chấm lại **58 câu trả lời đã lưu**, không bắt buộc train lại NB1–NB3. Lưu riêng lượt chấm cũ, cấu hình và raw score mới; không ghi đè để che lỗi.
3. Khi nhập bằng chứng 4B vào bài nộp, lưu lịch sử 0.6B, cập nhật `REFLECTION.md` bằng số 4B và ví dụ thực. Đường reference Colab trong adapter config cần được xử lý minh bạch khi chạy gatekeeper ở vị trí khác, không sửa checker để ép pass.
4. Chạy gatekeeper/audit cho bộ bài nộp nhất quán rồi mới commit/push. Không đánh dấu “cả hai judge đã đạt” với ZIP hiện tại.
