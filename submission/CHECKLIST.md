# Checklist Day22 — Trần Quốc Vương / 2A202602522

Checklist này đối chiếu README/rubric gốc và các mục người học đã gửi: tổng quan, chuẩn bị Colab, NB0–NB4, bài phản tư, danh sách bonus, nộp bài và thang điểm. Chỉ đánh dấu hoàn thành khi có bằng chứng thật; không đánh đồng việc chạy local với chạy đúng cấu hình Colab T4.

## Chuẩn bị và NB0

- [x] Tìm đúng repo Day22; đọc README, rubric, HARDWARE-GUIDE, docs/reference và mẫu phản tư.
- [x] Fork GitHub đã tồn tại và public: https://github.com/Neon310304/K4-L3-Track3-Day22-DPO-ORPO-Alignment. README không yêu cầu đổi tên repo theo mẫu riêng.
- [x] Có `IMPLEMENTATION_PROMPT.md`, hướng dẫn tái lập, môi trường Linux/CUDA riêng và `.env` local bị ignore.
- [x] Khai báo GPU 4 GiB, model Qwen3-0.6B override, hai judge NF4; không trình bày đây là kết quả Qwen3-4B/T4 mặc định.
- [x] NB0 tự cài negative log-sigmoid, qua assert tham chiếu/log 2 và kiểm gradient/ổn định số; notebook có output thực.
- [x] CPU tests và smoke GPU/import chạy qua; xem `submission/evidence/`.

## Phần bắt buộc đã hoàn thành

- [x] NB1: đủ 1.000 mẫu, response-only loss; loss log 2,0473 → 1,8390; SFT adapter và merged reference có thật local, hash tại `evidence/sft-reference.json`.
- [x] NB2: đủ 800/100 cặp tiếng Việt, split prompt không trùng; ba ví dụ có output; chosen dài hơn trong 65,875% cặp, có Parquet và `stats.json`.
- [x] NB3: reference là SFT đã gộp; đủ một epoch/100 bước; lưu chosen/rejected riêng train/held-out, metrics/runtime/VRAM thực. Lượt OOM và offload CPU được khai báo, không xoá log lỗi.
- [x] NB4: đủ tám câu cố định + 50 prompt held-out khác nhau; cùng greedy decode/batch 1/384 token; giữ nguyên raw outputs.
- [x] Nạp/chấm hai judge và áp dụng sanity filter gốc: Qwen đạt 7/12 nên bị loại; Llama 12/12 là judge duy nhất của kết quả chính. Có per-judge, agreement, CI, length-matched và thiên vị độ dài; không tự nhận hội đồng hai thành viên đã qua sanity.
- [x] Đủ bốn PNG và notebook NB0–NB4 có output thật, không dùng ảnh mẫu; bảng NB4 chỉ dàn lại từ raw.
- [x] REFLECTION có số thực, §3 ≥100 từ và §6 ≥150 từ, ví dụ hữu ích/an toàn; giải thích cả rejected reward tăng và judge trượt sanity.
- [x] CPU tests: **61 passed** (`evidence/pytest.log`); Ruff và Colab sync pass; `make verify` exit 0 (`evidence/verify.log`); **25/25** audit checks pass (`evidence/audit.json`).
- [x] Quét danh sách file có thể đưa lên Git không thấy key thật, `.env`, cache hoặc trọng số (`evidence/publishable-scan.json`); tests/rubric/gatekeeper gốc không bị nới lỏng.

## Kết luận đo được

- Held-out: 17 DPO thắng / 10 SFT thắng / 23 hoà; win rate tính hoà 0,5 là **57%, CI95% [47%, 67%]**. Chưa chứng minh DPO thắng SFT.
- Chẩn đoán tự động `INTENDED`, nhưng chosen **và rejected đều dương**, không khớp hoàn toàn mẫu lý tưởng chosen↑/rejected↓. Có giải thích trong REFLECTION §3.
- Cả hai model còn câu trả lời yếu hoặc không an toàn; sanity 12/12 không bảo đảm verdict đúng từng câu. Không coi đây là model sẵn sàng triển khai.
- Các trọng số chỉ giữ local; clone mới phải train/tái tạo reference trước verify. Không có API trả phí được gọi.

## Đối chiếu thang điểm

Điểm dưới đây là trọng số tiêu chí của đề, **không phải điểm tự chấm hoặc cam kết được giảng viên cho**. Các bằng chứng chưa commit/push vẫn chỉ nằm local.

| Tiêu chí | Điểm của đề | Bằng chứng / trạng thái |
|---|---:|---|
| NB0: loss qua kiểm tra | 6 | Notebook có output assert; tests bổ sung kiểm gradient và ổn định số. |
| NB0: giải thích displacement | 4 | REFLECTION §3 giải thích hiệu hai log-ratio bằng hai kịch bản toy. |
| NB1: SFT loss giảm, merged reference | 8 | Loss 2,0473 → 1,8390; model gộp thật local, hash tại `evidence/sft-reference.json`. |
| NB2: split prompt, ba mẫu | 8 | 800/100 cặp, assert disjoint; notebook in ba mẫu, REFLECTION §1 phân tích. |
| NB2: thiên vị độ dài | 4 | `02b-pref-length.png`, `data/pref/stats.json`: 65,875% chosen dài hơn. |
| NB3: DPO reference là SFT | 6 | `adapters/dpo/adapter_config.json` trỏ `/lab/models/sft-merged`; fingerprint split được kiểm. |
| NB3: reward train/held-out tách riêng | 10 | `03-dpo-reward-curves.png` và log_history lưu chosen/rejected ở cả hai split, không chỉ margin. |
| NB3: diễn giải chẩn đoán | 8 | REFLECTION §3 nói rõ nhãn INTENDED của hàm gốc và sự khác biệt với mẫu rejected giảm. |
| NB4: 8 cố định + ít nhất 50 held-out | 6 | 58 cặp raw và notebook có output, 50 prompt held-out khác nhau. |
| NB4: judge, CI, sanity, độ dài | 10 | Summary/raw/per-judge đủ chỉ số; Qwen bị loại vì sanity 7/12, hạn chế được khai báo. |
| Phản tư §3/§4/§6 | 20 | Có số thật, diễn giải, ví dụ hữu ích/an toàn và quyết định thực nghiệm; không có placeholder core. |
| Tái lập từ đầu | 5 | Có Docker/dependencies/env/hướng dẫn và từng stage đã thực thi bằng runner; **chưa chạy lại trọn `make pipeline` từ một clone/môi trường sạch mới với code cuối**. Không coi docs là bằng chứng của lượt chạy mới này. |
| `make verify` không lỗi | 5 | Đã exit 0 trong môi trường local `/lab`, log tại `evidence/verify.log`. |

Không tự khẳng định đạt 100/100. Model 0.6B thay cho 4B, judge NF4 chỉ còn một thành viên hợp lệ, và phạm vi kiểm chứng tái lập phải được người chấm đánh giá theo phần khai báo. DPO không thắng SFT không tự làm mất điểm nếu giải thích đúng bằng chứng.

## Đối chiếu hướng dẫn người học gửi

| Mục | Trạng thái | Bằng chứng / lưu ý |
|---|---|---|
| Chuẩn bị Colab T4 | Chưa thực hiện trên Colab | Bundle Colab đồng bộ, nhưng lượt đo thực dùng Docker/local GPU 4 GiB và Qwen3-0.6B; không phải Qwen3-4B/T4. |
| NB0 | Hoàn thành | Loss khớp tham chiếu và log 2; đã giải thích margin tăng khi chosen giảm nếu rejected giảm nhanh hơn trong REFLECTION §3. |
| NB1 | Hoàn thành trên model override | 1.000 mẫu, adapter và merged reference thật local; loss log 2,0473 → 1,8390, có ảnh. |
| NB2 | Hoàn thành | 800/100 cặp, assert split prompt không trùng, ba ví dụ đã phân tích; chosen dài hơn 65,875%, có ảnh/Parquet. |
| NB3 | Hoàn thành, kết quả cần diễn giải | 100 bước, eval 25/50/75/100, reward train/held-out riêng; chẩn đoán tự động INTENDED nhưng cả chosen/rejected dương, không đạt mẫu lý tưởng rejected giảm. |
| NB4 | Hoàn thành, hạn chế judge đã khai báo | 8 cố định + 50 held-out, đủ raw/summary/ảnh; cả hai RM được chấm nhưng Qwen 7/12 bị loại, Llama 12/12 được giữ. Win rate 57%, CI [47%, 67%], chưa chứng minh DPO thắng. |
| Độ dài và per-judge | Hoàn thành | Held-out câu dài hơn thắng 29,63%; length-matched 59,72% trên 36 cặp; win rate Qwen 49% vs Llama 57%; agreement 62,07% trên 58 cặp. |
| Bài phản tư | Hoàn thành | §1–§2 khai báo cấu hình/số đo thật; §3 có 419 từ phân cách bằng khoảng trắng ngoài bảng/ảnh; §4 có bảng và các ca h2/h3/s3/s4; §6 có 355 từ, đủ phương án thay thế/lý do/kết quả/thay đổi dự kiến. Các mục bonus ghi rõ chưa chạy. |
| Bonus | Chưa chạy | Không có số liệu để nhận điểm NB3b/NB5/NB6/NB7, beta-sweep, API cross-judge hoặc HF Hub. |

## Thao tác nộp còn lại

- [x] Kiểm tra lại GitHub API: repo Neon310304/K4-L3-Track3-Day22-DPO-ORPO-Alignment đang public.
- [x] Đủ năm notebook có output, bốn PNG, raw/summary eval, JSON DPO và REFLECTION trên đĩa; các artifact bắt buộc không bị Git ignore. `make verify` đã exit 0.
- [ ] Commit/push nội dung Day22: người học đã yêu cầu; đang kiểm tra file nộp trước khi thực hiện.
- [ ] Nộp link repo public vào LMS: người học thực hiện, không đồng nghĩa với push GitHub.
- [ ] Giữ repo public đến khi có điểm: người học duy trì; không thể xác nhận hoàn thành việc trong tương lai.
- [ ] NB3b, GGUF, benchmark, GRPO, beta-sweep, API cross-judge, HF Hub: chưa chạy, không tự nhận điểm bonus.

Không commit `.env`, credential, cache hay trọng số. Toàn bộ hướng dẫn người học đã gửi đến phần thang điểm đã được đối chiếu. File mới hoặc sửa chưa commit chưa được tính là bài đã nộp trên GitHub. Kết quả bất lợi hoặc CI chứa 0,5 vẫn hợp lệ nếu được giải thích trung thực.
