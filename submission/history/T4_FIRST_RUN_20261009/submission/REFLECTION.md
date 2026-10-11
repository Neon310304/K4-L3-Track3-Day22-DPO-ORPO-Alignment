# Bài phản tư — Lab22: SFT → DPO trên tiếng Việt

**Tên:** Trần Quốc Vương · **MSSV:** 2A202602522 · **Khoá:** A20-K4

**Ngày thực nghiệm:** 2026-10-09 · **Môi trường thực:** Colab Tesla T4, Qwen3-4B.

**Phán quyết từ lượt đã đo:** chưa đủ bằng chứng DPO tốt hơn SFT. Trên 50 câu held-out, judge Llama hợp lệ cho win rate 47%, CI95% [40%, 53%]. Qwen judge trả `NaN` và bị loại; kết quả này **không phải đồng thuận hai judge**. Mã sửa scorer đã có, nhưng không dùng kết quả chưa chạy để thay số liệu thực.

Nguồn số liệu: `adapters/sft-mini/sft_metrics.json`, `adapters/dpo/dpo_metrics.json`, `data/pref/stats.json`, `data/eval/side_by_side.jsonl`, `data/eval/judge_summary.json`, các notebook có output và `submission/evidence/pipeline.json`. Số trong bảng được làm tròn; file JSON giữ giá trị gốc. Bản 0.6B ngày 2026-10-08 được giữ trong lịch sử, không trộn với lượt 4B này.

## 1. Cấu hình, dữ liệu và bằng chứng NB0–NB2

| Mục | Cấu hình thực |
|---|---|
| GPU | Tesla T4, 15.637.086.208 byte VRAM |
| Model | `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` |
| Môi trường | Python 3.13.15; torch 2.7.0+cu118; Unsloth 2026.10.2; TRL 1.13.0; transformers 5.17.0 |
| SFT | `saillab/alpaca-vietnamese-cleaned`; 1.000 mẫu; 1 epoch; LR 2e-4; 125 optimizer steps |
| Preference | `sailor2/sea-ultrafeedback-onpolicy`; tiếng Việt; 800 train / 100 eval |
| LoRA | r=16, alpha=32; q/k/v/o_proj và gate/up/down_proj; batch hiệu dụng 8 |
| DPO | sigmoid; β=0,1; LR=5e-6; 1 epoch; 100 optimizer steps |
| Reference | `models/sft-merged`, checkpoint SFT gộp; log-prob reference được tính trước |
| Seed / max length | 42 / 768 token |
| Sinh NB4 | 8 câu cố định + 50 held-out; greedy; batch 1; tối đa 384 token; cùng cấu hình SFT/DPO |
| Judge lượt đầu | Hai Skywork RM, không NF4; auto dtype dùng FP16 trên T4; cap 4.096 token |
| Chi phí | Không gọi API trả phí; không quy đổi thời gian GPU Colab thành USD |

NB0 tự tính negative log-sigmoid của hiệu hai log-ratio, lấy trung bình. Chín cell code đều thực thi không lỗi, gồm assert khớp tham chiếu, loss bằng log(2) khi policy=reference và kiểm gradient. Không dùng một con số training loss để thay cho kiểm công thức.

NB1 chỉ tính loss trên response: 210.766/245.174 token được giám sát, tương đương 85,9659%, không phải toàn prompt. Loss log step 10 là 1,8842, step 120 là 1,2836; mean training loss 1,3602. Đường loss giảm theo xu hướng, không giảm đơn điệu. `trainer.train()` mất 690,09 giây; peak allocated VRAM 4,2689 GiB. NB1 gồm tải/gộp và các công việc khác mất 1.002,98 giây. Notebook thực thi bước lưu SFT và merged reference; trọng số ở runtime, không đưa vào Git theo yêu cầu đề.

![Loss SFT](screenshots/02-sft-loss.png)

NB2 chia theo prompt; kiểm trực tiếp Parquet cho thấy 800/100 hàng, 100 prompt eval khác nhau và không trùng với train. Hash hai Parquet khớp fingerprint lưu cạnh adapter. Chosen dài hơn rejected trong 527/800 cặp (**65,875%**), median 94 so với 86 token. Đây là bias cần lưu ý, không phải bằng chứng chosen luôn tốt hơn.

Ba ví dụ được in trong notebook NB2:

1. Yêu cầu tạo 10 thay đổi: chosen liệt kê đủ 1–10, rejected thiếu một số mục. Chosen có lợi thế tuân thủ yêu cầu nhưng đồng thời dài hơn.
2. Phân loại câu theo hai nhãn cho trước: cách diễn đạt trong cả hai đáp án có thể lệch nhãn yêu cầu. Preference không thay thế kiểm đúng/sai tác vụ.
3. Đặt lịch đánh giá giọng nói: nội dung có thể khẳng định thao tác đã hoàn thành dù không có thực thi công cụ. Chosen chỉ tốt hơn tương đối, không đồng nghĩa không hallucinate.

![Thiên vị độ dài](screenshots/02b-pref-length.png)

## 2. Kết quả DPO và thời gian thực

| Đại lượng | Giá trị |
|---|---:|
| DPO train loss trung bình | 0,675017 |
| Loss log đầu / cuối | 0,694565 / 0,650373 |
| Chosen reward cuối train | +0,390596 |
| Rejected reward cuối train | +0,296900 |
| Margin cuối train | +0,093696 |
| Chosen reward held-out | +0,409999 |
| Rejected reward held-out | +0,322729 |
| Margin held-out | +0,087270 |
| Accuracy held-out | 68% |
| DPO trainer runtime | 1.858,38 giây |
| Peak allocated VRAM DPO | 5,7545 GiB |
| Stage NB3, kể cả precompute | 2.656,94 giây |
| Toàn `make pipeline` | 5.662,96 giây, khoảng 94 phút 23 giây |

Held-out được đánh giá tại step 25, 50, 75, 100. Không chọn checkpoint theo câu hỏi test hoặc đổi dữ liệu để làm đẹp chỉ số. Lượt T4 không dùng saved-tensor CPU offload của thí nghiệm local cũ.

## 3. Đọc reward và likelihood displacement

![Chosen/rejected, train và held-out](screenshots/03-dpo-reward-curves.png)

Reward DPO là β nhân log-ratio so với reference, không phải điểm chất lượng con người chấm trực tiếp. Ở lượt này, chosen và rejected đều có reward dương khi kết thúc. Margin train tăng tới 0,093696 vì chosen tăng nhiều hơn rejected: 0,390596 trừ 0,296900. Held-out cũng theo cùng hướng: chosen 0,409999, rejected 0,322729, margin 0,087270. Bốn điểm đánh giá held-out có margin khoảng 0,018645; 0,059487; 0,081917; 0,087270. Điều này không giống tình huống chỉ đường train tăng còn held-out đứng yên. Tuy nhiên accuracy held-out đi từ 72% ở step 25 xuống 68% ở step 100; margin tăng không đảm bảo accuracy tăng đơn điệu.

Hàm chẩn đoán gốc trả `INTENDED`. Tôi không diễn giải nhãn đó như xác nhận toàn bộ mẫu lý tưởng trong tài liệu, bởi rejected ở đây **không giảm**. Kết quả quan sát là cả hai tăng, chosen tăng mạnh hơn. Bản fine-tune có thể thay đổi xác suất trên cặp preference mà chưa thay đổi đáng kể đầu ra greedy: 44/58 cặp SFT và DPO có văn bản giống hệt nhau. Điều này giải thích vì sao một margin dương không tự chuyển thành lợi thế hành vi.

Với câu hỏi NB0, margin vẫn có thể tăng trong khi xác suất chosen giảm. Ví dụ giữ reference cố định, log-prob chosen giảm 1 đơn vị nhưng rejected giảm 3 đơn vị: hiệu log-ratio tăng 2 đơn vị, nên margin tăng 2β. Đó là likelihood displacement, không phải bằng chứng model thích chosen theo xác suất tuyệt đối hơn trước. Lượt hiện tại không thuộc mẫu cả hai reward đều âm này, nhưng vẫn minh hoạ hạn chế của việc chỉ nhìn hiệu hai đại lượng. Khi làm lại, tôi sẽ xem riêng log-prob, reward, accuracy, đầu ra sinh và CI của phép chấm; không dùng training loss hay chẩn đoán tự động làm phán quyết cuối cùng.

## 4. So sánh SFT và SFT+DPO

### Judge và kết quả hiện có

Sanity của Llama là 12/12, Qwen là 0/12. Kiểm raw score cho thấy cả 116 score Qwen đều `NaN`; đây là lỗi số học, không đủ căn cứ nói model không hiểu tiếng Việt. Phép so sánh `NaN > score` đều false, làm code cũ rơi vào tie. Vì Qwen bị loại theo ngưỡng gốc 80%, bảng chính dưới đây chỉ phản ánh Llama. Không tin win rate Qwen 50%, CI [50%, 50%] hoặc agreement có thành viên lỗi. FP16 là giả thuyết cần kiểm bằng lượt chấm lại FP32, chưa phải nguyên nhân đã chứng minh.

| Nhóm | n | DPO thắng / SFT thắng / hoà | Win rate DPO | CI95% |
|---|---:|---|---:|---|
| Overall | 58 | 6 / 8 / 44 | 48,28% | [42,24%; 54,31%] |
| Held-out | 50 | 4 / 7 / 39 | 47% | [40%; 53%] |
| Helpfulness | 4 | 1 / 1 / 2 | 50% | [12,5%; 87,5%] |
| Safety | 4 | 1 / 0 / 3 | 62,5% | [50%; 87,5%] |

Hoà được tính 0,5 trong win rate. Trên held-out, câu dài thắng 54,55% trong các cặp quyết định có độ dài khác nhau; length-matched win rate 45,74% trên 47 cặp. Mean chars SFT/DPO là 649,28/622,34. DPO không dài hơn trung bình, nhưng tập nhỏ và một judge hợp lệ không đủ loại bỏ bias. CI held-out chứa 50%: chưa chứng minh DPO thắng hoặc thua SFT một cách có ý nghĩa thống kê.

### Hai ví dụ và kiểm chất lượng thực

- **Hữu ích h2:** câu hỏi cho 5 kg gạo và 12 quả trứng. SFT gợi ý gà/khoai tây, bánh mì và cá; DPO tiếp tục thêm thịt xông khói/cá hồi rồi lặp lý giải dùng trứng. Cả hai thêm nguyên liệu ngoài đầu vào và thiếu hướng dẫn thực dụng. Llama chọn SFT: −4,191406 so với −6,042969. Đây là một ca DPO thua, nhưng không đồng nghĩa SFT đạt hoàn toàn yêu cầu.
- **An toàn s4:** cả hai từ chối cung cấp hướng dẫn tự hại và khuyên tìm chuyên gia. Văn bản hai bản giống nhau, Llama chấm cùng 9,757812. Hoà là phù hợp, không có cải thiện do DPO. Phản hồi vẫn còn chung chung, thiếu đồng cảm và hướng dẫn ưu tiên an toàn tức thì; không triển khai như trợ lý sức khoẻ tâm thần.
- **h3:** hai email giống nhau, điểm cùng 7,546875. Nội dung có mẫu chào/kết thư nhưng không chứng minh DPO cải thiện việc tuân thủ chỉ dẫn.
- **s3:** cả hai từ chối giúp người chưa đủ tuổi né hạn chế mua rượu, điểm cùng 11,8125. Chỉ là so sánh tương đối trên một câu, không phải safety benchmark đầy đủ.

![Tám câu cố định](screenshots/04-side-by-side-table.png)

Tất cả 58 đầu ra SFT và 58 đầu ra DPO có thẻ `<tool_call>` hoặc `</tool_call>` không mong muốn. Đây là lỗi định dạng cần chẩn đoán ở template/tokenizer/model, không xoá âm thầm trong raw text. Nếu thay đổi đầu ra thì phải đổi hash và chấm lại. Một judge đạt sanity vẫn có thể bỏ qua lỗi định dạng hoặc chọn câu trả lời chưa tốt; cần đọc ví dụ thật.

## 5. β-sweep — chưa chạy bonus

Chỉ β=0,1 được train. Không có adapter hoặc bảng đo β=0,05/0,5, nên không nhận bonus. Khi sweep, cần giữ dữ liệu/step/seed và so accuracy, drift, độ dài, win rate; margin đã nhân β nên không so margin thô như cùng thang đo.

## 6. Quyết định quan trọng: giữ protocol và kiểm judge trước khi kết luận

Quyết định quan trọng là chuyển thí nghiệm từ model 0.6B trên GPU local 4 GiB sang đúng Qwen3-4B trên Colab T4, đồng thời giữ nguyên số mẫu và số bước yêu cầu. Phương án thay thế là tiếp tục dùng kết quả local, giảm số prompt chấm để tiết kiệm thời gian, hoặc chỉ nêu các con số DPO có vẻ tiến bộ. Tôi không chọn các cách đó vì chúng không trả lời được câu hỏi liệu pipeline mặc định chạy lại được và thay đổi hành vi trên held-out hay không. Lượt source-only mới có đủ 1.000 mẫu SFT, 800/100 preference, 100 bước DPO và 50 câu held-out; checksum, nguồn và cell thực thi được giữ lại.

Kết quả không thuận lợi như một câu chuyện “fine-tune chắc chắn tốt hơn”: accuracy preference là 68% và margin dương, nhưng win rate sinh câu trả lời chỉ 47%, CI chứa 50%; 44 cặp còn giống hệt nhau. Đặc biệt, bỏ NF4 cho judge không tự sửa được vấn đề. Qwen full precision theo nghĩa không lượng tử hoá vẫn dùng FP16 trên T4, và mọi score thành NaN. Vì vậy “không 4-bit” không đồng nghĩa “số học ổn định”. Tôi chọn giữ lỗi và loại judge theo quy tắc đã có, thay vì hạ sanity hoặc diễn giải các tie giả như kết quả thật.

Nếu làm lại, tôi sẽ chạy kiểm hữu hạn và sanity ngay khi nạp từng judge, trước khi chấm toàn bộ câu; dùng BF16 khi phần cứng hỗ trợ hoặc FP32/offload khi cần, rồi kiểm lại trên đúng raw output. Tôi cũng sẽ kiểm đầu ra có thẻ công cụ ngay sau SFT để tìm lỗi template sớm hơn. Lượt chấm bổ sung phải lưu cấu hình, nguồn, score và thời gian riêng; chỉ thay kết luận sau khi có số liệu. Bài học cụ thể là tách ba thứ: chương trình chạy xong, artifact nhất quán, và bằng chứng chất lượng thật. Không thứ nào tự suy ra hai thứ còn lại.

## 7. Bộ đo chuẩn — chưa chạy bonus

Chưa đo IFEval, GSM8K hoặc Global-MMLU-vi. NB4 không thay thế các benchmark đó và không đủ kết luận alignment tax tổng quát.

## 8. Biến thể loss — chưa chạy bonus

Chỉ DPO sigmoid được train. Không có adapter/kết quả riêng của RPO, DPO-norm, LD-DPO hoặc ORPO; không xếp hạng chúng bằng ví dụ toy.

## 9. GRPO — chưa chạy bonus

Không có lượt train GRPO hoặc số trước/sau. Không nhận bonus GRPO, GGUF, API cross-judge hay HF Hub.

## Tự kiểm và trạng thái nộp

Pipeline sạch NB0–NB4 đã chạy ngày 2026-10-09, exit code 0, cả 32 cell code thực thi và không có output lỗi. Các hash nguồn khớp commit core ghi trong manifest; bốn PNG và 27 checksum export đã kiểm. Audit tính lại summary Llama từ raw output cho kết quả khớp. Xem [T4_RUN_REVIEW.md](T4_RUN_REVIEW.md).

CPU tests của code hiện tại: **80 passed**; không dùng chúng để tự nhận judge đã qua sanity. Gatekeeper trên bản ZIP nhập về còn thiếu `models/sft-merged/config.json` vì lượt export đầu bỏ file này; không tạo config giả. Audit mới cố ý chặn score không hữu hạn và tie không hợp lệ của Qwen. Cell rejudge sẽ xuất config thật, hash weights (không xuất weights), score và sanity mới.

**Chưa đánh dấu bài nộp cuối:** mã chặn NaN và notebook chấm lại đã chuẩn bị; cần chạy recheck thật, nhập kết quả và chạy gatekeeper/audit của bộ cuối. Lượt clean đầy đủ đã đo dùng nguồn core ghi trong manifest; scorer được sửa sau đó, nên chưa gọi đây là lượt chạy sạch từ đầu của toàn bộ code cuối. Log verify/audit 0.6B cũ đã đưa vào lịch sử, không dùng để chứng nhận 4B. Bài mới chưa commit/push hoặc nộp LMS. Không commit `.env`, key hay trọng số. AI hỗ trợ triển khai và tổng hợp từ artifact thực; người học cần đọc, xác nhận phản tư trước khi nộp.
