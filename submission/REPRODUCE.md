# Tái lập Day22 — Qwen3-4B / Colab T4

## Chạy sạch toàn bộ

1. Mở `colab/Lab22_T4_CLEAN_RUN.ipynb` trên Colab, chọn Tesla T4 và runtime mới.
2. Chạy tất cả ba cell code. Launcher tạo môi trường Python 3.12 riêng, checkout source-only, giữ đủ 1.000/800/100 mẫu, 100 bước DPO và 50 prompt held-out; không lấy output/adapter cũ.
3. Tải ZIP bằng chứng và notebook đã thực thi. Kiểm pipeline exit code, nguồn, split, hash và raw score trước khi diễn giải kết quả.

Commit core được pin trong launcher; wrapper và bản sửa `lab22/judge.py` được nhúng rõ ràng, hash nguồn thực được lưu. Bản sửa chỉ đổi tính ổn định và kiểm lỗi judge, không đổi SFT/DPO hoặc dữ liệu. Trên T4, launcher dùng reward model FP32, `device_map=auto`, tối đa 11 GiB GPU và 6 GiB CPU, không NF4. Hai judge phải qua preflight Transformers vanilla với cùng 12 cặp sanity và ngưỡng 80% trước khi bắt đầu pipeline. Preflight đạt không thay thế kết quả sanity và score thực của NB4.

Colab host có thể dùng Python 3.13; không cài GPU stack trực tiếp vào host. Các job dùng `/content/day22-venv/bin/python`, torch 2.7.0+cu118 và wheel xformers chính thức cho Python 3.12. Wrapper đặt PATH của venv, cấu hình Jupyter/IPython riêng và kernelspec `IPythonKernel` với danh sách startup extensions rỗng, tránh nạp kernel/extension chỉ có trong host Colab. Cấu hình và phiên bản thực được lưu trong bằng chứng; giữ log nếu setup hoặc khởi động kernel thất bại.

## Chỉ sửa và chấm lại NB4, không train lại

Trên **notebook T4 cũ đang giữ `/content/day22-clean-t4`**, thêm một cell code cuối, dán toàn bộ nội dung `colab/Lab22_REJUDGE_CELL.txt` và chỉ chạy cell mới. Không Run all hoặc xoá runtime. Bản chia ba cell để tham khảo là `colab/Lab22_REJUDGE_ONLY.ipynb`; mở notebook mới có thể tạo runtime khác không giữ workspace hiện tại. Scorer chạy trong subprocess Transformers vanilla, không import Unsloth. Nó giữ nguyên 58 prompt/answer, lưu JSON strict không NaN, kiểm sanity cùng 12 cặp và ngưỡng 80%, lưu dữ liệu mới vào `data/eval/rechecked/` thay vì ghi đè lượt trước.

Model/judge cũ phải được giải phóng trước khi nạp reward model. Mỗi judge nạp lần lượt. FP32/offload dùng tối đa 10 GiB GPU và 6 GiB CPU; có thể chậm hơn FP16. Không đảm bảo cả hai judge qua trước khi đo. Nếu OOM hoặc lỗi số học, giữ log và dừng, không đổi ngưỡng để ép pass. Nếu thư mục rechecked đã tồn tại, dùng thư mục mới; không xoá bằng chứng lượt trước.

Nếu mất runtime, không cần train lại để rejudge: tạo T4 mới, cài stack, checkout source và giải nén ZIP bằng chứng đã lưu để phục hồi `data/eval/side_by_side.jsonl`; reward models có thể tải lại. Adapter/policy weights không cần cho thao tác chấm lại.

## Kiểm tra trước khi nộp

```bash
python -m pytest scripts/ -q
python scripts/build_colab.py --check
python scripts/build_clean_colab.py --check
make verify
python scripts/audit_submission.py
```

`make verify` phải chạy trong workspace tương ứng với reference adapter. Bản 4B ghi địa chỉ huấn luyện thật `/content/day22-clean-t4/models/sft-merged`; không sửa adapter config hoặc gatekeeper để che việc chuyển thư mục. Chạy gatekeeper trong Colab gốc hoặc mount repo vào đúng `/content/day22-clean-t4` trong Docker để kiểm metadata. Cell export giữ `config.json`, `generation_config.json` và tên/kích thước/SHA-256 của trọng số merged thật, không tạo config giả. Clone mới muốn chạy training/generation cần tái tạo merged reference và weights, vì ZIP bài làm không chứa trọng số.

Chỉ kết quả đã commit mới có trên GitHub. Không đưa `.env`, cache, models hoặc `.safetensors` vào repo/ZIP nộp. Đóng gói sau khi nhập recheck và cập nhật phản tư; ZIP bằng chứng lượt đầu không tự là bài nộp hoàn chỉnh.
