# Chạy lại Day22 của Trần Quốc Vương — 2A202602522

## Phạm vi

Phần bắt buộc NB0–NB4 theo README; không tự nhận điểm bonus chưa chạy. Máy local có RTX 3050 Ti 4 GiB nên dùng override Qwen3-0.6B thay vì Qwen3-4B của tier T4. Đây là thí nghiệm trên model nhỏ, không chứng minh kết quả của model 4B. Giữ cấu hình dữ liệu T4: 1.000 SFT, 800/100 preference, ít nhất 50 prompt held-out để judge.

## Windows + Docker Desktop

Docker phải dùng Linux containers và GPU NVIDIA hoạt động. Không chạy `make clean` nếu muốn giữ bằng chứng đã đo. Các trọng số và cache chỉ nằm local, không đưa lên Git.

Các lệnh build/run bên dưới dành cho lần tạo môi trường đầu. Nếu container `day22-lab` đã tồn tại nhưng dừng, dùng `docker start day22-lab` rồi chạy các lệnh `docker exec`; không tạo lại hoặc xoá container để kiểm checklist. Khi xong có thể dùng `docker stop day22-lab`, trọng số bind mount và venv trong container vẫn còn.

```powershell
docker build -f Dockerfile.local -t day22-lab:local .
New-Item -ItemType Directory -Path .cache -Force
$repo = (Get-Location).Path
docker run -d --name day22-lab --gpus all --ipc=host --mount "type=bind,source=$repo,target=/lab" --mount "type=bind,source=$repo\.cache,target=/cache" -e HF_HUB_DISABLE_XET=1 -e MPLBACKEND=Agg -e TOKENIZERS_PARALLELISM=false day22-lab:local
docker exec day22-lab uv venv /opt/day22-venv
docker exec day22-lab uv pip install --python /opt/day22-venv/bin/python --index-strategy unsafe-best-match -r requirements-local-gpu.txt
docker exec day22-lab /opt/day22-venv/bin/python -m ipykernel install --user --name python3 --display-name Day22
```

Tạo `.env` **local, bị ignore**, không có API key, với các giá trị:

```dotenv
COMPUTE_TIER=T4
BASE_MODEL=Qwen/Qwen3-0.6B
MAX_LEN=768
SFT_SLICE=1000
PREF_TRAIN=800
PREF_EVAL=100
DPO_BETA=0.1
DPO_LR=5e-6
DPO_EPOCHS=1
GEN_MAX_NEW_TOKENS=384
GEN_BATCH_SIZE=1
JUDGE_PROMPTS=50
JUDGE_RM_4BIT=1
JUDGE_RM_MAX_LENGTH=2048
DPO_SAVE_ON_CPU=1
```

```powershell
docker exec day22-lab /opt/day22-venv/bin/python scripts/verify.py --smoke
docker exec day22-lab /opt/day22-venv/bin/python scripts/run_pipeline.py
docker exec day22-lab /opt/day22-venv/bin/python scripts/render_comparison_table.py
docker exec day22-lab /opt/day22-venv/bin/python scripts/capture_judge_budget.py
docker exec day22-lab /opt/day22-venv/bin/python scripts/capture_environment.py
docker exec day22-lab /opt/day22-venv/bin/python -m pytest -q scripts/
docker exec day22-lab /opt/day22-venv/bin/python scripts/build_colab.py --check
docker exec -e PATH=/opt/day22-venv/bin:/usr/local/bin:/usr/bin:/bin day22-lab make verify
docker exec day22-lab /opt/day22-venv/bin/python scripts/audit_submission.py
```

Runner thực thi bằng Jupytext + nbconvert, giữ notebook có output và `submission/evidence/pipeline.json`. Mỗi stage có kernel riêng để giải phóng VRAM. Stage đã hoàn thành được bỏ qua; stage lỗi được lưu log và có thể chạy lại bằng `--stages nb3 nb4`. Không đổi cấu hình hoặc source đã chạy thành công rồi dùng lại kết quả cũ như cùng một thí nghiệm.

Ảnh bảng NB4 được dàn lại từ đúng `side_by_side.jsonl` bằng `render_comparison_table.py` để chữ không tràn ô; không sửa câu trả lời hay verdict. Notebook giữ output gốc của lượt thực thi.

`make pipeline` nguyên bản cũng chạy được trong container khi đặt PATH như lệnh verify; nó **chạy lại từ đầu**, không dùng cơ chế skip của runner. DPO reference được lưu ở `/lab/models/sft-merged`, nên các lệnh GPU/verify của bài local dùng cùng mount `/lab`. Clone mới không có trọng số: cần thực hiện pipeline trước `make verify`, không tạo config giả để vượt gate.

## Khác biệt với cấu hình mặc định

- CUDA 11.8 / torch 2.7.0 được chọn để khớp driver NVIDIA 546.33; Unsloth 2026.10.2, TRL 1.13.0 và transformers 5.17.0. TorchAO pin 0.13.0 vì bản mới dùng dtype mà torch này chưa có.
- Unsloth có thể tự ánh xạ base model sang checkpoint NF4 của cùng Qwen3-0.6B; cần đọc snapshot thực trong cache/evidence, không nhầm với train trên 4B.
- Hai reward model Skywork mặc định vẫn được giữ, nạp lần lượt ở NF4. Lượng tử hóa có thể đổi điểm/ordering; phải kiểm sanity trên chính phiên bản này. Context judge giới hạn 2.048 token, có thể cắt input dài; ghi nhận giới hạn trong phản tư.
- Ở run đã lưu, Qwen chỉ đúng 7/12 sanity và bị loại theo quy tắc gốc; Llama đúng 12/12, nên kết quả chính không phải đồng thuận hai judge. Đo tokenizer trên raw output cho thấy không input nào bị cắt bởi cap 2.048. Không đổi ngưỡng sanity để giữ Qwen trong panel.
- `GEN_BATCH_SIZE=1` chỉ giảm peak VRAM, cùng greedy decode và token ceiling cho SFT/DPO; không lọc câu hỏi để làm đẹp win rate.
- DPO lần đầu bị CUDA driver OOM ở backward dù checkpointing đã bật. `DPO_SAVE_ON_CPU=1` dùng saved-tensor offload của PyTorch, giữ objective/mask, đủ 800/100 cặp và `MAX_LEN=768`; đổi RAM/thời gian lấy VRAM. Lần lỗi vẫn có log/manifest, không dùng làm kết quả đã train thành công.
- Dependencies phục vụ bonus GGUF/vLLM không cần cho pipeline core; xem requirements gốc nếu chọn làm bonus.

## Nộp bài

Theo README/rubric: repo GitHub **public**, notebook NB0–NB4 giữ output, bốn PNG, kết quả eval/metrics, preference split và phản tư. Không có mẫu tên repo bắt buộc trong README. Không commit `.env`, cache, models hay adapter weights. Push và nộp LMS là hai thao tác riêng, chưa tự thực hiện nếu chưa được yêu cầu.
