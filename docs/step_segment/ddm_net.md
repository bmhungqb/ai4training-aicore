# DDM-Net: Step Segmentation Method & Experiment Tracker

> **Mục tiêu**: File duy nhất theo dõi toàn diện thông tin phương pháp (Method Info), trạng thái & các vấn đề kỹ thuật (Status & Issues), và kết quả thực nghiệm (Experiments & Results) của mô hình **DDM-Net** cho bài toán phân đoạn bước may công nghiệp (Step Segmentation).

---

## 1. Method Info (Thông tin Phương pháp)

### 1.1. Tổng quan & Kiến trúc (Overview & Architecture)
- **Phương pháp**: [DDM-Net](https://github.com/MCG-NJU/DDM) (*Dense-to-Discrete Motion Network for Generic Event Boundary Detection*), trích xuất và tối ưu từ blueprint [NVIDIA SOP Monitoring Blueprints](https://github.com/NVIDIA/sop-monitoring-blueprints/tree/main/microservices/sop-training-bp/microservices/ddm-training-ms/ddm).
- **Cơ chế cốt lõi**:
  - Tiếp cận phát hiện ranh giới sự kiện tổng quát (GEBD) theo mô hình phân loại nhị phân tất định (per-frame binary classifier).
  - Kết hợp hai nhánh: nhánh trích xuất đặc trưng hình ảnh RGB và nhánh biến thiên chuyển động dày đặc (DDM - Dense Difference Module).
  - Tích hợp **Co-Transformer Decoder** 6 lớp để mô hình hóa mối tương quan không-thời gian giữa RGB và DDM trước khi đưa vào các đầu dự đoán (classification heads).
- **Mã nguồn trong repository**: `src/step_segment/DDM-Net/`
  - Engine huấn luyện: PyTorch Lightning 2.x (`train_sop_lightning.py`, `pl_ddm_datamodule.py`).
  - Dataloader: `datasets/ddm_dataset.py` (sparse sampling qua PyAV cho training) và `datasets/ddm_val_dataset.py` (dense streaming qua decord cho validation).
  - Backbone hỗ trợ: ResNet-50 (`resnetGEBD.py`) và DINOv2 (`modeling/nvdinov2/` / `dinov2_vitb14`).
  - Cấu hình: `config/ddm_train_config.yaml` và `config/config.py`.

### 1.2. Định nghĩa Ranh giới Bước May (Step Boundary Definition)
Mỗi video có file nhãn vi-thao tác `action_segments_annotated.json` (tên thao tác tiếng Việt). DDM-Net được huấn luyện để phát hiện **ranh giới thay đổi thao tác (Generic Event Boundaries)**:
- Gộp các vi-phân đoạn liên tiếp có cùng `operation_name` thành một **step segment** duy nhất.
- Các phân đoạn rỗng (`""`) hoặc `"UNKNOWN"` được giữ nguyên thành các bước chuyển tiếp độc lập (idle/transition).
- Ranh giới giữa các step segment sau khi gộp chính là Ground Truth boundary để DDM-Net học.

### 1.3. Cấu trúc Thư mục & Dữ liệu
```text
ai4training-aicore-poc/
├── data/
│   ├── ddm_dataset/                            <-- Dataset chuẩn hóa cho DDM-Net
│   │   ├── train_annotation.json               <-- Format: {"video_id": [{"event":..., "start_timestamp":..., "end_timestamp":...}]}
│   │   ├── val_annotation.json
│   │   └── videos/                             <-- Symlinks tới file MP4 gốc
│   ├── cd1/chuyen1/
│   │   ├── cam-03_....mp4
│   │   ├── cam-03_....mask.png                 <-- ROI mask công nhân (nếu có)
│   │   ├── action_segments_annotated.json
│   │   └── step_segments.json                  <-- Sinh ra bởi tools/process_step_segments.py
│   └── ...
├── tools/
│   ├── process_step_segments.py                <-- Gộp action segments -> step_segments.json
│   └── prepare_ddm_dataset.py                  <-- Tạo split train/val & format DDM-Net
├── src/
│   └── step_segment/
│       └── DDM-Net/                            <-- Source code DDM-Net
│           ├── config/ddm_train_config.yaml
│           ├── datasets/
│           ├── modeling/
│           ├── train_sop_lightning.py
│           └── tools/run_train.sh
└── docs/
    └── step_segment/
        └── ddm_net.md                          <-- (File này) Tracker duy nhất
```

### 1.4. Thiết lập Môi trường & Hướng dẫn Huấn luyện
```bash
# 1. Tạo môi trường
conda create -n ddm python=3.10 -y
conda activate ddm
cd ai4training-aicore-poc
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r src/step_segment/DDM-Net/requirements.txt

# (Tùy chọn) Chỉ cần khi dùng nvdinov2 backbone (CUDA extension từ TAO):
cd src/step_segment/DDM-Net && pip install -e . && cd -

# 2. Tiền xử lý dữ liệu
python tools/process_step_segments.py
python tools/prepare_ddm_dataset.py --split-mode random --val-ratio 0.2 --seed 42
# Hoặc split theo folder:
# python tools/prepare_ddm_dataset.py --split-mode by_folder --val-folders cd18,cd19,cd20

# 3. Chạy huấn luyện (Single GPU)
bash src/step_segment/DDM-Net/tools/run_train.sh 1

# Multi-GPU (DDP 4 GPUs)
bash src/step_segment/DDM-Net/tools/run_train.sh 4

# Chạy nền với tmux/nohup
tmux new -s ddm_train
conda activate ddm
bash src/step_segment/DDM-Net/tools/run_train.sh 1 2>&1 | tee train.log
```

### 1.5. Các Siêu tham số Then chốt (Key Config Knobs)
- `dataset_config.frames_per_side`: Số frames lấy mẫu mỗi bên quanh frame trung tâm (mặc định ban đầu: 5, đề xuất nâng lên 8).
- `dataset_config.downsample`: Bước nhảy frame khi lấy mẫu (mặc định: 1, đề xuất 2).
- `dataset_config.min_change_dur`: Khoảng cách tối thiểu (giây) giữa 2 ranh giới.
- `training_config.model_ema` & `model_ema_start_epoch`: Bật Exponential Moving Average cho trọng số model để ổn định dự đoán.
- `training_config.checkpoint_top_k`: Lưu top K checkpoints có `val/f1_score` cao nhất.

---

## 2. Status & Issues (Hiện trạng & Các Vấn đề Kỹ thuật)

| Vấn đề | Mức độ | Hiện trạng | Tóm tắt giải pháp |
|---|---|---|---|
| **Cộng dồn 18 CrossEntropy loss phẳng** | 🔴 CRITICAL | Đang xử lý | Loss bị kẹt tại ~12.47 do 18 heads đoán 50/50 ($18 \times 0.693$); áp dụng trọng số `main_loss + 0.3 * aux_loss` |
| **Tràn RAM hệ thống lúc Validation (OOM)** | 🔴 CRITICAL | Đã có giải pháp | Memory leak do C++ buffer của `decord.VideoReader` + `num_workers > 0`; giảm workers, flush bộ đệm |
| **Cửa sổ thời gian quá hẹp (0.37s vs 1.84s)** | 🟡 HIGH | Đã có kế hoạch | Cửa sổ 11 frames chỉ bao phủ 15-20% thao tác; tăng `downsample: 2`, `frames_per_side: 8` để đạt span 1.10s |
| **Vùng thao tác nhỏ & Nhiễu hậu cảnh xưởng** | 🟡 MEDIUM | Đã có kế hoạch | Khung hình 1920x1080 chứa 70% nền thừa; tích hợp Bounding Box crop từ `*.mask.png` trước khi resize 224x224 |
| **Backbone DINOv2 đóng băng hoàn toàn** | 🟡 MEDIUM | Đang đánh giá | `freeze_backbone: true` khiến model không thích nghi miền may mặc; unfreeze 2 transformer blocks cuối |
| **LR Scheduler decay quá sớm** | 🟢 MINOR | Đang xử lý | Cosine decay tụt về `1e-10` ngay sau epoch 1; tăng `warmup_epochs: 3`, nâng `min_lr: 1e-6` |

---

### Chi tiết Issue 1: Training Bị Kẹt Flat Loss (~12.40) & F1-Score Không Cải Thiện
- **Hiện tượng**: Khi huấn luyện với backbone DINOv2 (`dinov2_vitb14`), train loss đi ngang quanh `12.20 - 12.50` suốt 6 epoch, val F1 dao động lẹt đẹt `0.39 - 0.45` rồi giảm dần.
- **Nguyên nhân cốt lõi**:
  1. Trong `train_sop_lightning.py`, hàm loss cộng dồn 18 heads (6 outputs kết hợp + 6 aux RGB + 6 aux DDM) không trọng số:
     $$\text{Loss} = \sum_{i=1}^6 L_{\text{comb}}^{(i)} + \sum_{i=1}^6 L_{\text{rgb}}^{(i)} + \sum_{i=1}^6 L_{\text{ddm}}^{(i)}$$
     Khi mô hình chưa học được gì, dự đoán nhị phân ngẫu nhiên cho loss $-\ln(0.5) \approx 0.693$. Với 18 heads: $18 \times 0.693 \approx 12.47$ — đúng bằng giá trị phẳng trên log.
  2. Gradient từ các layer nông đè bẹp tín hiệu gradient của main head ở layer cuối (`outputs[-1]`).
  3. `freeze_backbone: true` khóa hoàn toàn DINOv2 (vốn chỉ học từ ảnh tự nhiên, chưa biết domain may mặc).
  4. Cosine scheduler giảm LR quá nhanh về `min_lr: 1e-10`.
- **Giải pháp áp dụng**:
  - Gán trọng số ưu tiên main head: `loss = main_loss + 0.3 * (aux_loss / n_aux)`.
  - Nâng base LR lên `2e-4 - 3e-4`, tăng `warmup_epochs: 3`, đặt `min_lr: 1e-6`.
  - Mở unfreeze 2 blocks cuối của DINOv2 với LR nhỏ (`1e-5`).

---

### Chi tiết Issue 2: Tràn RAM Hệ Thống (OOM) Khi Bước Vào Validation
- **Hiện tượng**: Training chạy ổn định suốt epoch (RAM thấp), nhưng ngay khi vào `Validation DataLoader 0`, System RAM tăng vọt từ 2GB $\rightarrow$ 8GB $\rightarrow$ chạm đỉnh 16GB, dẫn tới Linux OOM Killer gửi `SIGKILL/SIGTERM`.
- **Nguyên nhân cốt lõi**:
  1. `DDMValStreamingDataset` sử dụng `decord.VideoReader(path, ctx=cpu(0), num_threads=0)`. Thư viện `decord` viết bằng C++ native tự duy trì buffer frame uncompressed trong RAM mà không giải phóng ngay giữa các sliding step.
  2. Validation quét dày đặc `temporal_stride=1` qua toàn bộ hàng nghìn frame của video validation dài.
  3. DataLoader `num_workers=2` hoặc `4` khởi tạo nhiều process độc lập, mỗi process mang một `VideoReader` riêng làm nhân bản buffer C++ lên nhiều lần.
  4. Mảng `self.validation_step_outputs` tích tụ scores và metadata của toàn bộ 700+ steps suốt epoch mà không giải phóng từng phần.
- **Giải pháp áp dụng**:
  - Đặt `val_config.workers: 0` hoặc `1` để triệt tiêu nhân bản C++ buffer đa tiến trình.
  - Tái cấu trúc cơ chế đọc frame của `DDMValStreamingDataset` hoặc giải phóng đối tượng reader định kỳ.
  - Xóa bộ đệm kết quả trung gian sau khi tính metric.

---

### Chi tiết Issue 3: Cửa Sổ Lấy Mẫu (0.37s) Quá Hẹp So Với Thời Lượng Thao Tác Thật
- **Thực trạng dữ liệu**: Thống kê 1,570 thao tác trong dataset:
  - Thời lượng trung vị (Median): **1.84s** (~55 frames ở 30 FPS).
  - Thời lượng trung bình (Mean): **2.90s** (~87 frames ở 30 FPS).
  - Tỷ lệ thao tác < 2s: **54.0%** (Train) và **64.5%** (Val).
- **Hạn chế**: Cấu hình cũ `frames_per_side: 5`, `downsample: 1` cho span:
  $$\text{Span} = 2 \times 5 + 1 = 11\text{ frames} \approx 0.37\text{s}$$
  Chỉ nhìn thấy $\approx 15\% - 20\%$ độ dài của một thao tác trung bình, hoàn toàn thiếu ngữ cảnh vĩ mô của hành động trước và sau.
- **Giải pháp**: Tăng `downsample: 2`, `frames_per_side: 8`:
  $$\text{Span} = (2 \times 8) \times 2 + 1 = 33\text{ frames} \approx 1.10\text{s}$$
  Bao quát trọn nửa sau thao tác trước và nửa đầu thao tác sau, giúp head transformer nhận diện biên chuyển tiếp rõ ràng mà chỉ tăng số frame nạp vào tensor từ 11 lên 17.

---

### Chi tiết Issue 4: Nhiễu Hậu Cảnh Xưởng & Tỷ Lệ Vùng Thao Tác Nhỏ
- **Thực trạng**: Khung hình camera xưởng may rộng $1920\times 1080$, vùng 2 bàn tay công nhân và mũi kim chỉ chiếm 30% - 40% diện tích. 60% - 70% còn lại là người qua lại, xe đẩy, ánh sáng đổi. Khi nén về $224\times 224$, chi tiết kim may bị nhòe và hậu cảnh gây nhiễu động học.
- **Giải pháp**: Tự động trích xuất Bounding Box từ `<video_stem>.mask.png` (đã có sẵn trong từng thư mục `data/cd*/chuyen*/`), thêm padding 10%, crop vùng làm việc rồi mới resize về $224\times 224$:
  - Tăng độ phân giải hiệu dụng gấp 2.5 - 3 lần.
  - Loại bỏ hoàn toàn nhiễu người di chuyển ở hậu cảnh.

---

## 3. Experiments & Results (Kết Quả Thực Nghiệm)

### 3.1. Quá trình Huấn luyện Thực nghiệm (DINOv2 Baseline - Epoch 0 đến 6)
- **Cấu hình**: `dinov2_vitb14`, `freeze_backbone: true`, `frames_per_side: 5`, `downsample: 1`, batch size 4, GPU RTX 5060 (16GB VRAM).
- **Nhật ký chỉ số qua các Epoch**:

| Epoch | Train Loss | Val F1-Score | Val Accuracy | Trạng thái / Ghi chú |
|:---:|:---:|:---:|:---:|:---|
| **Epoch 0** | 12.48 | **0.45299** | 24.2% | Bắt đầu huấn luyện, loss phẳng |
| **Epoch 1** | 12.35 | 0.39551 | 31.9% | F1 tụt |
| **Epoch 2** | 12.28 | **0.45455** | 19.7% | Đỉnh cao nhất (Top 1) |
| **Epoch 3** | 12.21 | 0.43421 | 47.8% | Accuracy tăng nhưng F1 giảm |
| **Epoch 4** | 12.24 | 0.40112 | 19.6% | Rớt khỏi top 3 |
| **Epoch 5** | 12.20 | 0.39870 | 21.3% | Rớt khỏi top 3 |
| **Epoch 6** | 12.22 | 0.41050 | 22.0% | Loss hoàn toàn đi ngang |

- **Nhận định**: Model bị **Underfitting nghiêm trọng** (không phải do thiếu dữ liệu; 25 video train chứa tới 1,190 boundaries và 904 steps/epoch ~ 3,616 cửa sổ huấn luyện). Nguyên nhân trực tiếp từ 18 head loss cộng phẳng và learning rate decay quá sâu.

### 3.2. Đo Đạc Tài Nguyên Hệ Thống
- **Tốc độ huấn luyện**: ~59-60 phút / epoch (904 steps, ~0.25 it/s).
- **VRAM GPU tiêu thụ**: ~8.5GB / 16GB (an toàn trong giới hạn RTX 5060).
- **System RAM lúc Train**: ~2.5GB - 3.5GB (ổn định).
- **System RAM lúc Validation (chưa tối ưu)**: Tăng vọt từ 2GB $\rightarrow$ 16GB (OOM leak).
- **System RAM lúc Validation (sau tối ưu worker=0)**: Duy trì ổn định ~4GB.

### 3.3. Lộ trình Thực nghiệm Tiếp theo (Experiment Roadmap)
1. **Thực nghiệm 1 (Loss & LR Fix)**: Huấn luyện lại với loss weighting (`main + 0.3*aux`), warmup 3 epochs, `min_lr: 1e-6`, base LR `2.5e-4`.
2. **Thực nghiệm 2 (Temporal Receptive Field)**: Đổi cấu hình sang `downsample: 2`, `frames_per_side: 8` (cửa sổ 1.1s).
3. **Thực nghiệm 3 (ROI Crop Integration)**: Bật Bounding Box crop từ `*.mask.png` và so sánh F1 với toàn khung hình.
4. **Thực nghiệm 4 (Partial Backbone Fine-tuning)**: Mở unfreeze 2 blocks cuối của DINOv2.
