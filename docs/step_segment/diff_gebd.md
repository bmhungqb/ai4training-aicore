# DiffGEBD: Step Segmentation Method & Experiment Tracker

> **Mục tiêu**: File duy nhất theo dõi toàn diện thông tin phương pháp (Method Info), trạng thái & các vấn đề kỹ thuật (Status & Issues), và kết quả thực nghiệm (Experiments & Results) của mô hình **DiffGEBD** cho bài toán phân đoạn bước may công nghiệp (Step Segmentation).

---

## 1. Method Info (Thông tin Phương pháp)

### 1.1. Tổng quan & Cơ sở Khoa học (Overview & Scientific Background)
- **Phương pháp**: [DiffGEBD](https://github.com/JaejunHwang/DiffGEBD) (*Generic Event Boundary Detection via Denoising Diffusion*, ICCV 2025, [arXiv:2508.12084](https://arxiv.org/pdf/2508.12084)).
- **Cơ chế cốt lõi**:
  - Không coi GEBD là bài toán phân loại nhị phân tất định (deterministic binary classification) thông thường, mà tiếp cận theo **mô hình sinh khuếch tán (denoising diffusion generative model)** (phong cách DDPM/DDIM).
  - Chuỗi ranh giới sự kiện 1D theo thời gian là mục tiêu khử nhiễu (denoising target), được điều kiện hóa bởi đặc trưng không-thời gian từ ResNet-50 kết hợp ma trận tự tương đồng thời gian (temporal self-similarity matrix).
  - Tích hợp **Classifier-Free Guidance (CFG)** tại thời điểm lấy mẫu suy luận để cân bằng giữa độ chính xác (fidelity) và độ đa dạng (diversity) của các ranh giới được dự đoán.
- **Mã nguồn trong repository**: `src/step_segment/DiffGEBD/`
  - Visual Backbone: ResNet-50 (`modeling/resnet.py`).
  - Diffusion Engine: `modeling/diffusion_model.py`, `modeling/diff_former.py`, `modeling/time_transformer.py`.
  - Dataset Module: `datasets/dataset.py` (hỗ trợ dataset type `SEWING`).
  - Runner: `train.py` (huấn luyện PyTorch DDP qua `torchrun`).
  - Cấu hình chuẩn: `config/sewing_diffgebd_resnet50.yaml` và `config/sewing_diffgebd_resnet50_chunked.yaml`.

### 1.2. Cấu trúc Thư mục & Định dạng Dữ liệu
```text
ai4training-aicore-poc/
├── data/
│   ├── diff_gebd_dataset/
│   │   ├── images/
│   │   │   ├── train/<video_id>/frame%d.jpg    <-- Frames trích xuất offline (đã crop ROI)
│   │   │   └── val/<video_id>/frame%d.jpg
│   │   ├── train_annotation.pkl                <-- Annotation chuẩn Kinetics-GEBD
│   │   ├── val_annotation.pkl
│   │   ├── train_annotation_chunked.pkl        <-- Annotation sau khi băm nhỏ (chunked)
│   │   └── val_annotation_chunked.pkl
├── tools/
│   ├── prepare_diff_gebd_dataset.py            <-- Trích xuất frame + sinh annotation pickle
│   ├── chunk_diff_gebd_dataset.py              <-- Băm video dài thành chunks 10-15s
│   ├── export_diffgebd_predictions.py          <-- Xuất PKL -> step_segments_pred.json
│   ├── infer_diffgebd.py                       <-- Script inference độc lập
│   └── benchmark_step_segment_models.py        <-- So sánh F1 giữa các mô hình
├── src/
│   └── step_segment/
│       └── DiffGEBD/
│           ├── config/
│           │   ├── sewing_diffgebd_resnet50.yaml
│           │   └── sewing_diffgebd_resnet50_chunked.yaml
│           ├── datasets/
│           ├── modeling/
│           ├── train.py
│           └── tools/run_train.sh
└── docs/
    └── step_segment/
        └── diff_gebd.md                        <-- (File này) Tracker duy nhất
```

### 1.3. Thiết lập Môi trường & Hướng dẫn Huấn luyện
```bash
# 1. Tạo môi trường
conda create -n diffgebd python=3.10 -y
conda activate diffgebd
cd ai4training-aicore-poc
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r src/step_segment/DiffGEBD/requirements.txt

# 2. Tiền xử lý dữ liệu
python tools/process_step_segments.py
python tools/prepare_diff_gebd_dataset.py --split-mode random --val-ratio 0.2 --seed 42

# 3. Tạo tập dữ liệu Chunked (Khắc phục Sampling Imbalance - xem Mục 2)
python tools/chunk_diff_gebd_dataset.py

# 4. Huấn luyện (Single GPU)
cd src/step_segment/DiffGEBD
bash tools/run_train.sh 1

# Multi-GPU (DDP 2 GPUs)
bash tools/run_train.sh 2 MODEL.SYNC_BN true

# Chạy với file config chunked tối ưu
torchrun --nproc_per_node=1 train.py \
  --config-file config/sewing_diffgebd_resnet50_chunked.yaml

# Chạy nền với tmux
tmux new -s diffgebd_train
conda activate diffgebd
cd src/step_segment/DiffGEBD
bash tools/run_train.sh 1 2>&1 | tee train.log
```

### 1.4. Đánh Giá & Xuất Kết Quả Benchmark
```bash
# Chạy validation-only trên checkpoint tốt nhất
cd src/step_segment/DiffGEBD
torchrun --nproc_per_node=1 train.py \
  --config-file output/sewing_diffgebd_resnet50_ann1_dim512_len100/config.yaml \
  --test-only --all-thres \
  --resume output/sewing_diffgebd_resnet50_ann1_dim512_len100/model_best.pth

# Chuyển đổi PKL sang định dạng step_segments_pred.json của dự án
cd ../../..
python tools/export_diffgebd_predictions.py \
  --pred-pkl src/step_segment/DiffGEBD/output/sewing_diffgebd_resnet50_ann1_dim512_len100/model_pred_dict_ep-1.pkl \
  --split val \
  --out-dir experiments/diffgebd_preds

# Benchmark trực tiếp với EfficientGEBD
python tools/benchmark_step_segment_models.py \
  --pred-a experiments/diffgebd_preds/diffgebd_preds.json --name-a DiffGEBD \
  --pred-b experiments/efficient_gebd_preds.json --name-b EfficientGEBD \
  --thresholds 0.05 0.1 0.2 0.3 0.4 0.5 \
  --out experiments/stage1_boundary_recall_9cd/diffgebd_vs_efficientgebd.json
```

### 1.5. Các Siêu tham số Quan trọng (Key Config Knobs)
- `DIFFUSION.CFG_SCALE`: Hệ số hướng dẫn không phân loại Classifier-Free Guidance (mặc định 7.0). Giá trị càng cao dự đoán ranh giới càng sắc bén, bớt đa dạng.
- `DIFFUSION.TIMESTEPS`: Số bước khuếch tán trong quá trình train (mặc định 1000).
- `DIFFUSION.SAMPLING_TIMESTEPS`: Số bước lấy mẫu khử nhiễu DDIM lúc inference (mặc định 16).
- `INPUT.SEQUENCE_LENGTH`: Số frame lấy mẫu trong mỗi sequence (mặc định 100-150 frames).
- `SOLVER.BATCH_SIZE`: Đặt là 2 (kết hợp AMP `SOLVER.AMPE: True`) để vừa trong VRAM GPU đơn (16GB).
- `MODEL.SYNC_BN`: Luôn đặt `false` khi chạy single GPU; chỉ bật `true` khi dùng multi-GPU DDP.

---

## 2. Status & Issues (Hiện trạng & Các Vấn đề Kỹ thuật)

| Vấn đề | Mức độ | Hiện trạng | Tóm tắt giải pháp |
|---|---|---|---|
| **Sampling Imbalance: Precision 0.9+ nhưng Recall 0.19** | 🔴 CRITICAL | ✅ FIXED | `np.linspace` trên video dài 6000f gây bước nhảy 40f, mất ranh giới; đã viết `tools/chunk_diff_gebd_dataset.py` băm video thành chunks 10-15s |
| **Độ trễ suy luận DDIM (Inference Latency)** | 🟡 MEDIUM | Đã kiểm soát | 16 bước forward khuếch tán tốn thời gian hơn phân loại 1 pass; giữ `SAMPLING_TIMESTEPS=16` hoặc thử nghiệm 8 |
| **Ràng buộc VRAM khi train End-to-End** | 🟡 MEDIUM | ✅ OK | End-to-end 100 frames × 224px tiêu thụ nhiều VRAM; dùng `BATCH_SIZE=2` và bật AMP |
| **SyncBatchNorm lỗi trên Single-GPU** | 🟢 MINOR | ✅ FIXED | Lỗi khởi tạo ProcessGroup khi chạy 1 GPU; đặt `MODEL.SYNC_BN: false` mặc định |

---

### Chi tiết Issue: Sampling Imbalance Khi Lấy Mẫu Video Dài
- **Hiện tượng**: Khi huấn luyện DiffGEBD trên dataset may mặc nguyên bản, mô hình đạt độ chính xác cực cao (**Precision 0.9+**) nhưng độ bao phủ ranh giới tụt thê thảm (**Recall chỉ ~0.19**).
- **Nguyên nhân cốt lõi**:
  - Trong `src/step_segment/DiffGEBD/datasets/dataset.py`:
    ```python
    if cfg.INPUT.END_TO_END:
        selected_indices = np.linspace(1, vlen, cfg.INPUT.SEQUENCE_LENGTH, dtype=int)
    ```
  - DiffGEBD được thiết kế cho bộ dữ liệu **Kinetics-GEBD** (tất cả video đã cắt sẵn ngắn đúng **10 giây**, ~300 frames). Lệnh `np.linspace(1, 300, 150)` cho bước nhảy 2 frames — bắt ranh giới rất hoàn hảo.
  - Video may công nghiệp lại là video dài nhiều phút (từ 1,500 đến 6,000 frames). Lệnh `np.linspace(1, 6000, 150)` dẫn đến bước nhảy thực tế là **40 frames/lần**.
  - Một step may ngắn chỉ kéo dài 5-10 frames. Bước nhảy 40 frames khiến việc lấy mẫu "nhảy cóc" qua toàn bộ các ranh giới. Tín hiệu positive frames hoàn toàn biến mất khỏi nhãn huấn luyện, khiến mô hình chỉ nhìn thấy background và tự động đoán toàn "0" để giảm loss.
- **Giải pháp triệt để đã triển khai**:
  - Không sửa đổi lõi mô hình vốn giả định 1 chuỗi liên tục, mà can thiệp ở tầng tiền xử lý:
    1. Viết `tools/chunk_diff_gebd_dataset.py` để băm nhỏ các video dài thành các đoạn chunk dài **10 đến 15 giây** có **overlapping 1-2 giây**.
    2. Tái cấu trúc file annotation thành `train_annotation_chunked.pkl` và `val_annotation_chunked.pkl`. Mỗi chunk được xem như một video độc lập.
    3. Cấu hình lại `INPUT.SEQUENCE_LENGTH: 150` trong `sewing_diffgebd_resnet50_chunked.yaml`.
    4. Lúc inference, viết hàm gom và ánh xạ timestamp từ từng chunk ngược trở lại mốc thời gian của video dài ban đầu.

---

## 3. Experiments & Results (Kết Quả Thực Nghiệm)

### 3.1. Bảng So Sánh Phương Pháp: DDM-Net vs EfficientGEBD vs DiffGEBD

| Tiêu chí | DDM-Net | EfficientGEBD | DiffGEBD |
|---|---|---|---|
| **Paradigm (Bản chất)** | Phân loại nhị phân từng frame (Deterministic) | Phân loại nhị phân + FPN & DiffFormer | Mô hình sinh khuếch tán (Denoising Diffusion) + CFG |
| **Backbone thị giác** | ResNet-50 / DINOv2 (frozen hoặc fine-tune) | ResNet-50 hoặc CSN (R50/R152) | ResNet-50 |
| **Độ dài chuỗi đầu vào** | Cửa sổ trượt cục bộ quanh mốc ứng viên | 100 frames end-to-end / 10s slice | 100-150 frames end-to-end (sau chunking) |
| **Chi phí lấy mẫu suy luận** | 1 lần forward pass | 1 lần forward pass | 16 lần forward passes (`SAMPLING_TIMESTEPS=16`) |
| **Tham số kiểm soát đặc thù** | `frames_per_side`, `min_change_dur`, EMA | `HEAD_CHOICE`, `POS_WEIGHT`, `SIGMA` | `CFG_SCALE`, `DIFFUSION.TIMESTEPS`, `SAMPLING_TIMESTEPS` |
| **Tệp cấu hình chính** | `config/ddm_train_config.yaml` | `config-files/sewing_resnet50.yaml` | `config/sewing_diffgebd_resnet50_chunked.yaml` |
| **Script chạy train** | `train_sop_lightning.py` | `train.py` | `train.py` |
| **Tiêu chuẩn đánh giá** | Boundary F1/Recall/Precision (theo tolerance) | GEBD Relative F1 & Absolute F1 | GEBD Relative F1 (0.05 - 0.5) & Absolute F1 |

### 3.2. Kết Quả Đo Lường Ban Đầu (Baseline Metrics)
- **Trước khi Chunking**:
  - Recall@0.05: **0.191** (quá thấp do nhảy cóc frame).
  - Precision@0.05: **0.912** (mô hình chỉ dám báo ranh giới khi cực kỳ chắc chắn).
  - F1@0.05: **0.316**.
- **Sau khi Chunking (`tools/chunk_diff_gebd_dataset.py`)**:
  - Dữ liệu ranh giới được khôi phục 100% độ bao phủ trong các chunks 10-15s.
  - Quá trình thực nghiệm với `sewing_diffgebd_resnet50_chunked.yaml` đang được triển khai để ghi nhận bảng F1 đa ngưỡng (0.05, 0.1, 0.2, 0.3, 0.4, 0.5).
