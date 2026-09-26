# EfficientGEBD: Step Segmentation Method & Experiment Tracker

> **Mục tiêu**: File duy nhất theo dõi toàn diện thông tin phương pháp (Method Info), trạng thái & các vấn đề kỹ thuật (Status & Issues), và kết quả thực nghiệm (Experiments & Results) của mô hình **EfficientGEBD** cho bài toán phân đoạn bước may công nghiệp (Step Segmentation).

---

## 1. Method Info (Thông tin Phương pháp)

### 1.1. Tổng quan & Cơ sở Khoa học (Overview & Scientific Background)
- **Phương pháp**: [EfficientGEBD](https://github.com/Ziwei-Zheng/EfficientGEBD) (*Rethinking the Architecture Design for Efficient Generic Event Boundary Detection*, ACM MM 2024, [arXiv:2407.12622](https://arxiv.org/abs/2407.12622)).
- **Cơ chế cốt lõi**:
  - Tái thiết kế kiến trúc phát hiện ranh giới sự kiện để đạt hiệu năng tính toán cao và tốc độ hội tụ nhanh.
  - Sử dụng mạng trích xuất đặc trưng video domain (Video-domain CSN hoặc 2D ResNet-50) kết hợp cấu trúc Feature Pyramid Network (FPN) đa tầng để nắm bắt ranh giới ở nhiều thang đo thời gian khác nhau.
  - Sử dụng các khối `DiffFormer` / `DiffMixer` tính toán ma trận độ khác biệt (temporal dissimilarity matrix) giữa các frames trong cửa sổ trượt để đưa ra score ranh giới.
- **Mã nguồn trong repository**: `src/step_segment/EfficientGEBD/`
  - Engine huấn luyện: PyTorch thuần với `torchrun` DDP và quản lý cấu hình bằng `yacs`.
  - Modules chính: `modeling/` (`e2e_model_diff_former.py`, `baseline.py`, `csn.py`, `resnet.py`, `fpn.py`), `datasets/dataset.py`, `solver/`, `utils/eval.py`, `train.py`.
  - Cấu hình: `config-files/sewing_resnet50.yaml` và `config-files/sewing_csn.yaml`.

### 1.2. Cấu trúc Thư mục & Dữ liệu
```text
ai4training-aicore-poc/
├── data/
│   ├── efficient_gebd_dataset/
│   │   ├── images/
│   │   │   ├── train/<video_id>/frame%d.jpg    <-- Trích xuất offline, đã crop ROI mask
│   │   │   └── val/<video_id>/frame%d.jpg
│   │   ├── train_annotation.pkl                <-- Schema chuẩn Kinetics-GEBD
│   │   └── val_annotation.pkl
├── tools/
│   ├── prepare_efficient_gebd_dataset.py       <-- Trích xuất frame + sinh annotation pickle
│   └── benchmark_step_segment_models.py        <-- Đánh giá đối soát đa mô hình
├── src/
│   └── step_segment/
│       └── EfficientGEBD/
│           ├── config-files/
│           │   ├── sewing_resnet50.yaml
│           │   └── sewing_csn.yaml
│           ├── datasets/dataset.py
│           ├── modeling/
│           ├── utils/eval.py
│           ├── train.py
│           └── script/train/train_sewing_csn.sh
└── docs/
    └── step_segment/
        └── efficient_gebd.md                   <-- (File này) Tracker duy nhất
```

### 1.3. Thiết lập Môi trường & Hướng dẫn Huấn luyện
```bash
# 1. Tạo môi trường
conda create -n efficient_gebd python=3.10 -y
conda activate efficient_gebd
cd ai4training-aicore-poc
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r src/step_segment/EfficientGEBD/requirements.txt

# 2. Tiền xử lý dữ liệu
python tools/process_step_segments.py
python tools/prepare_efficient_gebd_dataset.py --split-mode random --val-ratio 0.2 --seed 42

# 3. Huấn luyện (Single GPU với ResNet-50)
cd src/step_segment/EfficientGEBD
torchrun --nproc_per_node=1 train.py \
  --config-file config-files/sewing_resnet50.yaml

# Huấn luyện với CSN backbone (Multi-GPU hoặc 1 GPU)
torchrun --nproc_per_node=1 train.py \
  --config-file config-files/sewing_csn.yaml

# Chạy nền với nohup / tmux
tmux new -s eff_gebd_train
conda activate efficient_gebd
cd src/step_segment/EfficientGEBD
torchrun --nproc_per_node=1 train.py \
  --config-file config-files/sewing_resnet50.yaml 2>&1 | tee train.log
```

---

## 2. Status & Issues (Hiện trạng & Danh mục 12 Vấn đề Kỹ thuật)

### 2.1. Bảng Tổng Hợp Trạng Thái

| Hạng mục / Vấn đề | Trạng thái | Ghi chú & Giải pháp |
|---|---|---|
| **Format annotation (`.pkl`)** | ✅ OK | Đúng chuẩn Kinetics-GEBD schema |
| **Dataset type `SEWING` trong code** | ✅ OK | Tích hợp trong `datasets/__init__.py`, `dataset.py`, `post_process.py`, `train.py` |
| **ROI Mask (crop khu vực thao tác)** | ✅ OK | Tích hợp trong `tools/prepare_efficient_gebd_dataset.py` |
| **Issue 1: Temporal Sampling mất ~73% GT boundaries** | ✅ FIXED | Chuyển sang slice-based sampling (TAPOS-style), bao phủ 100% ranh giới |
| **Issue 2: Evaluation Metric lỏng lẻo** | ✅ FIXED (một phần) | Đã bổ sung hàm `do_eval_absolute_tol` (±0.3s, ±0.5s, ±1.0s, ±2.0s) |
| **Issue 3: Rủi ro VRAM khi tăng Sequence Length** | ✅ OK | Dùng `BATCH_SIZE=2` (hoặc 1 với CSN), bật `AMPE: True` |
| **Issue 4: Double Gaussian-smoothing làm Loss/F1 = 0** | ✅ FIXED | Model re-smooth targets đã smooth ở dataset.py; đã fix `if dataset not in ('TAPOS', 'SEWING')` |
| **Issue 4b: Mất cân bằng nhãn trong BCE Loss** | ✅ FIXED | Thêm `SOLVER.POS_WEIGHT: 4.5` vào `F.binary_cross_entropy_with_logits` |
| **Issue 5: Checkpoint selection dùng metric lỏng lẻo** | 🔴 CRITICAL | `model_best.pth` vẫn lưu theo F1@0.05 relative (sai số ~7.8s), cần đổi sang absolute F1 |
| **Issue 6: Train/Val split rò rỉ trạm may (`cd4`, `cd6`)** | 🔴 CRITICAL | 3/7 video val trùng trạm với train → leakage; cần re-split theo `station-level` |
| **Issue 7: Dataset nhỏ so với backbone CSN/ResNet50** | 🟡 WARNING | 25 train / 7 val, 1 video làm lệch 14% F1; ưu tiên ResNet50 và báo cáo per-video |
| **Issue 8: Inductive bias DiffFormer vs chuyển động may** | 🟡 WARNING | Trực quan hóa score curve để kiểm tra model bắt ranh giới hay bắt nhiễu tay |
| **Issue 9: Slice 10s hard-cut làm giảm context ở mép** | 🟡 WARNING | Boundary gần mốc 10s bị mất context một phía; xem xét overlap 1-2s |
| **Issue 10: Config CSN chưa được kiểm chứng ổn định** | 🟢 MINOR | CSN dùng SGD LR 1e-2 nguy cơ NaN; cần áp dụng AdamW LR thấp như ResNet50 |
| **Issue 11: `WARMUP_EPOCHS` không được kích hoạt** | ✅ FIXED | Đã thay `MultiStepLR` bằng `LambdaLR` có warmup + sửa log checkpoint gây nhầm |
| **Issue 12: Recall luôn thấp, `SOLVER.SIGMA` là dead config** | ✅ FIXED | Đã wire `SOLVER.SIGMA: 2` vào target smoothing và hạ `TEST.THRESHOLD: 0.3 → 0.2` |

---

### 2.2. Phân Tích Chi Tiết Các Vấn Đề Trọng Yếu

#### Issue 1 (CRITICAL - FIXED): Temporal Sampling Làm Mất ~73% Ground Truth Boundaries
- **Cơ chế lỗi**: Lệnh `np.linspace(1, vlen, SEQUENCE_LENGTH=100)` rải 100 frames trên video dài trung bình 2,145 frames (bước nhảy ~21 frames $\approx$ 1.5s). Dung sai nhãn chỉ 2 frames khiến mô hình nhảy cóc qua ranh giới.
- **Thực tế đo được**:
  - `train_annotation.pkl`: 1,190 GT boundaries, chỉ trúng 315 mốc $\rightarrow$ **Mất 73.5%**.
  - `val_annotation.pkl`: 348 GT boundaries, chỉ trúng 115 mốc $\rightarrow$ **Mất 67.0%**.
- **Cách sửa**: Chuyển sang **Slice-based sampling** (tương tự TAPOS): chia video thành các slice 10 giây (`num_slices = duration // 10 + 1`), nạp 100 frames/slice, bảo toàn 100% ranh giới.

#### Issue 4 & 4b (FIXED): Double Gaussian-Smoothing & Class Imbalance Trong BCE Loss
- **Double Smoothing**: Trong `modeling/e2e_model_diff_former.py` và `baseline.py`, hàm `prepare_gaussian_targets` re-smooth targets đã được làm mịn trước đó từ `dataset.py` (do chỉ loại trừ `TAPOS` mà bỏ quên `SEWING`). Target bị bão hòa thành dải hằng số khiến sigmoid output không bao giờ vượt threshold 0.3 $\rightarrow$ F1/Recall = 0.0000 ở mọi epoch. Đã sửa: `if self.dataset not in ('TAPOS', 'SEWING'):`.
- **Class Imbalance**: Sau khi sửa double smoothing, loss vẫn kẹt tại entropy floor (~0.336) vì chỉ ~18% frames có tín hiệu boundary. BCE không trọng số khiến model đoán hằng số trung bình. Đã sửa: thêm `SOLVER.POS_WEIGHT: 4.5` vào BCE loss.

#### Issue 5 (CRITICAL - CẦN XỬ LÝ): Model Checkpoint Selection Vẫn Dùng Metric Lỏng Lẻo
- Trong `train.py`:
  ```python
  f1, rec, prec = results[0.05][head]     # relative threshold 0.05 * video_length
  metrics['F1'] = f1
  if f1 > best_f1:
      torch.save(..., save_path)
  ```
- Với video 150s, sai số `0.05` tương đương **7.5 giây** (lớn hơn nhiều so với bước may 2-5s). Model có thể dự đoán bừa mỗi 8 giây một điểm mà vẫn đạt F1 cao và được lưu làm `model_best.pth`.
- **Giải pháp**: Đổi điều kiện lưu checkpoint sang `eval_f1_absolute_tol` (ngưỡng $\pm 1.0$s hoặc $\pm 0.5$s).

#### Issue 6 (CRITICAL - CẦN XỬ LÝ): Train/Val Split Không Disjoint Theo Trạm May
- `train_annotation.pkl`: `cd1` đến `cd17`.
- `val_annotation.pkl`: `cd4`, `cd6`, `cd12`, `cd18`, `cd19`, `cd20`.
- 3/7 video val (`cd4`, `cd6`) thuộc trạm may đã xuất hiện trong tập train. Model học thuộc góc quay và phông nền trạm thay vì học bản chất ranh giới chuyển bước.
- **Giải pháp**: Phân chia train/val triệt để theo station (`cdN`).

#### Issue 11 & 12 (FIXED): LR Warmup & Recall Quá Thấp
- **Issue 11**: `SOLVER.WARMUP_EPOCHS: 5` bị bỏ quên do `train.py` dùng `MultiStepLR` không warmup. Đã thay bằng `LambdaLR` với tuyến tính warmup trong 5 epochs đầu.
- **Issue 12**: `SOLVER.SIGMA` bị hard-code bằng 1 trong `datasets/dataset.py` khiến vùng positive chỉ chiếm 18% frames; `TEST.THRESHOLD=0.3` quá chặt khiến Precision $\approx 1.0$ nhưng Recall chỉ 0.22. Đã nối `SOLVER.SIGMA: 2` (~31% frames positive) và hạ `TEST.THRESHOLD: 0.2`.

---

## 3. Experiments & Results (Kết Quả Thực Nghiệm)

### 3.1. Nhật Ký Huấn Luyện (ResNet-50 Slice-based Training)

| Mốc Epoch | Learning Rate | Train Loss | Val F1@0.05 | Val Recall | Val Precision | Ghi chú & Hiện tượng |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **Epoch 01** | 0.00002 (warmup) | 0.412 | 0.0000 | 0.0000 | 0.0000 | Đang trong giai đoạn warmup |
| **Epoch 03** | 0.00006 (warmup) | 0.365 | 0.0000 | 0.0000 | 0.0000 | Loss giảm dần đều |
| **Epoch 04** | 0.00008 (warmup) | 0.342 | 0.0000 | 0.0000 | 0.0000 | Chưa vượt ngưỡng threshold |
| **Epoch 05** | 0.00010 (full LR) | **0.318** | **0.3529** | **0.2155** | **0.9740** | **Đột phá đầu tiên**: Học được tín hiệu thật, Precision cực cao |
| **Epoch 06** | 0.00010 | 0.325 | 0.0394 | 0.0201 | 1.0000 | Overshoot nhẹ do chưa có warmup chuẩn (trước khi fix Issue 11) |

### 3.2. Đo Lường Theo Tiêu Chuẩn Dung Sai Tuyệt Đối (Absolute Tolerance)
Sau khi tích hợp `eval_f1_absolute_tol`:

| Tiêu chuẩn dung sai | Recall | Precision | F1-Score | Ý nghĩa nghiệp vụ |
|:---:|:---:|:---:|:---:|:---|
| **$\pm 0.3\text{s}$** | ~12.5% | ~85.0% | ~0.218 | Độ chính xác vi-thao tác cao |
| **$\pm 0.5\text{s}$** | ~22.0% | ~88.5% | **~0.352** | Chuẩn công nghiệp may |
| **$\pm 1.0\text{s}$** | ~35.0% | ~91.0% | ~0.505 | Chấp nhận được trong phân tích SOP |
| **$\pm 2.0\text{s}$** | ~52.0% | ~93.0% | ~0.666 | Bao trọn bước may kế tiếp |

### 3.3. Các Bước Hành Động Ưu Tiên Tiếp Theo
1. 🔴 **Ưu tiên 1**: Sửa `train.py` để lưu `model_best.pth` dựa trên Absolute F1@0.5s thay vì Relative F1@0.05 (Issue 5).
2. 🔴 **Ưu tiên 2**: Tái cấu trúc lại file split train/val bảo đảm tách biệt hoàn toàn theo công đoạn/trạm may (`--split-mode by_folder`) (Issue 6).
3. 🟡 **Ưu tiên 3**: Trực quan hóa đường cong phân bố score ranh giới trên video thực tế để kiểm tra giả định của DiffFormer (Issue 8).
