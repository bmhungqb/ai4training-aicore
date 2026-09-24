# Kế Hoạch Triển Khai Huấn Luyện Step Segmentation Với DiffGEBD

## 1. Tổng Quan Mục Tiêu & Cơ Sở Khoa Học
- **Mục tiêu**: Triển khai và huấn luyện mô hình **DiffGEBD** (*Generic Event Boundary Detection via Denoising Diffusion - ICCV 2025*) cho bài toán phát hiện ranh giới bước may công nghiệp (Step Segmentation).
- **Mã nguồn gốc**: [JaejunHwang/DiffGEBD](https://github.com/JaejunHwang/DiffGEBD)
- **Bài báo**: [arXiv:2508.12084](https://arxiv.org/pdf/2508.12084)
- **Cơ chế cốt lõi của DiffGEBD**:
  - Không coi GEBD là bài toán phân loại nhị phân tất định (deterministic binary classification) thông thường, mà tiếp cận theo mô hình sinh khuếch tán (denoising diffusion generative model).
  - Tích hợp **Classifier-Free Guidance (CFG)** để kiểm soát độ phong phú (diversity) và độ chính xác (fidelity) của ranh giới sự kiện.
  - Sử dụng visual encoder (ResNet-50) trích xuất đặc trưng không gian - thời gian kết hợp temporal self-similarity matrix và DiffFormer/Transformer diffusion head để khử nhiễu dự đoán nhãn ranh giới 1D theo thời gian.
- **Tập dữ liệu**: Dữ liệu may công nghiệp (33 video) tại `data/**/step_segments.json` và video MP4 tương ứng (có hỗ trợ ROI mask thợ may).

---

## 2. Kiến Trúc Kỹ Thuật Đã Thống Nhất (Architecture & Design Alignment)

1. **Kiến trúc Framework & Vị trí Mã Nguồn**:
   - Giữ nguyên kiến trúc PyTorch thuần với `torchrun` DDP và quản lý cấu hình bằng `yacs` config (tương tự như cấu trúc chuẩn của EfficientGEBD và DiffGEBD gốc).
   - Thư mục mã nguồn độc lập: `src/step_segment/DiffGEBD/`:
     - `train.py`, `train.sh`, `test.sh`
     - `config/` (chứa `sewing_diffgebd_resnet50.yaml`, `config.yaml`)
     - `datasets/` (`dataset.py`, `__init__.py`)
     - `modeling/` (`basicGEBD.py`, `diff_former.py`, `diffusion_model.py`, `resnet.py`, `time_transformer.py`, `nn.py`, `config.py`)
     - `solver/` (`lr_schedulers.py`, `solver_utils.py`)
     - `utils/` (`eval.py`, `metrics.py`, `distribute.py`, `misc.py`, `sampler.py`)
     - `tools/` (`run_train.sh`, `export_predictions.py`)
     - `requirements.txt`

2. **Visual Backbone**:
   - Sử dụng **ResNet-50** mặc định theo công bố chính thức của DiffGEBD (pretrained ImageNet/Kinetics), cho phép huấn luyện end-to-end kết hợp AMP (Automatic Mixed Precision).

3. **Cấu Trúc Dataset Độc Lập (`data/diff_gebd_dataset/`)**:
   - Quản lý tách biệt hoàn toàn qua công cụ `tools/prepare_diff_gebd_dataset.py`.
   - Trích xuất offline frames có áp dụng **ROI Mask** (từ `tools/mask_editor`, file `*.mask.png` nếu có) để tập trung vào bàn máy may:
     - `data/diff_gebd_dataset/images/{train,val}/<video_id>/frame%d.jpg`
   - Sinh file annotation chuẩn Kinetics-GEBD định dạng pickle (`.pkl`):
     - `data/diff_gebd_dataset/train_annotation.pkl`
     - `data/diff_gebd_dataset/val_annotation.pkl`
   - Hỗ trợ 2 chế độ chia tập: `--split-mode random` (tỉ lệ val 20%) và `--split-mode by_folder` (theo danh sách mã công đoạn CD).

4. **Đánh Giá & Xuất Kết Quả Benchmark**:
   - Đánh giá F1, Precision, Recall chuẩn GEBD theo các ngưỡng khoảng cách tương đối (relative distance thresholds: 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5).
   - Xuất file dự đoán `model_pred_dict_ep*.pkl` và chuyển đổi sang định dạng `step_segments_pred.json` / báo cáo so sánh để đối chiếu trực tiếp với **DDM-Net** và **EfficientGEBD**.

---

## 3. Bố Cục Thư Mục (Directory Layout)

```text
ai4training-aicore-poc/
├── data/
│   ├── cd1/chuyen1/
│   │   ├── cam-03_....mp4
│   │   ├── cam-03_....mask.png               <-- ROI mask (nếu có)
│   │   ├── action_segments_annotated.json
│   │   └── step_segments.json                <-- Ground-truth steps
│   └── diff_gebd_dataset/                    <-- Độc lập cho DiffGEBD
│       ├── images/
│       │   ├── train/<video_id>/frame%d.jpg
│       │   └── val/<video_id>/frame%d.jpg
│       ├── train_annotation.pkl
│       └── val_annotation.pkl
├── tools/
│   ├── prepare_diff_gebd_dataset.py          <-- Chuẩn bị dataset cho DiffGEBD
│   └── export_diffgebd_predictions.py        <-- Convert prediction PKL -> step_segments_pred.json
├── src/
│   └── step_segment/
│       ├── DDM-Net/                          <-- NVIDIA SOP Lightning BP
│       ├── EfficientGEBD/                    <-- EfficientGEBD (ACM MM 2024)
│       └── DiffGEBD/                         <-- DiffGEBD (ICCV 2025)
│           ├── config/
│           │   └── sewing_diffgebd_resnet50.yaml
│           ├── datasets/
│           │   ├── __init__.py               <-- Hỗ trợ dataset SEWING
│           │   └── dataset.py                <-- Hỗ trợ dataset SEWING
│           ├── modeling/
│           │   ├── basicGEBD.py
│           │   ├── diff_former.py
│           │   ├── diffusion_model.py
│           │   ├── resnet.py
│           │   ├── time_transformer.py
│           │   ├── nn.py
│           │   └── config.py
│           ├── solver/
│           ├── utils/
│           ├── train.py                      <-- Main DDP engine
│           ├── requirements.txt
│           └── tools/
│               └── run_train.sh              <-- Server execution script
└── docs/
    └── step_segment_diffgebd_training_guidance.md <-- Hướng dẫn chạy train & benchmark
```

---

## 4. Các Bước Triển Khai Chi Tiết

### Giai đoạn 1: Chuẩn Bị Dữ Liệu (`tools/prepare_diff_gebd_dataset.py`)
- Quét toàn bộ `data/**/step_segments.json` (33 video công đoạn may).
- Với mỗi video:
  - Lấy các ranh giới chuyển tiếp giữa các bước thao tác (bỏ qua khoảng UNKNOWN đầu/cuối).
  - Kiểm tra và áp dụng ROI Mask (`*.mask.png`) nếu tồn tại để crop đúng vùng may của công nhân.
  - Trích xuất toàn bộ frames ra `data/diff_gebd_dataset/images/{train,val}/<video_id>/frame%d.jpg`.
  - Tính toán `fps`, `video_duration`, `substages_myframeidx`, `substages_timestamps`.
- Tạo `train_annotation.pkl` và `val_annotation.pkl` hỗ trợ cả 2 chế độ chia `--split-mode random` và `--split-mode by_folder`.

### Giai đoạn 2: Cài Đặt & Điều Chỉnh Mã Nguồn (`src/step_segment/DiffGEBD/`)
- Đồng bộ toàn bộ source code từ upstream [JaejunHwang/DiffGEBD](https://github.com/JaejunHwang/DiffGEBD) vào `src/step_segment/DiffGEBD/`.
- Cập nhật module `datasets/`:
  - Thêm dataset type `SEWING` vào `datasets/__init__.py` và `datasets/dataset.py`.
  - Thiết lập đường dẫn mặc định `data/diff_gebd_dataset/images` và đọc annotation `data/diff_gebd_dataset/{split}_annotation.pkl`.
  - Cấu hình template đọc frame: `frame{:d}.jpg`.
- Cập nhật `train.py`:
  - Trong `validate_end_to_end`, hỗ trợ đọc ground-truth annotation từ `data/diff_gebd_dataset/{split}_annotation.pkl` khi `dataset.name == 'SEWING'`.
- Tạo file cấu hình `config/sewing_diffgebd_resnet50.yaml`:
  - `MODEL.BACKBONE.NAME: 'resnet50'`
  - `INPUT.RESOLUTION: 224`
  - `INPUT.SEQUENCE_LENGTH: 100`
  - `INPUT.END_TO_END: True`
  - `DIFFUSION.TIMESTEPS: 1000`
  - `DIFFUSION.SAMPLING_TIMESTEPS: 16`
  - `DIFFUSION.CFG_SCALE: 7.0`
  - `SOLVER.BATCH_SIZE: 2` (phù hợp VRAM 1 GPU/2 GPU)
  - `SOLVER.AMPE: True`

### Giai đoạn 3: Script Thực Thi & Chuyển Đổi Kết Quả (`tools/`)
- Viết `src/step_segment/DiffGEBD/tools/run_train.sh` để dễ dàng chạy single-GPU hoặc multi-GPU:
  ```bash
  torchrun --nproc_per_node=1 train.py --config-file config/sewing_diffgebd_resnet50.yaml ...
  ```
- Viết `tools/export_diffgebd_predictions.py`:
  - Đọc file dự đoán `model_pred_dict_ep*.pkl`.
  - Chuyển đổi timestamp và frame index ranh giới thành danh sách `step_segments_pred.json` theo đúng cấu trúc dữ liệu của dự án.
  - Xuất bảng so sánh F1 / Recall / Precision tương thích với `experiments/stage1_boundary_recall_9cd`.

### Giai đoạn 4: Hướng Dẫn Huấn Luyện & Benchmark (`docs/`)
- Viết `docs/step_segment_diffgebd_training_guidance.md` hướng dẫn chi tiết từng bước:
  1. Cài đặt môi trường Python / Conda và PyTorch.
  2. Lệnh tiền xử lý dữ liệu.
  3. Lệnh chạy huấn luyện trên server (nohup / tmux).
  4. Lệnh chạy inference và xuất metrics.
  5. Bảng so sánh 3 phương pháp: **DDM-Net** vs **EfficientGEBD** vs **DiffGEBD**.
