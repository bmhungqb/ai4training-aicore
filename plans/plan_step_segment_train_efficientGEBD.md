# Kế Hoạch Triển Khai Huấn Luyện Step Segmentation Với EfficientGEBD

## 1. Tổng Quan Mục Tiêu
- Triển khai phương pháp **EfficientGEBD** (*Rethinking the Architecture Design for Efficient Generic Event Boundary Detection - ACM MM 2024*) cho bài toán phát hiện ranh giới bước thao tác may công nghiệp (Step Segmentation).
- **Mã nguồn gốc**: [Ziwei-Zheng/EfficientGEBD](https://github.com/Ziwei-Zheng/EfficientGEBD)
- **Bài báo**: [arXiv:2407.12622](https://arxiv.org/abs/2407.12622)
- **Tập dữ liệu**: Dữ liệu may công nghiệp tại `data/**/step_segments.json` (tương tự như DDM-Net nhưng chuẩn bị trong thư mục độc lập `efficient_gebd_dataset/`).

---

## 2. Kiến Trúc & Thiết Lập Kỹ Thuật (Aligned Requirements)

1. **Cấu trúc Source Code**:
   - Tách biệt hoàn toàn với DDM-Net, giữ nguyên cấu trúc chuẩn của repo gốc EfficientGEBD:
     `src/step_segment/EfficientGEBD/`
   - Gồm các thành phần:
     - `train.py`, `post_process.py`, `solver/`, `modeling/`, `datasets/`, `utils/`, `script/`
     - Huấn luyện qua `torchrun` (hỗ trợ single-GPU / multi-GPU DDP thuần) và quản lý cấu hình bằng `yacs` config.

2. **Backbone & Mô Hình**:
   - Sử dụng **Video-domain CSN** (`MODEL.BACKBONE.NAME: 'csn'` hoặc `'csn_r50'`) kết hợp với FPN và DiffFormer / DiffMixer heads theo thiết kế đề xuất tối ưu của bài báo.
   - Quản lý checkpoint pretrained `CSN-pretrained/` (ig65m) và tích hợp các module từ `mmaction2`/`mmengine`.

3. **Cấu Trúc Dataset Độc Lập (`efficient_gebd_dataset/`)**:
   - Thư mục riêng biệt: `data/efficient_gebd_dataset/` (hoặc `efficient_gebd_dataset/`) không dùng chung với `ddm_dataset/`.
   - Lưu trữ:
     - Trích xuất offline frames: `<dataset_dir>/images/{train,val}/<video_id>/frame%d.jpg`.
     - Áp dụng **ROI Mask** (từ `tools/mask_editor`, dùng `*.mask.png` nếu có) để crop đúng vùng thao tác thợ may tương tự như pipeline của DDM-Net.
     - Sinh file annotation chuẩn Kinetics-GEBD định dạng pickle (`.pkl`):
       - Chứa thông tin video metadata: `fps`, `num_frames`, `video_duration`, `path_frame`, và danh sách các mốc ranh giới `boundaries` / `f1_consis`.

---

## 3. Các Bước Triển Khai Chi Tiết

### Giai đoạn 1: Chuẩn Bị Dữ Liệu (`tools/prepare_efficient_gebd_dataset.py`)
- Quét toàn bộ `data/**/step_segments.json` và video tương ứng.
- Đọc ROI Mask `*.mask.png` (nếu có) thông qua helper tương tự `datasets/roi_mask.py`.
- Trích xuất các frames ra thư mục ảnh theo cấu trúc của EfficientGEBD:
  - `data/efficient_gebd_dataset/images/{train,val}/<video_name>/frame{:d}.jpg`
- Chuyển đổi timestamp các ranh giới từ `step_segments.json` sang mốc thời gian / frame index:
  - Tạo `data/efficient_gebd_dataset/train_annotation.pkl`
  - Tạo `data/efficient_gebd_dataset/val_annotation.pkl`
- Hỗ trợ chia tập train/val theo `--split-mode random` hoặc `--split-mode by_folder`.

### Giai đoạn 2: Cấu Trúc Repo & Mã Nguồn EfficientGEBD (`src/step_segment/EfficientGEBD/`)
- Đồng bộ/setup mã nguồn gốc của EfficientGEBD vào `src/step_segment/EfficientGEBD/`.
- Cập nhật các module:
  - `datasets/dataset.py`: Đảm bảo nạp đúng cấu trúc dataset trích xuất offline, load file pickle annotation may mặc.
  - `modeling/`: Đảm bảo các component CSN, ResNet, DiffMixer, DiffFormer, FPN head hoạt động trơn tru.
  - `modeling/config.py`: Thiết lập các biến mặc định phù hợp với bộ dữ liệu may.
  - Hướng dẫn thiết lập môi trường: bổ sung file `requirements.txt` chuyên biệt cho EfficientGEBD (PyTorch, mmaction2, mmengine, mmcv, yacs,...).

### Giai đoạn 3: Cấu Hình Huấn Luyện & Script Thực Thi
- Tạo file cấu hình yacs phù hợp cho bộ dữ liệu may (resolution, frame_per_side, downsample, sequence_length,...).
- Viết bash script chạy huấn luyện:
  - `src/step_segment/EfficientGEBD/script/train/train_sewing_csn.sh`
  - Hỗ trợ chạy trên máy cá nhân/server (1 GPU qua `torchrun --nproc_per_node 1 train.py ...`).
- Script kiểm thử nhanh (`--dry-run` hoặc sanity-check qua 1-2 video) để kiểm tra pipeline nạp dữ liệu, forward, backward, và validation.

### Giai đoạn 4: Đánh Giá & So Sánh (Benchmark)
- Thiết lập kịch bản đánh giá F1-score (với các ngưỡng dung sai $0.05, 0.1, 0.2, 0.3, 0.4, 0.5$).
- So sánh hiệu năng (Boundary Recall, F1, VRAM tiêu thụ, tốc độ huấn luyện và suy luận) giữa **EfficientGEBD** và **DDM-Net** trên cùng tập validation của bài toán phân đoạn bước may.

