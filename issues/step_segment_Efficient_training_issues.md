# EfficientGEBD Training — Dataset Alignment Issues

> Phân tích mức độ tương thích giữa setup EfficientGEBD hiện tại và dataset may công nghiệp (`data/efficient_gebd_dataset/`).
> Ngày phân tích: 2026-09-25

---

## Tóm Tắt Kết Quả

| Hạng mục | Trạng thái | Ghi chú |
|---|---|---|
| Format annotation (`.pkl`) | ✅ OK | Đúng chuẩn Kinetics-GEBD schema |
| Dataset type `SEWING` trong code | ✅ OK | Đã có trong `datasets/__init__.py`, `dataset.py`, `post_process.py`, `train.py` |
| ROI Mask (crop khu vực thao tác) | ✅ OK | Pipeline `.mask.png` đã tích hợp trong `prepare_efficient_gebd_dataset.py` |
| Temporal Sampling (lấy mẫu thời gian) | 🔴 CRITICAL | Mất **~73% GT boundaries** khi dùng `SEQUENCE_LENGTH=100` kiểu GEBD |
| Evaluation Metric (F1 tolerance) | 🟡 WARNING | Threshold tương đối (relative) quá lỏng lẻo với video dài nhiều bước may |
| VRAM / Memory khi tăng Sequence Length | 🟡 WARNING | Cần slice video trước khi forward để tránh OOM |

---

## Issue 1 (CRITICAL): Temporal Sampling Mất ~73% Ground Truth Boundaries

### Mô tả

EfficientGEBD trong chế độ `END_TO_END=True` (GEBD-style) lấy mẫu đúng `SEQUENCE_LENGTH=100` frames rải đều toàn bộ video bằng:

```python
selected_indices = np.linspace(1, vlen, cfg.INPUT.SEQUENCE_LENGTH, dtype=int)
```

### Đặc thù Dataset May Công Nghiệp

| Chỉ số | Giá trị |
|---|---|
| Số video (train+val) | 32 video |
| Độ dài trung bình | ~143 giây (~2,145 frames) |
| Độ dài tối đa | ~578 giây (~8,677 frames) |
| Số ranh giới (boundaries) trung bình | **48 ranh giới / video** |
| Khoảng cách trung bình giữa 2 bước may | ~2 - 3 giây |

### Vấn đề Phát Sinh

Với `SEQUENCE_LENGTH=100` trên video dài trung bình 2,145 frames:
- Bước nhảy giữa 2 frame lấy mẫu: **~21 frames (~1.5 giây)**
- Ngưỡng tolerance để gán nhãn positive: `0.3s × fps / 2 ≈ 2 frames`
- Một boundary **chỉ bị bắt** nếu có frame lấy mẫu rơi vào cửa sổ $\pm$2 frames quanh nó — xác suất rất thấp.

### Kết Quả Đo Thực Tế

```
=== data/efficient_gebd_dataset/train_annotation.pkl ===
Total GT boundaries  : 1190
Boundaries hit (100 frames uniform): 315  →  Mất 73.5%

=== data/efficient_gebd_dataset/val_annotation.pkl ===
Total GT boundaries  : 348
Boundaries hit (100 frames uniform): 115  →  Mất 67.0%
```

> [!CAUTION]
> Với setup hiện tại, 2/3 số bước may trong mỗi video **không bao giờ được thấy** bởi mạng trong cả quá trình training lẫn evaluation. Mô hình sẽ có bias nghiêm trọng về phía nhãn âm (background) và không thể học được phần lớn các ranh giới bước may thực tế.

### Nguyên Nhân Gốc Rễ

Dataset Kinetics-GEBD (target gốc của EfficientGEBD) có đặc thù:
- Video ngắn: ~10 giây, chứa **1 - 5 boundaries**, phân bố thưa.
- `SEQUENCE_LENGTH=100` là hợp lý: mỗi boundary bình quân cách nhau ~20 frames → xác suất bắt cao.

Dataset may công nghiệp của chúng ta ngược lại:
- Video dài hơn **10 - 20×**, boundaries dày đặc hơn **15 - 20×**.
- Cơ chế linspace uniform sampling không còn phù hợp.

### Giải Pháp Đề Xuất

Chuyển sang cơ chế **Sliding Window / Slice-based sampling** tương tự nhánh `TAPOS` đã có sẵn trong repo:

```python
# Tham khảo logic trong dataset.py (TAPOS branch, line ~181)
num_slices = int(video_duration // 10 + 1)   # mỗi slice ~10 giây
selected_indices = np.linspace(1, vlen, SEQUENCE_LENGTH * num_slices, dtype=int)
```

Kích hoạt logic xử lý `SEWING` tương tự `TAPOS` trong:
- `datasets/dataset.py` — phần `prepare_gebd_annotations`: thêm nhánh `SEWING` với slice-based sampling.
- `train.py` — phần post-processing eval: ghép lại scores của các slices trong cùng một video (đã có cho TAPOS, cần mở rộng cho SEWING).
- Config: thêm `DATASET_MODE: 'slice'` hoặc sử dụng `TAPOS`-style flag.

---

## Issue 2 (WARNING): Evaluation Metric Không Phản Ánh Đúng Bài Toán Thực Tế

### Mô tả

Hàm `do_eval` trong [`utils/eval.py`](../src/step_segment/EfficientGEBD/utils/eval.py) sử dụng **relative distance threshold**:

```python
if offset_arr[ann1_idx, min_idx] <= threshold * (ins_end - ins_start + 1):
    tp += 1
```

Với `rel_dis_thres = 0.05` và video dài 150 giây (~2,175 frames):
- **Sai số cho phép**: `0.05 × 2175 ≈ 109 frames ≈ 7.5 giây`

### Vấn đề

Một bước may công nghiệp điển hình kéo dài **2 - 5 giây**. Cho phép sai số lên tới 7.5 giây nghĩa là mô hình có thể báo ranh giới sai lệch cả một bước may và vẫn được tính là đúng (True Positive).

> [!WARNING]
> F1 score hiển thị trong training log sẽ bị **inflate** (phồng lên) so với chất lượng thực sự khi ứng dụng vào phân đoạn bước may. Không thể so sánh kết quả trực tiếp với benchmark pipeline hiện tại (`eval_boundary_recall.py`) vốn dùng ngưỡng tuyệt đối.

### Giải Pháp Đề Xuất

Bổ sung đánh giá theo **ngưỡng tuyệt đối (absolute tolerance)** song song:

| Ngưỡng tuyệt đối | Ý nghĩa |
|---|---|
| ±0.3s | Rất chính xác (dưới 1 bước thao tác nhỏ) |
| ±0.5s | Chính xác (nửa giây) |
| ±1.0s | Chấp nhận được |
| ±2.0s | Cho phép (ranh giới nằm trong bước tiếp theo) |

Cần thêm hàm `do_eval_absolute_tol(gt_dict, pred_dict, abs_tol_s=0.5)` vào `utils/eval.py` và gọi song song khi validation.

---

## Issue 3 (WARNING): VRAM Risk Khi Tăng Sequence Length

### Mô tả

Khi áp dụng slice-based sampling, mỗi slice ~100 frames × 224×224×3. Với backbone ResNet50 hoặc CSN, tensor shape là `[BATCH_SIZE, SEQUENCE_LENGTH, 3, H, W]`.

- `BATCH_SIZE=2`, `SEQUENCE_LENGTH=100`: `2 × 100 × 3 × 224 × 224 ≈ 1.2GB` chỉ riêng input tensor (fp32), chưa tính activation maps.
- Với CSN (R152) backbone, VRAM yêu cầu tổng thường từ **14GB đến 20GB** cho mỗi batch.

### Giải Pháp Đề Xuất

- Giữ `SEQUENCE_LENGTH=100` mỗi slice nhưng giảm `BATCH_SIZE=1` nếu dùng CSN R152.
- Bật `AMPE: True` (Automatic Mixed Precision) — đã có trong `sewing_csn.yaml`.
- Cân nhắc dùng `BACKBONE: resnet50` (đã có `sewing_resnet50.yaml`) cho giai đoạn debug/sanity check trước.

---

## Tệp Liên Quan

| Tệp | Vai trò |
|---|---|
| [`tools/prepare_efficient_gebd_dataset.py`](../tools/prepare_efficient_gebd_dataset.py) | Build annotation pickle từ `step_segments.json` |
| [`src/step_segment/EfficientGEBD/datasets/dataset.py`](../src/step_segment/EfficientGEBD/datasets/dataset.py) | Logic sampling frames, gán nhãn Gaussian |
| [`src/step_segment/EfficientGEBD/datasets/__init__.py`](../src/step_segment/EfficientGEBD/datasets/__init__.py) | Build dataloader, ROOT path mapping |
| [`src/step_segment/EfficientGEBD/utils/eval.py`](../src/step_segment/EfficientGEBD/utils/eval.py) | Hàm F1 evaluation (relative threshold) |
| [`src/step_segment/EfficientGEBD/train.py`](../src/step_segment/EfficientGEBD/train.py) | Training loop, validation & slice-merge logic (TAPOS) |
| [`src/step_segment/EfficientGEBD/config-files/sewing_resnet50.yaml`](../src/step_segment/EfficientGEBD/config-files/sewing_resnet50.yaml) | Config ResNet50 backbone |
| [`src/step_segment/EfficientGEBD/config-files/sewing_csn.yaml`](../src/step_segment/EfficientGEBD/config-files/sewing_csn.yaml) | Config CSN backbone |

---

## Thứ Tự Ưu Tiên Xử Lý

1. 🔴 **Issue 1** — Sửa Temporal Sampling → Slice-based (TAPOS-style) cho SEWING
2. 🟡 **Issue 2** — Bổ sung Absolute Tolerance Evaluation metric
3. 🟡 **Issue 3** — Tune VRAM config theo GPU thực tế
