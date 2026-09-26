# AI4Training Documentation & Model Trackers

Hệ thống tài liệu dự án được cấu trúc thành **các file theo dõi duy nhất theo từng phương pháp/mô hình (Single Source of Truth Trackers)**. Mỗi file tổng hợp đầy đủ 3 phần:
1. **Method Info**: Kiến trúc mô hình, hướng dẫn môi trường, lệnh huấn luyện và tham số.
2. **Status & Issues**: Trạng thái hiện tại, phân tích nguyên nhân gốc rễ và giải pháp cho các vấn đề kỹ thuật.
3. **Experiments & Results**: Nhật ký huấn luyện, bảng số liệu định lượng, benchmark và so sánh.

---

## Danh Mục Tài Liệu Theo Nhánh

### 1. Step Segmentation (`docs/step_segment/`)
Phát hiện ranh giới chuyển tiếp giữa các bước thao tác (generic event boundaries) bằng mô hình học sâu:

- [`ddm_net.md`](file:///home/manh-hung/Documents/work/WE/AI4Training/ai4training-aicore-poc/docs/step_segment/ddm_net.md):
  - **Mô hình**: DDM-Net (NVIDIA SOP Blueprint Architecture + PyTorch Lightning).
  - **Backbone**: ResNet-50 / DINOv2 (`dinov2_vitb14`).
  - **Issues trọng tâm**: Kẹt flat loss ~12.4 (cộng dồn 18 heads), tràn RAM lúc validation (decord C++ cache), mở rộng tầm nhìn thời gian (0.37s $\rightarrow$ 1.1s), Bounding Box crop từ mask.
- [`diff_gebd.md`](file:///home/manh-hung/Documents/work/WE/AI4Training/ai4training-aicore-poc/docs/step_segment/diff_gebd.md):
  - **Mô hình**: DiffGEBD (ICCV 2025 - Denoising Diffusion Generative Model + Classifier-Free Guidance).
  - **Backbone**: ResNet-50 + DiffFormer diffusion head.
  - **Issues trọng tâm**: Sampling Imbalance (Precision 0.9+ nhưng Recall 0.19 do linspace 40 frames), giải pháp video chunking 10-15s (`tools/chunk_diff_gebd_dataset.py`).
- [`efficient_gebd.md`](file:///home/manh-hung/Documents/work/WE/AI4Training/ai4training-aicore-poc/docs/step_segment/efficient_gebd.md):
  - **Mô hình**: EfficientGEBD (ACM MM 2024 - Rethinking Architecture for Efficient GEBD).
  - **Backbone**: Video-domain CSN (R50/R152) / ResNet-50 + FPN + DiffFormer.
  - **Issues trọng tâm**: 12 issues đã ghi nhận (lấy mẫu slice-based, double Gaussian-smoothing, class imbalance BCE `POS_WEIGHT: 4.5`, checkpoint selection theo absolute-tolerance, tránh leakage trạm may).

---

### 2. Action Segmentation & Evaluation (`docs/action_segment/`)
Phân đoạn thao tác vật lý thuần túy (Kinematic) và phân tích quy trình may qua VLM:

- [`stage1_kinematic.md`](file:///home/manh-hung/Documents/work/WE/AI4Training/ai4training-aicore-poc/docs/action_segment/stage1_kinematic.md):
  - **Thuật toán**: SAM 3 (theo dõi bàn tay) + SEA-RAFT (optical flow dày đặc) + Magnitude/Direction Kinematic Fusion (0 VLM, 0 API key).
  - **Cải tiến v2**: Co viền mask 3x3, dung hợp vận tốc median + RMS năng lượng khi tì cổ tay, nội suy tuyến tính lấp khoảng trống mất mask.
  - **Kết quả thực nghiệm**: Benchmark trên 9 công đoạn độc lập (Chuyền 1: 349 bước Ground Truth, 1,505 segments máy, Macro Recall 86.2%, Micro Recall 87.3%, MAE 0.213s).
- [`stage2_vlm_analysis.md`](file:///home/manh-hung/Documents/work/WE/AI4Training/ai4training-aicore-poc/docs/action_segment/stage2_vlm_analysis.md):
  - **Quy trình 4 bước**: Bước 1 Học tri thức chuyên gia (`expert_analysis.py`), Bước 2 Phân loại thao tác công nhân (`segment_classify.py`), Bước 3 Đánh giá vĩ mô (`macro_eval.py`), Bước 4 Chẩn đoán vi mô (`micro_eval.py`).
  - **Issues trọng tâm**: Khắc phục 4 hạn chế khi đưa chuỗi frame tĩnh vào VLM (nén mờ độ phân giải, nhiễu tĩnh, mất vector chuyển động, bùng nổ token/độ trễ) bằng Dynamic Action ROI Crop, Dual-View Composite, và Motion History Image (MHI) vector overlay.
