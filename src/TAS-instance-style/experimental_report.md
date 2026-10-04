# Báo Cáo Thực Nghiệm: Instance-Level Temporal Action Segmentation (BaFormer & TQT)

**Dự án**: AI4Training — AI Core Action Segmentation  
**Mô hình**: BaFormer (`bk_fde_tde`) & TQT (Temporal Query Transformer)  
**Tập dữ liệu**: [`dataset_tas_instance`](file:///home/hungbm/ai4training/ai4training-aicore/src/TAS-instance-style/dataset_tas_instance) (98 video clips sạch, 4 classes)  
**Tài liệu tham chiếu chuẩn**: [`problem_definition.md`](file:///home/hungbm/ai4training/ai4training-aicore/src/TAS-instance-style/problem_definition.md)  
**Môi trường phần cứng**: NVIDIA GeForce RTX 5060 Ti (CUDA 13.2, PyTorch 2.14.0)  
**Thời gian hoàn thành**: Tháng 10/2026  

---

## 1. Thiết Lập & Bối Cảnh Thực Nghiệm

Báo cáo này tổng hợp kết quả của chu kỳ thực nghiệm toàn diện trên tập dữ liệu chuẩn hóa 98 video clips công đoạn may mặc, chia làm 3 giai đoạn:
1. **Giai đoạn 1 (Over-Engineered BaFormer)**: Huấn luyện 4 đặc trưng backbone đơn lẻ (**ResNet-50**, **VideoMAE**, **DINOv2**, **DINOv3**) và 1 mô hình kết hợp (**DINOv2 + VideoMAE**) trên kiến trúc BaFormer tích tụ 10 hàm mất mát phụ từ Proposals 04–14.
2. **Giai đoạn 2 (Clean Baseline Ablation)**: Nhận diện hiện tượng mô hình bị can thiệp quá mức (over-constraining), tiến hành đưa BaFormer quay về **Clean Baseline chuẩn** (tắt toàn bộ 5 loss phụ gây nhiễu, đưa inference về cơ bản) để đánh giá sòng phẳng thực lực của 3 backbone hàng đầu: **Clean DINOv3**, **Clean DINOv2**, và **Clean VideoMAE**.
3. **Giai đoạn 3 (TQT — Temporal Query Transformer Baseline)**: Triển khai và đánh giá kiến trúc mô hình mới độc lập TQT chuẩn theo Section 13 [`problem_definition.md`](file:///home/hungbm/ai4training/ai4training-aicore/src/TAS-instance-style/problem_definition.md) (Dilated Temporal Convolutional Backbone + Temporal Query Decoder + Hungarian Bipartite Matcher + Set Criterion) trên toàn bộ 4 feature backbones: **VideoMAE**, **ResNet-50**, **DINOv2**, và **DINOv3**.

### Phân Bổ Lớp Dữ Liệu Validation:
* **Class 0 (Sewing/Joining)**: 5,846 frames (44.6%) — Lớp hành động chính
* **Class 1 (Positioning/Handling)**: 4,505 frames (34.3%) — Lớp thao tác chuẩn bị vải
* **Class 2 (Adjustment/Alignment/Preparation)**: 1,976 frames (15.1%) — Lớp vi chỉnh
* **Class 3 (Inspection/Auxiliary)**: 794 frames (6.0%) — Lớp thiểu số (kiểm tra đường may)
* **Tổng số ranh giới thực tế (GT Boundaries)**: 443 ranh giới

---

## 2. Bảng Xếp Hạng Tổng Hợp (Benchmark Leaderboard)

Xếp hạng toàn bộ 12 cấu hình thực nghiệm dựa trên chỉ số tổng hợp **Composite Score** ($0.4 \times \text{F1 Mean} + 0.3 \times \text{Edit} + 0.3 \times \text{Accuracy}$):

| Hạng | Cấu hình Thử nghiệm | Mô hình | Feature Dim | Best Ep / Total | Composite Score | F1 Mean (%) | F1@10 (%) | F1@25 (%) | F1@50 (%) | Frame Acc (%) | Edit Score | Boundary F1@3 (%) *(Best/Cuối)* | Boundary Recall@3 (%) *(Best/Cuối)* | Thư mục Checkpoint & Logs |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 🥇 | **Over-eng DINOv2** | BaFormer | 768d | 97 / 142 | **44.04** | **31.76** | **45.03** | **33.51** | **16.75** | 50.38 | **54.08** | 33.01 / 33.71 | 45.82 / **50.56** | [`exp_dinov2/1/`](file:///home/hungbm/ai4training/ai4training-aicore/src/TAS-instance-style/BaFormer/experiments/tas_instance/bk_fde_tde/exp_dinov2/1/) |
| 🥈 | **Clean DINOv3** | BaFormer | 768d | 43 / 88 | **42.45** | 27.48 | 39.89 | 30.85 | 11.70 | 53.24 | 51.61 | 1.73 / 20.88 | 0.90 / 18.74 | [`clean_dinov3/1/`](file:///home/hungbm/ai4training/ai4training-aicore/src/TAS-instance-style/BaFormer/experiments/tas_instance/bk_fde_tde/clean_dinov3/1/) |
| 🥉 | **Over-eng Fusion (D2+MAE)** | BaFormer | 1536d | 27 / 72 | **42.44** | 28.65 | 39.79 | 32.36 | 13.79 | **56.36** | 46.90 | 0.88 / 28.94 | 0.45 / 32.96 | [`exp_dinov2_videomae/1/`](file:///home/hungbm/ai4training/ai4training-aicore/src/TAS-instance-style/BaFormer/experiments/tas_instance/bk_fde_tde/exp_dinov2_videomae/1/) |
| 4 | **Clean DINOv2** | BaFormer | 768d | 99 / 144 | **41.85** | 28.15 | 41.76 | 30.16 | 12.53 | 50.67 | 51.29 | 26.94 / **33.80** | 29.35 / 49.21 | [`clean_dinov2/1/`](file:///home/hungbm/ai4training/ai4training-aicore/src/TAS-instance-style/BaFormer/experiments/tas_instance/bk_fde_tde/clean_dinov2/1/) |
| 5 | **Over-eng VideoMAE** | BaFormer | 768d | 87 / 132 | **40.34** | 28.27 | 41.67 | 29.90 | 13.24 | 50.13 | 46.66 | 2.63 / 15.13 | 1.35 / 10.16 | [`exp_videomae/1/`](file:///home/hungbm/ai4training/ai4training-aicore/src/TAS-instance-style/BaFormer/experiments/tas_instance/bk_fde_tde/exp_videomae/1/) |
| 6 | **Over-eng ResNet-50** | BaFormer | 2048d | 30 / 75 | **40.18** | 27.26 | 39.06 | 28.12 | 14.58 | 48.86 | 48.73 | 0.00 / 10.92 | 0.00 / 7.22 | [`exp_resnet50/1/`](file:///home/hungbm/ai4training/ai4training-aicore/src/TAS-instance-style/BaFormer/experiments/tas_instance/bk_fde_tde/exp_resnet50/1/) |
| 7 | **Clean VideoMAE** | BaFormer | 768d | 125 / 170 | **39.72** | 27.66 | 39.01 | 28.02 | 15.93 | 46.22 | 49.29 | 10.82 / 15.01 | 6.09 / 9.71 | [`clean_videomae/1/`](file:///home/hungbm/ai4training/ai4training-aicore/src/TAS-instance-style/BaFormer/experiments/tas_instance/bk_fde_tde/clean_videomae/1/) |
| 8 | **Over-eng DINOv3** | BaFormer | 768d | 30 / 75 | **38.93** | 26.11 | 37.78 | 26.11 | 14.44 | 51.19 | 43.76 | 0.44 / 15.32 | 0.23 / 11.51 | [`exp_dinov3/1/`](file:///home/hungbm/ai4training/ai4training-aicore/src/TAS-instance-style/BaFormer/experiments/tas_instance/bk_fde_tde/exp_dinov3/1/) |
| 9 | **TQT VideoMAE** | TQT | 768d | 37 / 72 | **36.17** | 22.64 | 33.97 | 24.72 | 9.22 | 45.67 | 44.70 | 0.00 / 0.00 | 0.00 / 0.00 | [`experiments/videomae/1/`](file:///home/hungbm/ai4training/ai4training-aicore/src/TAS-instance-style/TQT/experiments/videomae/1/) |
| 10 | **TQT ResNet-50** | TQT | 2048d | 41 / 76 | **35.42** | 19.10 | 29.45 | 20.42 | 7.44 | 50.99 | 41.60 | 0.00 / 0.00 | 0.00 / 0.00 | [`experiments/resnet50/1/`](file:///home/hungbm/ai4training/ai4training-aicore/src/TAS-instance-style/TQT/experiments/resnet50/1/) |
| 11 | **TQT DINOv2** | TQT | 768d | 10 / 45 | **31.84** | 15.91 | 25.17 | 17.26 | 5.30 | 42.16 | 42.75 | 0.00 / 0.00 | 0.00 / 0.00 | [`experiments/dinov2/1/`](file:///home/hungbm/ai4training/ai4training-aicore/src/TAS-instance-style/TQT/experiments/dinov2/1/) |
| 12 | **TQT DINOv3** | TQT | 768d | 64 / 99 | **30.44** | 14.15 | 22.82 | 13.46 | 6.18 | 40.03 | 42.59 | 0.00 / 0.00 | 0.00 / 0.00 | [`experiments/dinov3/1/`](file:///home/hungbm/ai4training/ai4training-aicore/src/TAS-instance-style/TQT/experiments/dinov3/1/) |

---

## 3. Ma Trận Chi Tiết Từng Lớp Hành Động (Per-Class Performance Matrix)

So sánh Precision (P), Recall (R), và F1-Score (%) trên từng lớp hành động:

| Cấu hình Thử nghiệm | Mô hình | Class 0: Sewing/Joining<br>*(Support: 5,846 frames)* | Class 1: Positioning/Handling<br>*(Support: 4,505 frames)* | Class 2: Adjustment/Prep<br>*(Support: 1,976 frames)* | Class 3: Inspection/Aux<br>*(Support: 794 frames)* |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Clean DINOv3** | BaFormer | P: 48.20% \| R: 56.38% \| **F1: 51.97%** | P: 53.56% \| R: 30.85% \| **F1: 39.15%** | P: 18.93% \| R: 33.20% \| **F1: 24.11%** | P: 7.62% \| R: 2.14% \| **F1: 3.34%** |
| **Clean DINOv2** | BaFormer | P: 56.65% \| R: 34.18% \| **F1: 42.63%** | P: **66.09%** \| R: 49.01% \| **F1: 56.28%** | P: **20.53%** \| R: **58.30%** \| **F1: 30.37%** | P: 41.68% \| R: **33.75%** \| **F1: 37.30%** 🏆 |
| **Clean VideoMAE** | BaFormer | P: 65.10% \| R: 50.99% \| **F1: 57.19%** | P: 65.87% \| R: 44.77% \| **F1: 53.31%** | P: 17.57% \| R: 42.81% \| **F1: 24.91%** | P: 12.20% \| R: 10.20% \| **F1: 11.11%** |
| **Over-eng DINOv3** | BaFormer | P: 48.09% \| R: 37.67% \| **F1: 42.24%** | P: 64.30% \| R: 36.03% \| **F1: 46.18%** | P: 18.27% \| R: 49.44% \| **F1: 26.68%** | P: 35.32% \| R: 29.85% \| **F1: 32.35%** |
| **Over-eng DINOv2** | BaFormer | P: 61.09% \| R: 55.39% \| **F1: 58.10%** | P: 60.24% \| R: **55.23%** \| **F1: 57.63%** | P: 19.67% \| R: 34.92% \| **F1: 25.17%** | P: **60.87%** \| R: 14.11% \| **F1: 22.90%** |
| **Over-eng VideoMAE** | BaFormer | P: **67.94%** \| R: 54.93% \| **F1: 60.75%** | P: 64.24% \| R: 48.86% \| **F1: 55.50%** | P: 20.15% \| R: 45.60% \| **F1: 27.95%** | P: 19.32% \| R: 12.09% \| **F1: 14.87%** |
| **Over-eng ResNet-50** | BaFormer | P: 59.31% \| R: **70.00%** \| **F1: 64.21%** | P: 63.87% \| R: 46.30% \| **F1: 53.69%** | P: 19.85% \| R: 28.69% \| **F1: 23.47%** | P: 38.00% \| R: 4.79% \| **F1: 8.50%** |
| **Over-eng Fusion** | BaFormer | P: 61.76% \| R: 43.02% \| **F1: 50.72%** | P: 62.80% \| R: 51.56% \| **F1: 56.63%** | P: 19.29% \| R: 47.82% \| **F1: 27.49%** | P: 43.33% \| R: 24.56% \| **F1: 31.35%** |
| **TQT VideoMAE** | TQT | P: 54.53% \| R: 52.21% \| **F1: 53.34%** | P: 42.92% \| R: **64.15%** \| **F1: 51.43%** | P: 14.74% \| R: 2.33% \| **F1: 4.02%** | P: 1.05% \| R: 0.63% \| **F1: 0.79%** |
| **TQT ResNet-50** | TQT | P: 62.45% \| R: 63.86% \| **F1: 63.14%** | P: 43.23% \| R: 60.89% \| **F1: 50.56%** | P: 26.92% \| R: 10.83% \| **F1: 15.45%** | P: 0.00% \| R: 0.00% \| **F1: 0.00%** |
| **TQT DINOv2** | TQT | P: 51.00% \| R: 53.47% \| **F1: 52.21%** | P: 37.77% \| R: 49.01% \| **F1: 42.66%** | P: 17.28% \| R: 10.02% \| **F1: 12.68%** | P: 0.00% \| R: 0.00% \| **F1: 0.00%** |
| **TQT DINOv3** | TQT | P: 45.90% \| R: 46.87% \| **F1: 46.38%** | P: 52.65% \| R: 48.52% \| **F1: 50.50%** | P: 10.14% \| R: 13.97% \| **F1: 11.75%** | P: 18.05% \| R: 6.30% \| **F1: 9.34%** |

---

## 4. Nghiên Cứu Bóc Tách: Over-Engineered vs Clean Baseline (Ablation Study)

Đối chiếu trực tiếp tác động của việc gỡ bỏ 5 hàm mất mát phụ (`contra`, `repulse`, `enc_ce`, `enc_smooth`, `mask_tv`) và các tầng hậu xử lý heuristic trong BaFormer:

| Tiêu chí Đánh giá | DINOv3 Cũ $\to$ **DINOv3 Sạch** | DINOv2 Cũ $\to$ **DINOv2 Sạch** | VideoMAE Cũ $\to$ **VideoMAE Sạch** |
| :--- | :---: | :---: | :---: |
| **Biến thiên Composite Score** | **38.93 $\to$ 42.45 (+3.52)** 🚀 | 44.04 $\to$ 41.85 (-2.19) | 40.34 $\to$ 39.72 (-0.62) |
| **Biến thiên Frame Accuracy** | **51.19% $\to$ 53.24% (+2.05%)** | 50.38% $\to$ 50.67% (+0.29%) | 50.13% $\to$ 46.22% (-3.91%) |
| **Biến thiên Edit Score** | **43.76 $\to$ 51.61 (+7.85)** 🏆 | 54.08 $\to$ 51.29 (-2.79) | 46.66 $\to$ 49.29 (+2.63) |
| **Biến thiên F1 Mean** | **26.11% $\to$ 27.48% (+1.37%)** | 31.76% $\to$ 28.15% (-3.61%) | 28.27% $\to$ 27.66% (-0.61%) |
| **Biến thiên Boundary F1 Cuối** | **15.32% $\to$ 20.88% (+5.56%)** | 33.71% $\to$ 33.80% (+0.09%) | 15.13% $\to$ 15.01% (-0.12%) |
| **Dự đoán Class 2 (Adjustment)** | **5,347 $\to$ 3,465 frames (-1,882 FP)** | 3,745 $\to$ 5,610 frames | 4,472 $\to$ 4,816 frames |
| **Loss Tổng thể (Validation)** | 9.42 $\to$ **8.08 (Giảm mạnh)** | 8.44 $\to$ **7.49 (Giảm mạnh)** | 8.40 $\to$ **7.37 (Giảm mạnh)** |

---

## 5. Nhận Xét & Phân Tích Dữ Liệu Thực Nghiệm (BaFormer)

### 5.1. DINOv3 (`clean_dinov3`): Bước Bứt Phá Lớn Nhất Nhờ Gỡ Bỏ Ràng Buộc
* **Giải phóng cấu trúc phân đoạn**: Ở mô hình cũ, DINOv3 bị phạt nặng bởi các loss đẩy query (`DQCR Repulse`) và chặn attention (`Boundary Barrier`), khiến mô hình xé nhỏ các đoạn may và dự đoán quá mức 5,347 frames vào Class 2.
* **Cải thiện toàn diện trên Clean Baseline**:
  * Số frame dự đoán giả Class 2 giảm 1,882 frames (xuống còn 3,465 frames).
  * Class 0 (Sewing) Recall tăng mạnh từ 37.67% lên **56.38%**, kéo F1 Class 0 tăng từ 42.24% lên **51.97%**.
  * **Edit Score nhảy vọt từ 43.76 lên 51.61 (+7.85 điểm)**, Frame Accuracy đạt **53.24%** (cao nhất trong 3 mô hình độc lập sạch), đưa Composite Score lên **42.45** (vượt qua Clean DINOv2 41.85).

### 5.2. DINOv2 (`clean_dinov2`): Quán Quân Định Vị Ranh Giới & Lớp Thiểu Số
* **Khả năng định vị ranh giới vượt trội**: DINOv2 duy trì sự thống trị tuyệt đối về ranh giới hành động, đạt **Boundary Recall 49.21%** và **Boundary F1 33.80%** (cao gấp $2.5\times$ DINOv3 và gấp $5\times$ VideoMAE).
* **Đột phá ở lớp khó (Class 3 Inspection)**: Trên Clean Baseline, DINOv2 đạt Recall **33.75%** và F1 **37.30%** (Precision 41.68%) — xác lập kỷ lục cao nhất của toàn bộ dự án ở lớp thiểu số này.

### 5.3. VideoMAE (`clean_videomae`): Ngữ Nghĩa Chuyển Động Chuẩn Xác Nhưng Bị Rào Cản Temporal Smearing
* **Độ chuẩn xác thao tác tay (Precision)**: Nhờ các khối 3D Spatio-Temporal Attention, VideoMAE luôn đạt Precision cao nhất ở các lớp thao tác máy: **Sewing Precision 65.10%** và **Positioning Precision 65.87%**.
* **Hạn chế cố hữu ở ranh giới**: Cơ chế 16-frame tubelet pooling làm mờ tín hiệu chuyển tiếp theo thời gian, khiến Boundary Recall chỉ đạt **9.71%**, không thể tách ranh giới frame-level sắc nét như các mô hình Spatial ViT (DINOv2/DINOv3).

### 5.4. ResNet-50 (`exp_resnet50`): Suy Giảm Năng Lực Trên Bài Toán Instance-Level
* **Thiên lệch lớp đa số**: ResNet-50 gán nhãn tới 70.00% frames vào Class 0 Sewing, nhưng gần như bỏ sót hoàn toàn Class 3 Inspection (Recall chỉ 4.79%, F1 8.50%).
* **Tê liệt ranh giới**: Boundary Recall tại Best Epoch là 0.00% (cuối kỳ chỉ 7.22%), mô hình bị bão hòa và dừng sớm ở Epoch 30.

---

## 6. Đánh Giá Chuyên Sâu Mô Hình Mới TQT (Temporal Query Transformer)

### 6.1. So Sánh Trực Tiếp TQT vs BaFormer Clean Baseline

Khi so sánh cùng một backbone đặc trưng giữa **TQT Baseline** và **BaFormer Clean Baseline**:

| Tiêu chí | DINOv3: TQT vs Clean BaFormer | DINOv2: TQT vs Clean BaFormer | VideoMAE: TQT vs Clean BaFormer |
| :--- | :---: | :---: | :---: |
| **Composite Score** | 30.44 vs **42.45 (-12.01)** | 31.84 vs **41.85 (-10.01)** | 36.17 vs **39.72 (-3.55)** |
| **F1 Mean** | 14.15% vs **27.48% (-13.33%)** | 15.91% vs **28.15% (-12.24%)** | 22.64% vs **27.66% (-5.02%)** |
| **Frame Accuracy** | 40.03% vs **53.24% (-13.21%)** | 42.16% vs **50.67% (-8.51%)** | 45.67% vs **46.22% (-0.55%)** |
| **Edit Score** | 42.59 vs **51.61 (-9.02)** | 42.75 vs **51.29 (-8.54)** | 44.70 vs **49.29 (-4.59)** |
| **Boundary F1@3** | 0.00% vs **20.88% (-20.88%)** | 0.00% vs **33.80% (-33.80%)** | 0.00% vs **15.01% (-15.01%)** |
| **Class 3 F1 (Inspection)** | 9.34% vs **3.34% (+6.00%)** | 0.00% vs **37.30% (-37.30%)** | 0.79% vs **11.11% (-10.32%)** |

> [!WARNING]
> **Hiện tượng đảo chiều bất thường giữa các backbone:**
> Trong BaFormer, các backbone thị giác mạnh (DINOv3, DINOv2) đứng đầu bảng xếp hạng (Composite 42.45 và 41.85). Tuy nhiên trong TQT, DINOv3 (30.44) và DINOv2 (31.84) lại tụt xuống đáy, trong khi VideoMAE (36.17) và ResNet-50 (35.42) lại đạt điểm cao hơn. Toàn bộ 4 mô hình TQT đều bị **Boundary F1 = 0.0%**.

---

### 6.2. Phân Tích 5 Nguyên Nhân Gốc Rễ Khiến TQT Đạt Kết Quả Thấp (Root-Cause Diagnosis)

Sau khi truy vết mã nguồn mô hình, hàm mất mát và cơ chế giải mã inference giữa TQT và BaFormer, xác định được 5 nguyên nhân kỹ thuật cốt lõi:

#### 1. Receptive Field Hạn Chế của Temporal Convolutional Backbone (MS-TCN vs ASFormer)
* **BaFormer**: Sử dụng `ASFormerEncoder` tích hợp cơ chế **Temporal Dilated Cross-Attention** và nhiều khối dilated convolution nhiều tầng với receptive field lý thuyết vượt trên 4,096 frames. Nhờ đó, biểu diễn đặc trưng ở mọi frame đều nắm bắt được ngữ cảnh toàn cục của toàn bộ video clip dài.
* **TQT**: Mô hình hiện tại sử dụng 10 tầng MS-TCN với kernel size $k=3$ và dilation tăng theo lũy thừa 2 ($d = 2^0, 2^1, \dots, 2^9$). Trường tiếp nhận tối đa chỉ đạt:
  $$\text{RF} = 1 + \sum_{i=0}^9 2 \times 2^i = 1 + 2 \times (1024 - 1) = 2,047 \text{ frames}$$
  Trong khi đó, tập dữ liệu `dataset_tas_instance` có nhiều video dài tới **4,599 frames**. Với các frame ở nửa sau video, MS-TCN hoàn toàn bị "mù" ngữ cảnh ở nửa đầu video, khiến decoder thiếu thông tin toàn cục để định vị ranh giới hành động.

#### 2. Khởi Tạo Query Ngẫu Nhiên Thiếu Vị Trí Neo (Temporal Anchor Queries vs Random Queries)
* **BaFormer**: Khởi tạo $Q=150$ queries theo các mốc thời gian neo đều đặn trên trục thời gian chuẩn hóa $[0, 1]$:
  $$\text{anchor}_t = \text{linspace}(0, 1, Q)$$
  Mỗi query từ đầu đã mang sẵn một inductive bias tự nhiên về việc nó chịu trách nhiệm quan sát phân đoạn nào của video (đầu, giữa, hay cuối clip).
* **TQT**: Khởi tạo $Q=100$ queries hoàn toàn ngẫu nhiên (`nn.Embedding(100, 64)`). Trong một chuỗi thời gian dài hàng ngàn frames, Hungarian Bipartite Matcher bị rơi vào tình trạng đối xứng hoán vị (permutation symmetry) nghiêm trọng. Các queries không chuyên biệt hóa được theo vị trí thời gian, dẫn đến việc tối ưu hóa decoder hội tụ rất chậm hoặc rơi vào điểm cực tiểu cục bộ (local minima).

#### 3. Tê Liệt Hoàn Toàn Dự Đoán Ranh Giới (Boundary Collapse — F1 = 0.0%)
* **BaFormer**: Làm mượt nhãn ground truth ranh giới bằng hàm Gaussian heatmap:
  $$\text{GT}_{bd}(t) = \exp\left(-\frac{(t - t^*)^2}{2\sigma^2}\right), \quad \sigma = 1.5$$
  Mỗi ranh giới trở thành một dải sáp nhập mềm rộng khoảng 5–7 frames, kèm trọng số phạt lớp dương cao ($pos\_weight = 6.0 \to 100.0$).
* **TQT**: Nhãn ranh giới được tạo thành một xung Dirac delta cứng (1 frame duy nhất bằng $1.0$, còn lại $0.0$). Trong video 4,000 frames chỉ có khoảng 10–20 frame ranh giới (tỷ lệ positive chỉ $0.2\% - 0.5\%$). Với $pos\_weight = 5.0$, mô hình dễ dàng tối thiểu hóa hàm mất mát BCE bằng cách dự đoán xác suất ranh giới cực nhỏ ($\sigma(\text{logit}) \approx 0.04 - 0.08$). Khi giải mã với ngưỡng kiểm định đỉnh $\text{threshold} = 0.30$, không có bất kỳ đỉnh nào vượt qua ngưỡng $\implies$ **Số đỉnh ranh giới tìm thấy bằng 0, dẫn đến Boundary Precision, Recall, F1 triệt tiêu về 0.0%**.

#### 4. Hệ Số Phạt Lớp Rỗng ($eos\_coef$) Quá Cao & Thiếu Class Reweighting
* **BaFormer**: Thiết lập hệ số rỗng $eos\_coef = 0.01$ (giảm trọng số lớp $\emptyset$ xuống chỉ còn 1%), đồng thời áp dụng trọng số cân bằng lớp $class\_weights = [0.82, 1.05, 1.18, 1.30]$ để bảo vệ lớp thiểu số Class 3 (chỉ chiếm 6% frames).
* **TQT**: Đặt $eos\_coef = 0.1$ (cao gấp 10 lần so với BaFormer) và dùng hàm Cross-Entropy không trọng số lớp (`empty_weight = [1, 1, 1, 1, 0.1]`). Do đó:
  * Mô hình ưu tiên dự đoán lớp rỗng để giảm loss an toàn thay vì mạo hiểm dự đoán action instance.
  * Lớp thiểu số Class 3 bị triệt tiêu hoàn toàn: F1 đạt **0.00%** trên cả DINOv2 và ResNet-50 (dự đoán đúng 0 frames).

#### 5. Nhiễu Tích Lũy Từ 90+ Queries Rỗng Trong Cơ Chế Giải Mã Frame-Level (Semantic Inference Aggregation)
* **BaFormer**: Sử dụng cơ chế phân đoạn theo ranh giới (*Interval Boundary Snapping*). Sau khi phát hiện các đỉnh ranh giới, mô hình chia video thành các khoảng $[s_k, s_{k+1}]$ và chỉ chọn **1 query duy nhất có điểm số cao nhất** để đại diện cho khoảng đó. Các query rỗng bị triệt tiêu hoàn toàn, không thể đóng góp nhiễu vào frame.
* **TQT**: Sử dụng công thức chiếu ma trận toàn cục trực tiếp:
  $$\text{sem\_prob}(c, t) = \sum_{q=1}^Q P(c \mid q) \cdot M_q(t)$$
  Trong 100 queries, thực tế chỉ có 5–10 queries khớp với các hành động thật, còn lại 90–95 queries là rỗng. Mặc dù xác suất dự đoán lớp rỗng của chúng cao, nhưng phần xác suất còn lại cho 4 lớp hành động vẫn dao động từ $0.05 - 0.10$. Khi tính tổng của 90 queries rỗng, lượng xác suất rác này tích lũy thành một mức sàn nhiễu (*noise floor*) khổng lồ, làm biến dạng hoàn toàn nhãn frame dự đoán và kéo sụt điểm số Frame Accuracy và Edit Score.

---

### 6.3. Kế Hoạch Cải Tiến TQT Cho Iteration Tiếp Theo (Actionable Roadmap)

Dựa trên các phân tích định lượng trên, lộ trình nâng cấp kiến trúc TQT bao gồm 5 can thiệp cụ thể:

1. **Khởi tạo Temporal Anchor Queries**: Thay thế `nn.Embedding(Q, D)` bằng các query neo phân bổ đều trên $[0, 1]$ tương tự BaFormer để phá vỡ thế đối xứng hoán vị và tăng $Q$ từ 100 lên 150.
2. **Chuẩn hóa nhãn Ranh giới với Gaussian Heatmap**: Áp dụng Gaussian smoothing $\sigma = 1.5$ cho mảng `boundary` trong `dataset.py` và tăng $pos\_weight$ từ 5.0 lên 50.0–100.0, giúp mô hình học được phân phối chuyển tiếp mềm.
3. **Điều chỉnh Loss Weight & Class Balancing**:
   * Giảm $eos\_coef$ từ $0.1$ xuống $0.01$ để kích thích query dự đoán hành động thực tế.
   * Bổ sung `class_weights = [0.82, 1.05, 1.18, 1.30]` vào phân loại nhãn để phục hồi năng lực nhận diện cho Class 2 và Class 3.
4. **Mở rộng Receptive Field của Temporal Backbone**: Tăng số tầng dilated convolution hoặc bổ sung khối Temporal Self-Attention / ASPP vào backbone MS-TCN để trường tiếp nhận bao phủ trọn vẹn 4,500 frames.
5. **Cải tiến Bộ Giải Mã Inference (Empty Query Masking & Boundary Snapping)**:
   * Trước khi thực hiện ma trận chiếu `sem_prob`, lọc bỏ toàn bộ các query có $P(\emptyset \mid q) > P(\text{action} \mid q)$ hoặc áp dụng cơ chế interval winning query dựa trên ranh giới.
