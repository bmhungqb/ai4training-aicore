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

### Phân Bổ Lớp Dữ Liệu & Ranh Giới (Validation):
* **Class 0 (Sewing/Joining)**: 5,846 frames (44.6%) — Lớp hành động chính
* **Class 1 (Positioning/Handling)**: 4,505 frames (34.3%) — Lớp thao tác chuẩn bị vải
* **Class 2 (Adjustment/Alignment/Preparation)**: 1,976 frames (15.1%) — Lớp vi chỉnh
* **Class 3 (Inspection/Auxiliary)**: 794 frames (6.0%) — Lớp thiểu số (kiểm tra đường may)
* **Tổng số ranh giới thực tế (GT Boundaries)**: **443 ranh giới** trên 10 video clips validation
* **Độ dung sai ranh giới (Tolerance)**: $\pm 3$ frames ($\approx 0.2$ giây ở 15 fps)

---

## 2. Bảng Xếp Hạng Tổng Hợp (Benchmark Leaderboard)

Xếp hạng toàn bộ 12 cấu hình thực nghiệm, tập trung trực tiếp vào **Năng Lực Định Vị Ranh Giới (Boundary Detection)** và **Chất Lượng Phân Đoạn (Segmentation F1 & Edit)**:

| Hạng | Cấu hình Thử nghiệm | Mô hình | Feature Dim | Best Ep / Total | Boundary F1@3 (%) *(Cuối)* | Boundary Recall@3 (%) | Boundary Precision@3 (%) | F1 Mean (%) | F1@10 (%) | F1@25 (%) | F1@50 (%) | Frame Acc (%) | Edit Score | Thư mục Checkpoint & Logs |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 🥇 | **Clean DINOv2** | BaFormer | 768d | 99 / 144 | **33.80%** | **49.21%** | 25.74% | **28.15%** | 41.76% | 30.16% | 12.53% | 50.67% | 51.29 | [`clean_dinov2/1/`](file:///home/hungbm/ai4training/ai4training-aicore/src/TAS-instance-style/BaFormer/experiments/tas_instance/bk_fde_tde/clean_dinov2/1/) |
| 🥈 | **Over-eng DINOv2** | BaFormer | 768d | 97 / 142 | **33.71%** | **50.56%** | 25.28% | **31.76%** | **45.03%** | **33.51%** | **16.75%** | 50.38 | **54.08** | [`exp_dinov2/1/`](file:///home/hungbm/ai4training/ai4training-aicore/src/TAS-instance-style/BaFormer/experiments/tas_instance/bk_fde_tde/exp_dinov2/1/) |
| 🥉 | **Over-eng Fusion (D2+MAE)** | BaFormer | 1536d | 27 / 72 | **28.94%** | 32.96% | 25.80% | 28.65% | 39.79% | 32.36% | 13.79% | **56.36%** | 46.90 | [`exp_dinov2_videomae/1/`](file:///home/hungbm/ai4training/ai4training-aicore/src/TAS-instance-style/BaFormer/experiments/tas_instance/bk_fde_tde/exp_dinov2_videomae/1/) |
| 4 | **Clean DINOv3** | BaFormer | 768d | 43 / 88 | **20.88%** | 18.74% | 23.58% | 27.48% | 39.89% | 30.85% | 11.70% | 53.24 | 51.61 | [`clean_dinov3/1/`](file:///home/hungbm/ai4training/ai4training-aicore/src/TAS-instance-style/BaFormer/experiments/tas_instance/bk_fde_tde/clean_dinov3/1/) |
| 5 | **Over-eng DINOv3** | BaFormer | 768d | 30 / 75 | **15.32%** | 11.51% | 22.87% | 26.11% | 37.78% | 26.11% | 14.44% | 51.19 | 43.76 | [`exp_dinov3/1/`](file:///home/hungbm/ai4training/ai4training-aicore/src/TAS-instance-style/BaFormer/experiments/tas_instance/bk_fde_tde/exp_dinov3/1/) |
| 6 | **Over-eng VideoMAE** | BaFormer | 768d | 87 / 132 | **15.13%** | 10.16% | 29.61% | 28.27% | 41.67% | 29.90% | 13.24% | 50.13 | 46.66 | [`exp_videomae/1/`](file:///home/hungbm/ai4training/ai4training-aicore/src/TAS-instance-style/BaFormer/experiments/tas_instance/bk_fde_tde/exp_videomae/1/) |
| 7 | **Clean VideoMAE** | BaFormer | 768d | 125 / 170 | **15.01%** | 9.71% | **33.08%** | 27.66% | 39.01% | 28.02% | 15.93% | 46.22 | 49.29 | [`clean_videomae/1/`](file:///home/hungbm/ai4training/ai4training-aicore/src/TAS-instance-style/BaFormer/experiments/tas_instance/bk_fde_tde/clean_videomae/1/) |
| 8 | **Over-eng ResNet-50** | BaFormer | 2048d | 30 / 75 | **10.92%** | 7.22% | 22.38% | 27.26% | 39.06% | 28.12% | 14.58 | 48.86 | 48.73 | [`exp_resnet50/1/`](file:///home/hungbm/ai4training/ai4training-aicore/src/TAS-instance-style/BaFormer/experiments/tas_instance/bk_fde_tde/exp_resnet50/1/) |
| 9 | **TQT VideoMAE** | TQT | 768d | 37 / 72 | **0.00%** | 0.00% | 0.00% | 22.64% | 33.97% | 24.72% | 9.22 | 45.67 | 44.70 | [`experiments/videomae/1/`](file:///home/hungbm/ai4training/ai4training-aicore/src/TAS-instance-style/TQT/experiments/videomae/1/) |
| 10 | **TQT ResNet-50** | TQT | 2048d | 41 / 76 | **0.00%** | 0.00% | 0.00% | 19.10% | 29.45% | 20.42 | 7.44 | 50.99 | 41.60 | [`experiments/resnet50/1/`](file:///home/hungbm/ai4training/ai4training-aicore/src/TAS-instance-style/TQT/experiments/resnet50/1/) |
| 11 | **TQT DINOv2** | TQT | 768d | 10 / 45 | **0.00%** | 0.00% | 0.00% | 15.91% | 25.17% | 17.26 | 5.30 | 42.16 | 42.75 | [`experiments/dinov2/1/`](file:///home/hungbm/ai4training/ai4training-aicore/src/TAS-instance-style/TQT/experiments/dinov2/1/) |
| 12 | **TQT DINOv3** | TQT | 768d | 64 / 99 | **0.00%** | 0.00% | 0.00% | 14.15% | 22.82 | 13.46 | 6.18 | 40.03 | 42.59 | [`experiments/dinov3/1/`](file:///home/hungbm/ai4training/ai4training-aicore/src/TAS-instance-style/TQT/experiments/dinov3/1/) |

---

## 3. Đánh Giá Toàn Diện Năng Lực Định Vị Ranh Giới (Boundary Detection In-Depth Analysis)

Do bài toán **Instance-Level Temporal Action Segmentation (ITAS)** đòi hỏi phải tách biệt các thao tác lặp lại của cùng một lớp hành động (ví dụ: `Sewing #1 | Sewing #2`), **ranh giới (Boundary)** là tín hiệu quyết định sự thành bại của toàn bộ hệ thống.

### 3.1. Bảng Đối Chiếu Chi Tiết Các Chỉ Số Ranh Giới

| Cấu hình Thử nghiệm | Mô hình | Boundary Loss | Boundary F1@3 (%) | Boundary Recall@3 (%) | Boundary Precision@3 (%) | Số Ranh Giới Phát Hiện Đúng (TP) | Số Cắt Nhầm (FP) | Số Cắt Bị Bỏ Sót (FN) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clean DINOv2** | BaFormer | **0.6368** | **33.80%** | **49.21%** | 25.74% | **218 / 443** | 629 | 225 |
| **Over-eng DINOv2** | BaFormer | 1.0258 | 33.71% | **50.56%** | 25.28% | **224 / 443** | 664 | 219 |
| **Over-eng Fusion** | BaFormer | 1.0196 | 28.94% | 32.96% | 25.80% | 146 / 443 | 420 | 297 |
| **Clean DINOv3** | BaFormer | 0.6426 | 20.88% | 18.74% | 23.58% | 83 / 443 | 269 | 360 |
| **Over-eng DINOv3** | BaFormer | 1.0191 | 15.32% | 11.51% | 22.87% | 51 / 443 | 172 | 392 |
| **Over-eng VideoMAE** | BaFormer | 1.0232 | 15.13% | 10.16% | 29.61% | 45 / 443 | 107 | 398 |
| **Clean VideoMAE** | BaFormer | 0.6357 | 15.01% | 9.71% | **33.08%** | 43 / 443 | **87** | 400 |
| **Over-eng ResNet-50** | BaFormer | 1.0252 | 10.92% | 7.22% | 22.38% | 32 / 443 | 111 | 411 |
| **TQT VideoMAE** | TQT | 0.2030 | 0.00% | 0.00% | 0.00% | 0 / 443 | 0 | 443 |
| **TQT ResNet-50** | TQT | 0.2025 | 0.00% | 0.00% | 0.00% | 0 / 443 | 0 | 443 |
| **TQT DINOv2** | TQT | 0.2060 | 0.00% | 0.00% | 0.00% | 0 / 443 | 0 | 443 |
| **TQT DINOv3** | TQT | 0.2023 | 0.00% | 0.00% | 0.00% | 0 / 443 | 0 | 443 |

---

### 3.2. Phân Tích Hiện Tượng & Cơ Chế Định Vị Ranh Giới

#### 1. DINOv2 Thống Trị Tuyệt Đối Về Nhận Diện Ranh Giới (Recall ~50%)
* **Độ sắc nét không gian (Spatial Feature Granularity)**: DINOv2 duy trì độ phân giải đặc trưng không gian cực kỳ chi tiết của từng frame đơn lẻ. Khi tay công nhân rời khỏi bàn máy may hoặc khi kéo vải để may đường tiếp theo, vector đặc trưng của DINOv2 thay đổi đột ngột giữa 2 frame liên tiếp, tạo ra gradient thời gian ($\Delta F_t = \|f_t - f_{t-1}\|$) rất mạnh.
* Kết quả: DINOv2 bắt được **218 – 224 trên tổng số 443 ranh giới thực tế** (Recall đạt **49.21% – 50.56%**), cao gấp **$2.6\times$ DINOv3**, gấp **$5\times$ VideoMAE**, và gấp **$7\times$ ResNet-50**.

#### 2. DINOv3: Bứt Phá Lớn Trên Clean Baseline Nhưng Vẫn Xếp Sau DINOv2 Về Ranh Giới
* Trên mô hình cũ (Over-engineered), DINOv3 bị phạt bởi Boundary Barrier Attention khiến ranh giới bị nén, Recall chỉ đạt **11.51%** (F1 15.32%).
* Khi đưa về Clean Baseline, Boundary Recall tăng lên **18.74%** và Boundary F1 đạt **20.88% (+5.56%)**, bắt được 83 ranh giới. Tuy nhiên, do đặc trưng pretrain của DINOv3 tối ưu hóa ngữ nghĩa cấp cao (semantic clustering) mạnh hơn biểu diễn pixel-level cục bộ, sự chuyển dịch giữa 2 frame liền kề êm hơn DINOv2, dẫn đến nhiều ranh giới vi mô bị bỏ qua.

#### 3. VideoMAE: Precision Ranh Giới Cao Nhất Nhưng Bị Rào Cản "Temporal Smearing"
* **Hiện tượng làm mờ theo thời gian**: VideoMAE áp dụng cơ chế 3D Spatio-Temporal Tubelet Embedding (16 frames gộp thành 1 tubelet). Cơ chế này gom thông tin động lực học của cả cửa sổ 16 frame, khiến điểm chuyển tiếp ranh giới chính xác ở frame $t$ bị dàn đều sang các frame lân cận $t \pm 8$.
* Hệ quả:
  * **Precision cao nhất (33.08%)**: Khi VideoMAE báo có ranh giới, độ tin cậy rất cao (chỉ cắt nhầm 87 lần).
  * **Recall thấp nhất (9.71%)**: VideoMAE bỏ sót tới 400 trên 443 ranh giới, không thể xác định điểm cắt sắc nét ở mức độ frame-level ($\pm 3$ frames).

#### 4. ResNet-50: Tê Liệt Hoàn Toàn Khả Năng Tách Phân Đoạn
* Đặc trưng 2D thuần túy không có attention toàn cục, thiếu khả năng phân biệt sự thay đổi tinh tế của thao tác tay. Recall ranh giới chỉ đạt **7.22%**, gần như toàn bộ video bị gộp thành một khối đồng nhất của lớp đa số.

#### 5. Tại Sao Toàn Bộ Các Chạy Của TQT Đều Bị Boundary F1 = 0.0%?
* **Cơ chế nhãn Dirac delta cứng**: TQT gán nhãn ranh giới bằng giá trị $1.0$ tại đúng 1 frame duy nhất, trong khi BaFormer dùng **Gaussian Heatmap** ($\sigma = 1.5$) để phân bổ xác suất mềm sang các frame lân cận.
* **Mất cân bằng cực độ & Pos Weight thấp**: Trong video 4,000 frames chỉ có ~10-20 frames ranh giới ($0.2\%$). Với trọng số phạt $pos\_weight = 5.0$, mô hình TQT đạt loss tối ưu bằng cách dự đoán xác suất ranh giới cho mọi frame ở mức $\approx 0.04 - 0.08$. Khi giải mã với ngưỡng NMS threshold $0.30$, không có điểm nào vượt qua ngưỡng $\implies$ **TP = 0, FP = 0, FN = 443, dẫn đến Boundary Precision = Recall = F1 = 0.0%**.

---

## 4. Ma Trận Chi Tiết Từng Lớp Hành Động (Per-Class Performance Matrix)

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

## 5. Nghiên Cứu Bóc Tách: Over-Engineered vs Clean Baseline (Ablation Study)

Đối chiếu trực tiếp tác động của việc gỡ bỏ 5 hàm mất mát phụ (`contra`, `repulse`, `enc_ce`, `enc_smooth`, `mask_tv`) và các tầng hậu xử lý heuristic trong BaFormer:

| Tiêu chí Đánh giá | DINOv3 Cũ $\to$ **DINOv3 Sạch** | DINOv2 Cũ $\to$ **DINOv2 Sạch** | VideoMAE Cũ $\to$ **VideoMAE Sạch** |
| :--- | :---: | :---: | :---: |
| **Boundary F1@3 Cuối** | **15.32% $\to$ 20.88% (+5.56%)** 🚀 | 33.71% $\to$ **33.80% (+0.09%)** | 15.13% $\to$ 15.01% (-0.12%) |
| **Boundary Recall@3** | **11.51% $\to$ 18.74% (+7.23%)** | 50.56% $\to$ 49.21% (-1.35%) | 10.16% $\to$ 9.71% (-0.45%) |
| **Boundary Precision@3** | 22.87% $\to$ **23.58% (+0.71%)** | 25.28% $\to$ **25.74% (+0.46%)** | 29.61% $\to$ **33.08% (+3.47%)** |
| **Frame Accuracy** | **51.19% $\to$ 53.24% (+2.05%)** | 50.38% $\to$ 50.67% (+0.29%) | 50.13% $\to$ 46.22% (-3.91%) |
| **Edit Score** | **43.76 $\to$ 51.61 (+7.85)** 🏆 | 54.08 $\to$ 51.29 (-2.79) | 46.66 $\to$ 49.29 (+2.63) |
| **F1 Mean** | **26.11% $\to$ 27.48% (+1.37%)** | 31.76% $\to$ 28.15% (-3.61%) | 28.27% $\to$ 27.66% (-0.61%) |
| **Dự đoán Class 2 (Adjustment)** | **5,347 $\to$ 3,465 frames (-1,882 FP)** | 3,745 $\to$ 5,610 frames | 4,472 $\to$ 4,816 frames |
| **Loss Tổng thể (Validation)** | 9.42 $\to$ **8.08 (Giảm mạnh)** | 8.44 $\to$ **7.49 (Giảm mạnh)** | 8.40 $\to$ **7.37 (Giảm mạnh)** |

---

## 6. Đánh Giá Chuyên Sâu Mô Hình Mới TQT (Temporal Query Transformer)

### 6.1. So Sánh Trực Tiếp TQT vs BaFormer Clean Baseline

Khi đối chiếu trên cùng một backbone đặc trưng giữa **TQT Baseline** và **BaFormer Clean Baseline**:

| Tiêu chí | DINOv3: TQT vs Clean BaFormer | DINOv2: TQT vs Clean BaFormer | VideoMAE: TQT vs Clean BaFormer |
| :--- | :---: | :---: | :---: |
| **Boundary F1@3** | 0.00% vs **20.88% (-20.88%)** | 0.00% vs **33.80% (-33.80%)** | 0.00% vs **15.01% (-15.01%)** |
| **Boundary Recall@3** | 0.00% vs **18.74% (-18.74%)** | 0.00% vs **49.21% (-49.21%)** | 0.00% vs **9.71% (-9.71%)** |
| **F1 Mean** | 14.15% vs **27.48% (-13.33%)** | 15.91% vs **28.15% (-12.24%)** | 22.64% vs **27.66% (-5.02%)** |
| **Frame Accuracy** | 40.03% vs **53.24% (-13.21%)** | 42.16% vs **50.67% (-8.51%)** | 45.67% vs **46.22% (-0.55%)** |
| **Edit Score** | 42.59 vs **51.61 (-9.02)** | 42.75 vs **51.29 (-8.54)** | 44.70 vs **49.29 (-4.59)** |
| **Class 3 F1 (Inspection)** | 9.34% vs **3.34% (+6.00%)** | 0.00% vs **37.30% (-37.30%)** | 0.79% vs **11.11% (-10.32%)** |

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

---

## 7. Benchmark Mới: Tập Dữ Liệu `dataset` (Chunk-Based Temporal Backbones)

### 7.1. Bối Cảnh & Khác Biệt So Với `dataset_tas_instance`

Song song với benchmark ở các mục 1–6 (dựa trên `dataset_tas_instance`, 98 clips, đặc trưng trích xuất **frame-by-frame** bằng ResNet-50/DINOv2/DINOv3/VideoMAE-sliding-window), một chu kỳ thực nghiệm độc lập đã được thực hiện trên tập dữ liệu **`dataset`** mới — xây dựng lại từ `original_data/processed_data/` + `balanced_split`, với 2 khác biệt cốt lõi:

1. **Nhãn 4 lớp, KHÔNG có class `background`**: Khác với `dataset_tas_instance` (có lớp "ranh giới rỗng"/gap), script `build_dataset.py` hấp thụ mọi khoảng trống (chủ yếu là đoạn đuôi video dài hơn đoạn annotation cuối, tới ~6.8s/100 frames) vào instance liền kề gần nhất thay vì tạo nhãn riêng — do đó mọi frame đều có nhãn hành động thật.
2. **Đặc trưng trích xuất theo CHUNK (clip-based), không phải frame-by-frame**: Với mỗi frame $t$, một cửa sổ `clip_len` frame liên tiếp CENTERED tại $t$ được đưa qua một backbone không-thời gian (3D CNN / Video Transformer) thực sự, rồi global-average-pool thành 1 vector/frame — khác hẳn cách ResNet-50/DINOv2/DINOv3 cũ xử lý 1 frame đơn lẻ mỗi lần. Ngoại lệ: **DINOv2** trên tập `dataset` mới vẫn chạy **frame-by-frame** (2D ViT thuần, không chunk) để làm đối chứng trực tiếp với các backbone chunk-based.
3. **ROI Mask Cropping**: Mọi frame được cắt theo bounding-box của mask ROI (`<source_video>.mask.png`, ~70–90% diện tích khung hình) trước khi resize, loại bỏ nền/viền không liên quan đến trạm máy may.

**Dữ liệu**: 86 video clips (69 train / 17 val=test, `balanced_split`).
**Phân bổ lớp (Validation, 17 clips, 19,156 frames)**:
* Class 0 (Preparation): 2,098 frames (11.0%)
* Class 1 (Positioning/Adjustment/Alignment): 10,981 frames (57.3%) — lớp đa số
* Class 2 (Sewing/Joining/Handling): 4,373 frames (22.8%)
* Class 3 (Inspection/Auxiliary): 1,704 frames (8.9%) — lớp thiểu số
* **Tổng số ranh giới thực tế (GT Boundaries, internal transitions)**: **612 ranh giới** trên 17 video clips validation
* **Độ dung sai ranh giới**: $\pm 3$ frames

**Mô hình & Cấu hình**: BaFormer Clean Baseline (`bk_fde_tde`), giống cấu hình mục 1–6 — CE + Mask + Dice + Standard Boundary losses, 5 loss phụ Proposal 04–14 tắt hoàn toàn (`contra/repulse/enc_ce/enc_smooth/mask_tv = 0.0`), `class_weights = [1.0, 1.0, 1.15, 1.25]` (4 lớp, không có background), `batch_size=4`, `base_lr=0.0005`, `early_stopping_patience=45`, 300 epochs max — **hyperparameters giống nhau 100% giữa 5 backbone** để đảm bảo so sánh công bằng.

### 7.2. Bảng Xếp Hạng (5 Backbones trên `dataset`)

| Hạng | Backbone | Loại trích xuất | Feature Dim | Best Ep / Total | Boundary F1@3 (%) | Boundary Recall@3 (%) | Boundary Precision@3 (%) | F1 Mean (%) | F1@10 (%) | F1@25 (%) | F1@50 (%) | Frame Acc (%) | Edit Score | Composite |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **S3D** | Chunk (3D CNN), 16f@224² | 1024d | 119 / 164 | **29.82%** | **36.93%** | 25.00% | **38.11%** | **53.05%** | **41.84%** | **19.43%** | **55.58%** | **61.08** | **50.24** |
| 🥈 | **I3D-R50** | Chunk (3D CNN), 8f@224² | 2048d | 122 / 167 | 27.09% | 33.66% | 22.66% | 39.27% | 53.90% | 44.15% | 19.76% | 52.97% | 61.80 | 50.14 |
| 🥉 | **MViT-v1-B** | Chunk (ViT-temporal), 16f@224² | 768d | 82 / 127 | 22.61% | 21.24% | 24.16% | **40.28%** | 54.33% | 45.20% | **21.31%** | **56.83%** | 59.97 | **51.15** |
| 4 | **VideoMAE** | Chunk (ViT-temporal), 16f@224² | 768d | **1 / 46** | 24.10%* | 22.88%* | 25.45%* | 22.12% | 35.14% | 23.35% | 7.86% | 48.43% | 53.08 | 39.30 |
| 5 | **DINOv2** | Frame-by-frame (2D ViT), 1f@224² | 768d | 70 / 115 | 9.09% | 5.39% | 28.95% | 35.75% | 50.70% | 39.75% | 16.82% | 51.92% | 58.19 | 47.33 |

`*` VideoMAE's best epoch is epoch 1 (model never improved over 45 subsequent epochs → early-stopped); its boundary numbers come from that single epoch, not a converged/stable state — see §7.4.2.

### 7.3. Ma Trận Chi Tiết Từng Lớp Hành Động (Per-Class Performance, tại Best Epoch)

| Backbone | Class 0: Preparation<br>*(2,098 frames, 11.0%)* | Class 1: Positioning/Adjustment/Alignment<br>*(10,981 frames, 57.3%)* | Class 2: Sewing/Joining/Handling<br>*(4,373 frames, 22.8%)* | Class 3: Inspection/Auxiliary<br>*(1,704 frames, 8.9%)* |
| :--- | :---: | :---: | :---: | :---: |
| **S3D** | P: 38.25% \| R: 19.16% \| **F1: 25.53%** | P: 67.23% \| R: 65.30% \| **F1: 66.25%** | P: 39.14% \| R: **57.79%** \| **F1: 46.67%** 🏆 | P: 28.08% \| R: 16.20% \| **F1: 20.54%** |
| **I3D-R50** | P: 36.62% \| R: 10.44% \| **F1: 16.25%** | P: 62.86% \| R: **68.64%** \| **F1: 65.62%** | P: 35.98% \| R: 36.91% \| **F1: 36.44%** | P: 23.59% \| R: **28.81%** \| **F1: 25.94%** |
| **MViT-v1-B** | P: **37.92%** \| R: 9.06% \| **F1: 14.62%** | P: 64.83% \| R: **76.44%** \| **F1: 70.16%** 🏆 | P: **41.26%** \| R: 44.77% \| **F1: 42.94%** | P: 27.34% \| R: 15.43% \| **F1: 19.73%** |
| **VideoMAE** | P: 6.91% \| R: 1.91% \| **F1: 2.99%** | P: 57.38% \| R: 65.24% \| **F1: 61.06%** | P: 24.35% \| R: 32.24% \| **F1: 27.75%** | P: 23.18% \| R: 4.11% \| **F1: 6.98%** |
| **DINOv2** | P: 31.37% \| R: 7.63% \| **F1: 12.27%** | P: **64.92%** \| R: 64.03% \| **F1: 64.47%** | P: 38.99% \| R: 40.64% \| **F1: 39.79%** | P: 21.58% \| R: **41.26%** \| **F1: 28.34%** 🏆 |

### 7.4. Phân Tích Hiện Tượng

#### 7.4.1. Backbone Chunk-Based (3D/Temporal) Vượt Trội So Với Frame-by-Frame DINOv2
Khác với benchmark `dataset_tas_instance` cũ (nơi DINOv2 frame-by-frame thống trị tuyệt đối về Boundary Recall, §3.2), trên tập `dataset` mới, **3 trong 4 backbone chunk-based (S3D, I3D-R50, MViT-v1-B) đều vượt DINOv2 ở hầu hết chỉ số**, đặc biệt là Boundary Recall (21–37% so với 5.39% của DINOv2) và F1 Mean (38–40% so với 35.75%). Nguyên nhân khả dĩ:
* **ROI mask cropping** đã loại bỏ phần lớn nền/nhiễu không liên quan trước khi đưa vào backbone, nên các đặc trưng không-thời gian (3D CNN/Video Transformer) tận dụng được tốt chuyển động tay/vải trong vùng ROI hẹp hơn, sắc nét hơn.
* **S3D và I3D-R50** là các mô hình nhận dạng hành động (action recognition) kinh điển, pretrained trên Kinetics-400 — vốn được thiết kế để nắm bắt chính xác các chuyển động lặp lại ngắn (short repeated motions), phù hợp tự nhiên với đặc trưng "các thao tác may lặp lại" của bài toán này.
* **DINOv2** (image-only pretraining, không có temporal context) phải suy luận chuyển động chỉ dựa trên sự khác biệt giữa 2 frame liên tiếp — khi ROI đã bị crop hẹp, biến động giữa các frame cũng giảm theo, làm giảm độ nhạy ranh giới tương đối so với lúc chạy trên full-frame (so sánh với 49–50% Boundary Recall của DINOv2 trên `dataset_tas_instance` cũ).

#### 7.4.2. VideoMAE (Chunk Mode) Bị Mất Ổn Định — Hội Tụ Ngay Epoch 1, Không Cải Thiện Thêm
VideoMAE là trường hợp bất thường duy nhất: đạt **Boundary F1 24.10%** ngay tại epoch 1 (cao thứ 2 trong 5 backbone), nhưng sau đó dao động mạnh và KHÔNG BAO GIỜ vượt lại điểm composite của epoch 1 trong suốt 45 epoch tiếp theo → bị early-stopping kích hoạt ở epoch 46. Theo dõi log chi tiết (`log_plain.txt`) cho thấy Boundary F1 rơi về 0.00–11.15% ở phần lớn các epoch sau, không có xu hướng hồi phục rõ ràng. Hai giả thuyết khả dĩ (chưa kiểm chứng sâu, cần thêm probe ở Diagnoser):
1. **Learning rate quá cao cho đặc trưng VideoMAE-chunk**: VideoMAE sliding-window 16-frame có độ "trơn" (smoothness) thời gian cao hơn S3D/I3D/MViT (tái khẳng định hiện tượng "Temporal Smearing" đã ghi nhận ở §3.2.3 cho VideoMAE frame-by-frame cũ) — khiến gradient ban đầu mạnh nhưng dễ dao động, không ổn định quanh một optimum tốt.
2. **Mismatch giữa input_dim=768 và đặc trưng cụ thể của backbone**: cùng input_dim 768 với MViT-v1-B nhưng động lực huấn luyện khác biệt hoàn toàn (MViT hội tụ ổn định đến epoch 82) — gợi ý vấn đề nằm ở đặc trưng/pretraining của VideoMAE chứ không phải ở kiến trúc ASFormerEncoder phía sau.

#### 7.4.3. S3D Đạt Boundary F1 Cao Nhất, MViT-v1-B Đạt F1 Mean/Frame Acc/Composite Cao Nhất
Không có một backbone áp đảo tuyệt đối toàn bộ chỉ số — có sự đánh đổi rõ:
* **S3D**: tối ưu cho **Boundary Detection** (F1@3 = 29.82%, Recall = 36.93% — cao nhất trong 5 backbone) nhờ Recall ranh giới vượt trội, dù Precision chỉ 25.00%.
* **MViT-v1-B**: tối ưu cho **chất lượng phân đoạn tổng thể** (F1 Mean 40.28%, Frame Acc 56.83%, Composite Score 51.15 — cao nhất) nhưng Boundary Recall thấp hơn hẳn (21.24%), cho thấy mô hình dự đoán đúng *nhãn lớp* tốt hơn nhưng kém nhạy với *thời điểm chuyển tiếp chính xác*.
* **I3D-R50**: cân bằng tốt giữa 2 nhóm chỉ số trên, đứng hạng 2 ở cả Boundary F1 (27.09%) và gần như ngang MViT ở F1 Mean (39.27%) — là lựa chọn "an toàn" nếu cần một backbone vừa định vị ranh giới tốt vừa phân loại nhãn ổn.

#### 7.4.4. Nhãn 4-Lớp Không-Background Giúp Class 2 (Sewing) Khá Hơn Hẳn So Với Benchmark Cũ
So với `dataset_tas_instance` cũ (Class 2 "Adjustment/Prep" F1 chỉ 24–30% ở mọi backbone, §4), trên tập `dataset` mới, **Class 2 (Sewing/Joining/Handling)** đạt F1 36–47% ở cả 5 backbone (cao nhất: S3D 46.67%) — một phần vì support frame của lớp này lớn hơn tương đối (22.8% vs 15.1% cũ), và một phần vì việc loại bỏ nhãn background giúp model không phải học thêm một "lớp nhiễu" chiếm ~1.9% dữ liệu (đã được hấp thụ vào các lớp hành động liền kề, xem §7.1).

### 7.5. Khuyến Nghị Tiếp Theo

1. **Chẩn đoán sâu VideoMAE-chunk instability** (§7.4.2): trước khi kết luận VideoMAE-chunk kém hơn các backbone khác, cần thử giảm `base_lr` (ví dụ 0.0005 → 0.0001–0.0002) hoặc tăng `warmup_epochs`, chạy lại 1 lần để loại trừ khả năng mất ổn định do optimizer/learning-rate thuần túy trước khi quy kết nguyên nhân về đặc trưng.
2. **Ensemble/Fusion S3D + MViT-v1-B**: do 2 backbone này bù trừ lẫn nhau rất rõ (S3D mạnh Boundary Recall, MViT mạnh Frame Acc/F1 Mean), nên thử nối đặc trưng (`feat_folders: ['features_s3d', 'features_mvit_v1_b']`, concat theo kênh) tương tự cách tiếp cận "Over-eng Fusion (D2+MAE)" ở §2 trên tập `dataset_tas_instance` cũ.
3. **Tăng Boundary Precision cho S3D**: Precision chỉ 25.00% (678 cắt nhầm / 904 dự đoán) — có thể cân nhắc tăng `peak_prominence`/`min_distance` trong post-processing hoặc tăng `bd_weight` để model tự tin hơn khi báo ranh giới, đổi lại có thể giảm nhẹ Recall.
4. **So sánh trực tiếp DINOv2 frame-by-frame vs DINOv2-nếu-chunk**: hiện tại DINOv2 là backbone duy nhất không chạy ở chế độ chunk trên tập `dataset`, nên chưa thể tách bạch "DINOv2 kém hơn vì thiếu temporal context" hay "vì input ROI bị crop hẹp hơn" — nếu muốn kết luận chắc chắn hơn ở §7.4.1, cần thử chạy DINOv2 qua `extract_chunk_features.py` ở chế độ sliding-window tương tự VideoMAE để so sánh cùng điều kiện.
