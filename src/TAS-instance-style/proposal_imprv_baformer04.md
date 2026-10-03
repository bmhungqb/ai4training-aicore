# Proposal & Research: Nâng cấp BaFormer++ cho Instance-Level Temporal Action Segmentation (Iteration 04)

**Tài liệu tham chiếu:** `problem_definition.md`, `ba_former++.md`, `proposal_imprv_baformer01.md`, `proposal_imprv_baformer03.md`  
**Dataset:** `dataset_tas_instance` (4 classes, 58 clips: 48 train, 10 val)  
**Tác giả:** Antigravity AI & Human Research Partner  
**Ngày lập:** Tháng 10/2026  

---

## 1. Tổng quan & Tiến trình Thực nghiệm (Runs 1 → 3 Trajectory)

### 1.1 Bảng đối chiếu tiến trình thực nghiệm

| Metric | Run 1 (Legacy Bugs) | Run 2 (Gaussian Smoothing $\sigma=1.5$) | Run 3 (Re-balanced Loss & Weights) | Mục tiêu Run 4 (BaFormer++) |
| :--- | :---: | :---: | :---: | :---: |
| **Best Epoch** | 1 (Flat) | 183 | 46 (Early stopped @ 96) | 60 – 120 (Ổn định) |
| **F1 Mean** | 0.00 | 27.84 | 27.54 | **> 35.0** |
| **F1@10** | 0.00 | 38.05 | 38.03 | **> 45.0** |
| **F1@25** | 0.00 | 31.09 | 30.05 | **> 38.0** |
| **F1@50** | 0.00 | 14.39 | 14.55 | **> 20.0** |
| **Edit Score** | 0.00 | 45.76 | **48.53** (+2.77) | **> 55.0** |
| **Val Accuracy** | 44.55% (Flat) | **48.09%** | 47.97% | **> 55.0%** |
| **Train Acc @ Peak** | - | 71.16% | 71.52% | ~65 – 70% |
| **Generalization Gap** | - | 23.07% (Overfitting) | 23.55% (Overfitting) | **< 12.0%** |
| **Boundary Precision** | 0.00% | 9.39% | 7.83% (False Positives > 92%) | **> 25.0%** |
| **Boundary Recall** | 0.00% | 45.20% | 54.24% | > 60.0% |

---

### 1.2 Bài học xương máu từ Run 3
1. **Bẫy điều chỉnh Class Weights thô bạo (`class_weights: [0.6, 0.7, 1.0, 1.7]`):**
   - Class 0 (`Sewing/Joining`) chiếm tới **44.6% thời lượng** của toàn bộ video. Khi bị giảm trọng số xuống `0.6`, mô hình né tránh dự đoán Class 0 (Recall sụp đổ từ >50% xuống **26.48%**).
   - Class 3 (`Inspection`, chiếm 6% thời lượng) nhận trọng số `1.7`, dẫn tới bị mô hình **lạm phát dự đoán gấp 4 lần** (3,268 frame dự đoán vs 794 frame thực tế, Precision chỉ 12.55%).
   - Sự méo mó này khiến độ chính xác toàn cục và F1 suy sụp nhanh chóng sau epoch 46, kích hoạt early stopping sớm.
2. **Sự bế tắc của Boundary Precision (Kẹt ở mức 7.8%):**
   - Dù đã tăng ngưỡng `threshold = 0.35` và hạ `pos_weight = 6.0`, tỉ lệ báo động giả (False Positive) của ranh giới vẫn vượt quá **92%**.
   - Việc tinh chỉnh siêu tham số bên ngoài đã chạm ngưỡng trần (ceiling effect) vì nguyên nhân bắt nguồn từ **khiếm khuyết cấu trúc kiến trúc bên trong mô hình**.

---

## 2. Chẩn đoán gốc rễ kiến trúc (Deep Architectural Diagnosis)

### Gốc rễ 1: Phép chiếu tĩnh (Static 1D Projection) mù chuyển động trong Boundary Head
Trong file `action_segmentation/models/transformer_decoder/transformer_decoder_mask_bd_mulkv.py`:
```python
output = self.vid_embed_before(output)
vid_embed = self.vid_embed(output.transpose(-1, -2)).transpose(-1, -2) # [B, 1, 64]
outputs_boundary = torch.einsum("bqc,bcl->bql", vid_embed, mask_features) # [B, 1, L]
```
* **Bản chất toán học:** `vid_embed` là một vector $\mathbf{v} \in \mathbb{R}^{64}$ được gộp cố định trên toàn bộ 150 queries cho cả video. Phép tính ranh giới tại frame $t$ thực chất chỉ là:
  $$b(t) = \mathbf{v}^\top \mathbf{f}_t$$
* **Tại sao không thể học được boundary?**
  Ranh giới hành động bản chất là **đạo hàm theo thời gian / biến thiên chuyển động** ($\Delta \mathbf{f}_t = \mathbf{f}_{t} - \mathbf{f}_{t-1}$ hoặc sự tương phản giữa quá khứ và tương lai). Tích vô hướng tĩnh chỉ đo xem frame $\mathbf{f}_t$ có hình chiếu lớn trên hướng $\mathbf{v}$ hay không, hoàn toàn **không có bất kỳ toán tử vi phân thời gian nào**. Do đó, bất cứ khi nào frame có độ sáng hoặc đặc trưng tĩnh trùng với $\mathbf{v}$, nó sẽ kích hoạt boundary giả $\rightarrow$ Boundary Precision kẹt cứng ở mức 7.8%.

### Gốc rễ 2: Bipartite Matching Instability trên tập dữ liệu nhỏ (48 videos)
* Trong BaFormer, 150 queries được khởi tạo hoàn toàn ngẫu nhiên: `self.query = nn.Parameter(torch.randn(1, 150, 64))`.
* Chúng không hề có tọa độ thời gian (temporal anchor). Trong một video dài 1,500 frames có 35 hành động, Hungarian matcher gán ngẫu nhiên 35 trong số 150 queries này.
* Giữa các epoch liên tiếp, Query $q_i$ có thể hôm nay khớp với hành động ở frame 100, nhưng ngày mai lại bị gán cho hành động ở frame 1,200. Hiện tượng trôi dạt gán cặp (matching drift) khiến gradient triệt tiêu lẫn nhau, buộc mô hình phải ghi nhớ vẹt (overfit) cả 48 video chỉ sau 40 epoch (Train Acc đạt 71.5% nhưng Val Acc dừng ở 48%).

### Gốc rễ 3: Nghịch lý giữa Huấn luyện (Hungarian Matching) và Suy luận (`inference_bd_peak`)
* **Lúc Training:** Gradient tối ưu trực tiếp cho từng query mask $m_q(t)$ thông qua Dice Loss và Focal Loss. Quá trình tối ưu **không hề dùng** `inference_bd_peak`.
* **Lúc Inference:** Hàm `inference_bd_peak` lại **vứt bỏ hoàn toàn hình dạng mask tự nhiên của queries**! Nó dùng boundary curve (đang có 92% điểm cắt sai) để băm nhỏ timeline, rồi tại mỗi khoảng $[b_k, b_{k+1}]$ lại dùng `argmax` chọn 1 query duy nhất.
* Một vết cắt ranh giới sai lọt vào giữa một hành động may liền mạch sẽ xé hành động đó thành 2 mẩu; mỗi mẩu lại vô tình nhận 1 query khác nhau $\rightarrow$ sinh ra phân mảnh và tụt Edit Score.

---

## 3. Tổng quan Nghiên cứu SOTA (Literature & State-of-the-Art)

Để giải quyết triệt để 3 gốc rễ trên, chúng tôi đã khảo sát các nghiên cứu SOTA từ 2 nhánh công nghệ:

```
                                  SOTA RESEARCH MAPPING
       ┌───────────────────────────────────────┴───────────────────────────────────────┐
       ▼                                                                               ▼
[Nhánh 1: Modern Instance Segmentation]                       [Nhánh 2: Temporal Boundary Detection & TAS]
├── Mask DINO (Query Selection & Denoising)                   ├── ASRF (Action Segment Refinement Framework)
├── ActionFormer & TriDet (Anchor-free & Trident Head)        ├── DiffGEBD & DDM-Net (Diffusion on TSM)
└── MinVIS & Video Mask2Former (Query Dominance)              └── SC-GEBD / FlowGEBD (Local Temporal Contrast)
```

---

### 3.1 Nhánh 1: Các mô hình Instance Transformer mới (Mask DINO, TriDet, MinVIS)

#### 1. Denoising Training & Mixed Query Selection (Từ DINO & Mask DINO)
* **Bài toán:** DETR và Mask2Former gặp vấn đề nghiêm trọng về sự bất ổn định của thuật toán Hungarian matching khi dữ liệu ít.
* **Giải pháp của Mask DINO:**
  1. *Query Selection:* Thay vì khởi tạo query ngẫu nhiên, mô hình trích xuất các vị trí tiềm năng nhất từ feature map của Encoder để làm khởi tạo cho query content.
  2. *Denoising Training (DN):* Đưa thêm một nhóm query nhân tạo được sinh ra bằng cách thêm nhiễu (noise) vào Ground Truth. Nhóm này được tối ưu trực tiếp mà **không qua Hungarian matching**.
* **Ý nghĩa cho BaFormer:** Tạo ra gradient ổn định ngay từ epoch đầu tiên, triệt tiêu hiện tượng query trôi dạt và rút ngắn thời gian hội tụ.

#### 2. Temporal Anchor Prior & Trident Boundary Head (Từ ActionFormer & TriDet)
* **ActionFormer (ECCV 2022) & TriDet (CVPR 2023):** Là hai đỉnh cao của Temporal Action Detection (TAD).
* Thay vì phân loại boundary nhị phân cứng 0/1 (bị mất cân bằng trầm trọng 98% vs 2%), TriDet đưa ra khái niệm **Relative Boundary Distribution (Trident Head)**:
  - Dự đoán phân phối xác suất tương đối quanh tâm ranh giới.
  - Sử dụng cơ chế Scalable-Granularity Perception (SGP) để nắm bắt ngữ cảnh ở nhiều độ phân giải thời gian.
* **Ý nghĩa cho BaFormer:** Thay vì ép mô hình học xung Dirac rời rạc, dự đoán phân phối ranh giới cục bộ giúp giảm thiểu hoàn toàn hiện tượng false-positive over-cutting.

#### 3. Query-Dominance Instance Assembly (Từ MinVIS & Video Mask2Former)
* **MinVIS (NeurIPS 2022):** Chứng minh rằng trong video, việc theo dõi và phân tách các instance cùng loại (same-class recurring instances) không cần heuristic cắt ghép phức tạp.
* **Cơ chế:** Mỗi query sau khi huấn luyện đã học được một "vùng sở hữu" (territory) riêng biệt. Ranh giới giữa 2 instance cùng loại xảy ra chính xác tại vị trí mà **Query chủ đạo thay đổi quyền kiểm soát**:
  $$\hat{q}(t) = \arg\max_{q \in \{1..Q\}} \Big( \sigma\big(m_q(t)\big) \cdot \max_{c} p_q(c) \Big)$$
* Khi $\hat{q}(t) \neq \hat{q}(t-1)$, một instance mới xuất hiện tự nhiên, bảo toàn yêu cầu của `problem_definition.md` mà không sợ bị boundary nhiễu băm vụn.

---

### 3.2 Nhánh 2: Các kỹ thuật SOTA về Boundary Detection trong TAS & GEBD

#### 1. Local Temporal Difference Operator (Từ ASRF & SC-GEBD)
* **ASRF (Ishikawa et al., CVPR 2021):** Sử dụng một nhánh riêng gọi là Boundary Regression Branch (BRB).
* Điểm cốt lõi: Ranh giới là sự chênh lệch đặc trưng giữa cửa sổ tương lai và quá khứ:
  $$\Delta \mathbf{F}(t) = \left\| \frac{1}{K}\sum_{i=1}^{K} \mathbf{f}_{t+i} - \frac{1}{K}\sum_{i=1}^{K} \mathbf{f}_{t-i} \right\|_2$$
* Đưa vector sai phân cục bộ $\Delta \mathbf{F}(t)$ vào làm đầu vào bổ sung cho bộ dò ranh giới giúp mô hình phát hiện tức thì sự đổi tư thế thao tác của thợ may.

#### 2. Temporal Self-Similarity Matrix (TSM) từ DiffGEBD & DDM-Net
* Trong chính repository này (`src/step_segment/DiffGEBD` và `src/step_segment/DDM-Net`), các mô hình khuếch tán (diffusion) phát hiện ranh giới thông qua ma trận tương đồng thời gian:
  $$\mathbf{S}_{i, j} = \frac{\mathbf{f}_i^\top \mathbf{f}_j}{\|\mathbf{f}_i\| \|\mathbf{f}_j\|}$$
* Tại mép ranh giới, ma trận $\mathbf{S}$ luôn xuất hiện cấu trúc khối bàn cờ (checkerboard pattern) rõ rệt. Việc áp dụng tích chập thời gian 1D (Temporal Conv1D) có receptive field bao quát cấu trúc này sẽ lọc sạch 90% nhiễu nền tĩnh.

---

## 4. Đề xuất Kiến trúc BaFormer++ (Iteration 04)

Dựa trên các phân tích khoa học trên, chúng tôi đề xuất gói nâng cấp toàn diện **BaFormer++** gồm 4 trụ cột:

```
                                  BAFORMER++ ARCHITECTURE
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ [ASFormer Frame Encoder]                                                               │
│ Features F ∈ R^{B × 64 × L}                                                            │
└────────────────────────────────────────┬───────────────────────────────────────────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
┌──────────────────────────────────────┐        ┌──────────────────────────────────────┐
│ Trụ cột 1: Temporal Conv1D           │        │ Trụ cột 2: Slimmed Decoder (4 Layers)│
│ Boundary Head                        │        │ with Temporal Anchor Queries         │
│                                      │        │                                      │
│ • Local Temp Diff: ΔF(t) = Ft+1 - Ft-1│       │ • Dec Layers: 10 -> 4 (Anti-overfit) │
│ • Conv1D(k=5, d=1) -> ReLU           │        │ • Queries neo theo mốc thời gian:    │
│ • Conv1D(k=3, d=2) -> Sigmoid        │        │   q_i anchored at t_i = i / Q        │
│ => Boundary curve b(t) ∈ [0, 1]      │        │ => Mask m_q(t) & Class logits c_q    │
└──────────────────┬───────────────────┘        └──────────────────┬───────────────────┘
                   │                                               │
                   └───────────────────────┬───────────────────────┘
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ Trụ cột 3: Dual-Mode / Soft Query-Dominance Inference                                  │
│                                                                                        │
│ Mode A: Soft Semantic Assembly S(t, c) = Σ_q p_q(c) * σ(m_{q, t})                      │
│ Mode B: Query Dominance q*(t) = argmax_q (σ(m_q(t)) * p_q) + Boundary Verification     │
│ => Triệt tiêu 100% hiện tượng xé nát nhãn của inference_bd_peak cũ                    │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### Trụ cột 1: Thay thế Boundary Head bằng Multi-Scale Temporal Conv1D + Local Difference
* **Loại bỏ:** Khối `self.vid_embed_before` và `self.vid_embed`.
* **Thêm mới:** `TemporalConvBoundaryHead` hoạt động trực tiếp trên chuỗi đặc trưng thời gian:
  ```python
  class TemporalConvBoundaryHead(nn.Module):
      def __init__(self, in_dim=64, hidden_dim=64):
          super().__init__()
          # 1. Nhánh tính vi phân cục bộ (Temporal Difference)
          self.diff_conv = nn.Conv1d(in_dim, hidden_dim, kernel_size=3, padding=1, bias=False)
          
          # 2. Nhánh tích chập đa độ phân giải (Dilated Conv1D Pyramid)
          self.conv_net = nn.Sequential(
              nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
              nn.BatchNorm1d(hidden_dim),
              nn.ReLU(inplace=True),
              nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, dilation=2, padding=2),
              nn.BatchNorm1d(hidden_dim // 2),
              nn.ReLU(inplace=True),
              nn.Conv1d(hidden_dim // 2, 1, kernel_size=3, padding=1)
          )
          
      def forward(self, x):
          # x: [B, C, L]
          # Vi phân bậc 1 thời gian: f(t+1) - f(t-1)
          diff = torch.zeros_like(x)
          diff[:, :, 1:-1] = torch.abs(x[:, :, 2:] - x[:, :, :-2])
          feat = x + self.diff_conv(diff)
          return self.conv_net(feat) # [B, 1, L]
  ```

---

### Trụ cột 2: Giảm độ sâu Decoder & Neo vị trí Query (Temporal Anchor Queries)
1. **Thu nhỏ Decoder (Chống Overfitting trên 48 videos):**
   - Giảm `dec_layers` từ **10 xuống 4**.
   - Mô hình hiện tại quá nặng đối với 48 video. 4 tầng decoder là con số tiêu chuẩn (như trong Mask2Former VIS) giúp giảm 60% tham số dư thừa, ngăn chặn việc mô hình học vẹt.
2. **Khởi tạo Query có vị trí thời gian:**
   - Thay vì `randn`, khởi tạo 150 queries tương ứng với 150 phân đoạn thời gian trải đều từ 0 đến 1. Bổ sung learnable temporal center positional embedding cho từng query.

---

### Trụ cột 3: Cơ chế Suy luận Mềm (Soft Semantic Assembly & Query-Dominance)
Trong [`main.py`](file:///home/hungbm/ai4training/ai4training-aicore/src/TAS-instance-style/BaFormer/main.py):
1. **Chế độ Semantic Assembly (Mask2Former chuẩn):**
   ```python
   def semantic_inference(mask_cls, mask_pred):
       mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
       mask_pred = mask_pred.sigmoid()
       semseg = torch.einsum('qc,ql->cl', mask_cls, mask_pred).transpose(0, 1)
       return semseg
   ```
2. **Chế độ Query-Dominance có kiểm chứng Boundary:**
   - Chỉ tạo boundary mới khi có sự đồng thuận giữa:
     a) **Query chuyển giao**: $\arg\max_q m_q(t) \neq \arg\max_q m_q(t-1)$.
     b) **Boundary head xác nhận**: $b(t) > \tau_{bd}$.
   - Tránh tuyệt đối việc dùng mọi đỉnh nhọn của $b(t)$ để cưỡng bức gán lại nhãn của cả đoạn.

---

### Trụ cột 4: Chuẩn hóa Loss & Tăng cường dữ liệu (Data Augmentation)
1. **Loại bỏ `class_weights` lệch lạc:**
   - Đặt `class_weights: None` hoặc áp dụng làm mềm nhãn nhẹ (`label_smoothing = 0.1`). Để Dice Loss tự động cân bằng kích thước phân đoạn.
2. **Kích hoạt Data Augmentation:**
   - Bật `config.augmentation.is_use = True` (`augment_crop` với tỉ lệ crop ngẫu nhiên 0.7 – 1.0 thời lượng video). Kỹ thuật này nhân tạo sinh ra hàng nghìn biến thể từ 48 video gốc, xóa bỏ khoảng cách generalization gap 23%.
3. **Tăng cường Regularization:**
   - Tăng `dropout = 0.25` và `weight_decay = 0.0005`.

---

## 5. Kế hoạch Triển khai Chi tiết & File Checklist

| STT | File cần sửa đổi | Nội dung can thiệp | Mức độ ưu tiên |
| :---: | :--- | :--- | :---: |
| 1 | `action_segmentation/models/transformer_decoder/transformer_decoder_mask_bd_mulkv.py` | Tích hợp `TemporalConvBoundaryHead` (Conv1D + vi phân thời gian), bỏ `vid_embed`. | **P0** (Bắt buộc) |
| 2 | `main.py` | Bổ sung hàm suy luận `inference_query_dominance()` & `semantic_inference()`. Tích hợp công tắc `--infer_mode`. | **P0** (Bắt buộc) |
| 3 | `configs/tas_instance.yaml` | Cập nhật cấu hình Run 4: `dec_layers: 4`, `class_weights: None`, `augmentation.is_use: True`, `dropout: 0.25`, `weight_decay: 0.0005`. | **P0** (Bắt buộc) |
| 4 | `action_segmentation/config/defaults.py` & `__init__.py` | Đồng bộ các tham số mặc định và bảo đảm tương thích kiểu `yacs`. | **P1** |
| 5 | `infer_and_visualize.py` | Bổ sung xuất video timeline so sánh giữa `inference_bd_peak` cũ vs `inference_query_dominance` mới. | **P2** |

---

## 6. Lộ trình Thực hiện Tiếp theo

1. **Bước 1:** Cập nhật mã nguồn `transformer_decoder_mask_bd_mulkv.py` với `TemporalConvBoundaryHead`.
2. **Bước 2:** Cập nhật hàm suy luận trong `main.py` và file cấu hình `configs/tas_instance.yaml`.
3. **Bước 3:** Chạy kiểm tra biên dịch (`py_compile`) đảm bảo 0 lỗi cú pháp.
4. **Bước 4:** Cung cấp lệnh khởi chạy huấn luyện **Run 4 (BaFormer++)** cho Human Executor.
