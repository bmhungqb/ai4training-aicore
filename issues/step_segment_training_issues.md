# Báo cáo Vấn đề: Training DDM-Net Bị Kẹt Loss và F1-Score Không Cải Thiện

## 1. Hiện tượng (Symptoms)
Quan sát qua các Epoch huấn luyện (từ Epoch 0 đến Epoch 6) với backbone DINOv2 (`dinov2_vitb14`):
- **Thời gian huấn luyện**: Khá chậm, trung bình **~59-60 phút / epoch** (904 steps, ~0.25 it/s).
- **Train Loss**: Đi ngang hoàn toàn (**flat loss**) quanh ngưỡng `12.20 - 12.50` qua suốt 6 epoch, hầu như không giảm.
- **Val F1-Score**: Dao động lẹt đẹt quanh mức **~0.43 - 0.45**, không hề có xu hướng cải thiện:
  - Epoch 0: `0.45299`
  - Epoch 1: `0.39551`
  - Epoch 2: `0.45455` (đỉnh cao nhất, sau đó bắt đầu giảm dần)
  - Epoch 3: `0.43421`
  - Epoch 4: Rớt khỏi top 3
  - Epoch 5: Rớt khỏi top 3
- **Val Accuracy**: Dao động rất bất thường (`24.2%` $\rightarrow$ `31.9%` $\rightarrow$ `19.7%` $\rightarrow$ `47.8%` $\rightarrow$ `19.6%`).

---

## 2. Phân tích Nguyên nhân Cốt lõi (Root Causes)

### A. Vấn đề Cộng dồn 18 CrossEntropy Losses không trọng số
Trong `train_sop_lightning.py`:
```python
loss = 0
for output in outputs:
    loss += self.criterion(output, target)
for rgb in rgbs:
    loss += self.criterion(rgb, target)
for ddm in ddms:
    loss += self.criterion(ddm, target)
```
- **Cấu trúc model**: Co-Transformer Decoder có 6 layer, trả về intermediate output ở mỗi layer cho cả 3 nhánh:
  - `outputs`: 6 outputs kết hợp giữa RGB & DDM
  - `rgbs`: 6 auxiliary outputs từ nhánh RGB
  - `ddms`: 6 auxiliary outputs từ nhánh DDM
  - $\Rightarrow$ **Tổng cộng có $6 \times 3 = 18$ lần tính `nn.CrossEntropyLoss()` cộng dồn trực tiếp.**
- **Tại sao Loss lại kẹt ở ~12.40?**
  - Đối với bài toán phân loại 2 lớp (nhị phân), nếu model dự đoán ngẫu nhiên 50/50 (chưa học được gì):
    $$\text{Loss} \approx -\ln(0.5) \approx 0.693$$
  - Khi cộng dồn 18 heads:
    $$18 \times 0.693 \approx 12.47$$
  - **Mức loss 12.40 trên log phản ánh chính xác trạng thái model đang đoán ngẫu nhiên ở toàn bộ 18 heads**, gradients từ các layer nông (layer 0, 1) đè bẹp và làm nhiễu gradient của main head ở layer cuối (`outputs[-1]`).

### B. Learning Rate & Scheduler Chưa Phù Hợp
Trong cấu hình `ddm_train_config.yaml`:
```yaml
learning_rate: 0.0001        # 1e-4
scheduler: "cosine"
warmup_epochs: 1
warmup_lr: 0.00001
epochs: 30
min_lr: 1e-10
```
- Do `freeze_backbone: true`, toàn bộ backbone DINOv2 đã bị đóng băng, chỉ có các module trên đỉnh (DDM Encoder/Decoder, Transformers, FC heads) được khởi tạo ngẫu nhiên (scratch) để train.
- Mức `learning_rate: 1e-4` đối với các module scratch là khá nhỏ, kết hợp với Cosine Decay decay về `min_lr: 1e-10` ngay sau epoch 1 khiến learning rate tụt quá nhanh, model bị mắc kẹt tại local minimum / plateau trước khi kịp học được đặc trưng có ý nghĩa.

### C. Backbone DINOv2 Bị Đóng Băng Hoàn Toàn (`freeze_backbone: true`)
- DINOv2 được pretrain trên tập dữ liệu tự nhiên quy mô lớn (LVD-142M / ImageNet), không có kiến thức tiên nghiệm về môi trường công nghiệp dệt may (vải vóc, chuyển động tay luồn chỉ, ép máy).
- Khi đóng băng hoàn toàn backbone, các vector đặc trưng trích xuất có thể không phân tách đủ rõ sự khác biệt giữa hai thao tác may liên tiếp, gây khó khăn cho Co-Transformer phía sau.

### D. Yếu Tố Dataset: Có phải do Dataset ít?
- **Số lượng video**: 25 train video, 7 val video (tổng cộng 32 video có nhãn, 1 video `cd7_chuyen1` rỗng nhãn).
- **Số lượng mẫu sliding window thực tế**:
  - Dù chỉ có 25 video train, nhưng tổng số ranh giới thao tác trong tập train là **1,190 boundaries** (trên 1,215 steps).
  - Với cách lấy mẫu của DDMDataset, mỗi epoch có **904 steps** ($\times 2$ samples/batch $\times 2$ positive/negative = ~3,616 cửa sổ 17-frame được duyệt qua mỗi epoch).
- **Kết luận về dataset**: 
  - **Dữ liệu không phải là nguyên nhân khiến loss bị kẹt**. Nếu do ít dữ liệu, model thường sẽ học rất nhanh trên tập train dẫn tới hiện tượng quá khớp (**Overfitting**: train loss giảm sâu về ~0 nhưng val loss tăng). Đằng này model bị **Underfitting nghiêm trọng** (train loss không hề giảm).
  - Tuy nhiên, độ đa dạng góc máy/người thao tác trong 25 video có thể tạo rào cản trần (ceiling) cho F1-score sau khi bài toán underfitting được giải quyết.

---

## 3. Đề Xuất Giải Pháp Khắc Phục (Action Items)

### 1. Điều chỉnh Loss Function & Auxiliary Loss Weighting
Thay vì cộng phẳng 18 loss với trọng số bằng nhau:
- **Tập trung tối ưu Main Head** (`outputs[-1]`):
  ```python
  main_loss = self.criterion(outputs[-1], target)
  aux_loss = 0
  for o in outputs[:-1]:
      aux_loss += self.criterion(o, target)
  for r in rgbs:
      aux_loss += self.criterion(r, target)
  for d in ddms:
      aux_loss += self.criterion(d, target)

  # Đặt trọng số nhỏ cho các nhánh phụ (ví dụ 0.2 - 0.3)
  n_aux = len(outputs[:-1]) + len(rgbs) + len(ddms)
  loss = main_loss + 0.3 * (aux_loss / n_aux)
  ```
- Cân nhắc sử dụng **Focal Loss** hoặc **Dice Loss** kết hợp CrossEntropy để làm nổi bật sự thay đổi nhanh tại ranh giới thay vì chỉ phạt phân loại nhị phân thông thường.

### 2. Tinh chỉnh Learning Rate & Scheduler
- Tăng learning rate cơ sở cho các head lên **`2e-4` đến `3e-4`**.
- Tăng `warmup_epochs: 2` hoặc `3` để ổn định các weights ngẫu nhiên.
- Tăng `min_lr` lên **`1e-6`** (thay vì `1e-10` làm LR biến mất ở các epoch sau).

### 3. Mở Unfreeze Một Phần Backbone (Fine-tuning)
- Nếu sau khi chỉnh loss và LR mà model vẫn khó phân biệt ranh giới, xem xét unfreeze các layer cuối của DINOv2 (ví dụ 2 transformer blocks cuối của ViT-B/14) với learning rate nhỏ hơn ($10\times$ nhỏ hơn head, ví dụ `1e-5`) để model học thích ứng với miền ảnh xưởng may.

### 4. Tăng tốc độ Validation
- Khắc phục bottleneck IO của dataloader bằng cách tăng `workers` trong `val_config` lên mức hợp lý (kết hợp với cơ chế giải phóng ram từ issue `step_segment_validating_issues.md`) để giảm thời gian chờ 5-6 phút mỗi epoch.
