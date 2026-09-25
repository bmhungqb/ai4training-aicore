# EfficientGEBD Training — Dataset Alignment Issues

> Phân tích mức độ tương thích giữa setup EfficientGEBD hiện tại và dataset may công nghiệp (`data/efficient_gebd_dataset/`).
> Ngày phân tích: 2026-09-25
> Cập nhật: 2026-09-26 — Issue 1-3 đã fix, phát hiện thêm Issue 4-9 sau khi retrain và phân tích sâu hơn.

---

## Tóm Tắt Kết Quả

| Hạng mục | Trạng thái | Ghi chú |
|---|---|---|
| Format annotation (`.pkl`) | ✅ OK | Đúng chuẩn Kinetics-GEBD schema |
| Dataset type `SEWING` trong code | ✅ OK | Đã có trong `datasets/__init__.py`, `dataset.py`, `post_process.py`, `train.py` |
| ROI Mask (crop khu vực thao tác) | ✅ OK | Pipeline `.mask.png` đã tích hợp trong `prepare_efficient_gebd_dataset.py` |
| Temporal Sampling (lấy mẫu thời gian) | ✅ FIXED | Issue 1 — chuyển sang slice-based sampling (TAPOS-style), coverage 100% |
| Evaluation Metric (F1 tolerance) | ✅ FIXED (một phần) | Issue 2 — đã có absolute-tolerance eval, nhưng **checkpoint selection vẫn dùng metric cũ** (xem Issue 5) |
| VRAM / Memory khi tăng Sequence Length | ✅ OK | Issue 3 — AMP + resnet50 config cho debug, batch size hợp lý |
| Double Gaussian-smoothing targets (SEWING) | ✅ FIXED | Issue 4 — model re-smooth targets đã được smooth sẵn ở dataset.py, khiến loss/F1 = 0 mọi epoch |
| Class imbalance (BCE loss) | ✅ FIXED (cần tune thêm) | Issue 4b — thêm `SOLVER.POS_WEIGHT`, giá trị 4.5 là ước lượng ban đầu |
| Model checkpoint selection dùng metric lỏng lẻo | 🔴 CRITICAL | Issue 5 — `model_best.pth` vẫn chọn theo F1@0.05 relative (inflate), không dùng absolute-tolerance |
| Train/Val split không disjoint theo trạm may | 🔴 CRITICAL | Issue 6 — 3/7 video val (`cd4`, `cd6`) trùng trạm với train → leakage |
| Dataset quá nhỏ so với backbone (CSN-R152/ResNet50) | 🟡 WARNING | Issue 7 — 25 video train / 7 video val, rủi ro overfit + metric nhiễu |
| Inductive bias của DiffFormer có thể không phù hợp sewing | 🟡 WARNING | Issue 8 — kiến trúc dựa trên frame dissimilarity, sewing có thể ngược lại |
| Slice 10s hard-cut làm mất context quanh điểm cắt | 🟡 WARNING | Issue 9 — WINDOW_SIZE chỉ ±0.8s, boundary gần mốc 10s bị giảm context |
| Config CSN chưa được tune ổn định (SGD, LR 1e-2, AMP) | 🟢 MINOR | Issue 10 — chỉ `sewing_resnet50.yaml` đã được tune sau sự cố NaN |
| `WARMUP_EPOCHS` chưa được dùng → metric dao động mạnh đầu training | ✅ FIXED | Issue 11 — thêm LR warmup thật (LambdaLR) + sửa log checkpoint gây nhầm lẫn |
| Recall luôn thấp (precision ≈ 1.0), `SOLVER.SIGMA` dead config | ✅ FIXED (cần tune thêm) | Issue 12 — wire `SOLVER.SIGMA` vào target smoothing, hạ `TEST.THRESHOLD` 0.3→0.2 |

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

## Issue 4 (FIXED): Double Gaussian-Smoothing Targets Cho SEWING → Loss/F1 = 0 Mọi Epoch

### Mô tả

Sau khi fix Issue 1 (chuyển SEWING sang slice-based sampling giống `TAPOS`), `datasets/dataset.py` đã **pre-compute Gaussian-smoothed targets** ngay trong nhánh slice (giống TAPOS). Tuy nhiên phần `forward()` của model (`modeling/e2e_model_diff_former.py`, `modeling/baseline.py`) chỉ skip bước smoothing lần 2 khi `self.dataset == 'TAPOS'`:

```python
if self.dataset != 'TAPOS':
    targets = prepare_gaussian_targets(targets)
```

Với `SEWING`, điều kiện này **luôn đúng** (vì `self.dataset == 'SEWING' != 'TAPOS'`) nên model re-smooth một target đã được smooth sẵn. `prepare_gaussian_targets` coi mọi giá trị non-zero là "tâm boundary" — vì target đã continuous (gaussian tail gần như khác 0 ở mọi vị trí), lần smooth thứ hai lan tỏa gaussian mới quanh gần như toàn bộ chuỗi, khiến target bị bão hòa (saturate) về gần hằng số. Model học theo target sai lệch này nên sigmoid output không bao giờ vượt `TEST.THRESHOLD=0.3` → recall/precision/F1 = 0.0000 ở **mọi** threshold, mọi epoch.

### Giải pháp đã áp dụng

```python
if self.dataset not in ('TAPOS', 'SEWING'):
    targets = prepare_gaussian_targets(targets)
```

Áp dụng cho cả `modeling/e2e_model_diff_former.py` và `modeling/baseline.py`.

---

## Issue 4b (FIXED — cần tune thêm): Class Imbalance Nghiêm Trọng Trong BCE Loss

### Mô tả

Sau khi fix Issue 4, loss vẫn plateau quanh **entropy floor** (~0.336) và không giảm thêm — model hội tụ về nghiệm "dự đoán hằng số ≈ giá trị trung bình target" thay vì học phân biệt theo thời gian. Đo thực tế trên `train_annotation.pkl` (sau slice-based sampling):

```
mean soft target value (Gaussian-smoothed) : ~0.18
fraction of frames với target > 0.5        : ~0.19
```

Vì chỉ ~18% frame có tín hiệu boundary, BCE thuần không có trọng số dễ hội tụ về nghiệm tầm thường (predict trung bình), khiến score không bao giờ vượt `TEST.THRESHOLD=0.3`.

### Giải pháp đã áp dụng

Thêm `SOLVER.POS_WEIGHT` (mặc định `1.0`, không ảnh hưởng config GEBD/TAPOS cũ), truyền vào `F.binary_cross_entropy_with_logits(..., pos_weight=...)` ở cả hai model. Set `POS_WEIGHT: 4.5` (≈ `(1-0.18)/0.18`) trong `sewing_resnet50.yaml` / `sewing_csn.yaml`.

> [!NOTE]
> `4.5` là ước lượng ban đầu dựa trên tỷ lệ target trung bình toàn dataset. Cần theo dõi precision/recall qua các epoch: nếu precision sụp đổ (quá nhiều false positive) thì giảm dần về 2–3; nếu recall vẫn thấp thì có thể tăng thêm hoặc cân nhắc Focal Loss thay vì pos_weight tĩnh.

---

## Issue 5 (CRITICAL): Model Checkpoint Selection Vẫn Dùng Metric Lỏng Lẻo (chưa fix)

### Mô tả

Issue 2 đã bổ sung absolute-tolerance evaluation (`eval_f1_absolute_tol`) và in ra bảng riêng khi validate, nhưng đây **chỉ là hiển thị**. Logic chọn checkpoint tốt nhất (`model_best.pth`) trong `train.py` vẫn dùng đúng metric lỏng lẻo của Issue 2:

```python
# train.py:484
f1, rec, prec = results[0.05][head]     # relative threshold 0.05 × video_length
metrics['F1'] = f1
...
# train.py:581
f1 = metrics_list[-1]['F1']             # <- dùng F1@0.05 relative để quyết định best checkpoint
if f1 > best_f1:
    torch.save(..., save_path)
```

Với video sewing trung bình ~2,300 frame, ngưỡng `0.05 × 2300 ≈ 115 frame ≈ 7.8 giây` — lớn hơn nhiều so với độ dài một bước may thực tế (2–5 giây).

### Rủi ro

- `model_best.pth` có thể được lưu dựa trên một tín hiệu gần như vô nghĩa với bài toán sewing (một model dự đoán một boundary mỗi ~8 giây có thể vẫn "điểm cao" theo metric này dù sai lệch cả một bước may).
- Checkpoint thực sự tốt hơn (theo absolute tolerance) có thể bị bỏ qua vì không "thắng" theo relative F1.

### Giải pháp đề xuất

- Với dataset `SEWING`, chuyển logic chọn best checkpoint sang absolute-tolerance F1 (ví dụ `±1.0s`) thay vì `results[0.05]`.
- Hoặc tối thiểu: log song song cả hai, và chọn checkpoint thủ công dựa trên absolute-tolerance table thay vì tin tưởng hoàn toàn vào `model_best.pth`.

---

## Issue 6 (CRITICAL): Train/Val Split Không Disjoint Theo Trạm May (chưa fix)

### Mô tả

Kiểm tra key của `train_annotation.pkl` / `val_annotation.pkl`:

```
train stations: cd1, cd2, cd3, cd4, cd5, cd6, cd7, cd8, cd9, cd10, cd11, cd13, cd14, cd15, cd16, cd17
val stations  : cd4, cd6, cd12, cd18, cd19, cd20
overlap       : cd4, cd6
```

Không có video (key) nào bị trùng chính xác giữa 2 tập (`cd4_chuyen2`/`cd4_chuyen3` ở train khác với `cd4_chuyen1` ở val), nhưng **cùng một trạm may / camera / góc quay / ánh sáng** (`cd4`, `cd6`) xuất hiện ở cả hai tập. 3/7 video val (43%) thuộc các trạm model đã "thấy" lúc train.

### Rủi ro

- Metric validation bị **inflate** cho các video thuộc trạm đã quen thuộc (model có thể học các đặc trưng riêng của trạm — background, ánh sáng, góc camera — thay vì đặc trưng chung của "ranh giới bước may").
- Không phản ánh đúng khả năng tổng quát hóa sang trạm/camera mới.

### Giải pháp đề xuất

- Re-split dataset theo **station-level** (`cdN`), đảm bảo train/val không share cùng station.
- Cân nhắc dùng k-fold theo station để có ước lượng ổn định hơn với dataset nhỏ.

---

## Issue 7 (WARNING): Dataset Quá Nhỏ So Với Kích Thước Backbone

### Mô tả

```
train: 25 video (397 slice 10s sau khi slice-based sampling)
val  : 7 video  (78 slice 10s)
```

CSN-R152 và ResNet50 là các backbone lớn (hàng chục triệu tham số), trong khi dữ liệu train hiệu dụng chỉ ~400 slice (nhiều slice có nội dung lặp lại cao trong cùng video). Val chỉ 7 video → **một video tốt/xấu có thể làm F1 dao động ~14%**.

### Giải pháp đề xuất

- Ưu tiên `sewing_resnet50.yaml` (backbone nhẹ hơn) cho giai đoạn hiện tại thay vì CSN-R152.
- Thêm augmentation mạnh hơn, weight decay, hoặc freeze thêm layer backbone để giảm overfit.
- Báo cáo F1 theo từng video val riêng lẻ, không chỉ số tổng hợp, để tránh đánh giá sai lệch do 1-2 video "ăn may".
- Cân nhắc k-fold cross-validation theo station khi dataset còn nhỏ.

---

## Issue 8 (WARNING): Inductive Bias Của DiffFormer Có Thể Không Phù Hợp Với Sewing

### Mô tả

Kiến trúc EfficientGEBD (`DiffFormer` + cosine similarity theo cửa sổ trượt) được thiết kế cho giả định của GEBD gốc: **boundary = độ khác biệt hình ảnh lớn giữa các frame lân cận** (đổi cảnh, đổi hành động trong video web đa dạng).

Với video may công nghiệp:
- Chuyển từ bước may này sang bước tiếp theo tại **cùng một trạm** thường rất mượt (cùng vải, cùng camera, cùng ánh sáng) → tín hiệu dissimilarity tại đúng boundary có thể **yếu**.
- Chuyển động tay/kim may nhanh **trong** một bước có thể tạo ra dissimilarity giữa các frame **lớn hơn** chính ranh giới thật → rủi ro false positive giữa bước, false negative tại ranh giới thật.

### Giải pháp đề xuất

- Không phải lỗi code — cần **validate sớm**: sau vài epoch, trực quan hóa predicted score curve của vài video val, so với GT boundary timestamps, để kiểm tra model có thực sự bắt tín hiệu chuyển bước hay chỉ bắt nhiễu chuyển động tay.
- Nếu tín hiệu dissimilarity yếu tại boundary thật, cân nhắc bổ sung thêm đặc trưng khác (optical flow, pose tay, tốc độ chuyển động) hoặc điều chỉnh sigma/half_dur cho phù hợp hơn.

---

## Issue 9 (WARNING): Slice 10s Hard-Cut Làm Giảm Context Quanh Điểm Cắt

### Mô tả

Sliced-based sampling (Issue 1) chia video thành các slice độc lập ~10 giây (`num_slices = duration // 10 + 1`), mỗi slice được forward riêng biệt (không có attention/context xuyên slice). Cửa sổ context cục bộ của model chỉ:

```
WINDOW_SIZE = 2*K + 1 = 17 sample × 0.1s/sample ≈ ±0.8 giây
```

Một boundary thật rơi gần điểm cắt 10 giây (trong khoảng ~1s quanh mốc cắt) sẽ bị giảm context ở một phía, vì slice bên cạnh được train như sample độc lập.

### Rủi ro

Với khoảng cách trung bình giữa 2 bước may ~2–3 giây, một tỷ lệ không nhỏ boundary sẽ rơi gần mốc 10s → có thể giảm recall hệ thống ở các vị trí này. Đây là rủi ro nhẹ/kỳ vọng được của phương pháp slice-based (cũng tồn tại sẵn ở nhánh `TAPOS`), không phải bug do thay đổi gần đây.

### Giải pháp đề xuất

- Kiểm tra tỷ lệ recall của các boundary nằm trong ±1s quanh mốc 10s so với các boundary khác, để định lượng ảnh hưởng thực tế.
- Nếu ảnh hưởng đáng kể, cân nhắc slice có overlap (ví dụ 10s slice với overlap 1-2s) thay vì cắt liền mạch.

---

## Issue 10 (MINOR): Config CSN Chưa Được Tune Ổn Định

### Mô tả

Sau sự cố NaN khi train, chỉ `sewing_resnet50.yaml` được điều chỉnh (AdamW, LR 1e-4, tắt AMP). `sewing_csn.yaml` vẫn giữ cấu hình gốc (`SGD`, `LR: 1e-2`, `AMPE: True`) — cùng tổ hợp có nguy cơ NaN trước đây. Fix `eps` trong cosine similarity (`e2e_model_diff_former.py`) là fix toàn cục nên áp dụng cho cả hai backbone, nhưng tổ hợp LR cao + AMP + dataset nhỏ với CSN-R152 **chưa được kiểm chứng thực tế**.

### Giải pháp đề xuất

- Nếu dùng `sewing_csn.yaml`, theo dõi sát loss những epoch đầu để phát hiện sớm NaN/instability.
- Cân nhắc áp dụng cùng công thức đã ổn định cho resnet50 (AdamW, LR thấp hơn, tắt AMP) nếu gặp lại sự cố.

### Ghi chú khác (cosmetic)

`TEST.RELDIS_THRESHOLD` trong các file yaml hiện **không được đọc** — `train.py` hard-code danh sách threshold (`[0.05]` hoặc sweep 10 giá trị) thay vì dùng giá trị từ config. Không gây lỗi nhưng gây nhầm lẫn khi đọc config.

---

## Issue 11 (FIXED): `WARMUP_EPOCHS` Chưa Được Dùng → LR Full Ngay Từ Epoch 0 → Metric Dao Động Mạnh

### Mô tả

`SOLVER.WARMUP_EPOCHS: 5` được định nghĩa trong config nhưng **chưa từng được dùng** ở `train.py`:

```python
scheduler = MultiStepLR(optimizer, milestones=cfg.SOLVER.MILESTONES)   # không có warmup
```

Do đó LR luôn ở mức full (`1e-4`) ngay từ epoch 0 (khớp với log luôn hiển thị `lr:0.00010`). Kết hợp với `POS_WEIGHT=4.5` (khuếch đại gradient trên các frame positive hiếm), training rất dễ overshoot khỏi một điểm tốt vừa tìm được, gây dao động mạnh giữa các epoch liên tiếp — quan sát thực tế:

```
Epoch04: F1@0.05 = 0.0000
Epoch05: F1@0.05 = 0.3529   (Recall 0.2155, Precision 0.9740)
Epoch06: F1@0.05 = 0.0394   (Recall 0.0201, Precision 1.0000)
```

Ngoài ra, dòng log `print('Saved to {}'.format(save_path))` nằm **ngoài** khối `if f1 > best_f1:` nên in ra mỗi epoch dù `model_best.pth` có thực sự được ghi đè hay không — dễ gây hiểu lầm rằng checkpoint tốt (epoch 5) đã bị mất khi thấy epoch 6 tệ hơn (thực tế `torch.save` chỉ chạy khi `f1 > best_f1` nên checkpoint tốt vẫn an toàn, chỉ có log gây nhầm lẫn).

### Giải pháp đã áp dụng

- Thay `MultiStepLR` bằng `LambdaLR` với warmup tuyến tính trong `WARMUP_EPOCHS` epoch đầu (`(epoch+1)/warmup_epochs`), sau đó áp dụng decay theo `MILESTONES`/`GAMMA` như cũ — khớp với ý định gốc của config.
- Sửa log: chỉ in `'Saved to {save_path}'` khi thực sự lưu checkpoint mới; nếu không cải thiện thì in rõ `'F1 {f1} did not improve over best {best_f1}, not saving.'`.

---

## Issue 12 (FIXED): Recall Luôn Thấp (Precision ≈ 1.0) — `SOLVER.SIGMA` Là Dead Config + Threshold Quá Chặt

### Mô tả

Sau khi các fix trước đã giúp training học được tín hiệu thật (không còn loss=0 hằng số), quan sát thực tế vẫn cho thấy **recall luôn rất thấp trong khi precision gần 1.0** (ví dụ Epoch05: Recall 0.22 / Precision 0.97). Model rất "thận trọng" — chỉ báo boundary khi cực kỳ chắc chắn, bỏ sót phần lớn các bước may thật. Đào sâu tìm ra 2 nguyên nhân:

**a) `SOLVER.SIGMA` được định nghĩa trong config nhưng chưa từng được dùng thật:**

```python
# datasets/dataset.py (trước fix)
labels_valid = prepare_gaussian_targets(labels_valid)   # luôn dùng sigma=1 mặc định, KHÔNG đọc cfg.SOLVER.SIGMA
```

Với `sigma=1` (mặc định hard-code), vùng "positive" quanh mỗi boundary rất hẹp — đo thực tế trên `train_annotation.pkl`:

```
sigma=1: mean soft target=0.1802, frac target>0.5=0.1863   (~18% frame có tín hiệu boundary)
sigma=2: mean soft target=0.2981, frac target>0.5=0.3117   (~31% frame có tín hiệu boundary)
```

Vùng positive quá hẹp khiến bài toán phân loại từng-frame trở nên khó và mất cân bằng hơn mức cần thiết, đẩy model về hướng "chỉ báo khi rất chắc" (recall thấp).

**b) `TEST.THRESHOLD=0.3` quá chặt so với mức độ calibration hiện tại của model:** với precision quan sát được gần 1.0 (dư địa lớn), threshold có thể hạ thấp để đánh đổi lấy recall mà không sợ sụp độ chính xác ngay lập tức.

### Giải pháp đã áp dụng

- `datasets/dataset.py`: truyền `sigma=cfg.SOLVER.SIGMA` vào `prepare_gaussian_targets(...)` thay vì dùng mặc định.
- Thêm `sigma{value}` vào tên file cache (`end_to_end_slice{L}_fps10_sigma{S}_...`) để thay đổi `SOLVER.SIGMA` tự động invalidate cache cũ, tránh tái sử dụng nhãn sai lệch.
- Tăng `SOLVER.SIGMA: 1 → 2` trong `sewing_resnet50.yaml` / `sewing_csn.yaml` (~31% frame positive thay vì ~18%).
- Giảm `TEST.THRESHOLD: 0.3 → 0.2` trong cả hai config để cho phép model "mạnh dạn" báo boundary hơn.

> [!NOTE]
> Đây là các tham số cần tune tiếp theo dữ liệu thực nghiệm: nếu recall vẫn thấp sau vài chục epoch, cân nhắc tăng `SIGMA` lên 2.5–3 hoặc giảm `TEST.THRESHOLD` xuống 0.15; nếu precision sụp quá nhanh thì lùi lại. Nên dùng bảng absolute-tolerance (Issue 2) để chọn threshold tối ưu thay vì cố định `0.3`/`0.2`.

---

## Tệp Liên Quan

| Tệp | Vai trò |
|---|---|
| [`tools/prepare_efficient_gebd_dataset.py`](../tools/prepare_efficient_gebd_dataset.py) | Build annotation pickle từ `step_segments.json` |
| [`src/step_segment/EfficientGEBD/datasets/dataset.py`](../src/step_segment/EfficientGEBD/datasets/dataset.py) | Logic sampling frames, gán nhãn Gaussian (nay đọc `SOLVER.SIGMA`) |
| [`src/step_segment/EfficientGEBD/datasets/__init__.py`](../src/step_segment/EfficientGEBD/datasets/__init__.py) | Build dataloader, ROOT path mapping |
| [`src/step_segment/EfficientGEBD/utils/eval.py`](../src/step_segment/EfficientGEBD/utils/eval.py) | Hàm F1 evaluation (relative + absolute tolerance) |
| [`src/step_segment/EfficientGEBD/train.py`](../src/step_segment/EfficientGEBD/train.py) | Training loop, validation, slice-merge logic, LR warmup, checkpoint selection |
| [`src/step_segment/EfficientGEBD/modeling/e2e_model_diff_former.py`](../src/step_segment/EfficientGEBD/modeling/e2e_model_diff_former.py) | E2EModelDiff forward/loss (Gaussian re-smoothing, pos_weight) |
| [`src/step_segment/EfficientGEBD/modeling/baseline.py`](../src/step_segment/EfficientGEBD/modeling/baseline.py) | BaseModel forward/loss (Gaussian re-smoothing, pos_weight) |
| [`src/step_segment/EfficientGEBD/modeling/config.py`](../src/step_segment/EfficientGEBD/modeling/config.py) | Định nghĩa `SOLVER.POS_WEIGHT`, `SOLVER.SIGMA` và các default config |
| [`src/step_segment/EfficientGEBD/config-files/sewing_resnet50.yaml`](../src/step_segment/EfficientGEBD/config-files/sewing_resnet50.yaml) | Config ResNet50 backbone |
| [`src/step_segment/EfficientGEBD/config-files/sewing_csn.yaml`](../src/step_segment/EfficientGEBD/config-files/sewing_csn.yaml) | Config CSN backbone |

---

## Thứ Tự Ưu Tiên Xử Lý

1. ✅ **Issue 1** — Sửa Temporal Sampling → Slice-based (TAPOS-style) cho SEWING *(đã fix)*
2. ✅ **Issue 2** — Bổ sung Absolute Tolerance Evaluation metric *(đã fix — hiển thị)*
3. ✅ **Issue 3** — Tune VRAM config theo GPU thực tế *(đã fix)*
4. ✅ **Issue 4 / 4b** — Fix double Gaussian-smoothing + thêm `POS_WEIGHT` chống class imbalance *(đã fix)*
5. 🔴 **Issue 5** — Chuyển checkpoint selection (`model_best.pth`) sang absolute-tolerance F1
6. 🔴 **Issue 6** — Re-split train/val theo station để tránh leakage
7. 🟡 **Issue 7** — Giảm overfit risk: ưu tiên resnet50, augmentation, báo cáo F1 theo từng video
8. 🟡 **Issue 8** — Validate trực quan xem model có bắt đúng tín hiệu boundary hay không
9. 🟡 **Issue 9** — Định lượng ảnh hưởng của slice hard-cut lên recall quanh mốc 10s
10. 🟢 **Issue 10** — Tune ổn định config CSN, dọn `TEST.RELDIS_THRESHOLD` không dùng
11. ✅ **Issue 11** — Thêm LR warmup thật + sửa log checkpoint gây nhầm lẫn *(đã fix)*
12. ✅ **Issue 12** — Wire `SOLVER.SIGMA` + hạ `TEST.THRESHOLD` để cải thiện recall *(đã fix, cần theo dõi/tune thêm)*
