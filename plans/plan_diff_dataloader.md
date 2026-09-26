# Kế hoạch cải tiến Dataloader cho DiffGEBD (Giải quyết vấn đề Sampling Imbalance)

## 1. Vấn đề hiện tại (Issue)
Trong quá trình huấn luyện DiffGEBD trên dataset Sewing, mô hình gặp hiện tượng mất cân bằng cực độ: **Precision rất cao (0.9+) nhưng Recall tụt thê thảm (0.19)**. 

### Nguyên nhân cốt lõi (Root Cause):
Sau khi phân tích mã nguồn tại `src/step_segment/DiffGEBD/datasets/dataset.py`, lỗi nằm ở cơ chế trích xuất chuỗi (Sequence Extraction) khi chạy ở chế độ `END_TO_END`:

```python
if cfg.INPUT.END_TO_END:
    selected_indices = np.linspace(1, vlen, cfg.INPUT.SEQUENCE_LENGTH, dtype=int)
```

- Lệnh `np.linspace` lấy toàn bộ độ dài của một video gốc (`vlen`) và ép (squeeze) nó thành đúng `SEQUENCE_LENGTH` frames (ví dụ: 150 frames).
- DiffGEBD ban đầu được thiết kế cho bộ dữ liệu **Kinetics-GEBD**, nơi **tất cả các video đều được cắt sẵn dài đúng 10 giây** (~300 frames). Do đó `np.linspace(1, 300, 150)` cho ra bước nhảy (downsample) là 2 - rất hoàn hảo.
- Tuy nhiên, các video trong dataset Sewing lại là video dài (nhiều phút). Ví dụ video 3-4 phút có thể lên tới 6000 frames. Lệnh `np.linspace(1, 6000, 150)` dẫn đến bước nhảy thực tế là **40 frames/lần**.
- Vì một step ngắn thường chỉ kéo dài 5-10 frames, việc nhảy bước tới 40 frames khiến cho quá trình lấy mẫu "nhảy cóc" qua toàn bộ các ranh giới (boundaries). Tín hiệu dương (positive frames) hoàn toàn biến mất khỏi nhãn huấn luyện, khiến mô hình chỉ nhìn thấy frames nền (background), dẫn đến việc mô hình tự động đoán "0" để giảm loss.

## 2. Đề xuất giải pháp (Proposed Solution)

Vì toàn bộ luồng pipeline của DiffGEBD (từ Dataloader, Decoder tới hàm Evaluate) đều được "hardcode" để giả định 1 video = 1 chuỗi tín hiệu duy nhất, ta không thể thay đổi cơ chế này bên trong lõi mô hình. Giải pháp triệt để là can thiệp ở khâu **Tiền xử lý dữ liệu (Data Preprocessing)**:

### Bước 1: Viết script Video Chunking (Băm nhỏ video)
Tạo một công cụ mới (`tools/chunk_diff_gebd_dataset.py`) để:
- Đọc các video dài trong dataset gốc và băm chúng ra thành các đoạn ngắn (chunks), mỗi chunk kéo dài khoảng **10 đến 15 giây**.
- Khuyến nghị áp dụng **Overlapping** (chồng lấn nhau khoảng 1-2 giây ở rìa) để đảm bảo không cắt trúng điểm ranh giới của các step.

### Bước 2: Tái cấu trúc Annotation File
- Khớp lại mốc thời gian (timestamps) của các ranh giới gốc vào hệ quy chiếu của từng chunk.
- Sinh ra file `train_annotation_chunked.pkl` và `val_annotation_chunked.pkl` mới.
- Trong file này, mỗi "chunk 10s" sẽ được hệ thống xem như một video độc lập (`vid` riêng biệt).
- *Tùy chọn:* Có thể loại bỏ bớt một tỷ lệ nhất định các chunk không chứa bất kỳ ranh giới nào (Negative chunks) để cân bằng lại tỷ lệ Positive/Negative cho tập Train.

### Bước 3: Cấu hình lại Model
- Đặt lại `SEQUENCE_LENGTH: 150` và `DOWNSAMPLE: 2` (hoặc cấu hình tương đương sao cho khớp với độ dài chunk 10s).
- Khi đó, lệnh `np.linspace` bên trong DiffGEBD sẽ hoạt động chính xác với mục đích ban đầu của nó, bảo toàn được mọi tín hiệu ranh giới.

### Bước 4: Xử lý Hậu kỳ (Post-processing lúc Inference)
- Khi inference (Validate/Test), mô hình sẽ dự đoán ranh giới trên từng chunk.
- Viết thêm hàm tổng hợp (Aggregate) để cộng dồn hoặc ánh xạ các mốc thời gian từ các chunk trở về mốc thời gian gốc của video dài.
