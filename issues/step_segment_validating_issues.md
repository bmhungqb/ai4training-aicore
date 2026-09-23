# Báo cáo Vấn đề: Tràn Bộ Nhớ RAM Hệ Thống Trong Giai Đoạn Validation Của DDM-Net

## 1. Hiện tượng (Symptom)
- **Giai đoạn Training**: Chạy hoàn toàn ổn định suốt Epoch 0 (1487/1487 steps trong ~31 phút), lượng RAM hệ thống (System RAM) và VRAM GPU (RTX 5060 16GB) được duy trì ở mức thấp, không bị phình to.
- **Giai đoạn Validation**: Ngay khi bước vào `Validation DataLoader 0`, lượng RAM hệ thống tăng liên tục theo thời gian (từ 2GB $\rightarrow$ 8GB $\rightarrow$ chạm đỉnh 16GB).
  - Nếu không can thiệp: Linux OOM Killer sẽ gửi tín hiệu `SIGKILL/SIGTERM` làm tiến trình chết đột ngột (`RuntimeError: DataLoader worker is killed by signal: Terminated / Killed`).
  - Nếu người dùng theo dõi và thấy cạn kiệt RAM, bắt buộc phải gửi `Ctrl + C` để cứu hệ thống.

---

## 2. Nguyên nhân cốt lõi (Root Causes)

Sự chênh lệch lớn về mức tiêu thụ RAM giữa Train và Validation là do hai cơ chế nạp dữ liệu hoàn toàn khác biệt:

### A. Cơ chế của Training (`DDMDataset` - Sparse Sampling)
- **Thư viện sử dụng**: `PyAV` (`av.open`).
- **Cách nạp frame**: Lấy mẫu thưa ngẫu nhiên (Sparse Sampling). Tại mỗi bước huấn luyện (`batch_size=4`), DataLoader chỉ lấy ngẫu nhiên 4 vị trí thời gian trong các video, giải mã đúng 11 frames ngữ cảnh quanh mốc đó (`frames_per_side=5`), sau đó **đóng file video và giải phóng đối tượng ngay lập tức** (`del decoder`, `del video_content`).
- **Kết quả**: Bộ nhớ RAM chỉ chứa vài tensor ảnh nhỏ (224x224), giải phóng liên tục nên RAM luôn ở mức thấp.

### B. Cơ chế của Validation (`DDMValStreamingDataset` - Dense Streaming)
Validation của DDM-Net cần tính toán điểm số F1-Score trên toàn bộ chuỗi thời gian của video, dẫn đến các nguyên nhân sau:

1. **Internal Cache của thư viện C++ `decord`**:
   - Validation sử dụng `decord.VideoReader`:
     ```python
     self.vr = VideoReader(path, ctx=cpu(0), num_threads=0)
     ```
   - Thư viện `decord` viết bằng native C++ và tự động duy trì một bộ nhớ đệm (packet buffer & uncompressed frame cache) dưới tầng native C++ trong System RAM.
   - Khi `DecordStreamingReader` duyệt qua hàng nghìn frame của một video, các buffer frame giải mã không được giải phóng ngay mà tích tụ dần.

2. **Duyệt dày đặc qua toàn bộ video (`temporal_stride=1`)**:
   - Khác với training chỉ lấy mẫu ngẫu nhiên vài điểm, validation phải trượt cửa sổ qua **từng frame một** từ đầu đến cuối 7 video validation dài hàng trăm giây. Số lượng frame uncompressed liên tục được bơm vào bộ nhớ với mật độ cực dày.

3. **Cộng hưởng đa tiến trình (`num_workers > 0`)**:
   - Khi DataLoader sử dụng `num_workers=2` (hoặc 4), PyTorch sẽ `fork()` các worker processes riêng biệt.
   - Mỗi worker process lại khởi tạo một phiên bản `decord.VideoReader` độc lập để đọc các phân đoạn video song song. Điều này làm nhân bản bộ nhớ đệm C++ lên gấp 2 hoặc 4 lần trong cùng một thời điểm.

4. **Tích tụ kết quả dự đoán trong suốt epoch (`self.validation_step_outputs`)**:
   - Toàn bộ scores, frame IDs và metadata của 700+ validation steps được `append` liên tục vào list `self.validation_step_outputs` của Lightning Module và chỉ được xử lý, giải phóng khi validation epoch kết thúc hoàn toàn.
