# Stage 1: Kinematic Action Segmentation Method & Experiment Tracker

> **Mục tiêu**: File duy nhất theo dõi toàn diện thông tin phương pháp (Method Info), trạng thái & các vấn đề kỹ thuật (Status & Issues), và kết quả thực nghiệm (Experiments & Results) của **Stage 1: Phân đoạn Động học Thao tác Vật lý (Kinematic Pre-Segmentation)** trong pipeline AI for Training.

---

## 1. Method Info (Thông tin Phương pháp)

### 1.1. Tổng quan & Nguyên lý (Overview & Principles)
- **Mục tiêu**: Tự động phát hiện các ranh giới vi-thao tác vật lý (Physical Action Boundaries) của công nhân may trực tiếp từ video thô mà **không cần VLM** và **không cần API key**.
- **Công nghệ cốt lõi**:
  - **SAM 3 (Segment Anything Model 3)**: Định vị và theo dõi mặt nạ 2 bàn tay/cánh tay của công nhân qua từng khung hình (`masks.npz`).
  - **SEA-RAFT**: Ước lượng trường dòng chảy quang học dày đặc (Dense Optical Flow) chất lượng cao (`flow.npz`).
  - **Magnitude / Direction Kinematic Fusion**: Dung hợp độ lớn vận tốc, hướng di chuyển và độ hỗn loạn chuyển động (turbulence) để tìm các điểm chuyển pha (thung lũng tốc độ, điểm dừng tay, đổi hướng may).
- **Đầu ra chính**:
  - `action_segments.json`: Danh sách các phân đoạn hành động kèm mốc thời gian `start_time_s`, `end_time_s`, `duration_s`.
  - `decomposed_motion.npz`: Vector phân rã động học (tốc độ, turbulence, likelihood).
  - `pipe1_report.json`: Báo cáo chi tiết các mốc cắt và thông số động học.
- **Mã nguồn trong repository**:
  - Script điều phối: `pipeline.py segment`
  - Sub-pipeline động học: `src/action_segment/segmentation/` & `src/action_segment/kinematic_pipeline/`

### 1.2. Bố Cục Thư Mục & Tài Nguyên
```text
ai4training-aicore-poc/
├── data/
│   ├── {cd_id}/chuyen{chuyen_id}/
│   │   ├── cam-03_....mp4                      <-- Video gốc
│   │   ├── cam-03_....mask.png                 <-- ROI mask công nhân (loại người ngoài)
│   │   └── chuyen1_segment.json                <-- Nhãn Ground Truth (nếu có)
│   └── {cd_id}/kinematic/{video_stem}/         <-- Output Stage 1 sinh ra
│       ├── action_segments.json                <-- Ranh giới vi-thao tác
│       ├── decomposed_motion.npz
│       ├── pipe1_report.json
│       ├── action_boundaries_dynamic.npy
│       └── motion_decomposition_smooth_plot.png
├── tools/
│   ├── eval_boundary_recall.py                 <-- Script đánh giá Recall 9 CĐ
│   └── export_eval_to_excel.py                 <-- Xuất kết quả chi tiết ra file Excel
├── experiments/
│   └── stage1_boundary_recall_9cd/
│       ├── evaluation_result_9cd.xlsx          <-- Bảng đối soát chi tiết 349 bước
│       └── eval_report.json                    <-- JSON kết quả benchmark
└── docs/
    └── action_segment/
        └── stage1_kinematic.md                 <-- (File này) Tracker duy nhất
```

### 1.3. Thiết Lập Môi Trường & Hướng Dẫn Thực Thi
```bash
# 1. Cài đặt hệ thống
sudo apt update && sudo apt install -y ffmpeg

# 2. Môi trường Python
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -r requirements-kinematic.txt

# 3. Đăng nhập Hugging Face (Bắt buộc cho SAM 3)
huggingface-cli login

# 4. Tải dữ liệu mẫu từ Google Drive (nếu chưa có)
gdown --id 1qwXBnvnvwC3THJAZetMWl0Rcr50MiIii -O data.zip
unzip data.zip

# 5. Chạy phân đoạn Stage 1:
# Chạy toàn bộ dữ liệu:
python pipeline.py segment --all-data --visualize

# Chạy theo công đoạn cụ thể (CĐ 1):
python pipeline.py segment --cong-doan 1 --visualize

# Chạy một video đơn lẻ:
python pipeline.py segment \
    --video data/1/cam-03_20260805_073527_cut_0_0-0_57.mp4 \
    --out-dir data/1/kinematic/cam-03_20260805_073527_cut_0_0-0_57 \
    --visualize

# Nếu gặp lỗi CUDA Out-Of-Memory trên GPU nhỏ:
python pipeline.py segment --all-data --resize-scale 0.25 --frame-step 2 --frame-by-frame
```

---

## 2. Status & Issues (Hiện trạng & Các Vấn đề Kỹ thuật)

### 2.1. Các Cải Tiến Thuật Toán Cốt Lõi Đã Giải Quyết (Stage 1 v2)
Trong file `src/action_segment/kinematic_pipeline/calculate_direction_magnitude.py`, 4 nút thắt vật lý đã được xử lý triệt để:

1. **Co viền Mask (`cv2.erode` kernel $3 \times 3$):**
   - Triệt tiêu hoàn toàn hiện tượng 1–3 pixel rìa mặt nạ SAM3 lấn ra mặt bàn may, loại bỏ gradient dòng chảy quang học giả gây nhiễu `turbulence` ảo.
2. **Bảo toàn cử động ngón tay khi tì cổ tay (Translation + RMS Energy):**
   - Thay thế việc tính `np.median(hand_flow)` thuần túy (vốn bị kéo về 0 khi công nhân tì cổ tay lên bàn) bằng công thức dung hợp năng lượng:
     $$\text{speed} = 0.5 \times v_{\text{median}} + 0.5 \times v_{\text{RMS}}$$
   - Duy trì vận tốc hoạt động khi ngón tay bấm/đẩy vải, ngăn chặn việc cắt nhầm ranh giới giữa chừng khi đang may.
3. **Nội suy thời gian (Temporal Interpolation) cho frame mất Mask:**
   - Xóa bỏ việc gán `speed = 0` khi SAM3 bị drop mask (chiếm 5% - 11% frames do chuyển động quá nhanh). Áp dụng `np.interp` bắc cầu qua các khoảng trống ngắn, dập tắt các "thung lũng dừng tay giả".
4. **Tối ưu siêu tham số động học ngành may:**
   - Hạ khoảng cách tối thiểu `min_distance` từ **1.5s $\rightarrow$ 0.5s** (bắt kịp các thao tác vi mô: lại mũi 0.4s, xoay góc 0.5s).
   - Ngưỡng động thích ứng: $\text{Threshold} = \text{local\_mean} + 0.7 \times \text{local\_std}$.
   - Trọng số đa phương thức: `Speed=0.40, Direction=0.40, Turbulence=0.20`, lọc góc xoay $25.0^\circ$.

---

### 2.2. Các Thách Thức Khi Chuyển Giao Sang Stage 2 (VLM Classification)

| Thách thức | Bản chất vấn đề | Ảnh hưởng |
|---|---|---|
| **Vỡ ngữ cảnh thời gian (Temporal Blindness)** | Stage 2 gửi 2–4 frame tĩnh sang VLM | Mất vector vận tốc, VLM không phân biệt được đẩy vải vs kéo vải vs nghỉ tay $\rightarrow$ Hallucination |
| **Bùng nổ chi phí & độ trễ (Latency/Cost)** | Stage 1 sinh 60–134 segments/video | Gọi tuần tự VLM 60–134 lần mất 3–5 phút/video; rủi ro rate limit và timeout |
| **Xung đột phân cấp (Granularity Mismatch)** | Micro-actions (0.6s) vs Macro-steps (2–6s) | VLM gán nhãn chập chờn (label flickering), logic gộp liên tiếp bị vô hiệu hóa |
| **Giới hạn kiến trúc Chat Completions** | Dùng API chat ảnh tĩnh thay vì video native | Không tối ưu cho dữ liệu chuỗi video công nghiệp |

---

## 3. Experiments & Results (Kết Quả Thực Nghiệm)

### 3.1. Thực Nghiệm 1: Đánh Giá Chuyên Sâu Công Đoạn 1 (CĐ 1: Diễu TP 4 cạnh nắp túi)
So sánh giữa Stage 1 ban đầu (Baseline) và Stage 1 sau cải tiến (v2) trên video chuẩn đối chiếu với `data/1/chuyen1_segment.json` (24 bước thao tác Ground Truth):

| Chỉ số đánh giá | Trước cải tiến (Baseline) | Sau cải tiến (Stage 1 v2) | Mức độ cải thiện |
|:---|:---:|:---:|:---:|
| **Boundary Recall (@0.5s)** | 44.0% | **88.0%** (22/25 mốc) | **+44.0% (Tăng gấp đôi độ nhạy)** 🚀 |
| **Boundary Recall (@1.0s)** | 88.0% | **100.0%** (25/25 mốc) | **Bắt trọn 100% ranh giới chuyên gia** |
| **Thao tác khớp CẢ 2 ĐẦU (Start & End)** | 12.5% (3/24 bước) | **79.2% (19/24 bước)** | **Tăng từ 3 bước lên 19 bước** 🚀 |
| **Thao tác khớp ÍT NHẤT 1 ĐẦU** | 66.7% | **100.0% (24/24 bước)** | **Độ bao phủ tuyệt đối** |
| **Boundary Precision (@0.5s)** | 48.0% | **45.9%** (28/61 vết cắt) | Ổn định (phù hợp tỷ lệ over-segment 2.4x) |
| **F1-Score (@0.5s)** | 45.9% | **60.3%** | **+14.4%** |
| **Frame Drop 0-Speed (Video chính)** | 21 frames | **0 frame** | Triệt tiêu 100% lỗi mất mask |

---

### 3.2. Thực Nghiệm 2: Benchmark Toàn Diện 9 Công Đoạn Độc Lập (Chuyền 1)
- **Tập dữ liệu**: Chuyền 1 gồm 9 công đoạn độc lập (CĐ 1, 2, 3, 4, 5, 6, 8, 9, 10).
- **Quy mô**: 9 video, **349 bước Ground Truth**, **363 mốc ranh giới**, máy phát hiện **1,505 phân đoạn** (1,514 ranh giới chuyển tiếp).

#### Bảng Kết Quả Chi Tiết 9 Công Đoạn (Dung sai $\pm 0.5$ giây):

| CĐ | Tên công đoạn | Số bước GT | Số mốc GT | Ranh giới máy | Tỷ lệ cắt | Boundary Recall | Độ lệch MAE (s) | Khớp CẢ 2 đầu | Khớp ÍT NHẤT 1 đầu |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | Diễu TP 4 cạnh túi lai x2 | 24 | 25 | 65 | 2.7x | **100.0%** | 0.135s | **24/24 (100.0%)** | 24/24 (100.0%) |
| **2** | Khóa lưỡi gà + doup đoạn cơi | 37 | 38 | 146 | 3.9x | **92.1%** | 0.203s | **31/37 (83.8%)** | 37/37 (100.0%) |
| **3** | Ráp chèn tay lót | 14 | 15 | 38 | 2.7x | **86.7%** | 0.231s | **10/14 (71.4%)** | 14/14 (100.0%) |
| **4** | May đáp túi lai | 11 | 12 | 36 | 3.3x | 75.0% | 0.198s | 5/11 (45.5%) | 11/11 (100.0%) |
| **5** | Rập lược TP cầu dk | 25 | 26 | 143 | 5.7x | **80.8%** | 0.210s | **17/25 (68.0%)** | 24/25 (96.0%) |
| **6** | Ráp ngang đô sau | 24 | 25 | 97 | 4.0x | **80.0%** | 0.238s | **15/24 (62.5%)** | 24/24 (100.0%) |
| **8** | Tra tay lót | 63 | 66 | 344 | 5.5x | **77.3%** | 0.233s | **38/63 (60.3%)** | 61/63 (96.8%) |
| **9** | Tra cổ chính | 40 | 41 | 143 | 3.6x | **92.7%** | 0.192s | **35/40 (87.5%)** | 40/40 (100.0%) |
| **10** | Tra tay chính | 111 | 115 | 502 | 4.5x | **91.3%** | 0.238s | **92/111 (82.9%)** | 110/111 (99.1%) |
| **Tổng** | **9 công đoạn** | **349** | **363** | **1,514** | **4.1x** | **Macro: 86.2%<br/>Micro: 87.3%** | **0.213s** | **267/349 (76.5%)** | **343/349 (98.3%)** |

#### Quét Dải Dung Sai Thời Gian (Tolerance Window Sweep):

| Cửa sổ dung sai | Macro Recall (%) | Micro Recall (%) | Số mốc GT trúng | Khớp CẢ 2 đầu | Khớp ÍT NHẤT 1 đầu | Sai lệch MAE (s) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **$\pm 0.25$s** | 50.5% | 52.1% | 189 / 363 | 28.1% (98/349) | 80.2% (280/349) | 0.126s |
| **$\pm 0.50$s (Chuẩn)** | **86.2%** | **87.3%** | **317 / 363** | **76.5% (267/349)** | **98.3% (343/349)** | **0.213s** |
| **$\pm 0.75$s** | **95.4%** | **95.6%** | **347 / 363** | **91.4% (319/349)** | **100.0% (349/349)** | **0.255s** |
| **$\pm 1.00$s** | **98.4%** | **98.6%** | **358 / 363** | **97.4% (340/349)** | **100.0% (349/349)** | **0.278s** |
| **$\pm 1.50$s** | **100.0%** | **100.0%** | **363 / 363** | **100.0% (349/349)** | **100.0% (349/349)** | **0.297s** |

### 3.3. File Dữ Liệu Thực Nghiệm Đi Kèm & Lệnh Tái Hiện
- File Excel chi tiết 349 bước: [`experiments/stage1_boundary_recall_9cd/evaluation_result_9cd.xlsx`](file:///home/manh-hung/Documents/work/WE/AI4Training/ai4training-aicore-poc/experiments/stage1_boundary_recall_9cd/evaluation_result_9cd.xlsx)
- File JSON kết quả: [`experiments/stage1_boundary_recall_9cd/eval_report.json`](file:///home/manh-hung/Documents/work/WE/AI4Training/ai4training-aicore-poc/experiments/stage1_boundary_recall_9cd/eval_report.json)
- Lệnh tái hiện:
  ```bash
  python -m tools.eval_boundary_recall --no-tune --exclude-cd 11 --out experiments/stage1_boundary_recall_9cd/eval_report.json
  python -m tools.export_eval_to_excel
  ```
