# FINE-TUNING INTENT DETECTION MODEL WITH BANKING DATASET

Dự án này tập trung vào việc áp dụng các kỹ thuật fine-tuning để giải quyết bài toán phân loại ý định (intent classification) trong lĩnh vực ngân hàng sử dụng dataset BANKING77 và thư viện Unsloth.

## 1. Giới thiệu Project

- **Mục tiêu**: Xây dựng mô hình phân loại ý định từ tin nhắn của khách hàng.
- **Dataset**: BANKING77 (chứa 77 loại ý định khác nhau trong lĩnh vực ngân hàng).
- **Công nghệ sử dụng**: Unsloth, Llama-3-8B (4-bit quantization), LoRA.

## 2. Cấu trúc thư mục

```
banking-intent-unsloth
|-- scripts
|   |-- train.py
|   |-- inference.py
|   |-- preprocess_data.py
|-- configs
|   |-- train.yaml
|   |-- inference.yaml
|-- sample_data
|   |-- train.csv
|   |-- test.csv
|-- train.sh
|-- inference.sh
|-- requirements.txt
|-- README.md
```

## 3. Hướng dẫn thiết lập môi trường

Khuyến khích sử dụng Google Colab với GPU T4 hoặc môi trường Linux có hỗ trợ CUDA.

```bash
# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

## 4. Chuẩn bị dữ liệu

Sử dụng script `preprocess_data.py` để tải và tiền xử lý dữ liệu từ Hugging Face. Script sẽ thực hiện lấy mẫu (sampling), chuẩn hóa văn bản và chia tập train/test.

```bash
python scripts/preprocess_data.py --fraction 1.0
```

## 5. Huấn luyện mô hình

Quá trình huấn luyện sử dụng Unsloth để tối ưu hóa tốc độ và bộ nhớ. Các tham số huấn luyện (batch size, learning rate, epochs...) được cấu hình tại `configs/train.yaml`.

```bash
# Chạy script huấn luyện
bash train.sh
```

Sau khi hoàn tất, checkpoint sẽ được lưu tại thư mục `models/banking_intent_model`.

## 6. Chạy Inference (Dự đoán)

Lớp `IntentClassification` trong `scripts/inference.py` được thiết kế để tải checkpoint và thực hiện dự đoán nhãn cho một tin nhắn đầu vào.

```bash
# Chạy script dự đoán mẫu
bash inference.sh
```

## 7. Video Demonstration

Video hướng dẫn chi tiết cách cài đặt, thực hiện huấn luyện và kết quả dự đoán của mô hình.

- **Link video**: [(Link Google Drive ở đây)](https://drive.google.com/file/d/1ltuor-ItAl8o4AB52wjA1uKEy4kq35AC/view?usp=sharing)

---

**Giảng viên hướng dẫn**: TS. Nguyễn Hồng Bửu Long
**Đơn vị**: Khoa Công nghệ Thông tin - Trường Đại học Khoa học Tự nhiên, ĐHQG-HCM
"""
