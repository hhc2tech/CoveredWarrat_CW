
# Covered Warrant (CW)

Chứng quyền có đảm bảo (Covered Warrant - CW) theo mô hình Black-Scholes, bao gồm:

---

# 📌 I. Giới thiệu nhanh

✅ **Covered Warrant là gì?**

Là chứng khoán do CTCK phát hành, cho phép người nắm giữ mua (Call CW) hoặc bán (Put CW) cổ phiếu cơ sở với giá xác định trong tương lai. Cần định giá hợp lý để tránh mua quá đắt/thiếu định giá.

---

# 📌 II. Mô hình định giá Black-Scholes (cho CW kiểu Châu Âu)

✅ **Công thức Black-Scholes:**

- Với CW mua (Call):

C = S · N(d₁) − K · e^(−rT) · N(d₂)

- Với CW bán (Put):

P = K · e^(−rT) · N(−d₂) − S · N(−d₁)

Trong đó:

d₁ = [ln(S/K) + (r + σ²/2) · T] / (σ · √T)  
d₂ = d₁ − σ · √T

---

### **Giải thích các biến:**

- S: Giá cổ phiếu hiện tại  
- K: Giá thực hiện  
- T: Thời gian đến ngày đáo hạn (tính theo năm)  
- r: Lãi suất phi rủi ro  
- σ: Độ biến động (volatility) của cổ phiếu  
- N(d): Hàm phân phối chuẩn tích lũy (CDF)

---

# 📌 III. Các hệ số nhạy cảm (Greek Letters)

✅ **Delta** – Đo độ nhạy của giá CW với giá tài sản cơ sở:

Delta = N(d₁) (Call), −N(−d₁) (Put)

✅ **Gamma** – Đo độ thay đổi của Delta:

Gamma = e^(−d₁² / 2) / [√(2π) · S · σ · √T]

---

# 📌 IV. Mở rộng với đặc điểm của CW tại Việt Nam

Tại Việt Nam, CW có những đặc điểm như:

- Tỷ lệ chuyển đổi (conversion ratio) không phải 1:1  
- Hiệu ứng pha loãng không tồn tại như quyền chọn thực  
- Giới hạn giá nên cần điều chỉnh giá lý thuyết:

Giá lý thuyết CW = (C · R) / (Tỷ lệ chuyển đổi)

Trong đó:  
- C là giá quyền theo mô hình Black-Scholes  
- R là hệ số điều chỉnh hoặc chiết khấu từ tổ chức phát hành
