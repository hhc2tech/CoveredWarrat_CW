# CoveredWarrat_CW
Chứng quyền có đảm bảo (Covered Warrant - CW) theo mô hình Black-Scholes, bao gồm:

# 📌 I. Giới thiệu nhanh

✅ **Covered Warrant là gì?**

Là chứng khoán do CTCK phát hành, cho phép người nắm giữ mua (Call CW) hoặc bán (Put CW) cổ phiếu cơ sở với giá xác định trong tương lai. Cần định giá hợp lý để tránh mua quá đắt/thiếu định giá.

---

# 📌 II. Mô hình định giá Black-Scholes (cho CW kiểu Châu Âu)

✅ **Công thức Black-Scholes:**

- Với CW mua (Call):

\[
C = S \cdot N(d_1) - K \cdot e^{-rT} \cdot N(d_2)
\]

- Với CW bán (Put):

\[
P = K \cdot e^{-rT} \cdot N(-d_2) - S \cdot N(-d_1)
\]

Trong đó:

\[
d_1 = \frac{\ln(S/K) + (r + \sigma^2/2)T}{\sigma \sqrt{T}}, \quad d_2 = d_1 - \sigma \sqrt{T}
\]

---

### **Giải thích các biến:**

- \( S \): Giá cổ phiếu hiện tại  
- \( K \): Giá thực hiện  
- \( T \): Thời gian đến ngày đáo hạn (tính theo năm)  
- \( r \): Lãi suất phi rủi ro  
- \( \sigma \): Độ biến động (volatility) của cổ phiếu  
- \( N(d) \): Hàm phân phối chuẩn tích lũy (CDF)

---

# 📌 III. Các hệ số nhạy cảm (Greek Letters)

✅ **Delta**: Đo độ nhạy của giá CW với giá tài sản cơ sở  
\[
\Delta = N(d_1) \quad (\text{Call}), \quad -N(-d_1) \quad (\text{Put})
\]

✅ **Gamma**: Đo độ thay đổi của Delta  
\[
\Gamma = \frac{N'(d_1)}{S \cdot \sigma \cdot \sqrt{T}} = \frac{e^{-d_1^2/2}}{\sqrt{2\pi} \cdot S \cdot \sigma \cdot \sqrt{T}}
\]

---

# 📌 IV. Mở rộng với đặc điểm của CW tại Việt Nam

Tại Việt Nam, CW có những đặc điểm như:

- **Tỷ lệ chuyển đổi** (conversion ratio) không phải 1:1  
- **Hiệu ứng pha loãng** không tồn tại như quyền chọn thực  
- **Giới hạn giá** nên cần điều chỉnh giá lý thuyết:

\[
\text{Giá lý thuyết CW} = \frac{C \cdot R}{\text{Tỷ lệ chuyển đổi}}
\]

> Trong đó \( C \) là giá quyền theo BS, \( R \) là hệ số điều chỉnh hoặc chiết khấu từ tổ chức phát hành.



