# Sử dụng một image Python làm nền
FROM python:3.9-slim

# Thiết lập thư mục làm việc bên trong container
WORKDIR /app

# Sao chép file requirements.txt vào container
COPY requirements.txt .

# Cài đặt các thư viện từ requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép các file mã nguồn khác vào container
COPY . .

# Chỉ định lệnh mặc định khi container khởi động
CMD ["python", "btc_trend_3m.py"]