Demo 1: 

Bug: 
1. Không thấy pipelines ở trong Admin => Khởi động lại images của docker. 
```bash
docker stop open-webui pipelines
docker rm open-webui pipelines
./dev-docker-up.sh
```

2. Tải lên .py mà gặp lỗi: Nonetype no attibute scoll => check thì thấy bug chưa có KEYs 

3. Bug khi tải lên .py thành công rồi mà không thấy models đâu. 
Check bug thì thấy lỗi xung đột thư viện: `docker exec -it pipelines pip install --upgrade pydantic`


Lỗi này liên quan đến vấn đề tương thích phiên bản của Pydantic. Có 2 cách giải quyết:

1. *Cách 1: Cập nhật phiên bản Pydantic*
docker exec -it pipelines pip install --upgrade pydantic

2. *Cách 2: Chỉ định phiên bản cụ thể* (khuyến nghị)
docker exec -it pipelines pip install "pydantic>=2.5.0,<3.0.0"

Sau đó khởi động lại container:
docker restart pipelines

*Giải thích lỗi:*
- Lỗi cannot import name 'can_be_positional' from 'pydantic._internal._utils' thường xảy ra do không tương thích giữa các phiên bản Pydantic
- Nguyên nhân có thể do:
  - Phiên bản Pydantic quá cũ
  - Xung đột giữa các dependency sử dụng các phiên bản Pydantic khác nhau
  - Langchain và các thư viện khác yêu cầu phiên bản Pydantic cụ thể

*Lưu ý:* Nếu vẫn gặp lỗi, bạn có thể thử:
1. Xóa và tạo lại container
2. Cập nhật các dependency khác
3. Kiểm tra requirements.txt để đảm bảo các phiên bản phù hợp


----
```
[LOG BUG WHEN DEMO 1: tổng 12h + 12h = 24h = 3 triệu] done: 6h chiều chủ nhật + 6h tối hôm CN 05.01.2025 VN đi Bão: Done được RUN CƠ BẢN VÀ CHẠY ĐƯỢC PIPELINES EXAMPLE thử custom pipeline qdant mà bug  + 6h xuyên từ 11h đến 6h sáng hôm thứ 6 hôm nay 10.01.2024 Done việc triển pipelines qdant từng bước từ việc test với qdant riêng, đến việc lắp code vào và import lên thấy model, sửa để UI hiển trùng với backend, sửa cấu trúc search để tìm dựa vào questions + từ 10h sáng đến 15h chiều (5h). 
```

vIDEO DEMO: https://youtu.be/m7VwaQ1iE5s