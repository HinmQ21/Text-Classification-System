# 🔤 Text Classification System

Hệ thống phân loại văn bản đa ngôn ngữ với khả năng xử lý real-time và batch processing sử dụng Machine Learning.

## 📋 Tổng quan

Đây là một hệ thống phân loại văn bản hoàn chỉnh với:

- **Frontend**: React.js với TypeScript
- **Backend**: FastAPI (Python) 
- **Machine Learning**: Transformer models với Hugging Face
- **Queue System**: Redis + RQ cho xử lý bất đồng bộ
- **Database**: SQLite với SQLAlchemy ORM
- **Deployment**: Docker & Docker Compose

## ✨ Tính năng chính

### 🎯 Phân loại văn bản
- **Sentiment Analysis**: Phân tích cảm xúc (Positive/Negative/Neutral)
- **Spam Detection**: Phát hiện thư rác
- **Topic Classification**: Phân loại chủ đề

### 🌍 Hỗ trợ đa ngôn ngữ
- Tự động phát hiện ngôn ngữ
- Dịch tự động sang tiếng Anh (Google Gemini API)
- Hỗ trợ 100+ ngôn ngữ

### ⚡ Xử lý đa dạng
- **Real-time**: Phân loại tức thì cho văn bản đơn
- **Batch Processing**: Xử lý hàng loạt với CSV files
- **Async Processing**: Xử lý bất đồng bộ với queue system

### 🔐 Xác thực & Quản lý
- Đăng ký/Đăng nhập người dùng
- JWT Authentication
- Lịch sử truy vấn
- Rate limiting

### 📊 Visualization & Export
- Biểu đồ confidence scores
- Export kết quả (CSV/JSON)
- Real-time progress tracking

## 🏗️ Kiến trúc hệ thống

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend API   │    │ Background      │
│   React + TS    │◄──►│   FastAPI       │◄──►│ Workers (RQ)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   Database      │    │   Redis         │
                       │   SQLite        │    │   Cache/Queue   │
                       └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   ML Models     │
                       │   Transformers  │
                       └─────────────────┘
```

## 🚀 Cài đặt và chạy

### Prerequisites

- Python 3.8+
- Node.js 16+
- Redis Server
- Git

### 1. Clone repository

```bash
git clone <repository-url>
cd text-classification-sys
```

### 2. Cài đặt Backend

```bash
cd backend

# Tạo virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Cài đặt dependencies
pip install -r requirements.txt
```

### 3. Cài đặt Frontend

```bash
cd frontend
npm install
```

### 4. Thiết lập môi trường

Tạo file `.env` trong thư mục `backend/`:

```env
# API Keys
GEMINI_API_KEY=your_gemini_api_key_here

# Database
DATABASE_URL=sqlite:///./text_classification.db

# JWT
SECRET_KEY=your_secret_key_here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Redis
REDIS_URL=redis://localhost:6379/0
```

### 5. Chạy hệ thống

#### Phương pháp 1: Manual startup

```bash
# 1. Khởi động Redis server
redis-server

# 2. Khởi động backend (từ thư mục backend/)
cd backend
python run_system.bat  # Hoặc chạy lệnh riêng lẻ:
# python start_workers.py --mode monitor
# python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 3. Khởi động frontend (từ thư mục frontend/)
cd frontend
npm start
```

#### Phương pháp 2: Docker Compose

```bash
docker-compose up --build
```

### 6. Truy cập ứng dụng

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **RQ Dashboard**: http://localhost:9181 (khi chạy manual)

## 📖 Sử dụng

### API Endpoints chính

#### 1. Phân loại văn bản đơn

```bash
POST /classify
{
  "text": "I love this product!",
  "model_type": "sentiment",
  "temperature": 0.7,
  "model_selection": "all",
  "enable_translation": true
}
```

#### 2. Phân loại bất đồng bộ

```bash
POST /classify/async
{
  "text": "Your text here",
  "model_type": "spam"
}
```

#### 3. Upload CSV cho batch processing

```bash
POST /classify/csv
Content-Type: multipart/form-data
- file: your_file.csv
- model_type: sentiment
- text_column: text
- batch_size: 10
```

#### 4. Xác thực người dùng

```bash
# Đăng ký
POST /auth/register
{
  "email": "user@example.com",
  "password": "password123"
}

# Đăng nhập
POST /auth/login
{
  "email": "user@example.com", 
  "password": "password123"
}
```

### Frontend Usage

1. **Văn bản đơn**: Nhập text vào form và chọn model type
2. **File CSV**: Upload file CSV với cột text
3. **Theo dõi tiến trình**: Xem real-time progress cho batch jobs
4. **Xem lịch sử**: Truy cập lịch sử các truy vấn đã thực hiện
5. **Export kết quả**: Download kết quả dưới dạng CSV/JSON

## 🔧 Cấu hình

### Model Configuration

File `backend/models/` chứa cấu hình cho các models:

- **Sentiment**: Phân tích cảm xúc
- **Spam**: Phát hiện spam  
- **Topic**: Phân loại chủ đề

### Queue Configuration

File `backend/config/redis_config.py`:

```python
REDIS_URL = "redis://localhost:6379/0"
QUEUE_NAMES = ["default", "classification", "csv_processing"]
```

### Worker Configuration

```bash
# Chạy workers với monitoring
python start_workers.py --mode monitor

# Chạy workers cụ thể
python start_workers.py --queues classification,csv_processing
```

## 📁 Cấu trúc thư mục

```
text-classification-sys/
├── backend/
│   ├── config/              # Cấu hình Redis, database
│   ├── models/              # Database models & schemas  
│   ├── services/            # Business logic services
│   ├── tasks/               # Background tasks
│   ├── data/                # Database & uploaded files
│   ├── main.py              # FastAPI application
│   ├── worker.py            # RQ worker
│   ├── start_workers.py     # Worker management
│   ├── requirements.txt     # Python dependencies
│   └── Dockerfile           # Backend container
├── frontend/
│   ├── src/                 # React source code
│   ├── public/              # Static files
│   ├── package.json         # Node dependencies
│   └── Dockerfile           # Frontend container
├── docker-compose.yml       # Multi-container setup
├── .gitignore
└── README.md
```

## 🧪 Testing

### Backend Tests

```bash
cd backend
python -m pytest tests/
```

### Frontend Tests

```bash
cd frontend
npm test
```

### API Testing

Sử dụng Swagger UI tại `http://localhost:8000/docs` để test các endpoints.

## 🚀 Deployment

### Production với Docker

```bash
# Build và deploy
docker-compose -f docker-compose.prod.yml up --build -d

# Scaling workers
docker-compose up --scale worker=3
```

### Environment Variables cho Production

```env
# Production settings
DEBUG=False
ALLOWED_HOSTS=your-domain.com
DATABASE_URL=postgresql://user:pass@db:5432/textclassification
REDIS_URL=redis://redis:6379/0
```

## 🔍 Monitoring

### RQ Dashboard

Truy cập http://localhost:9181 để xem:
- Active jobs
- Failed jobs  
- Worker status
- Queue statistics

### Health Checks

```bash
# System health
GET /health

# Queue status  
GET /queue/info

# Specific job status
GET /queue/status/{job_id}
```

## 🛠️ Troubleshooting

### Lỗi thường gặp

1. **Redis connection failed**
   ```bash
   # Khởi động Redis
   redis-server
   ```

2. **Model loading error**
   ```bash
   # Kiểm tra GEMINI_API_KEY trong .env
   # Đảm bảo internet connection
   ```

3. **Frontend proxy error**
   ```bash
   # Đảm bảo backend đang chạy trên port 8000
   # Kiểm tra CORS settings
   ```

4. **Worker not processing jobs**
   ```bash
   # Restart workers
   python start_workers.py --mode restart
   ```

### Debug Mode

```bash
# Backend với debug logs
DEBUG=True python -m uvicorn main:app --reload --log-level debug

# Frontend với debug
REACT_APP_DEBUG=true npm start
```

## 📄 License

MIT License - xem file LICENSE để biết thêm chi tiết.

## 🤝 Contributing

1. Fork repository
2. Tạo feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Tạo Pull Request

## 📞 Support

Nếu gặp vấn đề, vui lòng tạo issue trên GitHub repository hoặc liên hệ qua email.

---

**Made with ❤️ by Text Classification Team** 