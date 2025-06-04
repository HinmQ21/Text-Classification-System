# 🔤 Text Classification System

Hệ thống phân loại văn bản đa ngôn ngữ với khả năng xử lý real-time và batch processing sử dụng Machine Learning.

## 📋 Tổng quan

Đây là một hệ thống phân loại văn bản hoàn chỉnh với:

- **Frontend**: React.js với TypeScript
- **Backend**: FastAPI (Python) 
- **Machine Learning**: Transformer models với Hugging Face
- **Queue System**: Redis + RQ cho xử lý bất đồng bộ
- **Database**: MySQL với connection pooling và optimization
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
                       │   MySQL         │    │   Cache/Queue   │
                       └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   ML Models     │
                       │   Transformers  │
                       └─────────────────┘
```
## System flow

![system flow](docs/flow.jpg)

## 🚀 Cài đặt và chạy

### Prerequisites

- Python 3.9+
- Node.js 16+
- Redis Server
- Docker & Docker Compose (cho deployment)
- MySQL Server
- Git

### 1. Clone repository

```bash
git clone <repository-url>
cd text-classification-sys
```

### 2. Thiết lập Environment Variables

```bash
# Copy file cấu hình mẫu
cp env.example .env

# Chỉnh sửa file .env với các thông tin cần thiết
# Đặc biệt cần cập nhật:
# - GEMINI_API_KEY=your_actual_api_key
# - SECRET_KEY=your_secure_secret_key
```

### 3. Chạy với Docker (Khuyến nghị)

#### Development Mode

```bash
# Khởi động tất cả services
docker-compose up --build

# Chạy ở background
docker-compose up -d --build

# Xem logs
docker-compose logs -f

# Dừng services
docker-compose down
```

#### Production Mode

```bash
# Tạo file .env cho production với các giá trị thực tế
cp env.example .env

# Chỉnh sửa .env với các giá trị production:
# DEBUG=False
# SECRET_KEY=your-super-secure-secret-key
# GEMINI_API_KEY=your-actual-gemini-api-key

# Deploy production
docker-compose -f docker-compose.prod.yml up -d --build

# Scaling workers cho production
docker-compose -f docker-compose.prod.yml up -d --scale worker=3
```

### 4. Manual Setup (Development)

#### Backend Setup

```bash
cd backend

# Tạo virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Cài đặt dependencies
pip install -r requirements.txt

# Thiết lập .env file
cp ../env.example .env
# Chỉnh sửa .env với các giá trị cần thiết
```

#### Frontend Setup

```bash
cd frontend
npm install
```

#### Manual Startup

```bash
# 1. Khởi động Redis server
redis-server

# 2. Khởi động backend (từ thư mục backend/)
cd backend
python run_system.bat  # Windows
# Hoặc chạy lệnh riêng lẻ:
# python start_workers.py --mode monitor
# python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 3. Khởi động frontend (từ thư mục frontend/)
cd frontend
npm start
```

### 5. Truy cập ứng dụng

#### Development Mode
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **RQ Dashboard**: http://localhost:9181

#### Production Mode
- **Frontend**: http://localhost (port 80)
- **Backend API**: http://localhost:8000
- **RQ Dashboard**: http://localhost:9181

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

### Environment Variables

Các biến môi trường quan trọng trong file `.env`:

```env
# API Keys
GEMINI_API_KEY=your_gemini_api_key_here

# Security
SECRET_KEY=your-super-secret-jwt-key
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Database
DATABASE_URL=mysql+pymysql://app_user:your_password@localhost/text_classification

# Redis
REDIS_URL=redis://localhost:6379/0
```

### Docker Services

- **redis**: Message queue và cache
- **backend**: FastAPI application server
- **worker**: RQ background workers (có thể scale)
- **rq-dashboard**: Monitoring dashboard cho queues
- **frontend**: React application với Nginx

### Queue Configuration

```python
REDIS_URL = "redis://redis:6379/0"  # Docker
QUEUE_NAMES = ["default", "classification", "csv_processing"]
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
│   ├── logs/                # Log files
│   ├── main.py              # FastAPI application
│   ├── worker.py            # RQ worker
│   ├── start_workers.py     # Worker management
│   ├── requirements.txt     # Python dependencies
│   ├── Dockerfile           # Backend container
│   ├── docker-compose.yml   # Development setup
│   └── migrate_to_mysql.py  # Migration script
├── frontend/
│   ├── src/                 # React source code
│   ├── public/              # Static files
│   ├── package.json         # Node dependencies
│   ├── nginx.conf           # Nginx configuration
│   └── Dockerfile           # Frontend container
├── docker-compose.yml       # Development setup
├── docker-compose.prod.yml  # Production setup
├── env.example              # Environment variables template
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

### Docker Testing

```bash
# Test health checks
docker-compose ps
docker-compose logs backend
docker-compose logs worker

# Test specific service
docker-compose exec backend curl http://localhost:8000/health
```

## 🚀 Deployment

### Development Deployment

```bash
# Build và khởi động
docker-compose up --build

# Chạy ở background
docker-compose up -d --build

# Theo dõi logs
docker-compose logs -f
```

### Production Deployment

```bash
# Thiết lập environment variables cho production
cp env.example .env
# Chỉnh sửa .env với các giá trị production

# Deploy production stack
docker-compose -f docker-compose.prod.yml up -d --build

# Scale workers
docker-compose -f docker-compose.prod.yml up -d --scale worker=3

# Monitoring
docker-compose -f docker-compose.prod.yml logs -f
```

### Maintenance Commands

```bash
# Restart specific service
docker-compose restart backend

# Update and rebuild
docker-compose down
docker-compose pull
docker-compose up --build -d

# Clean up
docker-compose down --volumes --remove-orphans
docker system prune -a
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

# Docker container health
docker-compose ps
```

### Logs

```bash
# Xem logs tất cả services
docker-compose logs -f

# Xem logs specific service
docker-compose logs -f backend
docker-compose logs -f worker
docker-compose logs -f frontend

# Xem logs với timestamps
docker-compose logs -f -t
```

## 🛠️ Troubleshooting

### Lỗi thường gặp

1. **Redis connection failed**
   ```bash
   # Kiểm tra Redis service
   docker-compose ps redis
   docker-compose logs redis
   
   # Restart Redis
   docker-compose restart redis
   ```

2. **Backend không khởi động**
   ```bash
   # Kiểm tra logs
   docker-compose logs backend
   
   # Kiểm tra environment variables
   docker-compose exec backend env | grep -E "(REDIS|DATABASE|GEMINI)"
   ```

3. **Worker không xử lý jobs**
   ```bash
   # Kiểm tra worker status
   docker-compose logs worker
   
   # Restart workers
   docker-compose restart worker
   
   # Scale workers
   docker-compose up -d --scale worker=2
   ```

4. **Frontend không kết nối được backend**
   ```bash
   # Kiểm tra nginx config
   docker-compose exec frontend cat /etc/nginx/conf.d/default.conf
   
   # Kiểm tra network connectivity
   docker-compose exec frontend curl http://backend:8000/health
   ```

### Debug Commands

```bash
# Access container shell
docker-compose exec backend bash
docker-compose exec frontend sh

# Check environment variables
docker-compose exec backend env

# Check network connectivity
docker-compose exec backend ping redis
docker-compose exec frontend ping backend

# Monitor resource usage
docker stats
```

### Performance Tuning

```bash
# Scale workers theo nhu cầu
docker-compose up -d --scale worker=3

# Monitor resource usage
docker stats

# Optimize Redis memory
docker-compose exec redis redis-cli CONFIG GET maxmemory
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