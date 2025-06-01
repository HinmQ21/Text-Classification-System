# 🏗️ Tổng hợp Kiến trúc Hệ thống Text Classification

## 1. 🎨 Frontend Layer (Tầng Giao diện)

### **Công nghệ**: React.js + TypeScript
- **Giao diện người dùng**: Web interface hiện đại, responsive
- **Tính năng chính**:
  - ✅ Text input và file upload (CSV)
  - ✅ Model selection (Sentiment/Spam/Topic)
  - ✅ Real-time results visualization
  - ✅ Confidence scores với charts
  - ✅ Batch processing progress tracking
  - ✅ Export results (CSV/JSON)
  - ✅ Multi-language support indicator

### **Thư viện UI**:
- **Styling**: Tailwind CSS hoặc Material-UI
- **Charts**: Chart.js/Recharts cho visualization
- **File handling**: React-dropzone
- **State management**: Redux Toolkit hoặc Context API

---

## 2. 🚪 API Gateway (Tầng API)

### **Công nghệ**: FastAPI (Python)
- **Vai trò**: Central API endpoint, request routing
- **Tính năng**:
  - ✅ Request validation và sanitization
  - ✅ Authentication & Authorization (JWT)
  - ✅ Rate limiting (Redis-based)
  - ✅ API documentation (Swagger/OpenAPI)
  - ✅ CORS handling
  - ✅ Request/Response logging
  - ✅ Health checks
  - ✅ Save user history queries

### **Security Features**:
- Input validation với Pydantic
- Request size limits
- SQL injection protection
- XSS protection headers

---

## 3. ⚙️ Background Services (Tầng Xử lý nền)

### **Công nghệ**: Celery + Redis
- **Message Queue**: Redis làm broker
- **Task Processing**: Celery workers
- **Tính năng**:
  - ✅ Async text processing
  - ✅ Batch file processing
  - ✅ Multi-threading cho large datasets
  - ✅ Task monitoring và retry logic
  - ✅ Scalable worker processes
  - ✅ Task result caching

### **Worker Configuration**:
python
# Celery settings
CELERY_BROKER_URL = "redis://localhost:6379/0"
CELERY_RESULT_BACKEND = "redis://localhost:6379/0"
CELERY_WORKER_CONCURRENCY = 4
CELERY_TASK_ROUTES = {
    'text_classification.*': {'queue': 'classification'},
    'translation.*': {'queue': 'translation'}
}


---

## 4. 🔄 Text Processing Pipeline

### **4.1 Language Detection**
- **Thư viện**: langdetect hoặc polyglot
- **Chức năng**: Detect ngôn ngữ input text
- **Supported languages**: 100+ languages
- **Fallback**: Default to English nếu không detect được

### **4.2 Translation Service**
- **API**: Google Gemini API
- **Chức năng**: Translate non-English → English
- **Features**:
  - ✅ Batch translation cho efficiency
  - ✅ Caching translated texts
  - ✅ Fallback to other translation APIs
  - ✅ Translation quality scoring

### **4.3 Text Classification**
- **Models**: Transformer-based models
- **Preprocessing**: Tokenization, cleaning, normalization
- **Post-processing**: Confidence calculation, result formatting

---

## 5. 🤖 Machine Learning Models

### **5.1 Model Architecture**
Base Models: Transformer-based

### **5.2 Model Management**
<!-- - **Storage**: Model versioning với MLflow
- **Serving**: TorchServe hoặc ONNX Runtime
- **A/B Testing**: Model comparison framework
- **Monitoring**: Performance tracking, drift detection -->

### **5.3 Training Pipeline**
python
# Model training workflow
1. Data preprocessing & augmentation
2. Model fine-tuning với Hugging Face
3. Evaluation & validation
4. Model registration & versioning
5. Deployment to production


---

## 6. 🗄️ Database & Storage

### **6.1 PostgreSQL (Primary Database)**
sql
-- Core tables
users (id, email, created_at, api_key)
classification_jobs (id, user_id, status, model_type, created_at)
results (id, job_id, text, prediction, confidence, language)
query_history(id, model_type, input_text, prediction, confidence, created_at)

### **6.2 Redis (Cache & Queue)**
- **Session storage**: User sessions
- **Result caching**: Classification results
- **Rate limiting**: API request counting
- **Task queue**: Celery job queue

### **6.3 File Storage**
- **Local/S3**: Model files, uploaded CSVs
- **Structured**: Organized by date/user/model

---

## 7. 🐳 Infrastructure & Deployment

### **7.1 Containerization (Docker)**
dockerfile
# Multi-stage build
├── Frontend (Node.js → Nginx)
├── API Gateway (Python FastAPI)
├── ML Workers (Python + Models)
├── Redis (Cache/Queue)
└── PostgreSQL (Database)


### **7.2 Container Orchestration**
- **Development**: Docker Compose
- **Production**: Kubernetes hoặc Docker Swarm
- **Load Balancing**: Nginx reverse proxy
- **Service Discovery**: Consul hoặc built-in K8s

<!-- ### **7.3 Monitoring & Logging**
yaml
Monitoring Stack:
  - Prometheus (metrics collection)
  - Grafana (dashboards)
  - ELK Stack (logging)
  - Jaeger (distributed tracing)


--- -->

## 8. 🔧 External Services

### **8.1 Translation API**
- **Primary**: Google Gemini API
<!-- - **Backup**: Google Translate API, DeepL API
- **Rate limiting**: API quota management
- **Cost optimization**: Caching strategy -->

<!-- ### **8.2 Model Hosting**
- **Options**: 
  - Self-hosted với TorchServe
  - Cloud ML services (AWS SageMaker, GCP AI Platform)
  - Hugging Face Inference API (cho prototyping)

--- -->

## 9. 📊 Key Features Implementation

### **9.1 Multi-language Support**
python
async def process_text(text: str, model_type: str):
    # 1. Detect language
    detected_lang = detect_language(text)
    
    # 2. Translate if not English
    if detected_lang != 'en':
        text = await translate_text(text, target='en')
    
    # 3. Classify
    result = await classify_text(text, model_type)
    
    return {
        'original_language': detected_lang,
        'classification': result,
        'confidence': result.confidence
    }


### **9.2 Batch Processing**
python
@celery.task
def process_csv_batch(file_path: str, model_type: str):
    df = pd.read_csv(file_path)
    results = []
    
    for idx, row in df.iterrows():
        result = process_text(row['text'], model_type)
        results.append(result)
        
        # Update progress
        update_progress(job_id, (idx + 1) / len(df) * 100)
    
    return results


### **9.3 Real-time Updates**
- **WebSocket**: Real-time progress updates
- **Server-Sent Events**: Alternative cho WebSocket
- **Polling**: Fallback mechanism

---

## 10. 🚀 Scalability & Performance

### **10.1 Horizontal Scaling**
- **API Gateway**: Multiple FastAPI instances
- **Workers**: Auto-scaling Celery workers
- **Database**: Read replicas, connection pooling
- **Cache**: Redis Cluster

### **10.2 Performance Optimization**
- **Model optimization**: ONNX, TensorRT
- **Batch processing** cho efficiency
- **Result caching** với TTL
- **Database indexing** on frequent queries

### **10.3 Load Testing**
python
# Performance targets
- Single text classification: < 500ms
- Batch processing: 1000 texts/minute
- Concurrent users: 100+
- Uptime: 99.9%


---

## 11. 🔒 Security & Compliance

### **11.1 Security Measures**
- **Authentication**: JWT tokens
- **Authorization**: Role-based access
- **Data encryption**: At rest và in transit
- **Input validation**: Comprehensive sanitization
- **Rate limiting**: Per user/IP limits

### **11.2 Privacy & Compliance**
- **Data retention**: Configurable cleanup policies
- **GDPR compliance**: Data export/deletion
- **Audit logging**: All user actions logged
- **PII handling**: Text anonymization options

---

## 12. 📈 Monitoring & Analytics

### **12.1 System Metrics**
- **Performance**: Response times, throughput
- **Resources**: CPU, memory, disk usage
- **Errors**: Error rates, failure patterns
- **Business**: Classification accuracy, user engagement

### **12.2 Alerting**
yaml
Alerts:
  - High error rate (>5%)
  - Response time >1s
  - Queue length >1000
  - Model accuracy drop
  - API rate limit exceeded


---

## 13. 🚦 Deployment Strategy

### **13.1 CI/CD Pipeline**
yaml
stages:
  - test: Unit tests, integration tests
  - build: Docker images
  - deploy-staging: Staging environment
  - test-e2e: End-to-end tests
  - deploy-prod: Production deployment
  - monitor: Health checks


### **13.2 Blue-Green Deployment**
- **Zero downtime** deployments
- **Quick rollback** capability
- **Traffic switching** với load balancer

---

## 14. 💰 Cost Optimization

### **14.1 Resource Management**
- **Auto-scaling**: Scale based on demand
- **Spot instances**: For non-critical workloads
- **Resource pooling**: Shared resources
- **Caching**: Reduce API calls

### **14.2 Budget Monitoring**
- **Cost tracking**: Per service/user
- **Alerts**: Budget thresholds
- **Optimization**: Regular cost reviews

---

## 15. 🔄 Future Enhancements

### **15.1 Advanced Features**
- **Custom model training**: User-uploaded datasets
- **Multi-modal classification**: Text + images
- **Real-time streaming**: Kafka integration
- **Advanced analytics**: User behavior insights

### **15.2 Technology Upgrades**
- **Latest models**: GPT, Claude, Llama integration
- **Edge deployment**: Model serving at edge
- **Serverless**: Functions-as-a-Service
- **GraphQL**: Alternative to REST API

---