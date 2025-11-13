# System Testing Guide

## Complete Testing Checklist

### ✅ Phase 1: Setup Verification

```bash
# 1. Check Python version
python --version
# Expected: Python 3.10 or higher

# 2. Verify virtual environment
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# 3. Check installed packages
pip list | grep -E "fastapi|torch|xgboost|transformers"

# Expected output:
# fastapi==0.104.1
# torch==2.1.0
# xgboost==2.0.3
# transformers==4.35.2
```

### ✅ Phase 2: Backend Testing

#### Test 1: Start Backend Server

```bash
# Start the server
uvicorn backend.main:app --reload

# Expected output:
# INFO: Uvicorn running on http://0.0.0.0:8000
# INFO: Application startup complete
```

#### Test 2: Health Check

```bash
# In a new terminal, test health endpoint
curl http://localhost:8000/health

# Expected response:
{
  "status": "healthy",
  "models": {
    "soil_detector": false,  # true if model is loaded
    "fertilizer_advisor": true,
    "nlp_handler": false  # true if model is loaded
  }
}
```

#### Test 3: API Documentation

Open browser: `http://localhost:8000/docs`

Expected: Interactive Swagger UI showing all endpoints

#### Test 4: Test Endpoints

```bash
# Get supported crops
curl http://localhost:8000/api/v1/crops

# Get soil types
curl http://localhost:8000/api/v1/soil-types

# Get fertilizers
curl http://localhost:8000/api/v1/fertilizers
```

### ✅ Phase 3: Functional Testing

#### Test 1: Fertilizer Recommendation (No Image)

```bash
curl -X POST "http://localhost:8000/api/v1/recommend-fertilizer" \
  -H "Content-Type: application/json" \
  -d '{
    "soil_data": {
      "pH": 6.5,
      "Nitrogen": 45,
      "Phosphorus": 25,
      "Potassium": 35
    },
    "soil_type": "Loamy",
    "crop": "Wheat",
    "field_area": 1.0,
    "growth_stage": "Vegetative",
    "prefer_organic": false
  }'
```

Expected: JSON response with fertilizer recommendations

#### Test 2: Query Processing

```bash
curl -X POST "http://localhost:8000/api/v1/process-query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What fertilizer should I use for wheat?"
  }'
```

Expected: JSON response with intent and entities

#### Test 3: Soil Detection (With Demo Image)

```bash
# Create a test image or use existing one
curl -X POST "http://localhost:8000/api/v1/detect-soil" \
  -F "file=@path/to/soil_image.jpg"
```

Expected: JSON response with soil type and confidence

### ✅ Phase 4: Frontend Testing

#### Test 1: Open Frontend

1. Open `frontend/index.html` in browser
2. Check all sections load correctly:
   - ✅ Navigation bar
   - ✅ Hero section
   - ✅ Features section
   - ✅ Soil Detection form
   - ✅ Fertilizer Recommendation form
   - ✅ Query form

#### Test 2: Form Validation

1. Try submitting empty forms
2. Check validation messages appear
3. Upload invalid file types
4. Check error handling

#### Test 3: Complete Workflow

1. **Soil Detection:**
   - Upload soil image
   - Check loading spinner appears
   - Verify result display

2. **Complete Recommendation:**
   - Upload soil image
   - Select crop
   - Fill in NPK values
   - Submit form
   - Verify comprehensive results

3. **Query Processing:**
   - Enter query
   - Check language detection
   - Verify intent classification

### ✅ Phase 5: Integration Testing

#### Test 1: Run Quick Start Script

```bash
python quick_start.py
```

Expected output:
```
==============================================================
PLANT-SPECIFIC FERTILIZER & SOIL RECOMMENDATION SYSTEM
Quick Start Test Suite
==============================================================

TESTING FERTILIZER RECOMMENDATION SYSTEM
==============================================================

📊 Soil Test Results:
  pH: 6.5
  Nitrogen: 45
  ...

🌾 Generating recommendation for Wheat crop...

==============================================================
RECOMMENDATION RESULTS
==============================================================
...

ALL TESTS COMPLETED SUCCESSFULLY! ✅
```

#### Test 2: Docker Deployment

```bash
# Build and start containers
docker-compose up --build

# Test frontend
curl http://localhost:80

# Test backend
curl http://localhost:8000/health

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### ✅ Phase 6: Performance Testing

#### Test 1: Response Time

```bash
# Test API response time
time curl http://localhost:8000/api/v1/crops

# Expected: < 100ms
```

#### Test 2: Image Processing Time

```bash
# Test soil detection time
time curl -X POST "http://localhost:8000/api/v1/detect-soil" \
  -F "file=@soil_image.jpg"

# Expected: < 2 seconds (without GPU)
# Expected: < 500ms (with GPU)
```

#### Test 3: Concurrent Requests

```bash
# Install apache bench
apt-get install apache2-utils  # Linux
brew install ab  # Mac

# Run concurrent test
ab -n 100 -c 10 http://localhost:8000/health

# Expected: No errors, consistent response times
```

### ✅ Phase 7: Model Testing

#### Test 1: Train Soil Detector (Optional)

```bash
# Prepare sample images in data/soil_images/
# Then run training
python notebooks/train_soil_detector.py

# Check for:
# - Training progress
# - Validation accuracy
# - Model saved successfully
```

#### Test 2: Train Fertilizer Model

```bash
python notebooks/train_fertilizer_model.py

# Check for:
# - Dataset created
# - Training completed
# - Model saved
# - Feature importance displayed
```

### ✅ Phase 8: Error Handling

#### Test 1: Invalid Inputs

```bash
# Test invalid soil data
curl -X POST "http://localhost:8000/api/v1/recommend-fertilizer" \
  -H "Content-Type: application/json" \
  -d '{
    "soil_data": {
      "pH": -1,  # Invalid pH
      "Nitrogen": "abc"  # Invalid type
    }
  }'

# Expected: 422 Validation Error
```

#### Test 2: Missing Files

```bash
# Test without file upload
curl -X POST "http://localhost:8000/api/v1/detect-soil"

# Expected: 422 Unprocessable Entity
```

#### Test 3: Large File Upload

```bash
# Create large file (> 10MB)
dd if=/dev/zero of=large.jpg bs=1M count=15

# Try uploading
curl -X POST "http://localhost:8000/api/v1/detect-soil" \
  -F "file=@large.jpg"

# Expected: 413 Payload Too Large or handled gracefully
```

### ✅ Phase 9: Browser Compatibility

Test frontend in:
- ✅ Chrome
- ✅ Firefox
- ✅ Edge
- ✅ Safari
- ✅ Mobile browsers

### ✅ Phase 10: Deployment Testing

#### AWS Deployment Test

```bash
# Build Docker image
docker build -t agrismart:latest .

# Run locally
docker run -p 8000:8000 agrismart:latest

# Test
curl http://localhost:8000/health
```

#### GCP Deployment Test

```bash
# Test app engine configuration
gcloud app deploy deployment/gcp_deploy.yaml --no-promote
```

## Test Results Checklist

### Backend Tests
- [ ] Server starts successfully
- [ ] Health check returns 200
- [ ] API documentation accessible
- [ ] All endpoints respond
- [ ] Error handling works
- [ ] CORS configured correctly

### Frontend Tests
- [ ] All pages load
- [ ] Forms submit correctly
- [ ] Validation works
- [ ] Results display properly
- [ ] Responsive on mobile
- [ ] Cross-browser compatible

### Integration Tests
- [ ] Quick start script passes
- [ ] Image upload works
- [ ] Recommendations generated
- [ ] Query processing works
- [ ] Complete pipeline functional

### Performance Tests
- [ ] Response time < 1s
- [ ] Image processing < 2s
- [ ] Handles concurrent requests
- [ ] Memory usage acceptable
- [ ] No memory leaks

### Deployment Tests
- [ ] Docker builds successfully
- [ ] Docker compose works
- [ ] Cloud deployment ready
- [ ] Environment variables set
- [ ] Logs accessible

## Common Issues & Solutions

### Issue 1: Import Errors

**Symptom:** ModuleNotFoundError
**Solution:**
```bash
pip install -r requirements.txt
# Or
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

### Issue 2: Port Already in Use

**Symptom:** Address already in use
**Solution:**
```bash
# Find and kill process using port 8000
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:8000 | xargs kill -9
```

### Issue 3: CORS Errors

**Symptom:** Access-Control-Allow-Origin
**Solution:** Backend is running and CORS is configured in `backend/main.py`

### Issue 4: Model Not Found

**Symptom:** FileNotFoundError for .pth or .pkl
**Solution:** System runs in demo mode without models. Train models or models are optional.

### Issue 5: Out of Memory

**Symptom:** CUDA out of memory
**Solution:**
```python
# Reduce batch size in config.py
SOIL_DETECTION_CONFIG['batch_size'] = 16  # Reduce from 32
```

## Test Report Template

```
Test Date: _______________
Tester: _______________
Environment: _______________

Backend Tests:        [ PASS / FAIL ]
Frontend Tests:       [ PASS / FAIL ]
Integration Tests:    [ PASS / FAIL ]
Performance Tests:    [ PASS / FAIL ]
Deployment Tests:     [ PASS / FAIL ]

Issues Found:
1. _______________________________
2. _______________________________
3. _______________________________

Overall Status:       [ PASS / FAIL ]

Notes:
_____________________________________
_____________________________________
_____________________________________
```

## Automated Testing Script

Create `run_tests.py`:

```python
#!/usr/bin/env python3
import subprocess
import sys

def run_test(name, command):
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"{'='*60}")
    result = subprocess.run(command, shell=True)
    return result.returncode == 0

tests = [
    ("Health Check", "curl -f http://localhost:8000/health"),
    ("Get Crops", "curl -f http://localhost:8000/api/v1/crops"),
    ("Quick Start", "python quick_start.py"),
]

results = []
for name, cmd in tests:
    results.append((name, run_test(name, cmd)))

print(f"\n{'='*60}")
print("TEST SUMMARY")
print(f"{'='*60}")
for name, passed in results:
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status} - {name}")

all_passed = all(r[1] for r in results)
sys.exit(0 if all_passed else 1)
```

## CI/CD Testing (Future)

```yaml
# .github/workflows/test.yml
name: Test Suite

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.10
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: python quick_start.py
      - name: Build Docker
        run: docker build -t agrismart:test .
```

---

## ✅ Final Checklist

Before marking complete:
- [ ] All setup scripts work
- [ ] Backend starts without errors
- [ ] Frontend loads correctly
- [ ] At least one complete workflow tested
- [ ] Quick start script passes
- [ ] Documentation reviewed
- [ ] Deployment configuration tested

**System is ready for production when all items are checked!** 🚀

---

© 2025 AgriSmart - Testing Documentation
