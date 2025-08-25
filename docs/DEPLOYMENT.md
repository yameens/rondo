# Rondo Deployment Guide

This guide covers deploying Rondo to various platforms.

## 🚀 Quick Deploy Options

### **Option 1: Vercel + Railway + Supabase (Recommended)**

#### **Frontend (Vercel)**
1. **Connect Repository**:
   - Go to [vercel.com](https://vercel.com)
   - Import your GitHub repository: `https://github.com/yameens/rondo`
   - Set root directory to `frontend`

2. **Configure Build**:
   - Build Command: `npm run build`
   - Output Directory: `.next`
   - Install Command: `npm install`

3. **Environment Variables**:
   ```
   NEXT_PUBLIC_API_URL=https://your-backend.railway.app
   ```

4. **Deploy**: Click "Deploy"

#### **Backend (Railway)**
1. **Connect Repository**:
   - Go to [railway.app](https://railway.app)
   - Import your GitHub repository
   - Set root directory to `backend`

2. **Configure Service**:
   - Build Command: `pip install -r requirements-minimal.txt`
   - Start Command: `uvicorn app.demo:app --host 0.0.0.0 --port $PORT`

3. **Environment Variables**:
   ```
   DATABASE_URL=your-supabase-db-url
   REDIS_URL=your-redis-url
   SECRET_KEY=your-secret-key
   ```

4. **Deploy**: Railway will auto-deploy

#### **Database (Supabase)**
1. **Create Project**:
   - Go to [supabase.com](https://supabase.com)
   - Create new project

2. **Get Connection String**:
   - Go to Settings > Database
   - Copy connection string

3. **Update Backend**: Use connection string in Railway environment variables

---

### **Option 2: Render (All-in-One)**

1. **Connect Repository**:
   - Go to [render.com](https://render.com)
   - Import your GitHub repository

2. **Deploy Backend**:
   - Create new Web Service
   - Build Command: `pip install -r backend/requirements-minimal.txt`
   - Start Command: `cd backend && uvicorn app.demo:app --host 0.0.0.0 --port $PORT`

3. **Deploy Frontend**:
   - Create new Web Service
   - Build Command: `cd frontend && npm install && npm run build`
   - Start Command: `cd frontend && npm start`

4. **Environment Variables**: Set `NEXT_PUBLIC_API_URL` to your backend URL

---

### **Option 3: Railway (Complete Stack)**

1. **Deploy with Docker Compose**:
   ```bash
   # Install Railway CLI
   npm install -g @railway/cli
   
   # Login and deploy
   railway login
   railway init
   railway up
   ```

2. **Configure Services**:
   - PostgreSQL database (auto-provisioned)
   - Redis cache (auto-provisioned)
   - Backend service
   - Frontend service

---

## 🔧 Environment Variables

### **Backend Variables**
```env
# Database
DATABASE_URL=postgresql://user:pass@host:port/db

# Redis
REDIS_URL=redis://host:port

# Security
SECRET_KEY=your-secret-key-here

# Storage
SUPABASE_URL=your-supabase-url
SUPABASE_KEY=your-supabase-key

# Environment
ENVIRONMENT=production
DEBUG=false
```

### **Frontend Variables**
```env
# API Configuration
NEXT_PUBLIC_API_URL=https://your-backend-url.com

# Analytics (optional)
NEXT_PUBLIC_GA_ID=your-google-analytics-id

# Feature Flags
NEXT_PUBLIC_ENABLE_ANALYTICS=false
NEXT_PUBLIC_ENABLE_FEEDBACK=true
```

## 📊 Platform Comparison

| Platform | Frontend | Backend | Database | Storage | Cost | Difficulty |
|----------|----------|---------|----------|---------|------|------------|
| **Vercel** | ✅ | ❌ | ❌ | ❌ | Free | Easy |
| **Railway** | ✅ | ✅ | ✅ | ✅ | $5/mo | Medium |
| **Render** | ✅ | ✅ | ✅ | ✅ | Free | Medium |
| **Heroku** | ✅ | ✅ | ✅ | ✅ | $7/mo | Medium |
| **DigitalOcean** | ✅ | ✅ | ✅ | ✅ | $12/mo | Hard |

## 🚀 Recommended Deployment Path

### **For Beginners (Free)**
1. **Frontend**: Vercel
2. **Backend**: Render
3. **Database**: Supabase
4. **Storage**: Supabase Storage

### **For Production (Paid)**
1. **Frontend**: Vercel Pro
2. **Backend**: Railway
3. **Database**: Supabase Pro
4. **Storage**: Supabase Storage

## 🔍 Post-Deployment Checklist

- [ ] **Health Check**: Verify `/health` endpoint
- [ ] **CORS**: Check frontend-backend communication
- [ ] **File Upload**: Test file upload functionality
- [ ] **Database**: Verify database connections
- [ ] **SSL**: Ensure HTTPS is working
- [ ] **Performance**: Test with sample files
- [ ] **Monitoring**: Set up error tracking

## 🛠 Troubleshooting

### **Common Issues**

1. **CORS Errors**:
   - Add frontend URL to backend CORS settings
   - Check environment variables

2. **Database Connection**:
   - Verify connection string format
   - Check network access

3. **File Upload Failures**:
   - Check storage configuration
   - Verify file size limits

4. **Build Failures**:
   - Check dependency versions
   - Verify build commands

### **Support Resources**
- [Vercel Documentation](https://vercel.com/docs)
- [Railway Documentation](https://docs.railway.app)
- [Supabase Documentation](https://supabase.com/docs)
- [Render Documentation](https://render.com/docs)
