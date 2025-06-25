# Render Deployment Checklist

## ✅ Pre-Deployment Steps

### 1. **Repository Setup**
- [ ] Code is pushed to GitHub, GitLab, or Bitbucket
- [ ] All changes are committed and pushed to main branch

### 2. **Environment Variables Required**
Set these in your Render service dashboard under "Environment" tab:

#### **Authentication & Security**
- [ ] `SECRET_KEY` - Generate with: `openssl rand -hex 32`
- [ ] `ALGORITHM` - Set to: `HS256`
- [ ] `ACCESS_TOKEN_EXPIRE_MINUTES` - Set to: `30`
- [ ] `REFRESH_TOKEN_EXPIRE_DAYS` - Set to: `7`

#### **Email Configuration**
- [ ] `SMTP_HOST` - Your email provider (e.g., `smtp.gmail.com`)
- [ ] `SMTP_PORT` - Usually `587`
- [ ] `SMTP_USERNAME` - Your email address
- [ ] `SMTP_PASSWORD` - Your email password or app password
- [ ] `FROM_EMAIL` - Email address to send from

#### **OTP Configuration**
- [ ] `OTP_EXPIRE_MINUTES` - Set to: `10`

#### **AI Integration**
- [ ] `OPENAI_API_KEY` - Your OpenAI API key

#### **Optional**
- [ ] `FRONTEND_URL` - Your frontend domain (for CORS)

**Note:** `DATABASE_URL` is automatically provided by Render's PostgreSQL service.

## 🚀 Deployment Steps

### 1. **Create Render Account**
- [ ] Sign up at [render.com](https://render.com)
- [ ] Connect your Git provider (GitHub, GitLab, Bitbucket)

### 2. **Deploy via Blueprint**
- [ ] Click "New +" → "Blueprint"
- [ ] Select your repository
- [ ] Render will auto-detect `render.yaml`
- [ ] Review the services (web app + PostgreSQL database)
- [ ] Click "Apply"

### 3. **Configure Environment Variables**
- [ ] Go to your web service dashboard
- [ ] Click "Environment" tab
- [ ] Add all required environment variables from above
- [ ] Mark sensitive variables (API keys, passwords) as "secret"
- [ ] Save changes

### 4. **Monitor Deployment**
- [ ] Check build logs for any errors
- [ ] Wait for both services to deploy successfully
- [ ] Verify database migrations ran correctly

## 🔍 Post-Deployment Verification

### 1. **Health Check**
- [ ] Visit `https://your-app-name.onrender.com/health`
- [ ] Should return: `{"status": "healthy", "database": "connected"}`

### 2. **API Endpoints**
- [ ] Test root endpoint: `GET /`
- [ ] Test authentication endpoints if needed
- [ ] Verify CORS is working with your frontend

### 3. **Database**
- [ ] Check that all tables were created properly
- [ ] Verify migrations completed successfully
- [ ] Test basic CRUD operations

## 🛠️ Troubleshooting

### Build Fails
- [ ] Check build logs in Render dashboard
- [ ] Verify all dependencies in `requirements.txt`
- [ ] Ensure `build.sh` is executable

### Database Issues
- [ ] Verify PostgreSQL service is running
- [ ] Check migration logs
- [ ] Ensure `DATABASE_URL` is properly connected

### Environment Variable Issues
- [ ] Double-check all required env vars are set
- [ ] Verify no typos in variable names
- [ ] Ensure sensitive vars are marked as secret

### CORS Issues
- [ ] Add your frontend domain to `FRONTEND_URL`
- [ ] Check CORS configuration in `app/main.py`

## 📝 Production Considerations

### Security
- [ ] Update CORS origins to only allow your frontend domain
- [ ] Use strong, unique passwords for all services
- [ ] Regularly rotate API keys and secrets

### Performance
- [ ] Consider upgrading to paid database plan for production
- [ ] Monitor application performance and logs
- [ ] Set up proper error tracking

### Maintenance
- [ ] Set up automated backups for your database
- [ ] Monitor service health and uptime
- [ ] Keep dependencies updated

## 🎉 Success!

If all items are checked and your health endpoint returns `{"status": "healthy"}`, your application is successfully deployed on Render!

Your API will be available at: `https://your-service-name.onrender.com` 