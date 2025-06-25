# Render Deployment Guide

This guide will help you deploy your FastAPI application to Render.

## Prerequisites

1. A Render account (free tier available)
2. Your code pushed to a Git repository (GitHub, GitLab, or Bitbucket)

## Files Created for Render Deployment

- `render.yaml` - Render service configuration
- `build.sh` - Build script for dependency installation and migrations
- `env.example` - Template for environment variables
- `environment_variables.md` - Documentation for required environment variables

## Deployment Steps

### 1. Connect Your Repository

1. Log in to your Render dashboard
2. Click "New +" and select "Blueprint"
3. Connect your Git repository
4. Select the repository containing your FastAPI app

### 2. Configure Environment Variables

In your Render service dashboard, go to the "Environment" tab and add these variables:

#### Required Variables:
- `SECRET_KEY` - A secure random string (generate one using `openssl rand -hex 32`)
- `ALGORITHM` - Set to `HS256`
- `ACCESS_TOKEN_EXPIRE_MINUTES` - Set to `30`
- `REFRESH_TOKEN_EXPIRE_DAYS` - Set to `7`
- `SMTP_HOST` - Your email provider's SMTP host
- `SMTP_PORT` - SMTP port (usually 587)
- `SMTP_USERNAME` - Your email username
- `SMTP_PASSWORD` - Your email password or app-specific password
- `FROM_EMAIL` - Email address to send emails from
- `OTP_EXPIRE_MINUTES` - Set to `10`
- `OPENAI_API_KEY` - Your OpenAI API key

**Note**: `DATABASE_URL` will be automatically provided by the PostgreSQL service.

### 3. Deploy

1. Render will automatically detect the `render.yaml` file
2. It will create both the web service and PostgreSQL database
3. The build process will:
   - Install Python dependencies
   - Run Alembic migrations to set up the database schema
4. Your app will be available at the provided Render URL

## Important Notes

### Database
- The PostgreSQL database is automatically created and connected
- Migrations run automatically during the build process
- Database URL is automatically injected as an environment variable

### Security
- All sensitive environment variables should be marked as "secret" in Render
- The default database credentials have been removed from the code
- CORS is configured but should be restricted to your frontend domain in production

### Health Checks
- The `/health` endpoint is configured for Render's health checks
- This ensures your service is properly monitored

## Troubleshooting

### Build Failures
- Check the build logs in your Render dashboard
- Ensure all required environment variables are set
- Verify your requirements.txt includes all dependencies

### Database Issues
- Ensure the PostgreSQL service is running
- Check that migrations completed successfully
- Verify the DATABASE_URL is properly connected

### Application Errors
- Check the service logs in your Render dashboard
- Verify all environment variables are correctly set
- Test your application locally first

## Local Development

To run locally:

1. Copy `env.example` to `.env` and fill in your values
2. Set up a local PostgreSQL database
3. Run migrations: `alembic upgrade head`
4. Start the server: `uvicorn app.main:app --reload`

## Production Considerations

1. **CORS**: Update the CORS origins in `app/main.py` to only allow your frontend domain
2. **Environment Variables**: Ensure all production values are set
3. **Database**: Consider upgrading to a paid database plan for production workloads
4. **Monitoring**: Set up proper logging and monitoring
5. **SSL**: Render provides SSL certificates automatically 