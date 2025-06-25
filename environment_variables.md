# Environment Variables for Render Deployment

## Required Environment Variables

Set these environment variables in your Render service settings:

### Database Configuration
- `DATABASE_URL` - Automatically provided by Render PostgreSQL service

### JWT Configuration
- `SECRET_KEY` - A secure random string for JWT token signing
- `ALGORITHM` - Set to `HS256`
- `ACCESS_TOKEN_EXPIRE_MINUTES` - Set to `30`
- `REFRESH_TOKEN_EXPIRE_DAYS` - Set to `7`

### Email Configuration (SMTP)
- `SMTP_HOST` - Your SMTP server (e.g., `smtp.gmail.com`)
- `SMTP_PORT` - SMTP port (e.g., `587`)
- `SMTP_USERNAME` - Your email username
- `SMTP_PASSWORD` - Your email password or app password
- `FROM_EMAIL` - Email address to send from

### OTP Configuration
- `OTP_EXPIRE_MINUTES` - Set to `10`

### OpenAI Configuration
- `OPENAI_API_KEY` - Your OpenAI API key

## How to Set Environment Variables in Render

1. Go to your Render dashboard
2. Select your web service
3. Go to the "Environment" tab
4. Add each variable listed above
5. Make sure to mark sensitive variables (like API keys) as secret 