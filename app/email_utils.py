import aiosmtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from .config import settings


async def send_email(to_email: str, subject: str, body: str):
    """Send email using SMTP"""
    message = MIMEMultipart()
    message["From"] = settings.from_email
    message["To"] = to_email
    message["Subject"] = subject

    message.attach(MIMEText(body, "html"))

    # Gmail requires STARTTLS, not direct TLS
    async with aiosmtplib.SMTP(
            hostname=settings.smtp_host,
            port=settings.smtp_port,
            use_tls=False,  # Changed from True
            start_tls=True  # Added this
    ) as smtp:
        await smtp.login(settings.smtp_username, settings.smtp_password)
        await smtp.send_message(message)


async def send_otp_email(to_email: str, otp_code: str):
    """Send OTP verification email"""
    subject = "Verify Your Email - OTP"
    body = f"""
    <html>
        <body>
            <h2>Email Verification</h2>
            <p>Thank you for registering! Please use the following OTP to verify your email:</p>
            <h1 style="color: #333; background-color: #f0f0f0; padding: 10px; display: inline-block;">
                {otp_code}
            </h1>
            <p>This OTP will expire in {settings.otp_expire_minutes} minutes.</p>
            <p>If you didn't request this, please ignore this email.</p>
        </body>
    </html>
    """
    await send_email(to_email, subject, body)