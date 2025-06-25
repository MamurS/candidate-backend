from logging.config import fileConfig
from sqlalchemy import engine_from_config
from sqlalchemy import pool
from alembic import context
import os
import sys

# Add your project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# Now import your Base and models
from app.database import Base
from app.models import User, OTP, CVAnalysis, WorkPreferences  # Import all your models explicitly

# this is the Alembic Config object
config = context.config

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# IMPORTANT: Set the target metadata
target_metadata = Base.metadata

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    # Get database URL from environment variable, fallback to config
    url = os.getenv("DATABASE_URL") or config.get_main_option("sqlalchemy.url")
    
    # Handle Render's postgres:// vs SQLAlchemy's postgresql://
    if url and url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    # Override the sqlalchemy.url with environment variable if available
    configuration = config.get_section(config.config_ini_section, {})
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        # Handle Render's postgres:// vs SQLAlchemy's postgresql://
        if database_url.startswith("postgres://"):
            database_url = database_url.replace("postgres://", "postgresql://", 1)
        configuration["sqlalchemy.url"] = database_url
    
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()