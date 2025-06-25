#!/usr/bin/env bash
# exit on error
set -o errexit

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Wait for database to be ready (if needed)
echo "Running database migrations..."

# Run database migrations
python -m alembic upgrade head

echo "Build completed successfully!" 