#!/bin/bash
PORT=${PORT:-8000}
gunicorn --bind 0.0.0.0:$PORT --timeout 180 --workers 1 app:app
