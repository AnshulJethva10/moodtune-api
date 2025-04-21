# gunicorn.conf.py
workers = 1
timeout = 120  # Longer timeout for model loading and inference
bind = "0.0.0.0:10000"  # Render will override this with its PORT