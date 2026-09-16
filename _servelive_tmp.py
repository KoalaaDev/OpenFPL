"""Throwaway: serve the app with the live window forced open, to look at the
Live tab outside its real 24 h run-up. Not part of the project."""
import uvicorn
from app import live
live.OPEN_BEFORE = 30 * 24 * 3600      # pretend the deadline is imminent
live._cache["v"] = None
uvicorn.run("app.main:app", host="127.0.0.1", port=8413, log_level="warning")
