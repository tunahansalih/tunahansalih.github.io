#!/usr/bin/env python3
"""
Local development server for Tuna Meral's academic website.
Builds the site and serves on http://localhost:4000.
"""

import subprocess
import sys

def main():
    print("Building site...")
    subprocess.run([sys.executable, "build.py"], check=True)
    print("Starting server on http://localhost:4000 (Ctrl+C to stop)...")
    subprocess.run([sys.executable, "-m", "http.server", "4000"])

if __name__ == "__main__":
    main()
