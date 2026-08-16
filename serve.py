#!/usr/bin/env python3
"""
Local HTTP Server for ASL Studio Web Application
Serves the project root so you can test the Web App and Chrome Extension locally.
"""

import http.server
import socketserver
import webbrowser
import os
import sys

PORT = 8000

class ASLHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Enable CORS and standard caching headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        super().end_headers()

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    handler = ASLHTTPRequestHandler

    try:
        with socketserver.TCPServer(("", PORT), handler) as httpd:
            url = f"http://localhost:{PORT}/web/"
            print("=" * 60)
            print("  [*] ASL Studio & Translator Web App is Live!")
            print(f"  [>] URL: {url}")
            print("  Press Ctrl+C to stop the server.")
            print("=" * 60)
            
            # Open browser automatically if not run with --no-browser
            if "--no-browser" not in sys.argv:
                webbrowser.open(url)
                
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
    except OSError as e:
        print(f"\nPort {PORT} might already be in use: {e}")
        print(f"Try visiting http://localhost:{PORT}/web/ directly.")

if __name__ == "__main__":
    main()
