#!/usr/bin/env python3
"""Serve dashboard from mamba-prime dir, serving both stats files."""
import http.server, os
os.chdir('/home/phil/.gemini/antigravity/scratch/analysis_project/mamba-prime')
import socketserver
PORT = 8766
Handler = http.server.SimpleHTTPRequestHandler
Handler.extensions_map['.json'] = 'application/json'
with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"[SERVER] Mamba dashboard at http://localhost:{PORT}/dashboard/index.html")
    httpd.serve_forever()
