# Public tunnel to Streamlit (port 8501).
#
# If you see ERR_NGROK_121 (agent too old), update the binary:
#   & "$env:LOCALAPPDATA\Microsoft\WinGet\Links\ngrok.exe" update
# WinGet may still list 3.3.1; self-update pulls the current agent from ngrok.
#
# One-time authtoken: https://dashboard.ngrok.com/get-started/your-authtoken
#   PowerShell (paste your real token, do not use angle brackets):
#     & "$env:LOCALAPPDATA\Microsoft\WinGet\Links\ngrok.exe" config add-authtoken YOUR_TOKEN_HERE
#
#   If ngrok.yml has version "3" and ngrok errors, use v2 format:
#     version: "2"
#     authtoken: YOUR_TOKEN_HERE
#
# Terminal 1: .\scripts\run_streamlit_public.ps1
# Terminal 2: .\scripts\ngrok_tunnel.ps1
#
# Share the https://....ngrok-free.app URL. Ctrl+C to stop.

$ErrorActionPreference = "Stop"

$winGetNgrok = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Links\ngrok.exe"
$ngrok = if (Test-Path $winGetNgrok) { $winGetNgrok } else { "ngrok" }

$port = 8501
Write-Host "Starting ngrok tunnel -> http://localhost:$port" -ForegroundColor Cyan
Write-Host "Using: $ngrok" -ForegroundColor Gray
Write-Host "Ensure Streamlit is running first (.\scripts\run_streamlit_public.ps1)." -ForegroundColor Yellow
& $ngrok http $port
