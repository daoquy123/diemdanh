# Chạy điểm danh để máy khác truy cập được (cùng Wi‑Fi / LAN).
# Cách dùng: mở PowerShell tại thư mục project, chạy:
#   .\scripts\run_streamlit_public.ps1
#
# Trên máy khác: trình duyệt mở  http://<ĐỊA-CHỈ-IP-MÁY-CHỦ>:8501
#   (Windows: ipconfig  ->  IPv4 Address)
#
# Ra Internet: đã cài ngrok (winget). Terminal khác chạy: .\scripts\ngrok_tunnel.ps1

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

Write-Host "Đang khởi động Streamlit (lắng nghe 0.0.0.0:8501)..." -ForegroundColor Cyan
Write-Host "Máy cùng mạng truy cập: http://<IP-máy-này>:8501" -ForegroundColor Yellow
Write-Host "Nhấn Ctrl+C để dừng." -ForegroundColor Gray

if (Test-Path ".\.venv\Scripts\streamlit.exe") {
    & ".\.venv\Scripts\streamlit.exe" run ".\app\streamlit_app.py"
} else {
    streamlit run ".\app\streamlit_app.py"
}
