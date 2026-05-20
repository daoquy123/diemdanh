const video = document.getElementById("video");
const statusEl = document.getElementById("status");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const attendProgress = document.getElementById("attendProgress");
const attendPercent = document.getElementById("attendPercent");

let stream = null;
let timer = null;
let confirmingSince = null;
let confirmed = false;
const REQUIRED_SECONDS = 1.0;

function setConfirming(flag) {
  attendProgress.classList.toggle("hidden", !flag);
}

function setConfirmProgress(percent) {
  const p = Math.max(0, Math.min(100, Math.round(percent)));
  attendPercent.textContent = `${p}%`;
  attendProgress.style.background = `conic-gradient(#00b4ff ${p * 3.6}deg, rgba(0, 180, 255, 0.25) ${p * 3.6}deg)`;
}

async function startCamera() {
  if (stream) return;
  stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
  video.srcObject = stream;
}

function stopCamera() {
  if (!stream) return;
  stream.getTracks().forEach((t) => t.stop());
  stream = null;
}

function captureFrameBlob() {
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  return new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.8));
}

async function tickRecognize() {
  if (confirmed) return;
  const blob = await captureFrameBlob();
  if (!blob) return;
  const fd = new FormData();
  fd.append("frame", blob, "frame.jpg");
  const res = await fetch("/api/recognize", { method: "POST", body: fd });
  const data = await res.json();
  if (!data.ok) {
    statusEl.textContent = "Không thể nhận diện. Vui lòng thử lại.";
    confirmingSince = null;
    setConfirming(false);
    setConfirmProgress(0);
    return;
  }

  const recognized = data.results.find((r) => r.name && r.name !== "Unknown");
  if (!recognized) {
    confirmingSince = null;
    setConfirming(false);
    setConfirmProgress(0);
    statusEl.textContent = "Đưa rõ khuôn mặt vào khung hình.";
    return;
  }

  if (!confirmingSince) confirmingSince = Date.now();
  setConfirming(true);
  const elapsed = (Date.now() - confirmingSince) / 1000;
  const percent = (elapsed / REQUIRED_SECONDS) * 100;
  setConfirmProgress(percent);

  if (elapsed >= REQUIRED_SECONDS) {
    confirmed = true;
    setConfirming(false);
    setConfirmProgress(100);
    if (timer) clearInterval(timer);
    stopCamera();
    statusEl.textContent = `Đã điểm danh sinh viên ${recognized.name} thành công.`;
    return;
  }

  statusEl.textContent = `Đang xác nhận điểm danh... ${elapsed.toFixed(1)}/${REQUIRED_SECONDS}s`;
}

startBtn.addEventListener("click", async () => {
  await startCamera();
  confirmed = false;
  confirmingSince = null;
  setConfirming(false);
  setConfirmProgress(0);
  statusEl.textContent = "Đang tìm khuôn mặt sinh viên...";
  if (timer) clearInterval(timer);
  timer = setInterval(tickRecognize, 300);
});

stopBtn.addEventListener("click", () => {
  if (timer) clearInterval(timer);
  confirmed = false;
  confirmingSince = null;
  setConfirming(false);
  setConfirmProgress(0);
  stopCamera();
  statusEl.textContent = "Đã dừng.";
});
