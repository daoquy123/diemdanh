const video = document.getElementById("video");
const statusEl = document.getElementById("status");
const enrollBtn = document.getElementById("enrollBtn");
const stopBtn = document.getElementById("stopBtn");
const fullNameEl = document.getElementById("fullName");
const mssvEl = document.getElementById("mssv");
const scanProgress = document.getElementById("scanProgress");
const scanPercent = document.getElementById("scanPercent");

let stream = null;
let running = false;
let frames = [];
let timer = null;

function setScanning(isScanning) {
  scanProgress.classList.toggle("hidden", !isScanning);
  enrollBtn.disabled = isScanning;
  enrollBtn.textContent = isScanning ? "Đang quay..." : "🔴 Thêm sinh viên mới";
}

function setProgressPercent(percent) {
  const p = Math.max(0, Math.min(100, Math.round(percent)));
  scanPercent.textContent = `${p}%`;
  scanProgress.style.background = `conic-gradient(#0000cc ${p * 3.6}deg, rgba(255, 242, 0, 0.35) ${p * 3.6}deg)`;
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
  return new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.85));
}

async function uploadEnroll() {
  const fd = new FormData();
  fd.append("full_name", fullNameEl.value.trim());
  fd.append("mssv", mssvEl.value.trim());
  frames.forEach((b, i) => fd.append("frames", b, `frame_${i}.jpg`));
  const res = await fetch("/api/enroll", { method: "POST", body: fd });
  const data = await res.json();
  if (!res.ok) {
    statusEl.textContent = data.message || "Chưa hoàn thành. Vui lòng thử lại.";
    return;
  }
  statusEl.textContent = data.message || "Đã hoàn thành.";
}

async function startEnroll() {
  const fullName = fullNameEl.value.trim();
  const mssv = mssvEl.value.trim();
  if (fullName.length < 3 || mssv.length < 4) {
    statusEl.textContent = "Vui lòng nhập đầy đủ họ tên và MSSV.";
    return;
  }
  await startCamera();
  frames = [];
  running = true;
  setScanning(true);
  setProgressPercent(0);
  const started = Date.now();
  statusEl.textContent = "Đang quét...";

  timer = setInterval(async () => {
    if (!running) return;
    const elapsed = (Date.now() - started) / 1000;
    if (frames.length >= 100 || elapsed >= 30) {
      clearInterval(timer);
      running = false;
      setProgressPercent(100);
      statusEl.textContent = "Đang xử lý...";
      await uploadEnroll();
      setScanning(false);
      return;
    }
    const blob = await captureFrameBlob();
    if (blob) frames.push(blob);
    const percent = (frames.length / 100) * 100;
    setProgressPercent(percent);
    statusEl.textContent = `Đang quét... ${Math.round(percent)}%`;
  }, 200);
}

enrollBtn.addEventListener("click", startEnroll);
stopBtn.addEventListener("click", () => {
  running = false;
  if (timer) clearInterval(timer);
  stopCamera();
  setScanning(false);
  setProgressPercent(0);
  statusEl.textContent = "Đã dừng.";
});
