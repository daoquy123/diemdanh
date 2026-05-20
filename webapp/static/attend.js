const video = document.getElementById("video");
const snapshot = document.getElementById("snapshot");
const statusEl = document.getElementById("status");
const statusPanel = document.getElementById("statusPanel");
const statusTitle = document.getElementById("statusTitle");
const startBtn = document.getElementById("startBtn");
const captureBtn = document.getElementById("captureBtn");
const nativeCaptureBtn = document.getElementById("nativeCaptureBtn");
const pickBtn = document.getElementById("pickBtn");
const fileInput = document.getElementById("fileInput");
const nativeCaptureInput = document.getElementById("nativeCaptureInput");
const retakeBtn = document.getElementById("retakeBtn");
const stopBtn = document.getElementById("stopBtn");
const attendProgress = document.getElementById("attendProgress");
const attendPercent = document.getElementById("attendPercent");
const comparePanel = document.getElementById("comparePanel");
const cameraLabel = document.getElementById("cameraLabel");
const frameMeta = document.getElementById("frameMeta");
const liveIndicator = document.getElementById("liveIndicator");

let stream = null;
let timer = null;
let confirmingSince = null;
let confirmIdentity = null;
let confirmed = false;
let recognizing = false;

const REQUIRED_SECONDS = 1.5;
const MIN_SIM = 0.68;
const MIN_MARGIN = 0.06;

const STATUS_TITLES = {
  idle: "Sẵn sàng",
  live: "Đang live",
  busy: "Đang xử lý",
  success: "Đã xác nhận",
  warning: "Cần thử lại",
  error: "Có lỗi",
};

function setStatus(message, tone = "idle", title = "") {
  statusEl.textContent = message;
  if (statusTitle) statusTitle.textContent = title || STATUS_TITLES[tone] || STATUS_TITLES.idle;
  if (statusPanel) statusPanel.dataset.tone = tone;
}

function setCameraState(active, label = active ? "Camera bật" : "Tạm dừng") {
  if (!liveIndicator) return;
  liveIndicator.classList.toggle("is-on", active);
  liveIndicator.innerHTML = `<span></span> ${label}`;
}

function setCameraLabel(text) {
  if (cameraLabel) cameraLabel.textContent = text;
}

function setFrameMeta(text) {
  if (frameMeta) frameMeta.textContent = text;
}

function setConfirming(flag) {
  attendProgress.classList.toggle("hidden", !flag);
}

function setConfirmProgress(percent) {
  const p = Math.max(0, Math.min(100, Math.round(percent)));
  attendPercent.textContent = `${p}%`;
  attendProgress.style.background = `conic-gradient(#0000cc ${p * 3.6}deg, rgba(255, 242, 0, 0.35) ${p * 3.6}deg)`;
}

function pct(sim) {
  return Math.round((sim || 0) * 100);
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return "";
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function pickRecognized(data) {
  const th = data.threshold ?? MIN_SIM;
  const minMargin = data.min_margin ?? MIN_MARGIN;
  for (const r of data.results || []) {
    if (!r.name || r.name === "Unknown") continue;
    const sim = r.similarity || 0;
    const margin = r.margin ?? 0;
    if (sim >= th && margin >= minMargin) return r;
  }
  return null;
}

function failureMessage(data) {
  if ((data.gallery_count ?? 0) === 0) {
    return "Chưa có sinh viên nào trong hệ thống. Vào trang quản trị để đăng ký khuôn mặt trước.";
  }
  if (!data.results || data.results.length === 0) {
    return "Không thấy khuôn mặt. Bật đèn, nhìn thẳng camera và để mặt ở giữa khung.";
  }
  const th = data.threshold ?? MIN_SIM;
  const best = data.results.reduce((a, b) =>
    (a.similarity || 0) >= (b.similarity || 0) ? a : b
  );
  const top = best.top_matches?.[0];
  const near = top ? ` Gần nhất: ${top.name} (${pct(top.similarity)}%).` : "";
  if (best.name && best.name !== "Unknown") {
    const margin = best.margin ?? 0;
    const minMargin = data.min_margin ?? MIN_MARGIN;
    if ((best.similarity || 0) < th) {
      return `Thấy mặt nhưng độ giống thấp (${pct(best.similarity)}%, cần ≥${pct(th)}%).${near}`;
    }
    if (margin < minMargin) {
      return `Khó phân biệt với người khác (chênh ${pct(margin)}%, cần ≥${pct(minMargin)}%).${near}`;
    }
  }
  return `Thấy mặt nhưng chưa khớp sinh viên đã đăng ký (${pct(best.similarity)}%, cần ≥${pct(th)}%).${near}`;
}

function setButtonsBusy(busy) {
  recognizing = busy;
  captureBtn.disabled = busy;
  pickBtn.disabled = busy;
  startBtn.disabled = busy;
  retakeBtn.disabled = busy;
  if (nativeCaptureBtn) nativeCaptureBtn.disabled = busy;
  if (nativeCaptureInput) nativeCaptureInput.disabled = busy;
}

function showSnapshotFromBlob(blob) {
  const url = URL.createObjectURL(blob);
  if (snapshot._prevUrl) URL.revokeObjectURL(snapshot._prevUrl);
  snapshot._prevUrl = url;
  snapshot.src = url;
  snapshot.classList.remove("hidden");
  video.classList.add("offscreen");
  retakeBtn.classList.remove("hidden");
  setCameraLabel("Ảnh đang kiểm tra");
  setFrameMeta(formatBytes(blob.size));
  setCameraState(false, "Ảnh tĩnh");
}

function showLivePreview() {
  snapshot.classList.add("hidden");
  if (snapshot._prevUrl) {
    URL.revokeObjectURL(snapshot._prevUrl);
    snapshot._prevUrl = null;
  }
  snapshot.removeAttribute("src");
  video.classList.remove("offscreen");
  retakeBtn.classList.add("hidden");
  setCameraLabel("Webcam trực tiếp");
}

async function waitForVideoReady() {
  await video.play().catch(() => {});
  for (let i = 0; i < 60; i++) {
    if (video.videoWidth > 0 && video.videoHeight > 0) {
      setFrameMeta(`${video.videoWidth} × ${video.videoHeight}`);
      return;
    }
    await new Promise((r) => setTimeout(r, 100));
  }
  throw new Error("Camera chưa có hình. Đợi 1-2 giây rồi chụp lại.");
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("Trình duyệt không hỗ trợ webcam. Dùng HTTPS, Camera máy hoặc Tải ảnh.");
  }
  if (!stream) {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    video.srcObject = stream;
  }
  showLivePreview();
  retakeBtn.classList.remove("hidden");
  setCameraState(true, "Camera bật");
  await waitForVideoReady();
}

function stopCamera() {
  if (!stream) {
    setCameraState(false, "Tạm dừng");
    return;
  }
  stream.getTracks().forEach((t) => t.stop());
  stream = null;
  video.srcObject = null;
  setCameraState(false, "Tạm dừng");
  setFrameMeta("Chưa có khung hình");
}

function stopLiveTimer() {
  if (timer) {
    clearInterval(timer);
    timer = null;
  }
}

function canvasToBlob(canvas, quality = 0.92) {
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => {
        if (blob) {
          resolve(blob);
          return;
        }
        try {
          const dataUrl = canvas.toDataURL("image/jpeg", quality);
          fetch(dataUrl)
            .then((r) => r.blob())
            .then(resolve)
            .catch(reject);
        } catch (e) {
          reject(e);
        }
      },
      "image/jpeg",
      quality
    );
  });
}

async function captureFrameBlob() {
  await waitForVideoReady();
  const w = video.videoWidth;
  const h = video.videoHeight;
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, w, h);
  return canvasToBlob(canvas);
}

function hideComparePanel() {
  if (!comparePanel) return;
  comparePanel.classList.add("hidden");
  comparePanel.innerHTML = "";
}

function renderComparePanel(data, faceIndex = 0) {
  if (!comparePanel) return;
  const face = data.results?.[faceIndex];
  const ranking = face?.gallery_ranking;
  if (!ranking?.length) {
    hideComparePanel();
    return;
  }
  const th = data.threshold ?? MIN_SIM;
  const minMargin = data.min_margin ?? MIN_MARGIN;
  const top1 = ranking[0];
  const top2 = ranking[1];
  const margin = face?.margin ?? (top1 && top2 ? top1.similarity - top2.similarity : 0);
  const marginPct = pct(margin);
  const marginWarn = margin < minMargin;

  let html = `<div class="compare-head">`;
  html += `<div><p class="eyebrow">Gallery</p><h2>So sánh ảnh với ${ranking.length} sinh viên</h2></div>`;
  html += `<span class="compare-threshold">Ngưỡng ${pct(th)}%</span>`;
  html += `</div>`;
  html += `<p class="compare-meta">Điểm nhận diện dùng kết quả tốt nhất giữa prototype đã lọc nhiễu và frame enroll. Chênh #1-#2 cần ≥ ${pct(minMargin)}% (đang ${marginPct}%). Detect ${pct(face?.detection_score ?? 0)}%.</p>`;
  if (marginWarn && top1 && top2) {
    html += `<p class="compare-warn">#1 ${escapeHtml(top1.name)} (${pct(top1.similarity)}%) gần #2 ${escapeHtml(top2.name)} (${pct(top2.similarity)}%). Nên nhìn thẳng, tăng sáng hoặc enroll lại nếu vẫn dễ nhầm.</p>`;
  }
  html += `<div class="compare-table-wrap"><table class="compare-table"><thead><tr><th>#</th><th>MSSV / tên</th><th>Điểm</th><th>Proto</th><th>Max frame</th></tr></thead><tbody>`;

  ranking.forEach((row, i) => {
    const sim = row.similarity || 0;
    const pass = sim >= th;
    const isTop1 = i === 0;
    const fullName = row.full_name ? `<br><small>${escapeHtml(row.full_name)}</small>` : "";
    const rowCls = isTop1 ? "is-top1" : pass ? "is-pass" : "";
    html += `<tr class="${rowCls}">`;
    html += `<td>${i + 1}${isTop1 ? " ★" : ""}</td>`;
    html += `<td class="compare-mssv">${escapeHtml(row.name)}${fullName}`;
    if (isTop1 && face?.name) {
      html += `<br><small class="compare-verdict">Kết luận: ${escapeHtml(face.name)}</small>`;
    }
    html += `</td>`;
    html += `<td class="compare-pct-cell"><strong>${pct(sim)}</strong>%</td>`;
    html += `<td>${pct(row.similarity_proto ?? 0)}%</td>`;
    html += `<td>${pct(row.similarity_max_frame ?? 0)}%</td>`;
    html += `</tr>`;
  });

  html += `</tbody></table></div>`;
  comparePanel.innerHTML = html;
  comparePanel.classList.remove("hidden");
}

async function recognizeBlob(blob, { compareAll = false } = {}) {
  if (!blob || blob.size < 100) {
    throw new Error("Ảnh rỗng hoặc quá nhỏ.");
  }
  const fd = new FormData();
  fd.append("frame", blob, "frame.jpg");
  const q = compareAll ? "?compare_all=1" : "";
  const res = await fetch(`/api/recognize${q}`, { method: "POST", body: fd });
  let data;
  try {
    data = await res.json();
  } catch {
    throw new Error(`Server trả lỗi (${res.status}). Kiểm tra uvicorn đang chạy.`);
  }
  if (!res.ok) {
    throw new Error(data.message || `Lỗi server ${res.status}`);
  }
  return data;
}

async function runSnapshotRecognize(blob, sourceLabel) {
  setButtonsBusy(true);
  setConfirming(false);
  setConfirmProgress(0);
  confirmingSince = null;
  confirmIdentity = null;
  setStatus("Đang gửi ảnh lên server nhận diện...", "busy");

  try {
    const data = await recognizeBlob(blob, { compareAll: true });
    if (!data.ok) {
      hideComparePanel();
      setStatus(data.message || "Không thể nhận diện. Vui lòng thử lại.", "error");
      return;
    }
    if (!data.results?.length) {
      hideComparePanel();
      setStatus(failureMessage(data), "warning");
      return;
    }
    renderComparePanel(data, 0);
    const recognized = pickRecognized(data);
    if (recognized) {
      confirmed = true;
      stopLiveTimer();
      setStatus(
        `Đã điểm danh (${sourceLabel}) sinh viên ${recognized.name} với độ giống ${pct(recognized.similarity)}%.`,
        "success"
      );
      return;
    }
    setStatus(`${failureMessage(data)} Xem bảng so sánh bên dưới.`, "warning");
  } catch (err) {
    hideComparePanel();
    setStatus(err.message || String(err), "error");
    console.error(err);
  } finally {
    setButtonsBusy(false);
  }
}

async function processImageFile(file, sourceLabel) {
  if (!file || !file.type.startsWith("image/")) {
    setStatus("File không phải ảnh.", "error");
    return;
  }
  stopLiveTimer();
  confirmed = false;
  setConfirming(false);
  setConfirmProgress(0);
  showSnapshotFromBlob(file);
  await runSnapshotRecognize(file, sourceLabel);
}

async function tickRecognize() {
  if (confirmed || recognizing) return;
  recognizing = true;
  try {
    const blob = await captureFrameBlob();
    const data = await recognizeBlob(blob);
    const recognized = pickRecognized(data);
    if (!recognized) {
      confirmingSince = null;
      setConfirming(false);
      setConfirmProgress(0);
      setStatus(failureMessage(data), "warning");
      return;
    }
    if (!confirmingSince || confirmIdentity !== recognized.name) {
      confirmingSince = Date.now();
      confirmIdentity = recognized.name;
    }
    setConfirming(true);
    const elapsed = (Date.now() - confirmingSince) / 1000;
    setConfirmProgress((elapsed / REQUIRED_SECONDS) * 100);
    if (elapsed >= REQUIRED_SECONDS) {
      confirmed = true;
      setConfirming(false);
      setConfirmProgress(100);
      stopLiveTimer();
      stopCamera();
      setStatus(`Đã điểm danh sinh viên ${recognized.name} thành công.`, "success");
      return;
    }
    setStatus(
      `Đang xác nhận ${recognized.name}... ${elapsed.toFixed(1)}/${REQUIRED_SECONDS}s (${pct(recognized.similarity)}%).`,
      "live"
    );
  } catch (err) {
    setStatus(err.message || String(err), "error");
  } finally {
    recognizing = false;
  }
}

captureBtn.addEventListener("click", async () => {
  stopLiveTimer();
  confirmed = false;
  hideComparePanel();
  setConfirming(false);
  setConfirmProgress(0);

  try {
    setStatus("Đang mở webcam...", "busy");
    await startCamera();
    setStatus("Đang chụp khung hình hiện tại...", "busy");
    const blob = await captureFrameBlob();
    showSnapshotFromBlob(blob);
    await runSnapshotRecognize(blob, "ảnh chụp");
  } catch (err) {
    setStatus(err.message || "Không chụp được. Thử Camera máy hoặc Tải ảnh.", "error");
    console.error(err);
  }
});

if (nativeCaptureInput) {
  nativeCaptureInput.addEventListener("change", async () => {
    const file = nativeCaptureInput.files?.[0];
    nativeCaptureInput.value = "";
    if (!file) return;
    await processImageFile(file, "camera máy");
  });
}

if (nativeCaptureBtn && nativeCaptureInput) {
  nativeCaptureBtn.addEventListener("click", () => {
    stopLiveTimer();
    confirmed = false;
    hideComparePanel();
    setStatus("Đang mở camera của thiết bị...", "busy");
    nativeCaptureInput.click();
  });
}

pickBtn.addEventListener("click", () => {
  stopLiveTimer();
  confirmed = false;
  hideComparePanel();
  setStatus("Chọn ảnh khuôn mặt để kiểm tra.", "idle");
  fileInput.click();
});

fileInput.addEventListener("change", async () => {
  const file = fileInput.files?.[0];
  fileInput.value = "";
  if (!file) return;
  await processImageFile(file, "ảnh tải lên");
});

retakeBtn.addEventListener("click", async () => {
  confirmed = false;
  confirmingSince = null;
  confirmIdentity = null;
  setConfirming(false);
  setConfirmProgress(0);
  showLivePreview();
  setStatus("Đang mở lại webcam...", "busy");
  try {
    await startCamera();
    setStatus("Sẵn sàng. Chụp lại hoặc bật Live.", "idle");
  } catch (err) {
    setStatus(err.message || "Bật camera hoặc chọn ảnh từ máy.", "error");
  }
});

startBtn.addEventListener("click", async () => {
  confirmed = false;
  confirmingSince = null;
  confirmIdentity = null;
  hideComparePanel();
  setConfirming(false);
  setConfirmProgress(0);
  showLivePreview();
  try {
    setStatus("Đang mở webcam...", "busy");
    await startCamera();
    setCameraState(true, "Đang live");
    setStatus("Đang tìm khuôn mặt sinh viên...", "live");
    stopLiveTimer();
    timer = setInterval(tickRecognize, 400);
  } catch (err) {
    setStatus(err.message || "Không mở được webcam.", "error");
  }
});

stopBtn.addEventListener("click", () => {
  stopLiveTimer();
  confirmed = false;
  confirmingSince = null;
  confirmIdentity = null;
  hideComparePanel();
  setConfirming(false);
  setConfirmProgress(0);
  stopCamera();
  showLivePreview();
  setCameraState(false, "Tạm dừng");
  setStatus("Đã dừng camera và luồng nhận diện.", "idle");
});

setStatus("Chọn Live để điểm danh liên tục hoặc chụp một ảnh để kiểm tra.", "idle");
setCameraState(false, "Tạm dừng");
