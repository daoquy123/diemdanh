const video = document.getElementById("video");
const snapshot = document.getElementById("snapshot");
const statusEl = document.getElementById("status");
const startBtn = document.getElementById("startBtn");
const captureBtn = document.getElementById("captureBtn");
const pickBtn = document.getElementById("pickBtn");
const fileInput = document.getElementById("fileInput");
const nativeCaptureInput = document.getElementById("nativeCaptureInput");
const retakeBtn = document.getElementById("retakeBtn");
const stopBtn = document.getElementById("stopBtn");
const attendProgress = document.getElementById("attendProgress");
const attendPercent = document.getElementById("attendPercent");
const comparePanel = document.getElementById("comparePanel");

let stream = null;
let timer = null;
let confirmingSince = null;
let confirmIdentity = null;
let confirmed = false;
let recognizing = false;

const REQUIRED_SECONDS = 1.5;
const MIN_SIM = 0.68;
const MIN_MARGIN = 0.06;

function setConfirming(flag) {
  attendProgress.classList.toggle("hidden", !flag);
}

function setConfirmProgress(percent) {
  const p = Math.max(0, Math.min(100, Math.round(percent)));
  attendPercent.textContent = `${p}%`;
  attendProgress.style.background = `conic-gradient(#00b4ff ${p * 3.6}deg, rgba(0, 180, 255, 0.25) ${p * 3.6}deg)`;
}

function pct(sim) {
  return Math.round((sim || 0) * 100);
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
    return "Chưa có sinh viên nào trong hệ thống. Vào Thêm sinh viên mới để đăng ký khuôn mặt trước.";
  }
  if (!data.results || data.results.length === 0) {
    return "Không thấy khuôn mặt. Bật đèn, nhìn thẳng camera, để mặt ở giữa khung (một người).";
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
      return `Thấy mặt nhưng độ giống thấp (${pct(best.similarity)}%, cần ≥${pct(th)}%).${near} Bật sáng hơn hoặc đăng ký lại.`;
    }
    if (margin < minMargin) {
      return `Khó phân biệt với người khác (chênh ${pct(margin)}%, cần ≥${pct(minMargin)}%).${near} Nhìn thẳng camera, thử lại.`;
    }
  }
  return `Thấy mặt nhưng chưa khớp sinh viên đã đăng ký (${pct(best.similarity)}%, cần ≥${pct(th)}%).${near} Đăng ký lại hoặc bật sáng hơn.`;
}

function setButtonsBusy(busy) {
  recognizing = busy;
  captureBtn.disabled = busy;
  pickBtn.disabled = busy;
  startBtn.disabled = busy;
  if (nativeCaptureInput) nativeCaptureInput.disabled = busy;
}

function showSnapshotFromBlob(blob) {
  const url = URL.createObjectURL(blob);
  if (snapshot._prevUrl) URL.revokeObjectURL(snapshot._prevUrl);
  snapshot._prevUrl = url;
  snapshot.src = url;
  snapshot.classList.remove("hidden");
  video.classList.add("offscreen");
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
}

async function waitForVideoReady() {
  await video.play().catch(() => {});
  for (let i = 0; i < 60; i++) {
    if (video.videoWidth > 0 && video.videoHeight > 0) return;
    await new Promise((r) => setTimeout(r, 100));
  }
  throw new Error("Camera chưa có hình (videoWidth=0). Đợi 1–2 giây rồi chụp lại.");
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("Trình duyệt không hỗ trợ camera. Dùng HTTPS hoặc Chọn ảnh từ máy.");
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
  await waitForVideoReady();
}

function stopCamera() {
  if (!stream) return;
  stream.getTracks().forEach((t) => t.stop());
  stream = null;
  video.srcObject = null;
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
  const margin =
    face?.margin ??
    (top1 && top2 ? top1.similarity - top2.similarity : 0);

  const marginPct = pct(margin);
  const marginWarn = margin < minMargin;

  let html = `<h2>So sánh ảnh chụp với gallery (${ranking.length} người)</h2>`;
  html += `<p class="compare-meta">Điểm = max(frame enroll tốt nhất, prototype đã lọc nhiễu) + lật ảnh TTA. `;
  html += `Ngưỡng ≥ ${pct(th)}%, chênh #1−#2 ≥ ${pct(minMargin)}% (đang ${marginPct}%${marginWarn ? ", <strong>dễ nhầm</strong>" : ""}). Detect ${pct(face?.detection_score ?? 0)}%.</p>`;
  if (marginWarn && top1 && top2) {
    html += `<p class="compare-warn">#1 ${top1.name} (${pct(top1.similarity)}%) vs #2 ${top2.name} (${pct(top2.similarity)}%) — chênh ${marginPct}%. Enroll lại MSSV của bạn: đủ sáng, nhìn thẳng, 100 frame.</p>`;
  }
  html += `<table class="compare-table"><thead><tr><th>#</th><th>MSSV / tên</th><th>Điểm</th><th>Proto</th><th>Max frame</th></tr></thead><tbody>`;

  ranking.forEach((row, i) => {
    const sim = row.similarity || 0;
    const pass = sim >= th;
    const isTop1 = i === 0;
    const label = row.full_name ? `${row.name}<br><small>${row.full_name}</small>` : row.name;
    const rowCls = isTop1 ? "is-top1" : pass ? "is-pass" : "";
    html += `<tr class="${rowCls}">`;
    html += `<td>${i + 1}${isTop1 ? " ★" : ""}</td>`;
    html += `<td class="compare-mssv">${label}`;
    if (isTop1 && face?.name) {
      html += `<br><small class="compare-verdict">Kết luận: ${face.name}</small>`;
    }
    html += `</td>`;
    html += `<td class="compare-pct-cell"><strong>${pct(sim)}</strong> %</td>`;
    html += `<td>${pct(row.similarity_proto ?? 0)} %</td>`;
    html += `<td>${pct(row.similarity_max_frame ?? 0)} %</td>`;
    html += `</tr>`;
  });

  html += `</tbody></table>`;
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
  statusEl.textContent = "Đang nhận diện ảnh...";

  try {
    const data = await recognizeBlob(blob, { compareAll: true });
    if (!data.ok) {
      hideComparePanel();
      statusEl.textContent = data.message || "Không thể nhận diện. Vui lòng thử lại.";
      return;
    }
    if (!data.results?.length) {
      hideComparePanel();
      statusEl.textContent = failureMessage(data);
      return;
    }
    renderComparePanel(data, 0);
    const recognized = pickRecognized(data);
    if (recognized) {
      confirmed = true;
      stopLiveTimer();
      statusEl.textContent = `Đã điểm danh (${sourceLabel}) sinh viên ${recognized.name} — độ giống ${pct(recognized.similarity)}%. Xem bảng so sánh bên dưới.`;
      return;
    }
    statusEl.textContent = `${failureMessage(data)} Xem bảng so sánh bên dưới.`;
  } catch (err) {
    hideComparePanel();
    statusEl.textContent = err.message || String(err);
    console.error(err);
  } finally {
    setButtonsBusy(false);
  }
}

async function processImageFile(file, sourceLabel) {
  if (!file || !file.type.startsWith("image/")) {
    statusEl.textContent = "File không phải ảnh.";
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
  try {
    const blob = await captureFrameBlob();
    const data = await recognizeBlob(blob);
    const recognized = pickRecognized(data);
    if (!recognized) {
      confirmingSince = null;
      setConfirming(false);
      setConfirmProgress(0);
      statusEl.textContent = failureMessage(data);
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
      statusEl.textContent = `Đã điểm danh sinh viên ${recognized.name} thành công.`;
      return;
    }
    statusEl.textContent = `Đang xác nhận ${recognized.name}... ${elapsed.toFixed(1)}/${REQUIRED_SECONDS}s (${pct(recognized.similarity)}%)`;
  } catch (err) {
    statusEl.textContent = err.message || String(err);
  }
}

/** Chụp từ khung video đang mở */
captureBtn.addEventListener("click", async () => {
  stopLiveTimer();
  confirmed = false;
  hideComparePanel();
  setConfirming(false);
  setConfirmProgress(0);

  try {
    statusEl.textContent = "Đang mở camera...";
    await startCamera();
    statusEl.textContent = "Đang chụp...";
    const blob = await captureFrameBlob();
    showSnapshotFromBlob(blob);
    await runSnapshotRecognize(blob, "ảnh chụp");
  } catch (err) {
    statusEl.textContent = err.message || "Không chụp được. Thử «Camera máy» hoặc «Chọn ảnh».";
    console.error(err);
  }
});

/** Mobile: mở app Camera chụp 1 tấm */
if (nativeCaptureInput) {
  nativeCaptureInput.addEventListener("change", async () => {
    const file = nativeCaptureInput.files?.[0];
    nativeCaptureInput.value = "";
    if (!file) return;
    await processImageFile(file, "camera máy");
  });
}

const nativeCaptureBtn = document.getElementById("nativeCaptureBtn");
if (nativeCaptureBtn && nativeCaptureInput) {
  nativeCaptureBtn.addEventListener("click", () => {
    stopLiveTimer();
    confirmed = false;
    hideComparePanel();
    nativeCaptureInput.click();
  });
}

pickBtn.addEventListener("click", () => {
  stopLiveTimer();
  confirmed = false;
  hideComparePanel();
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
  statusEl.textContent = "Đang mở lại camera...";
  try {
    await startCamera();
    statusEl.textContent = "Sẵn sàng. Chụp lại hoặc bắt đầu live.";
  } catch (err) {
    statusEl.textContent = err.message || "Bật camera hoặc chọn ảnh từ máy.";
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
    await startCamera();
    statusEl.textContent = "Đang tìm khuôn mặt sinh viên (live)...";
    stopLiveTimer();
    timer = setInterval(tickRecognize, 400);
  } catch (err) {
    statusEl.textContent = err.message || "Không mở được camera.";
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
  statusEl.textContent = "Đã dừng.";
});
