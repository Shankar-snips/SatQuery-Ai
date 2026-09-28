const API_BASE = "https://satquery-ai-backend-oawm.onrender.com";
let currentMode = "single";
let currentStep = "upload";
let selectedFiles = {}; // slotIndex -> File
let uploadedImageIds = []; // parallel to slot indices
let rsAdapted = false;

const MODE_CONFIG = {
  single: { slots: 1, titles: ["Image"], hints: ["single image"] },
  cross_modal: {
    slots: 2,
    titles: ["Optical image", "SAR image"],
    hints: ["optical / multispectral", "synthetic aperture radar"],
  },
  bi_temporal: {
    slots: 2,
    titles: ["Before", "After"],
    hints: ["earlier date", "later date"],
  },
};

const EXAMPLES = {
  single: [
    "Describe the land-cover and major objects visible in this image.",
    "Highlight the water body referred to in the query.",
    "How many built-up clusters are visible?",
  ],
  cross_modal: [
    "Use the optical and SAR images together to identify built-up and water-covered regions.",
  ],
  bi_temporal: [
    "What changed between these two dates, and where did the change occur?",
    "Has the built-up area increased, decreased, or remained unchanged?",
  ],
};

const RUN_STEPS = [
  { key: "compatibility_check_passed", label: "Checking image compatibility" },
  { key: "query_classified", label: "Classifying query intent" },
  { key: "tasks_selected", label: "Selecting specialist tool(s)" },
  { key: "tool_executed", label: "Executing RS-adapted model(s)" },
  { key: "outputs_merged", label: "Merging outputs & scoring confidence" },
];

const statusBadge = document.getElementById("status-badge");
const statusLabel = document.getElementById("status-label");
const tmImages = document.getElementById("tm-images");
const tmAdapted = document.getElementById("tm-adapted");
const tmLastRun = document.getElementById("tm-lastrun");

const stepsNav = document.getElementById("steps-nav");
const modeList = document.getElementById("mode-list");
const uploadBtn = document.getElementById("upload-btn");
const runBtn = document.getElementById("run-btn");
const queryInput = document.getElementById("query-input");
const exampleChips = document.getElementById("example-chips");
const resetBtn = document.getElementById("reset-btn");

const runProgress = document.getElementById("run-progress");
const runProgressSteps = document.getElementById("run-progress-steps");
const resultWrap = document.getElementById("result-wrap");
const errorPanel = document.getElementById("error-panel");

function toast(msg, type = "ok") {
  const stack = document.getElementById("toast-stack");
  const el = document.createElement("div");
  el.className = `toast ${type === "err" ? "err" : ""}`;
  el.textContent = msg;
  stack.appendChild(el);
  setTimeout(() => el.remove(), 4200);
}
async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    const data = await res.json();
    rsAdapted = !!data.rs_adapted;
    statusLabel.textContent = `online · ${data.device}${rsAdapted ? " · RS-adapted" : ""}`;
    statusBadge.classList.add("ok");
    tmAdapted.textContent = rsAdapted ? "active" : "base weights";
  } catch (e) {
    statusLabel.textContent = "backend offline";
    statusBadge.classList.add("bad");
    tmAdapted.textContent = "unknown";
  }
}
checkHealth();

function goToStep(step) {
  currentStep = step;
  document
    .querySelectorAll(".view")
    .forEach((v) => v.classList.remove("active"));
  document.getElementById(`view-${step}`).classList.add("active");
  document.querySelectorAll(".step").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.step === step);
  });
}

stepsNav.addEventListener("click", (e) => {
  const btn = e.target.closest(".step");
  if (!btn) return;
  const target = btn.dataset.step;
  if (target === "query" && uploadedImageIds.length === 0) {
    toast("Upload image(s) first.", "err");
    return;
  }
  if (
    target === "results" &&
    resultWrap.classList.contains("hidden") &&
    errorPanel.classList.contains("hidden")
  ) {
    toast("Run a query first.", "err");
    return;
  }
  goToStep(target);
});

document.querySelectorAll("[data-goto]").forEach((btn) => {
  btn.addEventListener("click", () => goToStep(btn.dataset.goto));
});

function markStepDone(step) {
  const el = document.querySelector(`.step[data-step="${step}"]`);
  if (el) el.classList.add("done");
}

modeList.addEventListener("click", (e) => {
  const btn = e.target.closest(".mode-option");
  if (!btn) return;
  document
    .querySelectorAll(".mode-option")
    .forEach((b) => b.classList.remove("active"));
  btn.classList.add("active");
  currentMode = btn.dataset.mode;
  renderSlots();
  renderExamples();
  resetUploadState();
});

function renderSlots() {
  const cfg = MODE_CONFIG[currentMode];
  [0, 1].forEach((i) => {
    const zone = document.querySelector(`.dropzone[data-slot="${i}"]`);
    zone.classList.toggle("hidden", i >= cfg.slots);
    document.getElementById(`slot-title-${i}`).textContent =
      cfg.titles[i] || `Image ${i + 1}`;
    document.getElementById(`slot-hint-${i}`).textContent = cfg.hints[i] || "";
  });
  document.getElementById("compare-wrap").classList.add("hidden");
}

function renderExamples() {
  exampleChips.innerHTML = "";
  (EXAMPLES[currentMode] || []).forEach((ex) => {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.textContent = ex;
    chip.onclick = () => {
      queryInput.value = ex;
      queryInput.focus();
    };
    exampleChips.appendChild(chip);
  });
}

function resetUploadState() {
  selectedFiles = {};
  uploadedImageIds = [];
  runBtn.disabled = true;
  uploadBtn.textContent = "Upload & continue";
  [0, 1].forEach((i) => {
    document.getElementById(`preview-${i}`).innerHTML = "";
    document.getElementById(`meta-${i}`).innerHTML = "";
    document
      .querySelector(`.dropzone[data-slot="${i}"]`)
      .classList.remove("filled");
    document.getElementById(`file-${i}`).value = "";
  });
  tmImages.textContent = "0";
}

function fmtBytes(b) {
  if (b < 1024) return `${b} B`;
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`;
  return `${(b / (1024 * 1024)).toFixed(1)} MB`;
}

function handleFileForSlot(idx, file) {
  if (!file) return;
  selectedFiles[idx] = file;
  const zone = document.querySelector(`.dropzone[data-slot="${idx}"]`);
  zone.classList.add("filled");

  const preview = document.getElementById(`preview-${idx}`);
  const meta = document.getElementById(`meta-${idx}`);
  const ext = file.name.split(".").pop().toUpperCase();

  if (file.type.startsWith("image/")) {
    const url = URL.createObjectURL(file);
    preview.innerHTML = `<img src="${url}" alt="${file.name}" />`;
    const img = new Image();
    img.onload = () => {
      meta.innerHTML = `<span>${ext}</span><span>${img.naturalWidth}×${img.naturalHeight}px</span><span>${fmtBytes(file.size)}</span>`;
    };
    img.src = url;
    maybeUpdateCompareSlider(idx, url);
  } else {
    preview.innerHTML = "";
    meta.innerHTML = `<span>${ext}</span><span>${fmtBytes(file.size)}</span><span>geospatial — no browser preview</span>`;
  }

  runBtn.disabled = true; // must re-upload after any file change
  uploadBtn.textContent = "Upload & continue";
  document
    .querySelector(`.dropzone[data-slot="${idx}"]`)
    .classList.remove("filled-uploaded");
}

[0, 1].forEach((idx) => {
  const input = document.getElementById(`file-${idx}`);
  const zone = document.querySelector(`.dropzone[data-slot="${idx}"]`);

  input.addEventListener("change", (e) =>
    handleFileForSlot(idx, e.target.files[0]),
  );

  ["dragenter", "dragover"].forEach((evt) =>
    zone.addEventListener(evt, (e) => {
      e.preventDefault();
      zone.classList.add("drag-over");
    }),
  );
  ["dragleave", "drop"].forEach((evt) =>
    zone.addEventListener(evt, (e) => {
      e.preventDefault();
      zone.classList.remove("drag-over");
    }),
  );
  zone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files[0];
    if (file) {
      input.files = e.dataTransfer.files;
      handleFileForSlot(idx, file);
    }
  });
});

function maybeUpdateCompareSlider(idx, url) {
  if (currentMode !== "bi_temporal") return;
  const wrap = document.getElementById("compare-wrap");
  if (idx === 0) document.getElementById("compare-img-b").src = url; // full underlay = "after" side visible by default
  if (idx === 1) document.getElementById("compare-img-a").src = url;
  if (selectedFiles[0] && selectedFiles[1]) {
    document.getElementById("compare-img-b").src = URL.createObjectURL(
      selectedFiles[1],
    ); // right/base = after
    document.getElementById("compare-img-a").src = URL.createObjectURL(
      selectedFiles[0],
    ); // left/clip = before
    wrap.classList.remove("hidden");
  }
}

const compareRange = document.getElementById("compare-range");
const compareClip = document.getElementById("compare-clip");
const compareHandle = document.getElementById("compare-handle");
compareRange.addEventListener("input", () => {
  const pct = compareRange.value;
  compareClip.style.width = `${pct}%`;
  compareHandle.style.left = `${pct}%`;
  const sliderWidth = document.getElementById("compare-slider").offsetWidth;
  document.getElementById("compare-img-a").style.width = `${sliderWidth}px`;
});

uploadBtn.addEventListener("click", async () => {
  const cfg = MODE_CONFIG[currentMode];
  const files = [];
  for (let i = 0; i < cfg.slots; i++) {
    if (!selectedFiles[i]) {
      toast(`Please choose ${cfg.titles[i].toLowerCase()}.`, "err");
      return;
    }
    files.push(selectedFiles[i]);
  }
  const form = new FormData();
  files.forEach((f) => form.append("files", f));

  uploadBtn.textContent = "Uploading…";
  uploadBtn.disabled = true;
  try {
    const res = await fetch(`${API_BASE}/api/upload`, {
      method: "POST",
      body: form,
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    uploadedImageIds = data.uploaded.map((u) => u.image_id);
    runBtn.disabled = false;
    uploadBtn.textContent = "✓ Uploaded";
    tmImages.textContent = String(uploadedImageIds.length);
    markStepDone("upload");
    toast(`${uploadedImageIds.length} image(s) uploaded.`);
    goToStep("query");
  } catch (e) {
    toast("Upload failed: " + e.message, "err");
    uploadBtn.textContent = "Upload & continue";
  } finally {
    uploadBtn.disabled = false;
  }
});

runBtn.addEventListener("click", async () => {
  const query = queryInput.value.trim();
  if (!query) {
    toast("Type a question first.", "err");
    return;
  }
  if (uploadedImageIds.length === 0) {
    toast("Upload image(s) first.", "err");
    return;
  }

  goToStep("results");
  resultWrap.classList.add("hidden");
  errorPanel.classList.add("hidden");
  runProgress.classList.remove("hidden");
  renderRunProgress([]);

  runBtn.disabled = true;

  try {
    const res = await fetch(`${API_BASE}/api/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        image_ids: uploadedImageIds,
        input_mode: currentMode,
      }),
    });
    const data = await res.json();
    const seenSteps = (data.trace || []).map((t) => t.step);
    await animateProgress(seenSteps);

    runProgress.classList.add("hidden");
    if (data.error) {
      document.getElementById("error-text").textContent = data.error;
      errorPanel.classList.remove("hidden");
      toast("Run failed — see details.", "err");
    } else {
      renderResult(data);
      resultWrap.classList.remove("hidden");
      markStepDone("query");
      markStepDone("results");
      tmLastRun.textContent = new Date().toLocaleTimeString();
      toast("Analysis complete.");
    }
  } catch (e) {
    runProgress.classList.add("hidden");
    document.getElementById("error-text").textContent =
      "Request failed: " + e.message;
    errorPanel.classList.remove("hidden");
    toast("Request failed.", "err");
  } finally {
    runBtn.disabled = false;
  }
});

function renderRunProgress(doneSteps) {
  runProgressSteps.innerHTML = "";
  RUN_STEPS.forEach((s, i) => {
    const div = document.createElement("div");
    div.className = "rp-step";
    if (doneSteps.includes(s.key)) div.classList.add("done");
    else if (i === doneSteps.length) div.classList.add("active");
    div.innerHTML = `<span class="rp-dot"></span><span>${s.label}</span>`;
    runProgressSteps.appendChild(div);
  });
}

function animateProgress(seenSteps) {
  return new Promise((resolve) => {
    let i = 0;
    const done = [];
    const tick = () => {
      if (i < RUN_STEPS.length) {
        done.push(RUN_STEPS[i].key);
        renderRunProgress(done);
        i++;
        setTimeout(tick, 220);
      } else {
        resolve();
      }
    };
    tick();
  });
}
function confidenceBand(v) {
  if (v >= 0.7) return "high";
  if (v >= 0.35) return "medium";
  return "low";
}
const BAND_COLOR = { high: "#4fd8c4", medium: "#ffb545", low: "#ff5a52" };

function renderResult(data) {
  document.getElementById("answer-text").textContent = data.final_answer;

  const pct = Math.round((data.overall_confidence || 0) * 100);
  const band = confidenceBand(data.overall_confidence || 0);
  const gauge = document.getElementById("confidence-gauge");
  gauge.style.setProperty("--pct", pct);
  gauge.style.setProperty("--cyan", BAND_COLOR[band]);
  gauge.style.background = `conic-gradient(${BAND_COLOR[band]} calc(var(--pct) * 1%), #1f3350 0)`;
  document.getElementById("gauge-value").textContent = `${pct}%`;

  const grid = document.getElementById("evidence-grid");
  grid.innerHTML = "";
  const urls = data.evidence_urls || [];
  document
    .getElementById("evidence-empty")
    .classList.toggle("hidden", urls.length > 0);
  urls.forEach((url) => {
    const img = document.createElement("img");
    img.src = `${API_BASE}${url}`;
    img.addEventListener("click", () => window.open(img.src, "_blank"));
    grid.appendChild(img);
  });

  const traceList = document.getElementById("trace-list");
  traceList.innerHTML = "";
  (data.trace || []).forEach((entry) => {
    const li = document.createElement("li");
    const { step, timestamp, ...rest } = entry;
    const details = Object.entries(rest)
      .map(([k, v]) => `${k}: ${JSON.stringify(v)}`)
      .join(" · ");
    li.innerHTML = `<strong>${step.replace(/_/g, " ")}</strong>${details}`;
    traceList.appendChild(li);
  });

  document.getElementById("trace-json").textContent = JSON.stringify(
    data,
    null,
    2,
  );

  const reportLink = document.getElementById("report-link");
  if (data.report_url) {
    reportLink.href = `${API_BASE}${data.report_url}`;
    reportLink.style.display = "inline-flex";
  } else {
    reportLink.style.display = "none";
  }
}

document.getElementById("tab-bar").addEventListener("click", (e) => {
  const btn = e.target.closest(".tab-btn");
  if (!btn) return;
  document
    .querySelectorAll(".tab-btn")
    .forEach((b) => b.classList.remove("active"));
  document
    .querySelectorAll(".tab-panel")
    .forEach((p) => p.classList.remove("active"));
  btn.classList.add("active");
  document.getElementById(`tab-${btn.dataset.tab}`).classList.add("active");
});

document.getElementById("copy-btn").addEventListener("click", () => {
  const text = document.getElementById("answer-text").textContent;
  navigator.clipboard.writeText(text).then(() => toast("Answer copied."));
});

resetBtn.addEventListener("click", () => {
  resetUploadState();
  queryInput.value = "";
  resultWrap.classList.add("hidden");
  errorPanel.classList.add("hidden");
  runProgress.classList.add("hidden");
  document.querySelectorAll(".step").forEach((s) => s.classList.remove("done"));
  tmLastRun.textContent = "—";
  goToStep("upload");
  toast("Session reset.");
});

renderSlots();
renderExamples();
document.addEventListener("DOMContentLoaded", () => {
  const landingPage = document.getElementById("landing-page");
  const startButton = document.getElementById("start-btn");
  const appShell = document.getElementById("app-shell");

  if (!landingPage || !startButton || !appShell) {
    return;
  }

  startButton.addEventListener("click", () => {
    landingPage.classList.add("landing-hidden");

    setTimeout(() => {
      appShell.classList.add("active");
      landingPage.style.display = "none";
    }, 550);
  });
});