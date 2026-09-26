import { ReplayDataSource } from "./data-source.js";

const STAGE_LABELS = {
  assessment: "退化感知",
  planning: "智能规划",
  completed: "完成",
  "super-resolution": "超分辨率",
  denoising: "去噪",
  "motion deblurring": "运动去模糊",
  "defocus deblurring": "失焦去模糊",
  dehazing: "去雾",
  deraining: "去雨",
  brightening: "暗光增强",
  "jpeg compression artifact removal": "压缩伪影去除",
};

const SOURCE_LABELS = {
  depictqa: "DEPICTQA",
  llm: "LLM · DECISION",
  agent: "AGENT",
  executor: "EXECUTOR",
};

const SEVERITY_VALUES = {
  "very low": 15,
  low: 32,
  medium: 55,
  high: 78,
  "very high": 100,
};

const $ = (selector) => document.querySelector(selector);
const elements = {
  runTitle: $("#run-title"),
  sourceLabel: $("#source-label"),
  runId: $("#run-id"),
  invocationCount: $("#invocation-count"),
  elapsedTime: $("#elapsed-time"),
  totalTime: $("#total-time"),
  stageRail: $("#stage-rail"),
  eventSource: $("#event-source"),
  eventTime: $("#event-time"),
  eventTitle: $("#event-title"),
  eventSummary: $("#event-summary"),
  openQa: $("#open-qa"),
  inputImage: $("#input-image"),
  currentImage: $("#current-image"),
  currentImageLabel: $("#current-image-label"),
  currentImageIndex: $("#current-image-index"),
  imageChangeBadge: $("#image-change-badge"),
  currentCard: $(".current-card"),
  reasoning: $("#reasoning-content"),
  timeline: $("#timeline"),
  eventCounter: $("#event-counter"),
  playToggle: $("#play-toggle"),
  playIcon: $("#play-icon"),
  restart: $("#restart"),
  scrubber: $("#scrubber"),
  controlStatus: $("#control-status"),
  controlEvent: $("#control-event"),
  qaDialog: $("#qa-dialog"),
  dialogSource: $("#dialog-source"),
  dialogQuestion: $("#dialog-question"),
  dialogAnswer: $("#dialog-answer"),
  dialogUsage: $("#dialog-usage"),
  closeDialog: $("#close-dialog"),
  errorScreen: $("#error-screen"),
  errorMessage: $("#error-message"),
};

class ReplayController {
  constructor(manifest) {
    this.manifest = manifest;
    this.events = manifest.events;
    this.interactions = new Map(
      manifest.interactions.map((interaction) => [interaction.id, interaction]),
    );
    this.duration = Math.max(manifest.run.duration_seconds, 0.01);
    this.currentTime = 0;
    this.currentIndex = 0;
    this.speed = 4;
    this.playing = false;
    this.lastFrame = null;
    this.lastImage = null;
    this.bind();
    this.renderStatic();
    this.seek(0);
  }

  bind() {
    elements.playToggle.addEventListener("click", () => this.toggle());
    elements.restart.addEventListener("click", () => {
      this.pause();
      this.seek(0);
    });
    elements.scrubber.addEventListener("input", (event) => {
      this.seek((Number(event.target.value) / 100) * this.duration);
    });
    document.querySelectorAll("[data-speed]").forEach((button) => {
      button.addEventListener("click", () => {
        this.speed = Number(button.dataset.speed);
        document.querySelectorAll("[data-speed]").forEach((item) => {
          item.classList.toggle("active", item === button);
        });
      });
    });
    elements.openQa.addEventListener("click", () => this.showInteraction());
    elements.closeDialog.addEventListener("click", () => elements.qaDialog.close());
    elements.qaDialog.addEventListener("click", (event) => {
      if (event.target === elements.qaDialog) elements.qaDialog.close();
    });
  }

  renderStatic() {
    const { run, stages } = this.manifest;
    elements.runTitle.textContent = run.title;
    elements.sourceLabel.textContent = run.source_label;
    elements.runId.textContent = run.id;
    elements.invocationCount.textContent = run.n_invocations ?? "—";
    elements.totalTime.textContent = formatTime(this.duration);
    elements.inputImage.src = run.input_image;
    elements.currentImage.src = run.input_image;
    elements.stageRail.innerHTML = stages
      .map(
        (stage, index) => `
          <div class="stage-item" data-stage="${escapeHtml(stage.id)}">
            <span class="stage-dot">${String(index + 1).padStart(2, "0")}</span>
            <span>${escapeHtml(stage.label)}</span>
          </div>
        `,
      )
      .join("");
    elements.timeline.innerHTML = this.events
      .map(
        (event, index) => `
          <article class="timeline-event" data-index="${index}" tabindex="0">
            <div class="timeline-time">${formatTime(event.at)}</div>
            <div class="timeline-content">
              <small>${escapeHtml(SOURCE_LABELS[event.source] || event.source.toUpperCase())}</small>
              <strong>${escapeHtml(event.title)}</strong>
              <p>${escapeHtml(event.summary)}</p>
            </div>
          </article>
        `,
      )
      .join("");
    elements.timeline.querySelectorAll(".timeline-event").forEach((item) => {
      const activate = () => {
        this.pause();
        this.seek(this.events[Number(item.dataset.index)].at);
      };
      item.addEventListener("click", activate);
      item.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") activate();
      });
    });
  }

  toggle() {
    if (this.playing) {
      this.pause();
      return;
    }
    if (this.currentTime >= this.duration) this.seek(0);
    this.playing = true;
    this.lastFrame = performance.now();
    elements.playIcon.textContent = "Ⅱ";
    elements.controlStatus.textContent = `播放中 · ${this.speed}×`;
    requestAnimationFrame((time) => this.tick(time));
  }

  pause() {
    this.playing = false;
    this.lastFrame = null;
    elements.playIcon.textContent = "▶";
    elements.controlStatus.textContent =
      this.currentTime >= this.duration ? "回放完成" : "已暂停";
  }

  tick(now) {
    if (!this.playing) return;
    const delta = (now - this.lastFrame) / 1000;
    this.lastFrame = now;
    this.seek(Math.min(this.duration, this.currentTime + delta * this.speed), false);
    if (this.currentTime >= this.duration) {
      this.pause();
      return;
    }
    requestAnimationFrame((time) => this.tick(time));
  }

  seek(time, pause = true) {
    if (pause && this.playing) this.pause();
    this.currentTime = Math.max(0, Math.min(time, this.duration));
    let index = 0;
    this.events.forEach((event, candidate) => {
      if (event.at <= this.currentTime) index = candidate;
    });
    const changedEvent = index !== this.currentIndex;
    this.currentIndex = index;
    this.renderFrame(this.events[index], changedEvent);
  }

  renderFrame(event, changedEvent) {
    elements.elapsedTime.textContent = formatTime(this.currentTime);
    const progress = (this.currentTime / this.duration) * 100;
    elements.scrubber.value = progress;
    elements.scrubber.style.setProperty("--progress", `${progress}%`);
    elements.eventSource.textContent = SOURCE_LABELS[event.source] || event.source;
    elements.eventTime.textContent = `T+${formatTime(event.at)}`;
    elements.eventTitle.textContent = event.title;
    elements.eventSummary.textContent = event.summary;
    elements.openQa.hidden = !event.qa_id;
    elements.controlEvent.textContent = `${String(this.currentIndex + 1).padStart(2, "0")} · ${event.title}`;
    elements.eventCounter.textContent = `${this.currentIndex + 1} / ${this.events.length}`;

    const image = this.latestImage(this.currentIndex);
    if (image && image !== this.lastImage) {
      elements.currentImage.src = image;
      this.lastImage = image;
      elements.currentCard.classList.remove("changed");
      requestAnimationFrame(() => elements.currentCard.classList.add("changed"));
    } else if (changedEvent) {
      elements.currentCard.classList.remove("changed");
    }
    elements.currentImageLabel.textContent =
      event.kind === "completed"
        ? "最终复原结果"
        : event.data?.tool
          ? `${displayTool(event.data.tool)} 输出`
          : "当前处理结果";
    elements.currentImageIndex.textContent =
      event.kind === "completed" ? "✓" : String(this.imageSequence()).padStart(2, "0");
    elements.imageChangeBadge.textContent =
      event.kind === "completed"
        ? "处理完成"
        : event.image
          ? "新图像已生成"
          : "沿用上一结果";

    this.renderStages(event.stage);
    this.renderTimeline();
    this.renderReasoning(event);
  }

  latestImage(index) {
    for (let candidate = index; candidate >= 0; candidate -= 1) {
      if (this.events[candidate].image) return this.events[candidate].image;
    }
    return this.manifest.run.input_image;
  }

  imageSequence() {
    const images = new Set([this.manifest.run.input_image]);
    this.events.slice(0, this.currentIndex + 1).forEach((event) => {
      if (event.image) images.add(event.image);
    });
    return images.size;
  }

  renderStages(activeStage) {
    const stageIds = this.manifest.stages.map((stage) => stage.id);
    const activeIndex = Math.max(0, stageIds.indexOf(activeStage));
    document.querySelectorAll(".stage-item").forEach((item, index) => {
      item.classList.toggle("done", index < activeIndex);
      item.classList.toggle("active", index === activeIndex);
    });
  }

  renderTimeline() {
    const items = elements.timeline.querySelectorAll(".timeline-event");
    items.forEach((item, index) => {
      item.classList.toggle("past", index < this.currentIndex);
      item.classList.toggle("active", index === this.currentIndex);
    });
    const active = items[this.currentIndex];
    active?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }

  renderReasoning(event) {
    const data = event.data || {};
    if (event.kind === "assessment" && data.assessments?.length) {
      elements.reasoning.innerHTML = `
        <div class="assessment-grid">
          ${data.assessments
            .map(
              (item) => `
                <div class="assessment-item" data-level="${escapeHtml(item.severity)}">
                  <header><span>${escapeHtml(item.label)}</span><strong>${escapeHtml(item.severity)}</strong></header>
                  <div class="severity-track"><i style="width:${SEVERITY_VALUES[item.severity] || 0}%"></i></div>
                </div>
              `,
            )
            .join("")}
        </div>
      `;
      return;
    }

    if (event.kind === "decision" || event.kind === "plan") {
      const planEvent =
        event.kind === "plan"
          ? event
          : this.events.find((item) => item.kind === "plan");
      const plan = planEvent?.data?.plan || this.manifest.run.plan;
      const thought =
        event.data.thought ||
        this.events.find((item) => item.kind === "decision")?.data?.thought ||
        "根据退化状态和历史经验生成有序恢复计划。";
      elements.reasoning.innerHTML = `
        <div class="decision-layout">
          <p>${escapeHtml(thought)}</p>
          <div class="plan-flow">
            ${plan
              .map(
                (stage, index) => `
                  ${index ? '<span class="plan-arrow">→</span>' : ""}
                  <span class="plan-node">${escapeHtml(STAGE_LABELS[stage] || stage)}</span>
                `,
              )
              .join("")}
          </div>
        </div>
      `;
      return;
    }

    const icon =
      event.kind === "completed"
        ? "✓"
        : event.kind === "selection"
          ? "◆"
          : event.source === "depictqa"
            ? "◎"
            : "→";
    const supporting = event.data.severity
      ? `量化判断：${event.data.severity}。该结果写入工作记忆，并用于工具选择或进入下一阶段。`
      : event.kind === "stage_start"
        ? "Agent 按照已生成的任务顺序调用专用图像复原工具。"
        : event.kind === "completed"
          ? "所有计划任务均已完成，最终结果已从真实运行目录载入。"
          : "该事件来自 AgenticIR 的真实工作流日志。";
    elements.reasoning.innerHTML = `
      <div class="simple-reason">
        <div class="reason-icon">${icon}</div>
        <div>
          <h4>${escapeHtml(event.title)}</h4>
          <p>${escapeHtml(supporting)}</p>
        </div>
      </div>
    `;
  }

  showInteraction() {
    const event = this.events[this.currentIndex];
    const interaction = this.interactions.get(event.qa_id);
    if (!interaction) return;
    elements.dialogSource.textContent = interaction.source_label;
    elements.dialogQuestion.textContent = interaction.question;
    elements.dialogAnswer.textContent = interaction.answer;
    const usage = interaction.usage || {};
    const parts = [];
    if (usage.prompt_tokens != null) parts.push(`PROMPT ${usage.prompt_tokens} TOKENS`);
    if (usage.completion_tokens != null) parts.push(`COMPLETION ${usage.completion_tokens} TOKENS`);
    if (usage.cost_usd != null) parts.push(`COST $${usage.cost_usd.toFixed(5)}`);
    elements.dialogUsage.textContent =
      parts.join("  ·  ") || "原始图像 Base64 与系统提示词已从演示副本中移除";
    elements.qaDialog.showModal();
  }
}

function formatTime(seconds) {
  const safe = Math.max(0, Number(seconds) || 0);
  const minutes = Math.floor(safe / 60);
  const remainder = safe - minutes * 60;
  return `${String(minutes).padStart(2, "0")}:${remainder.toFixed(1).padStart(4, "0")}`;
}

function displayTool(tool) {
  const aliases = {
    diffbir: "DiffBIR",
    mprnet: "MPRNet",
    restormer: "Restormer",
    swinir_50: "SwinIR",
    xrestormer: "X-Restormer",
    drbnet: "DRBNet",
    maxim: "MAXIM",
  };
  return aliases[tool] || tool.replaceAll("_", " ");
}

function escapeHtml(value) {
  const div = document.createElement("div");
  div.textContent = String(value ?? "");
  return div.innerHTML;
}

async function start() {
  try {
    const manifest = await new ReplayDataSource().load();
    new ReplayController(manifest);
  } catch (error) {
    elements.errorMessage.textContent = error.message;
    elements.errorScreen.hidden = false;
  }
}

start();
