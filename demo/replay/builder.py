from __future__ import annotations

import ast
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TIMESTAMP_RE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})"
    r" - (?P<level>[A-Z]+)$"
)
QA_RE = re.compile(
    r"\*\*Question\*\*\s*(?P<question>.*?)\s*"
    r"\*\*Answer \(from (?P<source>[^)]+)\)\*\*\s*"
    r"(?P<answer>.*?)(?=\n\s*\*\*Question\*\*|\Z)",
    re.DOTALL,
)
IMAGE_RE = re.compile(r"!\[image\]\(data:image/[^\n]+\)")
SEVERITY_RE = re.compile(
    r"^Severity of (?P<degradation>.+?) of (?P<target>.+?) is "
    r"(?P<severity>very low|low|medium|high|very high)\.$"
)
EXECUTING_RE = re.compile(r"^Executing (?P<stage>.+?) on (?P<target>.+?)\.\.\.$")
RESULT_RE = re.compile(
    r"^(?P<label>.+?) result: (?P<target>.+?) with "
    r"(?P<severity>very low|low|medium|high|very high) severity\.$"
)

STAGE_LABELS = {
    "super-resolution": "超分辨率",
    "denoising": "去噪",
    "motion deblurring": "运动去模糊",
    "defocus deblurring": "失焦去模糊",
    "dehazing": "去雾",
    "deraining": "去雨",
    "brightening": "暗光增强",
    "jpeg compression artifact removal": "压缩伪影去除",
}
DEGRADATION_LABELS = {
    "low resolution": "低分辨率",
    "noise": "噪声",
    "motion blur": "运动模糊",
    "defocus blur": "失焦模糊",
    "blur": "模糊",
    "haze": "雾",
    "rain": "雨",
    "dark": "暗光",
    "jpeg compression artifact": "JPEG 压缩伪影",
}


def _clean_markdown(text: str) -> str:
    text = IMAGE_RE.sub("[图像输入已从演示数据中移除]", text)
    return text.replace("\\>", ">").strip()


def parse_qa_log(path: Path) -> list[dict[str, Any]]:
    """Parse model interactions without retaining images or system prompts."""

    text = path.read_text(encoding="utf-8")
    interactions: list[dict[str, Any]] = []
    for index, match in enumerate(QA_RE.finditer(text), start=1):
        source_raw = match.group("source").strip()
        source = "depictqa" if "depict" in source_raw.lower() else "llm"
        question = _clean_markdown(match.group("question"))
        answer = _clean_markdown(match.group("answer"))

        usage: dict[str, Any] = {}
        usage_match = re.search(
            r"Token usage so far: (\d+) prompt tokens, (\d+) completion tokens",
            answer,
        )
        cost_match = re.search(r"Cost so far: \$([0-9.]+)", answer)
        if usage_match:
            usage["prompt_tokens"] = int(usage_match.group(1))
            usage["completion_tokens"] = int(usage_match.group(2))
        if cost_match:
            usage["cost_usd"] = float(cost_match.group(1))
        answer = re.split(r"\nToken usage so far:", answer, maxsplit=1)[0].strip()

        parsed: Any = None
        try:
            if source == "depictqa":
                parsed = ast.literal_eval(answer)
            else:
                parsed = json.loads(answer)
        except (SyntaxError, ValueError, json.JSONDecodeError):
            parsed = None

        interactions.append(
            {
                "id": f"qa-{index:03d}",
                "source": source,
                "source_label": "DepictQA" if source == "depictqa" else source_raw,
                "question": question,
                "answer": answer,
                "parsed": parsed,
                "usage": usage,
            }
        )
    return interactions


def parse_workflow_log(path: Path) -> list[dict[str, Any]]:
    """Parse the timestamp/message pairs written by AgenticIR's logger."""

    records: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    message_lines: list[str] = []

    def flush() -> None:
        nonlocal current, message_lines
        if current is not None:
            current["message"] = "\n".join(message_lines).strip()
            records.append(current)
        current = None
        message_lines = []

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        timestamp_match = TIMESTAMP_RE.match(raw_line)
        if timestamp_match:
            flush()
            current = {
                "timestamp": datetime.strptime(
                    timestamp_match.group("timestamp"), "%Y-%m-%d %H:%M:%S,%f"
                ),
                "level": timestamp_match.group("level"),
            }
        elif current is not None and (raw_line or message_lines):
            message_lines.append(raw_line)
    flush()
    return records


class ReplayBuilder:
    """Build a browser-safe replay bundle from one completed run."""

    def __init__(self, run_dir: Path, output_dir: Path, title: str | None = None):
        self.run_dir = run_dir.resolve()
        self.output_dir = output_dir.resolve()
        self.title = title or "复杂退化图像智能复原"
        self.logs_dir = self.run_dir / "logs"
        self.summary_path = self.logs_dir / "summary.json"
        self.workflow_path = self.logs_dir / "workflow.log"
        self.qa_path = self.logs_dir / "llm_qa.md"
        self.assets_dir = self.output_dir / "assets"
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        missing = [
            path
            for path in (self.summary_path, self.workflow_path, self.qa_path)
            if not path.is_file()
        ]
        if missing:
            joined = ", ".join(str(path) for path in missing)
            raise FileNotFoundError(f"运行目录缺少演示所需文件：{joined}")

    def build(self) -> Path:
        summary = json.loads(self.summary_path.read_text(encoding="utf-8"))
        workflow = parse_workflow_log(self.workflow_path)
        interactions = parse_qa_log(self.qa_path)
        if not workflow:
            raise ValueError(f"工作流日志为空：{self.workflow_path}")

        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.assets_dir.mkdir(parents=True)

        image_map, gallery = self._copy_tree_images(summary["tree"])
        input_image = self._copy_named_image(
            self._resolve_logged_path(summary["tree"]["img_path"]),
            "00-input.png",
        )
        result_path = self.run_dir / "result.png"
        result_image = (
            self._copy_named_image(result_path, "99-result.png")
            if result_path.is_file()
            else gallery[-1]["image"]
        )

        events = self._build_events(workflow, interactions, image_map, result_image)
        initial_plan = summary.get("plan", {}).get("initial") or []
        execution_path = summary.get("execution_path", {}).get("subtasks") or []
        plan = initial_plan or execution_path
        stages = self._build_stages(plan, events)
        duration = max(event["at"] for event in events) if events else 0.0

        manifest = {
            "schema_version": 1,
            "mode": "replay",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "run": {
                "id": self.run_dir.name,
                "title": self.title,
                "source_label": "真实运行记录回放",
                "started_at": workflow[0]["timestamp"].isoformat(),
                "duration_seconds": round(duration, 3),
                "status": "completed",
                "input_image": input_image,
                "result_image": result_image,
                "plan": plan,
                "n_invocations": summary.get("n_invocations"),
            },
            "stages": stages,
            "interactions": interactions,
            "gallery": gallery,
            "events": events,
        }
        manifest_path = self.output_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return manifest_path

    def _resolve_logged_path(self, logged_path: str) -> Path:
        path = Path(logged_path)
        if path.is_file():
            candidate = path.resolve()
        else:
            try:
                run_index = path.parts.index(self.run_dir.name)
            except ValueError as exc:
                raise FileNotFoundError(f"无法定位日志中的图片：{logged_path}") from exc
            candidate = self.run_dir.joinpath(*path.parts[run_index + 1 :]).resolve()
        try:
            candidate.relative_to(self.run_dir)
        except ValueError as exc:
            raise ValueError(f"日志图片不在运行目录内：{candidate}") from exc
        if not candidate.is_file():
            raise FileNotFoundError(f"日志中的图片不存在：{candidate}")
        return candidate

    def _copy_named_image(self, source: Path, filename: str) -> str:
        destination = self.assets_dir / filename
        shutil.copy2(source, destination)
        return f"data/current/assets/{filename}"

    def _copy_tree_images(
        self, tree: dict[str, Any]
    ) -> tuple[dict[tuple[str, str], str], list[dict[str, Any]]]:
        image_map: dict[tuple[str, str], str] = {}
        gallery: list[dict[str, Any]] = []
        index = 1

        def walk(node: dict[str, Any], parent_chain: list[str]) -> None:
            nonlocal index
            for stage, stage_node in node.get("children", {}).items():
                for tool, tool_node in stage_node.get("tools", {}).items():
                    safe_stage = re.sub(r"[^a-z0-9]+", "-", stage.lower()).strip("-")
                    safe_tool = re.sub(r"[^a-z0-9]+", "-", tool.lower()).strip("-")
                    filename = f"{index:02d}-{safe_stage}-{safe_tool}.png"
                    image = self._copy_named_image(
                        self._resolve_logged_path(tool_node["img_path"]), filename
                    )
                    image_map[(stage, tool)] = image
                    gallery.append(
                        {
                            "id": f"image-{index:03d}",
                            "stage": stage,
                            "stage_label": STAGE_LABELS.get(stage, stage),
                            "tool": tool,
                            "tool_label": _display_tool(tool),
                            "degradation": tool_node.get("degradation"),
                            "severity": tool_node.get("severity"),
                            "selected": stage_node.get("best_tool") == tool,
                            "image": image,
                            "chain": parent_chain + [f"{stage}@{tool}"],
                        }
                    )
                    index += 1
                    walk(tool_node, parent_chain + [f"{stage}@{tool}"])

        walk(tree, [])
        return image_map, gallery

    def _build_events(
        self,
        workflow: list[dict[str, Any]],
        interactions: list[dict[str, Any]],
        image_map: dict[tuple[str, str], str],
        result_image: str,
    ) -> list[dict[str, Any]]:
        start = workflow[0]["timestamp"]
        depictqa_queue = [
            qa["id"] for qa in interactions if qa["source"] == "depictqa"
        ]
        llm_queue = [qa["id"] for qa in interactions if qa["source"] == "llm"]
        assessment_qa = depictqa_queue.pop(0) if depictqa_queue else None
        decision_qa = llm_queue.pop(0) if llm_queue else None
        current_stage: str | None = None
        current_image: str | None = None
        events: list[dict[str, Any]] = []

        for record in workflow:
            message = record["message"]
            at = round((record["timestamp"] - start).total_seconds(), 3)
            base = {
                "id": f"event-{len(events) + 1:03d}",
                "at": at,
                "timestamp": record["timestamp"].isoformat(),
                "source": "agent",
                "kind": "log",
                "stage": current_stage or "assessment",
                "title": "运行事件",
                "summary": message,
                "image": current_image,
                "qa_id": None,
                "data": {},
            }

            if message.startswith("Evaluation:"):
                assessments = _literal_list(message[len("Evaluation:") :].strip())
                base.update(
                    source="depictqa",
                    kind="assessment",
                    stage="assessment",
                    title="DepictQA 全局退化感知",
                    summary=_assessment_summary(assessments),
                    qa_id=assessment_qa,
                    data={"assessments": _assessment_items(assessments)},
                )
            elif message.startswith("Insights:"):
                thought = message[len("Insights:") :].strip()
                base.update(
                    source="llm",
                    kind="decision",
                    stage="planning",
                    title="LLM 检索经验并形成排序依据",
                    summary=thought,
                    qa_id=decision_qa,
                    data={"thought": thought},
                )
            elif message.startswith("Plan:"):
                plan = _literal_list(message[len("Plan:") :].strip())
                base.update(
                    source="llm",
                    kind="plan",
                    stage="planning",
                    title="生成恢复计划",
                    summary=" → ".join(STAGE_LABELS.get(item, item) for item in plan),
                    qa_id=decision_qa,
                    data={"plan": plan},
                )
            elif executing := EXECUTING_RE.match(message):
                current_stage = executing.group("stage")
                base.update(
                    source="executor",
                    kind="stage_start",
                    stage=current_stage,
                    title=f"开始{STAGE_LABELS.get(current_stage, current_stage)}",
                    summary=f"处理对象：{_display_chain(executing.group('target'))}",
                    data={"target": executing.group("target")},
                )
            elif severity := SEVERITY_RE.match(message):
                target = severity.group("target")
                tool = target.rsplit("@", 1)[-1]
                current_image = image_map.get((current_stage or "", tool), current_image)
                qa_id = depictqa_queue.pop(0) if depictqa_queue else None
                degradation = severity.group("degradation")
                level = severity.group("severity")
                base.update(
                    source="depictqa",
                    kind="tool_assessment",
                    stage=current_stage or "execution",
                    title=f"DepictQA 评估 {_display_tool(tool)}",
                    summary=(
                        f"处理后{DEGRADATION_LABELS.get(degradation, degradation)}"
                        f"程度：{level}"
                    ),
                    image=current_image,
                    qa_id=qa_id,
                    data={
                        "tool": tool,
                        "degradation": degradation,
                        "severity": level,
                    },
                )
            elif result := RESULT_RE.match(message):
                target = result.group("target")
                tool = target.rsplit("@", 1)[-1]
                current_image = image_map.get((current_stage or "", tool), current_image)
                base.update(
                    source="agent",
                    kind="selection",
                    stage=current_stage or "execution",
                    title=f"选择 {_display_tool(tool)}",
                    summary=(
                        f"{STAGE_LABELS.get(current_stage or '', current_stage or '')}"
                        f"完成，残余程度 {result.group('severity')}"
                    ),
                    image=current_image,
                    data={"tool": tool, "severity": result.group("severity")},
                )
            elif message.startswith("Restoration result:"):
                current_image = result_image
                base.update(
                    source="agent",
                    kind="completed",
                    stage="completed",
                    title="图像复原完成",
                    summary=f"最终路径：{_display_chain(message.split(':', 1)[1].strip('. '))}",
                    image=result_image,
                    data={"result": message.split(":", 1)[1].strip(". ")},
                )

            events.append(base)
        return events

    def _build_stages(
        self, plan: list[str], events: list[dict[str, Any]]
    ) -> list[dict[str, str]]:
        execution_stages = list(plan)
        if not execution_stages:
            execution_stages = list(
                dict.fromkeys(
                    event["stage"]
                    for event in events
                    if event["stage"] not in {"assessment", "planning", "completed"}
                )
            )
        stage_ids = ["assessment", "planning", *execution_stages, "completed"]
        labels = {
            "assessment": "退化感知",
            "planning": "智能规划",
            "completed": "完成",
            **STAGE_LABELS,
        }
        return [{"id": stage, "label": labels.get(stage, stage)} for stage in stage_ids]


def _literal_list(value: str) -> list[Any]:
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []


def _assessment_items(items: list[Any]) -> list[dict[str, str]]:
    return [
        {
            "degradation": str(item[0]),
            "label": DEGRADATION_LABELS.get(str(item[0]), str(item[0])),
            "severity": str(item[1]),
        }
        for item in items
        if isinstance(item, (list, tuple)) and len(item) == 2
    ]


def _assessment_summary(items: list[Any]) -> str:
    significant = [
        item
        for item in _assessment_items(items)
        if item["severity"] in {"medium", "high", "very high"}
    ]
    if not significant:
        return "未检测到需要处理的中高程度退化"
    return "；".join(f"{item['label']} {item['severity']}" for item in significant)


def _display_tool(tool: str) -> str:
    aliases = {
        "diffbir": "DiffBIR",
        "mprnet": "MPRNet",
        "restormer": "Restormer",
        "swinir_50": "SwinIR",
        "xrestormer": "X-Restormer",
        "drbnet": "DRBNet",
        "maxim": "MAXIM",
    }
    return aliases.get(tool, tool.replace("_", " ").title())


def _display_chain(chain: str) -> str:
    chunks = re.split(r"-(?=[a-z][a-z -]+@)", chain)
    displayed: list[str] = []
    for chunk in chunks:
        if "@" in chunk:
            stage, tool = chunk.rsplit("@", 1)
            displayed.append(
                f"{STAGE_LABELS.get(stage, stage)} / {_display_tool(tool)}"
            )
        else:
            displayed.append(chunk)
    return " → ".join(displayed)
