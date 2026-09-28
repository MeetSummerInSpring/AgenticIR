from pathlib import Path
from ast import literal_eval
import hashlib
import shutil
import logging
from time import localtime, strftime
from typing import Optional
from uuid import uuid4

import cv2
import json

from llm import GPT4, DepictQA
from . import prompts
from executor import executor, Tool, ToolExecutionError, ToolRunResult
from utils.img_tree import ImgTree
from utils.logger import get_logger
from utils.misc import sorted_glob
from utils.custom_types import *
from utils.episode_store import EPISODE_SCHEMA_VERSION, EpisodeStore
from utils.schedule_memory import ScheduleMemory
from utils.tool_selector import ToolSelector


class IRAgent:
    """
    Args:
        input_path (Path): Path to the input image.
        output_dir (Path): Path to the output directory, in which a directory will be created.
        llm_config_path (Path, optional): Path to the config file of LLM. Defaults to Path("config.yml").
        evaluate_degradation_by (str, optional): The method of degradation evaluation, "depictqa" or "gpt4v". Defaults to "depictqa".
        with_retrieval (bool, optional): Whether to schedule with retrieval. Defaults to True.
        schedule_experience_path (Path | None, optional): Path to structured scheduling memory. Defaults to Path("memory/schedule_rules.json").
        with_reflection (bool, optional): Whether to reflect on the results of tools. Defaults to True.
        reflect_by (str, optional): The method of reflection on results of tools, "depictqa" or "gpt4v". Defaults to "depictqa".
        with_rollback (bool, optional): Whether to roll back when failing in one subtask. Defaults to True.
        silent (bool, optional): Whether to suppress the console output. Defaults to False.
        episode_store_path (Path, optional): Append-only runtime event database. Defaults to Path("memory/episodes.sqlite3").
        tool_profile_path (Path, optional): Tool ranking policy and cost hints. Defaults to Path("memory/tool_profiles.json").
        max_tools_per_subtask (int | None, optional): Optional Top-K override.
        tool_exploration_rate (float | None, optional): Optional exploration-rate override.
        random_seed (int, optional): Seed for controlled tool exploration. Defaults to 0.
    """

    def __init__(
        self,
        input_path: Path,
        output_dir: Path,
        llm_config_path: Path = Path("config.yml"),
        evaluate_degradation_by: str = "depictqa",
        with_retrieval: bool = True,
        schedule_experience_path: Optional[Path] = Path(
            "memory/schedule_rules.json"
        ),
        with_reflection: bool = True,
        reflect_by: str = "depictqa",
        with_rollback: bool = True,
        silent: bool = False,
        episode_store_path: Path = Path("memory/episodes.sqlite3"),
        tool_profile_path: Path = Path("memory/tool_profiles.json"),
        max_tools_per_subtask: Optional[int] = None,
        tool_exploration_rate: Optional[float] = None,
        random_seed: int = 0,
    ) -> None:
        self._prepare_dir(input_path, output_dir)
        self._init_state()
        self._config(
            evaluate_degradation_by,
            with_retrieval,
            with_reflection,
            reflect_by,
            with_rollback,
        )
        self._create_components(
            llm_config_path=llm_config_path,
            schedule_experience_path=schedule_experience_path,
            silent=silent,
            episode_store_path=episode_store_path,
            tool_profile_path=tool_profile_path,
            max_tools_per_subtask=max_tools_per_subtask,
            tool_exploration_rate=tool_exploration_rate,
            random_seed=random_seed,
        )
        self._set_constants()
        self._record_run_started()

    def _init_state(self) -> None:
        self.plan: list[Subtask] = []
        self.work_mem: dict = {
            "plan": {"initial": [], "adjusted": [
                # {
                #     "failed": [...] + [...],
                #     "new": [...] + [...]
                # }
            ]},
            "execution_path": {"subtasks": [], "tools": []},
            "n_invocations": 0,
            "degradation_state": {},
            "tool_rankings": [],
            "tree": {
                "img_path": str(self.img_tree_dir / "0-img" / "input.png"),
                "best_descendant": None,
                "state": {},
                "children": {
                    # `subtask1`: {
                    #     "best_tool": ...,
                    #     "tools": {
                    #         `tool1`: {
                    #             "degradation": ...,
                    #             "severity": ...,
                    #             "img_path": ...,
                    #             "best_descendant": ...,
                    #             "children": {...}
                    #         },
                    #         ...
                    #     }
                    # }
                },
            },
        }
        self.cur_node = self.work_mem["tree"]

    def _config(
        self,
        evaluate_degradation_by: str,
        with_retrieval: bool,
        with_reflection: bool,
        reflect_by: str,
        with_rollback: bool
    ) -> None:
        assert evaluate_degradation_by in {"gpt4v", "depictqa"}
        self.evaluate_degradation_by = evaluate_degradation_by
        self.with_retrieval = with_retrieval
        assert reflect_by in {"gpt4v", "depictqa"}
        self.with_reflection = with_reflection
        self.reflect_by = reflect_by
        self.with_rollback = with_rollback

    def _create_components(
        self,
        llm_config_path: Path,
        schedule_experience_path: Optional[Path],
        silent: bool,
        episode_store_path: Path,
        tool_profile_path: Path,
        max_tools_per_subtask: Optional[int],
        tool_exploration_rate: Optional[float],
        random_seed: int,
    ) -> None:
        self.qa_logger = get_logger(
            logger_name="IRAgent QA",
            log_file=self.qa_path,
            console_log_level=logging.WARNING,
            file_format_str="%(message)s",
            silent=silent,
        )
        workflow_format_str = "%(asctime)s - %(levelname)s\n%(message)s\n"
        self.workflow_logger = get_logger(
            logger_name="IRAgent Workflow",
            log_file=self.workflow_path,
            console_format_str=workflow_format_str,
            file_format_str=workflow_format_str,
            silent=silent,
        )

        self.gpt4 = GPT4(
            config_path=llm_config_path,
            logger=self.qa_logger,
            silent=silent,
            system_message=prompts.system_message,
        )
        self.depictqa = None
        if self.evaluate_degradation_by == "depictqa" or self.reflect_by == "depictqa":
            self.depictqa = DepictQA(logger=self.qa_logger, silent=silent)

        self.schedule_memory = None
        self.schedule_memory_path = schedule_experience_path
        if self.with_retrieval:
            if schedule_experience_path is None:
                raise ValueError("Structured scheduling experience is required.")
            self.schedule_memory = ScheduleMemory(schedule_experience_path)

        self.executor = executor
        self.episode_store = EpisodeStore(episode_store_path)
        self.episode_store_path = Path(episode_store_path)
        self.tool_profile_path = Path(tool_profile_path)
        self.tool_selector = ToolSelector(
            profile_path=tool_profile_path,
            episode_store=self.episode_store,
            top_k_override=max_tools_per_subtask,
            exploration_rate_override=tool_exploration_rate,
            seed=random_seed,
        )
        self.random_seed = random_seed
        self._last_schedule_rule_ids: list[str] = []

    def _set_constants(self) -> None:
        self.degra_subtask_dict: dict[Degradation, Subtask] = {
            "low resolution": "super-resolution",
            "noise": "denoising",
            "motion blur": "motion deblurring",
            "defocus blur": "defocus deblurring",
            "haze": "dehazing",
            "rain": "deraining",
            "dark": "brightening",
            "jpeg compression artifact": "jpeg compression artifact removal",
        }
        self.subtask_degra_dict: dict[Subtask, Degradation] = {
            v: k for k, v in self.degra_subtask_dict.items()
        }
        self.degradations = set(self.degra_subtask_dict.keys())
        self.subtasks = set(self.degra_subtask_dict.values())
        self.levels: list[Level] = ["very low", "low", "medium", "high", "very high"]

    def run(
        self,
        plan: Optional[list[Subtask]] = None,
        cache: Optional[Path] = None,
    ) -> None:
        try:
            if plan is not None:
                self.plan = plan.copy()
                self.work_mem["plan"]["initial"] = plan.copy()
                self._append_episode_event(
                    event_type="schedule_decision",
                    state=self._episode_state(),
                    action={"source": "manual", "agenda": plan},
                    outcome={"plan": plan, "retrieved_rule_ids": []},
                )
                self._dump_summary()
            else:
                self.propose()

            while self.plan:
                success = self.execute_subtask(cache)
                if plan is None and self.with_rollback and not success:
                    self.roll_back()
                    self.reschedule()
            self._record_res()
        except Exception as exc:
            self._append_episode_event(
                event_type="run_failed",
                state=self._episode_state(),
                outcome={
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            raise

    def propose(self) -> None:
        evaluation = self.evaluate_degradation()
        agenda = self.extract_agenda(evaluation)
        plan = self.schedule(agenda)

        self.work_mem["plan"]["initial"] = plan.copy()
        self._dump_summary()
        self.workflow_logger.info(f"Plan: {plan}")
        self.plan = plan

    def extract_agenda(
        self,
        evaluation: list[tuple[Degradation, Level]],
    ) -> list[Subtask]:
        agenda: list[Subtask] = []
        image = cv2.imread(self.cur_node["img_path"])
        if image is None:
            raise ValueError(f"Cannot read image: {self.cur_node['img_path']}")
        if max(image.shape[:2]) < 300:
            agenda.append("super-resolution")
        for degradation, severity in evaluation:
            if self.levels.index(severity) >= 2:
                agenda.append(self.degra_subtask_dict[degradation])
        return agenda

    def evaluate_degradation(self) -> list[tuple[Degradation, Level]]:
        if self.evaluate_degradation_by == "gpt4v":
            evaluation = self.evaluate_degradation_by_gpt4v()
        else:
            evaluation = literal_eval(
                self.depictqa(
                    Path(self.cur_node["img_path"]),
                    task="eval_degradation",
                )
            )

        degradation_state = dict(evaluation)
        self.work_mem["degradation_state"] = degradation_state
        self.cur_node["state"] = degradation_state.copy()
        self.workflow_logger.info(f"Evaluation: {evaluation}")
        self._append_episode_event(
            event_type="degradation_evaluated",
            state=self._episode_state(),
            outcome={
                "evaluation": degradation_state,
                "evaluator": self.evaluate_degradation_by,
            },
        )
        return evaluation

    def evaluate_degradation_by_gpt4v(self) -> list[tuple[Degradation, Level]]:
        def check_evaluation(evaluation: object):
            assert isinstance(evaluation, list), "Evaluation should be a list."
            rsp_degradations = set()
            for ele in evaluation:
                assert isinstance(
                    ele, dict
                ), "Each element in evaluation should be a dict."
                assert set(ele.keys()) == {
                    "degradation",
                    "thought",
                    "severity",
                }, f"Invalid keys: {ele.keys()}."
                degradation = ele["degradation"]
                rsp_degradations.add(degradation)
                severity = ele["severity"]
                assert severity in self.levels, f"Invalid severity: {severity}."
            assert rsp_degradations == self.degradations - {
                "low resolution"
            }, f"Invalid degradation: {rsp_degradations}."

        evaluation = literal_eval(
            self.gpt4(
                prompt=prompts.gpt_evaluate_degradation_prompt,
                img_path=Path(self.cur_node["img_path"]),
                format_check=check_evaluation,
            )
        )
        evaluation = [(ele["degradation"], ele["severity"]) for ele in evaluation]
        return evaluation

    def schedule(
        self,
        agenda: list[Subtask],
        ps: str = "",
    ) -> list[Subtask]:
        self._last_schedule_rule_ids = []
        if len(agenda) <= 1:
            plan = agenda.copy()
            source = "deterministic"
        else:
            degradations = [
                self.subtask_degra_dict[subtask]
                for subtask in agenda
            ]
            if self.with_retrieval:
                plan = self.schedule_w_retrieval(degradations, agenda, ps)
                source = "structured_memory"
            else:
                plan = self.schedule_wo_retrieval(degradations, agenda, ps)
                source = "llm"

        self._append_episode_event(
            event_type="schedule_decision",
            state=self._episode_state(),
            action={
                "source": source,
                "agenda": agenda,
                "postscript": ps,
            },
            outcome={
                "plan": plan,
                "retrieved_rule_ids": self._last_schedule_rule_ids,
            },
        )
        return plan

    def schedule_w_retrieval(
        self,
        degradations: list[Degradation],
        agenda: list[Subtask],
        ps: str,
    ) -> list[Subtask]:
        def check_order(schedule: object):
            assert isinstance(schedule, dict), "Schedule should be a dict."
            assert set(schedule.keys()) == {"thought", "order"}, (
                f"Invalid keys: {schedule.keys()}."
            )
            order = schedule["order"]
            assert isinstance(order, list), "Order should be a list."
            assert len(order) == len(agenda) and set(order) == set(agenda), (
                f"{order} is not a permutation of {agenda}."
            )

        if self.schedule_memory is None:
            raise RuntimeError("Schedule memory was not initialized.")
        experience, rule_ids = self.schedule_memory.render_for_prompt(
            degradations,
            agenda,
        )
        self._last_schedule_rule_ids = rule_ids
        schedule = self.gpt4(
            prompt=prompts.schedule_w_retrieval_prompt.format(
                degradations=degradations,
                agenda=agenda,
                experience=experience,
            ) + ps,
            format_check=check_order,
        )
        schedule = literal_eval(schedule)
        self.workflow_logger.info(
            f"Retrieved schedule rules: {rule_ids}. "
            f"Insights: {schedule['thought']}"
        )
        return schedule["order"]

    def reason_to_schedule(
        self, degradations: list[Degradation], agenda: list[Subtask]
    ) -> str:
        insights = self.gpt4(
            prompt=prompts.reason_to_schedule_prompt.format(
                degradations=degradations, agenda=agenda
            ),
        )
        self.workflow_logger.info(f"Insights: {insights}")
        return insights

    def schedule_wo_retrieval(
        self, degradations: list[Degradation], agenda: list[Subtask], ps: str
    ) -> list[Subtask]:
        insights: str = self.reason_to_schedule(degradations, agenda)

        def check_order(order: object):
            assert isinstance(order, list), "Order should be a list."
            assert len(order) == len(agenda) and set(order) == set(agenda), (
                f"{order} is not a permutation of {agenda}."
            )

        order = self.gpt4(
            prompt=prompts.schedule_wo_retrieval_prompt.format(
                degradations=degradations, agenda=agenda, insights=insights
            ) + ps,
            format_check=check_order,
        )
        return literal_eval(order)

    def execute_subtask(self, cache: Optional[Path]) -> bool:
        subtask = self.plan.pop(0)
        parent_node = self.cur_node
        state_before = self._episode_state()
        subtask_dir, degradation, toolbox = self._prepare_for_subtask(subtask)
        results_by_level: dict[str, list[Path]] = {}
        success = True

        for tool in toolbox:
            self.work_mem["n_invocations"] += 1
            tool_dir = subtask_dir / f"tool-{tool.tool_name}"
            output_dir = tool_dir / "0-img"
            output_dir.mkdir(parents=True)

            try:
                if cache is None:
                    run_result = tool(
                        input_dir=Path(parent_node["img_path"]).parent,
                        output_dir=output_dir,
                        silent=True,
                    )
                    output_path = run_result.output_path
                else:
                    destination = output_dir / "output.png"
                    relative_path = destination.relative_to(self.img_tree_dir)
                    source = cache / relative_path
                    destination.symlink_to(source)
                    Tool._validate_image(destination, role="cached output")
                    output_path = destination
                    run_result = ToolRunResult(
                        tool_name=tool.tool_name,
                        subtask=subtask,
                        duration_seconds=None,
                        output_path=output_path,
                    )
            except ToolExecutionError as exc:
                self._append_episode_event(
                    event_type="tool_attempt",
                    state=state_before,
                    action={"subtask": subtask, "tool": tool.tool_name},
                    transition={
                        "duration_seconds": exc.duration_seconds,
                    },
                    outcome={
                        "status": "execution_error",
                        "quality_success": None,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                    provenance={"tool_class": type(tool).__name__},
                )
                self.workflow_logger.warning(
                    "Candidate execution failed; trying the next registered candidate: %s", exc
                )
                continue

            try:
                if self.with_reflection:
                    degradation_level = self.evaluate_tool_result(
                        output_path,
                        degradation,
                    )
                else:
                    degradation_level = "none"
            except Exception as exc:
                self._append_episode_event(
                    event_type="tool_attempt",
                    state=state_before,
                    action={"subtask": subtask, "tool": tool.tool_name},
                    transition=self._tool_transition(run_result, output_path),
                    outcome={
                        "status": "evaluation_error",
                        "quality_success": None,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                    provenance={"tool_class": type(tool).__name__},
                )
                raise

            next_node = self._record_tool_res(output_path, degradation_level)
            quality_success = (
                degradation_level in {"very low", "low"}
                if self.with_reflection
                else None
            )
            self._append_episode_event(
                event_type="tool_attempt",
                state=state_before,
                action={"subtask": subtask, "tool": tool.tool_name},
                transition={
                    **self._tool_transition(run_result, output_path),
                    "state_after": next_node["state"],
                },
                outcome={
                    "status": "ok",
                    "quality_success": quality_success,
                    "severity_after": degradation_level,
                },
                provenance={"tool_class": type(tool).__name__},
            )

            results_by_level.setdefault(degradation_level, []).append(output_path)
            if not self.with_reflection or degradation_level == "very low":
                selected_level = degradation_level
                best_tool_name = tool.tool_name
                best_image_path = output_path
                break
        else:
            for result_level in self.levels[1:]:
                if result_level not in results_by_level:
                    continue
                candidates = results_by_level[result_level]
                self.workflow_logger.info("Searching for the best tool...")
                best_image_path = self.search_best_by_comp(candidates)
                best_tool_name = self._get_name_stem(
                    best_image_path.parents[1].name
                )
                selected_level = result_level
                if result_level != "low":
                    success = False
                break
            else:
                raise RuntimeError(
                    f"No valid result was produced for subtask {subtask!r}."
                )

        parent_node["children"][subtask]["best_tool"] = best_tool_name
        self.cur_node = parent_node["children"][subtask]["tools"][best_tool_name]
        self.work_mem["degradation_state"] = self.cur_node["state"].copy()
        if self.with_rollback and not success:
            self.cur_node["best_descendant"] = str(best_image_path)
            done_subtasks, _ = self._get_execution_path(
                Path(self.cur_node["img_path"])
            )
            self.work_mem["plan"]["adjusted"].append(
                {
                    "failed": f"{done_subtasks} + {self.plan}",
                    "new": None,
                }
            )

        self._append_episode_event(
            event_type="tool_selected",
            state=state_before,
            action={"subtask": subtask, "tool": best_tool_name},
            transition={"state_after": self.cur_node["state"]},
            outcome={
                "status": "selected",
                "quality_success": success,
                "severity_after": selected_level,
                "remaining_plan": self.plan,
            },
        )
        self._dump_summary()
        self._render_img_tree()
        self.workflow_logger.info(
            f"{subtask.capitalize()} result: "
            f"{self._img_nickname(self.cur_node['img_path'])} "
            f"with {selected_level} severity."
        )
        return success

    def evaluate_tool_result(self, img_path: Path, degradation: Degradation) -> Level:
        if self.reflect_by == "gpt4v":
            level = self.evaluate_tool_result_by_gpt4v(img_path, degradation)
        else:
            level = literal_eval(
                self.depictqa(
                    img_path=img_path, task="eval_degradation", degradation=degradation
                )
            )[0][1]
        return level

    def evaluate_tool_result_by_gpt4v(
        self, img_path: Path, degradation: Degradation
    ) -> Level:
        def check_tool_res_evaluation(evaluation: object):
            assert isinstance(evaluation, dict), "Evaluation should be a dict."
            assert set(evaluation.keys()) == {
                "thought",
                "severity",
            }, f"Invalid keys: {evaluation.keys()}."
            severity = evaluation["severity"]
            assert severity in self.levels, f"Invalid severity: {severity}."

        degra_level = literal_eval(
            self.gpt4(
                prompt=prompts.gpt_evaluate_tool_result_prompt.format(
                    degradation=degradation
                ),
                img_path=img_path,
                format_check=check_tool_res_evaluation,
            )
        )["severity"]
        return degra_level

    def search_best_by_comp(self, candidates: list[Path]) -> Path:
        """Compares multiple images to decide the best one."""

        best_img = candidates[0]
        for i in range(1, len(candidates)):
            cur_img = candidates[i]
            self.workflow_logger.info(
                f"Comparing {self._img_nickname(best_img)} and {self._img_nickname(cur_img)}..."
            )

            choice = self.compare_quality(best_img, cur_img)

            if choice == "latter":
                best_img = cur_img
                self.workflow_logger.info(
                    f"{self._img_nickname(best_img)} is better."
                )
            elif choice == "former":
                self.workflow_logger.info(
                    f"{self._img_nickname(best_img)} is better."
                )
            else:  # neither; keep the former
                self.workflow_logger.info(
                    f"Hard to decide. Keeping {self._img_nickname(best_img)}."
                )
        self.workflow_logger.info(
            f"{self._img_nickname(best_img)} is selected as the best."
        )
        return best_img

    def compare_quality(self, img1: Path, img2: Path) -> str:
        if self.reflect_by == "gpt4v":
            choice = self.compare_quality_by_gpt4v(img1, img2)
        else:
            choice = self.depictqa(img_path=[img1, img2], task="comp_quality")
        return choice

    def compare_quality_by_gpt4v(self, img1: Path, img2: Path) -> str:
        def check_comparison(comparison: object):
            assert isinstance(comparison, dict), "Comparison should be a dict."
            assert set(comparison.keys()) == {
                "thought",
                "choice",
            }, f"Invalid keys: {comparison.keys()}."
            assert comparison["choice"] in {
                "former",
                "latter",
                "neither",
            }, f"Invalid choice: {comparison['choice']}."

        comparison: dict = literal_eval(
            self.gpt4(
                prompt=prompts.gpt_compare_prompt,
                img_path=[img1, img2],
                format_check=check_comparison,
            )
        )
        return comparison["choice"]

    def roll_back(self) -> None:
        # backtrack
        self._backtrack()
        step = 1
        while self._fully_expanded():
            self.workflow_logger.info(
                f"All execution paths from {self._img_nickname(self.cur_node['img_path'])} "
                f"lead to severe degradation.")
            self._set_best_desc()
            if self.cur_node != self.work_mem["tree"]:
                step += 1
                self._backtrack()
            else:
                break
        self.workflow_logger.info(
            f"Roll back for {step} step(s) "
            f"to {self._img_nickname(self.cur_node['img_path'])} "
            f"with agenda {self.plan}."
        )

        # compromise
        if self._fully_expanded():  # back to root
            self._to_best_desc(Path(self.cur_node["best_descendant"]))
            self.workflow_logger.info(
                "All execution paths from the input lead to severe degradation.\n"
                f"Compromise: jump to {self._img_nickname(self.cur_node['img_path'])} "
                f"with agenda {self.plan}."
            )
            assert not self._fully_expanded() or not self.plan, \
                "Invalid compromise: cannot go on or terminate."
        
        # check
        done_subtasks, _ = self._get_execution_path(Path(self.cur_node['img_path']))
        done_subtasks, plan = set(done_subtasks), set(self.plan)
        assert done_subtasks & plan == set(), \
            f"Invalid plan: {done_subtasks} & {plan} != ∅."
        assert done_subtasks | plan == set(self.work_mem["plan"]["initial"]), (
            f"Invalid plan: {done_subtasks} | {plan} != "
            f"{self.work_mem['plan']['initial']}.")

    def _fully_expanded(self) -> bool:
        return len(self.plan) == len(self.cur_node["children"])

    def _set_best_desc(self) -> None:
        candidates = [
            Path(subtask_res["tools"][subtask_res["best_tool"]]["best_descendant"])
            for subtask_res in self.cur_node["children"].values()
        ]
        self.workflow_logger.info("Searching for the best descendant...")
        best_img_path = self.search_best_by_comp(candidates)
        self.cur_node["best_descendant"] = str(best_img_path)

    def _to_best_desc(self, best_desc_path: Path):
        self.cur_node = self._img_path_to_node(best_desc_path)
        done_subtasks, _ = self._get_execution_path(best_desc_path)
        self.plan = list(set(self.plan) - set(done_subtasks))

    def _backtrack(self) -> None:
        """Returns to the parent of the current node (update plan and cur_node)."""
        this_subtask = self.degra_subtask_dict[self.cur_node["degradation"]]
        self.plan.insert(0, this_subtask)

        parent_img_path = next(
            Path(self.cur_node["img_path"]).parents[3].glob("0-img/*.png")
        )
        self.cur_node = self._img_path_to_node(parent_img_path)
        self.workflow_logger.info(
            f"Back to {self._img_nickname(self.cur_node['img_path'])}.")

    def _img_path_to_node(self, img_path: Path) -> dict:
        subtasks, tools = self._get_execution_path(img_path)
        node = self.work_mem["tree"]
        for subtask, tool in zip(subtasks, tools):
            node = node["children"][subtask]["tools"][tool]
        return node

    def reschedule(self) -> None:
        if not self.plan:
            return
        
        if not self.cur_node["children"]:
            # compromise, pick up the failed plan
            done_subtasks, _ = self._get_execution_path(Path(self.cur_node['img_path']))
            for adjusted_plan in self.work_mem["plan"]["adjusted"]:
                failed = adjusted_plan["failed"]
                failed_done, failed_planned = failed.split(" + ")
                failed_done = literal_eval(failed_done)
                failed_planned = literal_eval(failed_planned)
                if failed_done == done_subtasks:
                    self.plan = failed_planned
                    self.workflow_logger.info(f"Pick up the failed plan {failed_done} + {failed_planned}.")
                    break
            else:
                raise Exception(f"Invalid rescheduling: no failed plan found when processing {self.work_dir}.")

        elif len(self.plan) == len(self.cur_node["children"]) + 1:
            next_agenda = list(self.cur_node["children"])
            next_plan = self.schedule(next_agenda)
            top_subtask = list(set(self.plan)-set(next_agenda))[0]
            self.plan = [top_subtask] + next_plan

        else:
            done_top_subtasks = list(self.cur_node["children"])
            assert len(self.plan) - len(done_top_subtasks) > 1
            if len(done_top_subtasks) == 1:
                failed_tries_str = done_top_subtasks[0]
            else:
                failed_tries_str = 'any of ' + ', '.join(done_top_subtasks)
            reschedule_ps = prompts.reschedule_ps_prompt.format(
                failed_tries=failed_tries_str)
            self.plan = self.schedule(agenda=self.plan, ps=reschedule_ps)

            if self.plan[0] in done_top_subtasks:
                invalid_plan = self.plan.copy()
                for i, subtask in enumerate(self.plan):
                    if subtask not in done_top_subtasks:
                        self.plan[0], self.plan[i] = self.plan[i], self.plan[0]
                        break
                self.workflow_logger.warning(
                    f"Invalid rescheduling: the first subtask of {invalid_plan} "
                    f"in {done_top_subtasks}. Swapping it with {self.plan[0]}.")

        # record update
        done_subtasks, _ = self._get_execution_path(Path(self.cur_node['img_path']))
        assert set(done_subtasks+self.plan) == set(self.work_mem["plan"]["initial"]), \
            (f"Invalid adjusted plan: {done_subtasks} ∪ {self.plan} "
             f"!= {self.work_mem['plan']['initial']}.")
        self.work_mem["plan"]["adjusted"][-1]["new"] = f"{done_subtasks} + {self.plan}"
        self._dump_summary()

        self.workflow_logger.info(f"Adjusted plan: {self.plan}.")

    def _prepare_for_subtask(
        self,
        subtask: Subtask,
    ) -> tuple[Path, Degradation, list[Tool]]:
        self.workflow_logger.info(
            f"Executing {subtask} on "
            f"{self._img_nickname(self.cur_node['img_path'])}..."
        )
        subtask_dir = (
            Path(self.cur_node["img_path"]).parents[1]
            / f"subtask-{subtask}"
        )
        subtask_dir.mkdir()

        degradation = self.subtask_degra_dict[subtask]
        registered_tools = tuple(self.executor.toolbox_router[subtask])
        toolbox, diagnostics = self.tool_selector.select(
            subtask,
            registered_tools,
        )
        ranking_record = {
            "subtask": subtask,
            "image_path": self.cur_node["img_path"],
            "ranked_tools": diagnostics,
            "selected_tools": [tool.tool_name for tool in toolbox],
        }
        self.work_mem["tool_rankings"].append(ranking_record)
        self.workflow_logger.info(
            "Tool ranking: "
            + json.dumps(ranking_record, ensure_ascii=False)
        )
        self._append_episode_event(
            event_type="tool_candidates_ranked",
            state=self._episode_state(),
            action={"subtask": subtask},
            outcome=ranking_record,
        )
        return subtask_dir, degradation, toolbox

    def _record_tool_res(
        self,
        img_path: Path,
        degradation_level: Level,
    ) -> dict:
        tool_name = self._get_name_stem(img_path.parents[1].name)
        subtask = self._get_name_stem(img_path.parents[2].name)
        degradation = self.subtask_degra_dict[subtask]

        self.workflow_logger.info(
            f"Severity of {degradation} of {self._img_nickname(img_path)} "
            f"is {degradation_level}."
        )

        current_children = self.cur_node["children"]
        if subtask not in current_children:
            current_children[subtask] = {"best_tool": None, "tools": {}}
        if tool_name in current_children[subtask]["tools"]:
            raise ValueError(
                f"Duplicate tool result for {subtask}@{tool_name}."
            )

        next_state = self.cur_node.get("state", {}).copy()
        if degradation_level != "none":
            next_state[degradation] = degradation_level
        next_node = {
            "degradation": degradation,
            "severity": degradation_level,
            "img_path": str(img_path),
            "best_descendant": None,
            "state": next_state,
            "children": {},
        }
        current_children[subtask]["tools"][tool_name] = next_node
        return next_node

    def _record_res(self) -> None:
        self.res_path = Path(self.cur_node["img_path"])
        self.workflow_logger.info(
            f"Restoration result: {self._img_nickname(self.res_path)}."
        )
        subtasks, tools = self._get_execution_path(self.res_path)
        self.work_mem["execution_path"]["subtasks"] = subtasks
        self.work_mem["execution_path"]["tools"] = tools
        self._dump_summary()

        copied_result = self.work_dir / "result.png"
        shutil.copy(self.res_path, copied_result)
        self._append_episode_event(
            event_type="run_finished",
            state=self._episode_state(),
            action={
                "subtasks": subtasks,
                "tools": tools,
            },
            transition={
                "result_path": str(copied_result),
                "result_sha256": self._hash_file(copied_result),
            },
            outcome={
                "status": "completed",
                "n_invocations": self.work_mem["n_invocations"],
            },
        )
        print(f"Result saved in {self.res_path}.")

    def _record_run_started(self) -> None:
        self._append_episode_event(
            event_type="run_started",
            state={
                "image_path": str(self.root_input_path),
                "image_sha256": self._hash_file(self.root_input_path),
                "degradations": {},
                "execution_path": {"subtasks": [], "tools": []},
                "remaining_plan": [],
            },
            outcome={"status": "started"},
        )

    def _append_episode_event(
        self,
        *,
        event_type: str,
        state: Optional[dict] = None,
        action: Optional[dict] = None,
        transition: Optional[dict] = None,
        outcome: Optional[dict] = None,
        provenance: Optional[dict] = None,
    ) -> str:
        base_provenance = {
            "episode_schema_version": EPISODE_SCHEMA_VERSION,
            "evaluate_degradation_by": self.evaluate_degradation_by,
            "reflect_by": self.reflect_by,
            "schedule_memory_path": self.schedule_memory_path,
            "tool_profile_path": self.tool_profile_path,
            "random_seed": self.random_seed,
        }
        base_provenance.update(provenance or {})
        return self.episode_store.append_event(
            run_id=self.run_id,
            event_type=event_type,
            state=state,
            action=action,
            transition=transition,
            outcome=outcome,
            provenance=base_provenance,
        )

    def _episode_state(self) -> dict:
        image_path = Path(self.cur_node["img_path"])
        subtasks, tools = self._get_execution_path(image_path)
        return {
            "image_path": str(image_path),
            "image_sha256": self._hash_file(image_path),
            "degradations": self.cur_node.get("state", {}).copy(),
            "execution_path": {
                "subtasks": subtasks,
                "tools": tools,
            },
            "remaining_plan": self.plan.copy(),
        }

    @staticmethod
    def _tool_transition(
        result: ToolRunResult,
        output_path: Path,
    ) -> dict:
        return {
            "duration_seconds": result.duration_seconds,
            "output_path": str(output_path),
            "output_sha256": IRAgent._hash_file(output_path),
            "command": list(result.command) if result.command else None,
        }

    @staticmethod
    def _hash_file(path: Path) -> str:
        digest = hashlib.sha256()
        with Path(path).open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _get_execution_path(self, img_path: Path) -> tuple[list[Subtask], list[ToolName]]:
        """Returns the execution path of the restored image (list of subtask and tools)."""
        exe_path = self._img_tree.get_execution_path(img_path)
        if not exe_path:
            return [], []
        subtasks, tools = zip(*exe_path)
        return list(subtasks), list(tools)

    def _prepare_dir(self, input_path: Path, output_dir: Path) -> None:
        """Sets attributes: `work_dir, img_tree_dir, log_dir, qa_path, workflow_path, summary_path`. Creates necessary directories, which will be like
        ```
        output_dir
        └── {task_id}(work_dir)
            ├── img_tree
            │   └── 0-img
            │       └── input.png
            └── logs
                ├── summary.json
                ├── workflow.log
                ├── llm_qa.md
                └── img_tree.html
        ```
        """

        task_id = f"{input_path.stem}-{strftime('%y%m%d_%H%M%S', localtime())}"
        self.run_id = f"{task_id}-{uuid4().hex[:12]}"
        self.work_dir = output_dir / task_id
        self.work_dir.mkdir(parents=True)

        self.img_tree_dir = self.work_dir / "img_tree"
        self.img_tree_dir.mkdir()

        self.log_dir = self.work_dir / "logs"
        self.log_dir.mkdir()
        self.qa_path = self.log_dir / "llm_qa.md"
        self.workflow_path = self.log_dir / "workflow.log"
        self.work_mem_path = self.log_dir / "summary.json"

        rqd_input_dir = self.img_tree_dir / "0-img"
        rqd_input_dir.mkdir()
        rqd_input_path = rqd_input_dir / "input.png"
        self.root_input_path = rqd_input_path
        shutil.copy(input_path, rqd_input_path)

        self._render_img_tree()

    def _img_nickname(self, img_path: str | Path) -> str:
        """Image name to display in log, showing the execution path."""        
        if isinstance(img_path, str):
            img_path = Path(img_path)
        subtasks, tools = self._get_execution_path(img_path)
        if not subtasks:
            return "input"
        return "-".join([f"{subtask}@{tool}" 
                         for subtask, tool in zip(subtasks, tools)])

    def _get_name_stem(self, name: str) -> str:
        return name[name.find("-") + 1 :]

    @property
    def _img_tree(self) -> ImgTree:
        return ImgTree(self.img_tree_dir, html_dir=self.log_dir)

    def _render_img_tree(self) -> None:
        self._img_tree.to_html()

    def _dump_summary(self) -> None:
        with open(self.work_mem_path, "w") as f:
            json.dump(self.work_mem, f, indent=2)
