from __future__ import annotations

import json
import tempfile
import unittest
import socket
from unittest import mock
import importlib.util
import re
from unittest.mock import patch
from pathlib import Path

from h3serve.config import ServicePaths
from h3serve.models import MODEL_FILES, model_status


class ReleaseLayoutTest(unittest.TestCase):
    def test_busy_port_fails_before_application_or_model_construction(self) -> None:
        from h3serve import app as app_module

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
            occupied.bind(("127.0.0.1", 0))
            occupied.listen(1)
            port = occupied.getsockname()[1]
            args = mock.Mock(
                host="127.0.0.1", port=port, api_key=None,
                release_root=Path(__file__).resolve().parents[1],
                data_dir=Path(__file__).resolve().parents[1] / "data",
                max_queued_jobs=1, engine="reference", lazy_load=False,
                unified_console=True,
                memory_profile="auto",
            )
            with mock.patch.object(app_module, "parse_args", return_value=args), mock.patch.object(
                app_module, "create_app"
            ) as create_app:
                with self.assertRaisesRegex(SystemExit, "already in use"):
                    app_module.main()
                create_app.assert_not_called()

    def test_fixed_engine_start_scripts_are_present(self) -> None:
        root = Path(__file__).resolve().parents[1]
        fidelity = root / "scripts/start-fidelity.sh"
        turbo = root / "scripts/start-turbo.sh"
        unified = root / "scripts/start.sh"
        stop = root / "scripts/stop.sh"
        resolver = root / "scripts/_runtime.sh"
        self.assertTrue(fidelity.is_file())
        self.assertTrue(turbo.is_file())
        self.assertTrue(unified.is_file())
        self.assertTrue(stop.is_file())
        self.assertTrue(resolver.is_file())
        self.assertIn("--engine original", fidelity.read_text(encoding="utf-8"))
        self.assertIn("--engine lora", turbo.read_text(encoding="utf-8"))
        self.assertIn("h3_configure_runtime", resolver.read_text(encoding="utf-8"))
        self.assertIn("--unified-console", unified.read_text(encoding="utf-8"))
        self.assertIn("kill -INT", stop.read_text(encoding="utf-8"))
        self.assertIn("server.py", stop.read_text(encoding="utf-8"))

    def test_control_panel_has_no_engine_switch_input(self) -> None:
        root = Path(__file__).resolve().parents[1]
        html = (root / "static/index.html").read_text(encoding="utf-8")
        app = (root / "static/app.js").read_text(encoding="utf-8")
        self.assertNotIn('name="engine"', html)
        css = (root / "static/infinite-video.css").read_text(encoding="utf-8")
        script = (root / "static/infinite-video.js").read_text(encoding="utf-8")
        self.assertIn('id="infiniteReferences" class="reference-attachments" hidden', html)
        self.assertIn('id="infiniteReferenceFileDrop"', html)
        self.assertIn('id="infiniteReferenceFiles"', html)
        self.assertIn('id="infiniteReferenceImages" name="reference_images"', html)
        self.assertIn('id="infiniteReferenceAudios" name="reference_audios"', html)
        self.assertIn("function distributeInfiniteReferenceFiles(", script)
        self.assertIn("const maximum = Math.max(1, 15 - overlap);", script)
        self.assertNotIn("project?.window_count ? Math.max(1, 15 - overlap) : 15", script)
        self.assertIn('新增时长与参考上一段合计最多 15 秒。', html)
        self.assertIn('.infinite-shell [hidden]{display:none!important}', css)
        self.assertIn("engine !== 'reference'", script)
        self.assertIn("engine === 'reference'", script)
        self.assertIn("project.can_retry_tail", script)
        self.assertIn("失败尾段已移除，正在重新提交", script)
        self.assertIn('id="engineBanner"', html)
        self.assertIn('id="engineLobby"', html)
        self.assertIn('<strong>X-MinimaxH3</strong>', html)
        self.assertIn('id="chooseWorkspace"', html)
        self.assertIn('id="workspaceDialog"', html)
        self.assertIn('id="exitEngine"', html)
        self.assertIn('id="createPage" hidden', html)
        self.assertIn('class="workspace-tabs" aria-label="工作区" hidden', html)
        self.assertIn('id="tasksPage"', html)
        self.assertIn('data-page="infinite"', html)
        self.assertIn('id="infinitePage"', html)
        self.assertIn('id="infiniteWindowForm"', html)
        project_form = html[
            html.index('<form id="infiniteProjectForm"'):
            html.index('</form>', html.index('<form id="infiniteProjectForm"'))
        ]
        self.assertEqual(
            re.findall(r'name="([^"]+)"', project_form),
            [
                "workflow_version", "creation_mode", "title", "preview_resolution",
                "final_resolution", "aspect_ratio", "model_variant",
                "sampling_steps", "final_sampling_steps",
            ],
        )
        self.assertNotIn('name="acceleration"', project_form)
        self.assertNotIn('name="second_pass_acceleration"', project_form)
        self.assertIn('id="infiniteWindowAccelerationValue"', html)
        self.assertIn('data-project-mode="online"', project_form)
        self.assertIn('data-project-mode="json"', project_form)
        self.assertIn('id="infiniteProjectJsonInput"', project_form)
        self.assertNotIn('>记忆与窗口</span>', project_form)
        self.assertNotIn('name="overview"', project_form)
        self.assertNotIn('name="visual_memory_capacity"', project_form)
        self.assertIn('id="infiniteMemoryEnabled"', html)
        self.assertIn('name="overlap_seconds" type="range" min="0"', html)
        self.assertIn("q('#infiniteMemoryEnabled')", script)
        self.assertIn('class="infinite-window-progress"', script)
        self.assertIn('--window-progress:${percent}%', script)
        self.assertIn('.infinite-window-progress{', css)
        self.assertIn('id="infiniteProjectStepTrack"', project_form)
        self.assertIn('id="infiniteProjectFirstPassSteps"', project_form)
        self.assertNotIn('name="preview_branch_steps"', project_form)
        self.assertNotIn('id="infiniteProjectPreviewEnabled"', project_form)
        self.assertIn('id="infiniteProjectLoraEnabled" type="checkbox" checked', project_form)
        self.assertIn('name="model_variant" class="visually-hidden-control"', project_form)
        self.assertIn('class="field infinite-aspect-field"', project_form)
        self.assertIn('class="infinite-inference-grid"', project_form)
        self.assertNotIn('infinite-preview-branch-card', project_form)
        self.assertIn('.infinite-geometry-grid{grid-template-columns:minmax(420px,1fr) 92px', css)
        self.assertIn("event.target.checked ? 'lora' : 'base'", script)
        self.assertIn('data-infinite-final-resolution="1080p"', project_form)
        self.assertIn('Math.abs(nearest - value) <= 10', script)
        self.assertIn('>画面与输出</span>', project_form)
        self.assertIn('>推理参数</span>', project_form)
        self.assertIn('id="deleteInfiniteProject"', html)
        self.assertNotIn('id="discardInfiniteTail"', html)
        self.assertNotIn('id="infiniteSecondSamplingForm"', html)
        self.assertIn('id="infiniteWindowList"', html)
        self.assertIn('id="addInfiniteWindowPrompt"', html)
        self.assertIn('id="collapseInfiniteWindowEditor"', html)
        self.assertIn('class="panel infinite-card" hidden', html)
        self.assertNotIn('<input name="overview" type="hidden"', html)
        self.assertNotIn("form.elements.overview", script)
        self.assertIn('draftStorageKey', script)
        self.assertIn("promptDrafts.push(inheritedWindowValues())", script)
        self.assertIn('promptDrafts.splice(index, 1)', script)
        self.assertIn('id="infiniteJsonInput"', html)
        self.assertIn('infinite-project-template.json', html)
        self.assertIn('id="infiniteFinalSamplingForm"', html)
        self.assertIn('/batch', script)
        self.assertIn('/final-sampling', script)
        self.assertIn('/static/infinite-video.js', html)
        self.assertIn('20260911-ref-files-duration-budget-v1', html)
        self.assertIn('/static/infinite-video.css?v=20260911-ref-files-duration-budget-v1', html)
        self.assertIn("window.H3InfiniteStudio?.activate(currentEngine, options)", app)
        self.assertIn('id="conversationFeed"', html)
        self.assertIn('class="conversation-composer panel"', html)
        self.assertIn('class="submit-button composer-submit"', html)
        self.assertIn('aria-label="发送生成任务" disabled', html)
        self.assertEqual(html.count('title="发送生成任务"'), 1)
        self.assertLess(html.index('id="formMessage"'), html.index('id="storyboardTitle"'))
        self.assertIn('class="conversation-history panel"', html)
        self.assertNotIn('id="geometryText"', html)
        self.assertLess(html.index('id="exitEngine"'), html.index('<main'))
        self.assertNotIn('id="cpuUsage"', html)
        self.assertIn('id="hostMemoryUsage"', html)
        self.assertIn('>主机物理内存（驻留 / 总量）</span>', html)
        self.assertIn('>H3视频服务（驻留总量）</span>', html)
        self.assertIn('分配给H3服务的内存硬上限', app)
        self.assertIn('不会预占', app)
        self.assertIn('id="vramUsage"', html)
        self.assertIn('data.service_memory', app)
        self.assertIn('data.memory', app)
        self.assertIn('hostMemory.occupied_gib', app)
        self.assertIn('serviceMemory.resident_gib', app)
        self.assertIn('serviceMemory.resident_process_count', app)
        self.assertIn('含模型映射', app)
        self.assertNotIn('内存额度计费', app)
        self.assertIn('id="inferenceDrawer"', html)
        self.assertIn('id="videoSettingsDrawer"', html)
        self.assertIn('<details class="config-box" id="videoSettingsDrawer">', html)
        self.assertIn('<details class="config-box" id="inferenceDrawer">', html)
        self.assertIn('<summary class="config-box-head">', html)
        self.assertNotIn('name="size_mode"', html)
        self.assertNotIn('id="customSizeFields"', html)
        self.assertIn('id="creationResolutionSlider" class="total-step-slider" type="range" min="360" max="1440" step="1"', html)
        self.assertIn('id="freeformDurationValue">5.0 秒</output></span><input name="duration_seconds" type="range" min="1" max="15" step="0.5"', html)
        self.assertIn('单次生成任务最多15秒；更长内容请使用长视频创作。', html)
        self.assertIn("function currentMaxDuration()", app)
        self.assertIn("return 15;", app)
        self.assertIn('id="infiniteProjectResolutionTrack"', html)
        self.assertIn('id="infiniteProjectResolutionSlider" class="first-step-slider" type="range" min="360" max="1440" step="1"', html)
        self.assertIn('id="infiniteProjectFinalResolutionSlider" class="total-step-slider" type="range" min="360" max="1440" step="1"', html)
        self.assertNotIn("slider.disabled = hasKeyframes", app)
        self.assertNotIn("使用首帧或尾帧时暂不支持渐进分辨率", app)
        self.assertIn('name="duration_seconds" type="range" min="1" max="15" step="0.25"', html)
        self.assertIn('name="workflow_version" type="hidden" value="3"', html)
        self.assertIn('与单视频创作使用同一条双节点采样轨道', html)
        self.assertIn("Math.abs(nearest - value) <= 10", app)
        self.assertIn("field.value = `${value}p`", app)
        for resolution in ("360p", "480p", "540p", "720p", "900p", "1080p"):
            self.assertIn(f'data-resolution="{resolution}"', html)
        self.assertIn('data-resolution="1440p">1440P</button>', html)
        self.assertIn('id="queuedJobs"', html)
        self.assertIn('id="selectAllHistory"', html)
        self.assertIn('id="clearHistorySelection"', html)
        self.assertIn('id="deleteSelectedHistory"', html)
        self.assertIn('id="historySelectionCount"', html)
        self.assertIn('data-history-select', app)
        self.assertIn("/api/v1/jobs/records", app)
        self.assertNotIn('id="shotList"', html)
        self.assertNotIn('data-prompt-mode=', html)
        self.assertNotIn('id="structuredPromptEditor"', html)
        self.assertIn('id="freeformPromptEditor"', html)
        self.assertIn('id="freeformPrompt" name="prompt"', html)
        self.assertNotIn('id="compiledPrompt"', html)
        self.assertIn('class="freeform-writing-guide"', html)
        self.assertIn('<b>撰写建议</b>', html)
        self.assertIn('<code>overall_soundscape:</code>', html)
        self.assertIn('<code>non_diegetic_music:</code>', html)
        self.assertIn('id="freeformDurationField"', html)
        self.assertNotIn('id="freeformDurationField" hidden', html)
        self.assertNotIn('id="referenceStyleOpening"', html)
        self.assertNotIn('整体画面与连续性', html)
        self.assertNotIn('id="referenceDefinitions"', html)
        self.assertNotIn('id="referenceRetention"', html)
        self.assertNotIn('id="referenceSummary"', html)
        self.assertNotIn('id="overallSoundscape"', html)
        self.assertNotIn('id="bgmStyle"', html)
        self.assertNotIn('id="enhanceReferences"', html)
        self.assertNotIn('id="enhanceVisuals"', html)
        self.assertNotIn('id="enhanceSound"', html)
        self.assertIn('id="referenceFileDrop"', html)
        self.assertIn('id="referenceFiles"', html)
        self.assertIn('id="globalReferenceImageResolution"', html)
        self.assertIn('id="globalReferenceVideoResolution"', html)
        self.assertIn('始终按原宽高比等比例缩小', html)
        self.assertIn('原画幅尺寸、完整构图和视频时长不变', html)
        self.assertNotIn('id="mimoApiKeyInput"', html)
        self.assertNotIn('id="generationLimitEditor"', html)
        self.assertNotIn('id="generationLimitStatus"', html)
        self.assertNotIn('serverGenerationLimitSettings', app)
        self.assertIn("[['w4a8', 'W4A8 轻量权重'], ['int8', 'INT8 高质量权重']]", app)
        self.assertIn('分配给H3服务的内存硬上限', app)
        self.assertIn('系统保留6GiB', app)
        self.assertIn('data-memory-weight', app)
        self.assertNotIn('＞64GB 高速模式', app)
        self.assertNotIn('≤64GB 兼容模式', app)
        self.assertNotIn("function setPromptEditorMode(mode)", app)
        self.assertNotIn("function syncStructuredEditorState()", app)
        self.assertIn("if (document.hidden || uiPollPromise)", app)
        self.assertIn("服务响应超时；请检查8090端口转发", app)
        self.assertNotIn("reconcileEngineState().catch(() => {}); }, 1500", app)
        self.assertNotIn("promptEditorMode", app)
        self.assertNotIn("compileStoryboard", app)
        self.assertIn("if (!$('#freeformPrompt').value.trim())", app)
        self.assertNotIn("/studio/prompt-enhancements", app)
        self.assertNotIn("/api/v1/prompt-enhancements", app)
        self.assertNotIn("referenceMediaPayload", app)
        self.assertNotIn("currentReferenceProtocol", app)
        self.assertIn('font-size:10px', (root / "static" / "storyboard.css").read_text(encoding="utf-8"))
        self.assertNotIn('id="upscaleEnabled"', html)
        self.assertNotIn('name="upscale_resolution"', html)
        self.assertIn('name="sampling_steps" class="total-step-slider" type="range" min="1" max="30"', html)
        self.assertIn('id="firstPassSteps" class="first-step-slider"', html)
        self.assertIn('id="generationStepTrack"', html)
        self.assertIn('id="firstPassResolutionSlider"', html)
        self.assertIn('id="previewEnabled"', html)
        self.assertIn('id="singleSecondSamplingWindowEnabled"', html)
        self.assertIn('id="globalSecondSamplingWindowSeconds" type="range" min="3" max="15"', html)
        self.assertIn('id="singleSecondSamplingSigmaScale" type="range" min="0.25" max="1" step="0.05" value="1"', html)
        self.assertIn("form.set('selflift_temporal_window_enabled', windowed ? 'true' : 'false')", app)
        self.assertIn("form.set('selflift_temporal_window_enabled', 'false')", app)
        self.assertIn('id="loraAccelerationEnabled" type="checkbox"', html)
        self.assertIn('name="model_variant" class="visually-hidden-control"', html)
        self.assertNotIn('<span>推理路线</span><select name="model_variant">', html)
        self.assertNotIn('id="previewBranchSteps"', html)
        self.assertIn('id="globalPreviewSteps" type="range" min="1" max="4" step="1" value="2"', html)
        self.assertNotIn('class="progressive-settings-grid preview-only-settings-grid"', html)
        self.assertIn('Preview Step', html)
        self.assertIn('name="acceleration" type="range" min="0" max="100"', html)
        self.assertIn('name="second_pass_acceleration" type="range" min="0" max="100"', html)
        self.assertIn('一采加速力度', html)
        self.assertIn('二采加速力度', html)
        self.assertNotIn('name="memory_mode"', html)
        self.assertNotIn('显存执行后端', html)
        self.assertNotIn('id="checkpointEnabled"', html)
        self.assertNotIn('name="checkpoint_step" type="range"', html)
        self.assertIn('中途预览', html)
        self.assertIn('第一次采样分辨率', html)
        self.assertNotIn('class="upscale-setting"', html)
        self.assertIn('name="execution_mode" type="hidden" value="complete"', html)
        self.assertIn('id="engineLoadProgress"', html)
        self.assertIn('id="engineLoadBar"', html)
        self.assertIn('renderEngineLoadProgress', app)
        self.assertNotIn('name="checkpoint_preview_steps" type="range"', html)
        self.assertNotIn('id="globalCheckpointPreviewSteps"', html)
        self.assertNotIn('id="globalCheckpointPreviewResolution"', html)
        self.assertIn('class="service-settings-drawer checkpoint-preview-settings"', html)
        self.assertIn('id="globalPreviewSteps" type="range" min="1" max="4"', html)
        self.assertIn("serverPreviewSettings", app)
        self.assertIn("configureServerPreview", app)
        self.assertIn("form.set('preview_mode', 'pause')", app)
        self.assertIn("form.set('preview_step_index', String(firstSteps - 1))", app)
        self.assertIn("form.set('second_pass_acceleration', String(secondPassAcceleration))", app)
        self.assertIn("form.set('acceleration_transition_step', String(firstSteps))", app)
        self.assertIn("form.delete('preview_branch_steps')", app)
        self.assertIn("Number(globalPreviewPolicy.steps)", app)
        self.assertNotIn('data-second-sampling=', app)
        self.assertNotIn("form.set('preview_branch_steps', String(Math.max(1, totalSteps - firstSteps)))", app)
        self.assertIn("form.set('checkpoint_preview_resolution', 'source')", app)
        self.assertIn("20260831-auto-resource-budget-v1", html)
        self.assertNotIn('id="secondSamplingDialog"', html)
        self.assertNotIn('id="secondSamplingForm"', html)
        self.assertNotIn('高清二次采样', html)
        self.assertIn('id="videoRepairDialog"', html)
        self.assertIn('id="videoRepairForm"', html)
        self.assertNotIn('id="videoRepairMode"', html)
        self.assertNotIn('id="videoRepairGrid"', html)
        self.assertNotIn('id="videoRepairCanvas"', html)
        self.assertNotIn('id="videoRepairMaxFaces"', html)
        self.assertNotIn('id="videoRepairMagnification"', html)
        self.assertNotIn('id="videoRepairSteps"', html)
        self.assertIn('id="videoRepairAcceleration" type="range" min="0" max="100" step="1" value="50"', html)
        self.assertIn('id="globalFaceRepairCanvas" type="range" min="192" max="1088" step="32" value="768"', html)
        self.assertIn('id="globalFaceRepairCapacity"', html)
        self.assertIn("serverFaceRepairSettings", app)
        self.assertIn("configureServerFaceRepair", app)
        self.assertIn("canvas_size:Number(canvasSize)", app)
        self.assertIn("acceleration:Number($('#videoRepairAcceleration').value)", app)
        self.assertIn('class="compact-video-repair-dialog"', html)
        self.assertIn('data-video-repair=', app)
        self.assertIn('/video-repair', app)
        self.assertIn('id="globalSecondSamplingWindowEnabled"', html)
        self.assertIn('id="globalSecondSamplingWindowSeconds"', html)
        self.assertIn('id="globalSecondSamplingWindowControls"', html)
        self.assertIn('class="service-settings-drawer second-sampling-window-settings"', html)
        self.assertEqual(html.count('class="service-settings-drawer '), 5)
        self.assertNotIn('<span>主机内存预算</span>', html)
        self.assertIn("serverSecondSamplingWindowSettings", app)
        self.assertIn("configureServerSecondSamplingWindow", app)
        self.assertIn('name="sigma_scale" type="range" min="0.25" max="1"', html)
        self.assertIn('id="clearLatentCache"', html)
        self.assertNotIn('name="width" type="range"', html)
        self.assertNotIn('name="height" type="range"', html)
        self.assertIn("function syncDurationControl()", app)
        self.assertNotIn('id="qualitySlider"', html)
        self.assertNotIn('id="advancedDrawer"', html)
        self.assertNotIn('data-upscale-resolution="720p"', html)
        self.assertNotIn('data-upscale-resolution="2k"', html)
        self.assertIn('data-resolution="1080p">1080P</button>', html)
        self.assertIn('data-resolution="720p">720P</button>', html)
        self.assertNotIn('id="secondSamplingResolution"', html)
        self.assertNotIn('data-second-resolution=', html)
        self.assertNotIn('>2K（实验）</option>', html)
        self.assertNotIn('id="upscaleEnabled" name="upscale_enabled" type="checkbox" checked', html)
        self.assertLess(html.index('id="firstDrop"'), html.index('id="freeformPrompt"'))

    def test_open_settings_drawer_uses_full_row(self) -> None:
        root = Path(__file__).resolve().parents[1]
        controls_css = (root / "static" / "composer-controls.css").read_text()
        self.assertIn(".config-drawer[open]{grid-column:1/-1}", controls_css)

    def test_dual_slider_handles_stay_separated_when_values_match(self) -> None:
        root = Path(__file__).resolve().parents[1]
        controls_css = (root / "static" / "composer-controls.css").read_text()
        self.assertIn(
            ".dual-step-range .first-step-slider{z-index:3;transform:translateY(-10px)}",
            controls_css,
        )
        self.assertIn(
            ".dual-step-range .total-step-slider{z-index:2;transform:translateY(10px)}",
            controls_css,
        )

    def test_default_linux_service_launch_refreshes_runtime_mirror(self) -> None:
        root = Path(__file__).resolve().parents[1]
        launcher = (root / "scripts/linux_runtime_exec.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('sync_mode="${H3_LINUX_SYNC:-auto}"', launcher)
        self.assertIn('"${default_service_launch}" == "1"', launcher)

    def test_freeform_prompt_submission_keeps_reference_tools(self) -> None:
        root = Path(__file__).resolve().parents[1]
        script = (root / "static/app.js").read_text(encoding="utf-8")
        for required in (
            "reference_media",
            "function renderReferencePreviews(",
            "function renderConversation()",
            "function bindFileDrop(",
            "function removeReferenceFile(",
            "function insertReferenceToken(",
            "function distributeReferenceFiles(",
            "function updateReferenceMentionMenu(",
            "function refreshResources()",
            "form = new FormData(event.target)",
        ):
            self.assertIn(required, script)
        self.assertIn(
            "setInterval(() => { if (!document.hidden) refreshResources(); }, 1000);",
            script,
        )

    def test_default_paths_are_inside_release_root(self) -> None:
        root = Path(__file__).resolve().parents[1]
        paths = ServicePaths.defaults(root)
        self.assertEqual(paths.release_root, root)
        self.assertEqual(paths.model_dir, root / "models")
        self.assertEqual(paths.output_dir, root / "output")
        self.assertEqual(paths.minimax_source_dir, root / "runtime/vendor/MiniMax-H3")
        self.assertEqual(paths.lightx_source_dir, root / "runtime/vendor/LightX2V")
        self.assertEqual(paths.flashvsr_source_dir, root / "third_party/flashvsr")
        self.assertEqual(
            paths.flashvsr_model_dir, root / "models/upscalers/flashvsr-v1.1"
        )
        self.assertEqual(
            paths.flashvsr_python_executable,
            (root / "runtime/flashvsr-venv/bin/python").absolute(),
        )
        self.assertEqual(
            paths.turbo_curve_path,
            root / "backends/turbo/custom_node/h3_silu_temb_grid.safetensors",
        )

    def test_original_engine_does_not_require_lora_weight(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            for role, (folder, filename) in MODEL_FILES.items():
                if role == "lora":
                    continue
                path = root / folder / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            status = model_status(root)
            self.assertTrue(status["engines"]["original"]["ready"])
            self.assertFalse(status["engines"]["lora"]["ready"])
            self.assertFalse(status["engines"]["reference_lora"]["ready"])

    def test_virtualenv_python_symlink_is_not_resolved_away(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            base = root / "base-python"
            base.touch()
            venv_python = root / "venv-python"
            venv_python.symlink_to(base)
            with patch.dict("os.environ", {"H3_SERVE_PYTHON": str(venv_python)}):
                paths = ServicePaths.defaults(root)
            self.assertEqual(paths.python_executable, venv_python.absolute())

    def test_model_downloader_prepares_optional_64gb_qwen_cache(self) -> None:
        root = Path(__file__).resolve().parents[1]
        script = (root / "scripts/download_models.py").read_text(encoding="utf-8")
        self.assertIn("--skip-local-qwen-cache", script)
        self.assertIn("materialize_local_checkpoint", script)

    def test_model_downloader_maps_root_w4_checkpoint_into_dit_folder(self) -> None:
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location(
            "h3_download_models_test",
            root / "scripts/download_models.py",
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model_root = root / "models"
        target = (
            model_root
            / "diffusion_models/minimax_h3_fl2va_pruned_w4a8_mixed.safetensors"
        )
        local_dir = module._download_local_dir(
            model_root,
            target,
            {
                "filename": "minimax_h3_fl2va_pruned_w4a8_mixed.safetensors",
                "install_path": (
                    "diffusion_models/"
                    "minimax_h3_fl2va_pruned_w4a8_mixed.safetensors"
                ),
            },
        )
        self.assertEqual(local_dir, model_root / "diffusion_models")

    def test_preflight_checks_real_host_memory_capacity(self) -> None:
        root = Path(__file__).resolve().parents[1]
        script = (root / "scripts/preflight.py").read_text(encoding="utf-8")
        self.assertIn("host_memory_supported", script)
        self.assertIn("resolve_host_memory_profile", script)
        self.assertTrue((root / "docs/DEPLOY_64GB_WSL.md").is_file())

    def test_comfyui_api_connector_is_part_of_the_release(self) -> None:
        root = Path(__file__).resolve().parents[1]
        connector = root / "integrations/comfyui"
        self.assertTrue((connector / "install_local.py").is_file())
        self.assertTrue((connector / "h3serve_connector/nodes.py").is_file())
        build_script = (root / "scripts/build_release.sh").read_text(encoding="utf-8")
        self.assertIn("integrations", build_script)

    def test_flashvsr_alignment_box_preserves_the_full_composition(self) -> None:
        root = Path(__file__).resolve().parents[1]
        worker = root / "scripts/flashvsr_worker.py"
        spec = importlib.util.spec_from_file_location("flashvsr_worker_contract", worker)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # A 640x352 H3 frame delivered at short-edge 720 becomes 1310x720.
        # The 128-aligned model canvas is 1408x768, but its explicit content
        # box must preserve the full composition instead of centre-cropping it.
        geometry = module.fit_geometry(640, 352, 1408, 768)
        self.assertEqual(geometry, (1396, 768, 6, 0, 1402, 768))

    def test_temporal_second_sampling_release_is_reproducible(self) -> None:
        root = Path(__file__).resolve().parents[1]
        self.assertTrue(
            (root / "third_party/flashvsr/diffsynth/__init__.py").is_file()
        )
        self.assertTrue((root / "third_party/flashvsr/LICENSE").is_file())
        self.assertTrue(
            (
                root
                / "wheels/block_sparse_attn-0.0.2-cp311-cp311-linux_x86_64.whl"
            ).is_file()
        )
        install = (root / "scripts/install.sh").read_text(encoding="utf-8")
        self.assertIn("flashvsr-venv", install)
        self.assertIn("torch==2.6.0+cu124", install)
        manifest = json.loads(
            (root / "models/manifest.json").read_text(encoding="utf-8")
        )
        temporal_roles = {
            item["role"]
            for item in manifest["artifacts"]
            if item.get("profile") == "temporal"
        }
        self.assertEqual(
            temporal_roles,
            {
                "temporal_second_sampling_dit",
                "temporal_second_sampling_lq_projection",
                "temporal_second_sampling_decoder",
                "temporal_second_sampling_prompt",
            },
        )

    def test_public_surfaces_do_not_contain_research_release_names(self) -> None:
        root = Path(__file__).resolve().parents[1]
        files = [
            root / "README.md",
            root / "static/index.html",
            root / "static/app.js",
            root / "h3serve/backend.py",
        ]
        forbidden = (r"\bV[5-8](?:[.\-])", r"Fast\s+Max", r"final[_ ]audit", r"hot\d")
        for path in files:
            text = path.read_text(encoding="utf-8")
            for pattern in forbidden:
                self.assertIsNone(
                    re.search(pattern, text, re.IGNORECASE),
                    f"research name pattern {pattern} leaked through {path.name}",
                )


if __name__ == "__main__":
    unittest.main()
