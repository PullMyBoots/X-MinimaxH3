from __future__ import annotations

import json
import tempfile
import threading
import unittest
from unittest.mock import patch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from h3serve_connector.client import H3ServeClient
from h3serve_connector.nodes import (
    H3ServeAdvancedGenerate,
    H3ServeConnection,
    H3ServeConnectionEnglish,
    H3ServeFL2VAPresetGenerate,
    H3ServeFL2VAPresetGenerateEnglish,
    H3ServeRef2VAPresetGenerate,
    H3ServeRef2VAPresetGenerateEnglish,
    H3ServeFL2VACheckpointSubmit,
    H3ServeRef2VACheckpointSubmit,
    H3ServeCheckpointPreview,
    H3ServeCheckpointResume,
    H3ServeSecondSampling,
    H3ServeSecondSamplingEnglish,
    QUALITY,
    _acceleration_value,
    _add_reference_resolution_fields,
    _max_native_duration,
    _preset_geometry,
    _require_ready_engine,
    _source_job_id_from_video,
)


class NativeDurationBudgetTest(unittest.TestCase):
    def test_invalid_legacy_acceleration_falls_back_to_zero(self):
        self.assertEqual(_acceleration_value(float("nan")), 0.0)
        self.assertEqual(_acceleration_value("原始权重"), 0.0)
        self.assertEqual(_acceleration_value(50), 50.0)

    def test_preset_aspect_ratios_share_the_pixel_frame_budget(self):
        self.assertEqual(_preset_geometry("1080p", "16:9"), (1920, 1088))
        self.assertEqual(_max_native_duration(1920, 1088), 8)
        self.assertEqual(_max_native_duration(1440, 1088), 243 / 24)
        self.assertEqual(_max_native_duration(1088, 1088), 328 / 24)

    def test_generation_nodes_expose_transparent_long_horizon_but_checkpoint_stays_native(self):
        preset = H3ServeFL2VAPresetGenerate.INPUT_TYPES()["required"]
        advanced = H3ServeAdvancedGenerate.INPUT_TYPES()["required"]
        english = H3ServeFL2VAPresetGenerateEnglish.INPUT_TYPES()["required"]
        checkpoint = H3ServeFL2VACheckpointSubmit.INPUT_TYPES()["required"]
        self.assertEqual(preset["duration_seconds"][1]["max"], 60.0)
        self.assertEqual(advanced["duration_seconds"][1]["max"], 60.0)
        self.assertEqual(english["duration_seconds"][1]["max"], 60.0)
        self.assertEqual(checkpoint["duration_seconds"][1]["max"], 15.0)


class Handler(BaseHTTPRequestHandler):
    requests = []
    engine = "original"

    def log_message(self, *_args):
        pass

    def _send_json(self, document, status=200):
        content = json.dumps(document).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        if self.path == "/api/v1/options":
            self._send_json({
                "current_engine": self.__class__.engine,
                "engine_control": {"switching": False},
            })
        elif self.path == "/readyz":
            status = "ready" if self.__class__.engine else "idle"
            self._send_json({"status": status}, 200 if self.__class__.engine else 503)
        elif self.path == "/api/v1/jobs/job-1":
            self._send_json({"id": "job-1", "status": "succeeded", "progress": {"percent": 100}})
        elif self.path == "/api/v1/jobs/job-1/video":
            content = b"fake-mp4"
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == "/api/v1/jobs/job-1/preview":
            content = b"fake-preview-mp4"
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        else:
            self._send_json({"status": "ok"})

    def do_POST(self):
        content = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        self.__class__.requests.append((self.headers.get("Content-Type"), content))
        self._send_json({"id": "job-1", "status": "queued", "progress": {"percent": 0}}, 202)


class ClientTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.client = H3ServeClient(f"http://127.0.0.1:{cls.server.server_port}")

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_json_submit_wait_and_download(self):
        job = self.client.submit({"mode": "preset", "prompt": "test"}, {})
        self.assertEqual(job["id"], "job-1")
        completed = self.client.wait(
            "job-1", poll_seconds=0.01, progress=lambda _job: None,
            cancel_check=lambda: None,
        )
        self.assertEqual(completed["status"], "succeeded")
        with tempfile.TemporaryDirectory() as directory:
            output = self.client.download("job-1", Path(directory) / "result.mp4")
            self.assertEqual(output.read_bytes(), b"fake-mp4")

    def test_generation_and_second_sampling_match_the_console_surface(self):
        preset = H3ServeFL2VAPresetGenerate.INPUT_TYPES()["required"]
        self.assertIn("1080p", preset["resolution"][0])
        self.assertNotIn("upscale", preset)
        self.assertEqual(preset["sampling_steps"][1]["max"], 30)
        advanced = H3ServeAdvancedGenerate.INPUT_TYPES()["required"]
        self.assertEqual(advanced["width"][1]["max"], 1920)
        self.assertEqual(advanced["height"][1]["max"], 1920)
        second = H3ServeSecondSampling.INPUT_TYPES()["required"]
        self.assertEqual(second["目标分辨率"][0], ["720P", "1080P", "1440P"])
        self.assertEqual(second["二次采样步数"][1]["min"], 1)
        self.assertEqual(second["二次采样步数"][1]["max"], 8)
        self.assertEqual(second["加速力度"][1]["default"], 75.0)

    def test_reference_resolution_controls_map_to_shared_api_fields(self):
        fields = {}
        _add_reference_resolution_fields(fields, {
            "参考图片分辨率": "保持原样",
            "参考视频分辨率": "480P",
        })
        self.assertEqual(fields, {
            "reference_image_resolution": "original",
            "reference_video_resolution": "480p",
        })

        inherited = {}
        _add_reference_resolution_fields(inherited, {
            "参考图片分辨率": "使用服务端设置",
            "参考视频分辨率": "使用服务端设置",
        })
        self.assertEqual(inherited, {})

    def test_multipart_upload(self):
        self.client.submit(
            {"mode": "preset", "prompt": "test"},
            {"first_frame": ("first.png", b"png-data")},
        )
        content_type, body = Handler.requests[-1]
        self.assertIn("multipart/form-data", content_type)
        self.assertIn(b'name="first_frame"', body)
        self.assertIn(b"png-data", body)

    def test_engine_readiness_gate(self):
        engine, _ = _require_ready_engine(self.client)
        self.assertEqual(engine, "original")

        Handler.engine = None
        try:
            with self.assertRaisesRegex(RuntimeError, "尚未选择引擎"):
                _require_ready_engine(self.client)
        finally:
            Handler.engine = "original"

    def test_second_sampling_video_wire_recovers_the_source_job(self):
        class Video:
            def get_stream_source(self):
                return "/tmp/output/h3_serve/source-job-123.mp4"

        self.assertEqual(_source_job_id_from_video(Video()), "source-job-123")

        class ForeignVideo:
            def get_stream_source(self):
                return "/tmp/output/foreign.mp4"

        with self.assertRaisesRegex(RuntimeError, "H3 Serve生成节点"):
            _source_job_id_from_video(ForeignVideo())

    def test_second_sampling_node_projects_console_controls_to_the_api(self):
        source_video = object()
        connection = {"server_url": "http://127.0.0.1:8090"}
        with patch(
            "h3serve_connector.nodes._run_second_sampling",
            return_value=("second-video",),
        ) as run:
            result = H3ServeSecondSampling().sample(
                connection, source_video, "1080P", 4,
                "增强 · Denoise 0.25", 75.0,
            )
        self.assertEqual(result, ("second-video",))
        run.assert_called_once_with(connection, source_video, {
            "resolution": "1080p",
            "steps": 4,
            "strength": "enhance",
            "acceleration": 75.0,
        })

    def test_english_second_sampling_projects_the_same_api_contract(self):
        source_video = object()
        connection = {"server_url": "http://127.0.0.1:8090"}
        with patch(
            "h3serve_connector.nodes._run_second_sampling",
            return_value=("second-video",),
        ) as run:
            result = H3ServeSecondSamplingEnglish().sample_english(
                connection, source_video, "1080p", 4,
                "Enhance · Denoise 0.25", 75.0,
            )
        self.assertEqual(result, ("second-video",))
        run.assert_called_once_with(connection, source_video, {
            "resolution": "1080p",
            "steps": 4,
            "strength": "enhance",
            "acceleration": 75.0,
        })

    def test_conditioning_workflows_expose_disjoint_inputs(self):
        fl2va = H3ServeFL2VAPresetGenerate.INPUT_TYPES()["optional"]
        ref2va = H3ServeRef2VAPresetGenerate.INPUT_TYPES()["optional"]
        self.assertEqual(set(fl2va), {"first_frame", "last_frame"})
        self.assertEqual(
            set(ref2va),
            {*(f"Picture {index}" for index in range(1, 10)),
             *(f"Video {index}" for index in range(1, 4)),
             *(f"Audio {index}" for index in range(1, 4))},
        )
        self.assertNotIn("参考素材", ref2va)

    def test_simple_generation_exposes_steps_and_acceleration_directly(self):
        schema = H3ServeRef2VAPresetGenerate.INPUT_TYPES()["required"]
        self.assertNotIn("quality", schema)
        self.assertEqual(schema["sampling_steps"][1]["default"], 8)
        self.assertEqual(schema["acceleration"][1]["default"], 0.0)
        self.assertNotIn("提示词增强", schema)
        self.assertNotIn("background_music", schema)
        self.assertNotIn("参考图片分辨率", schema)
        self.assertNotIn("参考视频分辨率", schema)
        fl2va = H3ServeFL2VAPresetGenerate.INPUT_TYPES()["required"]
        self.assertNotIn("background_music", fl2va)
        self.assertNotIn("提示词增强", fl2va)
        self.assertNotIn("参考图片分辨率", fl2va)
        self.assertNotIn("参考视频分辨率", fl2va)
        self.assertTrue(fl2va["prompt"][1]["forceInput"])
        self.assertEqual(fl2va["preview_mode"][0], ["关闭", "开启"])
        self.assertEqual(fl2va["预览位置"][1]["default"], 6)
        self.assertEqual(fl2va["预览位置"][1]["min"], 1)
        self.assertEqual(fl2va["预览位置"][1]["max"], 19)
        self.assertNotIn("预览分辨率", fl2va)
        self.assertNotIn("LoRA预览步数", fl2va)
        self.assertNotIn("upscale", fl2va)
        self.assertEqual(fl2va["断点任务ID"][1]["default"], "")
        self.assertEqual(fl2va["断点动作"][1]["default"], "新建任务")
        for name in ("prompt", "preview_mode", "预览位置"):
            self.assertEqual(schema[name], fl2va[name])

    def test_english_creator_surface_is_fully_english_and_equivalent(self):
        fl2va = H3ServeFL2VAPresetGenerateEnglish.INPUT_TYPES()
        ref2va = H3ServeRef2VAPresetGenerateEnglish.INPUT_TYPES()
        required = fl2va["required"]
        self.assertEqual(required["model_variant"][0], ["Base weights", "LoRA Turbo"])
        self.assertEqual(required["preview_mode"][0], ["Off", "On"])
        self.assertEqual(required["checkpoint_action"][1]["default"], "New task")
        self.assertIn("preview_step", required)
        self.assertNotIn("预览位置", required)
        self.assertEqual(set(fl2va["optional"]), {"first_frame", "last_frame"})
        self.assertIn("Picture 1", ref2va["optional"])
        self.assertEqual(H3ServeConnectionEnglish.RETURN_NAMES, ("connection",))

        prompt = "integrated_multimodal_description: one shot"
        with patch("h3serve_connector.nodes._run", return_value=("ok",)) as run:
            result = H3ServeFL2VAPresetGenerateEnglish().generate_english(
                connection={"server_url": "http://127.0.0.1:8090"},
                prompt=prompt,
                resolution="480p",
                aspect_ratio="16:9",
                duration_seconds=5.0,
                sampling_steps=8,
                acceleration=25.0,
                model_variant="Base weights",
                preview_mode="Off",
                preview_step=6,
                seed=4404,
                checkpoint_job_id="",
                checkpoint_action="New task",
            )
        self.assertEqual(result, ("ok",))
        fields = run.call_args.args[1]
        self.assertEqual(fields["prompt"], prompt)
        self.assertEqual(fields["model_variant"], "base")
        self.assertEqual(fields["preview_mode"], "off")

    def test_creator_preview_stops_at_formal_checkpoint(self):
        with patch(
            "h3serve_connector.nodes._run_interactive_checkpoint",
            return_value=(None, "preview"),
        ) as run_checkpoint, patch("h3serve_connector.nodes._run") as run_complete:
            result = H3ServeFL2VAPresetGenerate().generate(
                连接={"server_url": "http://127.0.0.1:8090"},
                prompt="one continuous shot", resolution="720p", aspect_ratio="16:9",
                duration_seconds=5.0, sampling_steps=12, acceleration=50.0,
                model_variant="原始权重", preview_mode="开启", seed=4404,
                预览位置=6, 断点任务ID="", 断点动作="新建任务",
            )
        self.assertEqual(result, (None, "preview"))
        run_complete.assert_not_called()
        fields = run_checkpoint.call_args.args[1]
        self.assertEqual(fields["execution_mode"], "checkpoint")
        self.assertEqual(fields["checkpoint_step"], 6)
        self.assertTrue(fields["checkpoint_retain"])
        self.assertTrue(fields["checkpoint_preview"])
        self.assertNotIn("checkpoint_preview_resolution", fields)
        self.assertNotIn("checkpoint_preview_steps", fields)
        self.assertEqual(fields["preview_mode"], "off")

    def test_creator_preview_repairs_shifted_new_task_sentinel(self):
        with patch(
            "h3serve_connector.nodes._run_interactive_checkpoint",
            return_value=(None, "preview"),
        ) as run_checkpoint:
            result = H3ServeFL2VAPresetGenerate().generate(
                连接={"server_url": "http://127.0.0.1:8090"},
                prompt="one continuous shot", resolution="720p",
                aspect_ratio="16:9", duration_seconds=5.0,
                sampling_steps=12, acceleration=50.0,
                model_variant="原始权重", preview_mode="开启", seed=4404,
                预览位置=6, 断点任务ID="新建任务", 断点动作="新建任务",
            )
        self.assertEqual(result, (None, "preview"))
        run_checkpoint.assert_called_once()

    def test_creator_preview_resumes_existing_checkpoint_without_resubmitting(self):
        with patch(
            "h3serve_connector.nodes._resume_interactive_checkpoint",
            return_value=("final", "preview"),
        ) as resume, patch("h3serve_connector.nodes._run_interactive_checkpoint") as submit:
            result = H3ServeFL2VAPresetGenerate().generate(
                连接={"server_url": "http://127.0.0.1:8090"},
                prompt="unchanged", resolution="720p", aspect_ratio="16:9",
                duration_seconds=5.0, sampling_steps=12, acceleration=50.0,
                model_variant="原始权重", preview_mode="开启", seed=4404,
                断点任务ID="job-checkpoint", 断点动作="继续生成",
            )
        self.assertEqual(result, ("final", "preview"))
        resume.assert_called_once_with(
            {"server_url": "http://127.0.0.1:8090"}, "job-checkpoint",
        )
        submit.assert_not_called()

    def test_freeform_editor_has_no_hidden_native_required_shot_control(self):
        app_source = (
            Path(__file__).parents[3] / "static" / "app.js"
        ).read_text(encoding="utf-8")
        self.assertNotIn('maxlength="6000" required aria-label="SHOT', app_source)

    def test_ref2va_prompt_is_forwarded_without_soundtrack_rewrite(self):
        prompt = "  subject_definitions:\n<Subject 1> from <Picture 1>.\n  "
        with patch("h3serve_connector.nodes._run", return_value=("ok",)) as run:
            H3ServeRef2VAPresetGenerate().generate(
                连接={"server_url": "http://127.0.0.1:8090"},
                prompt=prompt,
                resolution="480p",
                aspect_ratio="16:9",
                duration_seconds=5.0,
                sampling_steps=8,
                acceleration=0.0,
                model_variant="原始权重",
                preview_mode="关闭",
                seed=4404,
            )
        fields = run.call_args.args[1]
        self.assertEqual(fields["prompt"], prompt)
        self.assertNotIn("non_diegetic_music", fields["prompt"])

    def test_fl2va_prompt_is_forwarded_as_one_raw_string(self):
        prompt = "  任意连续文本\nintegrated_multimodal_description: 保持原样。  "
        with patch("h3serve_connector.nodes._run", return_value=("ok",)) as run:
            H3ServeFL2VAPresetGenerate().generate(
                连接={"server_url": "http://127.0.0.1:8090"},
                prompt=prompt,
                resolution="480p",
                aspect_ratio="16:9",
                duration_seconds=5.0,
                sampling_steps=8,
                acceleration=0.0,
                model_variant="原始权重",
                preview_mode="关闭",
                seed=4404,
            )
        fields = run.call_args.args[1]
        self.assertEqual(fields["prompt"], prompt)

    def test_generation_nodes_are_direct_output_nodes(self):
        self.assertIs(H3ServeFL2VAPresetGenerate.OUTPUT_NODE, True)
        self.assertIs(H3ServeRef2VAPresetGenerate.OUTPUT_NODE, True)

    def test_checkpoint_nodes_separate_submit_preview_and_resume(self):
        fl2va = H3ServeFL2VACheckpointSubmit.INPUT_TYPES()
        ref2va = H3ServeRef2VACheckpointSubmit.INPUT_TYPES()
        for schema in (fl2va, ref2va):
            required = schema["required"]
            self.assertIn("sampling_steps", required)
            self.assertIn("acceleration", required)
            self.assertIn("model_variant", required)
            self.assertNotIn("quality", required)
            self.assertEqual(required["sampling_steps"][1]["default"], 8)
            self.assertEqual(required["checkpoint_step"][1]["default"], 3)
        self.assertEqual(set(fl2va["optional"]), {"first_frame", "last_frame"})
        self.assertIn("Picture 1", ref2va["optional"])
        self.assertIn("Audio 1", ref2va["optional"])
        self.assertEqual(
            ref2va["required"]["参考图片分辨率"][1]["default"],
            "使用服务端设置",
        )
        self.assertEqual(
            ref2va["required"]["参考视频分辨率"][1]["default"],
            "使用服务端设置",
        )
        self.assertEqual(
            H3ServeFL2VACheckpointSubmit.RETURN_NAMES,
            ("任务ID", "断点任务详情"),
        )
        self.assertEqual(H3ServeCheckpointPreview.RETURN_NAMES, ("断点预览",))
        self.assertEqual(H3ServeCheckpointResume.RETURN_NAMES, ("最终视频",))

    def test_advanced_node_default_is_valid_for_base_and_lora(self):
        required = H3ServeAdvancedGenerate.INPUT_TYPES()["required"]
        self.assertEqual(required["sampling_steps"][1]["default"], 8)

    def test_creator_nodes_only_expose_outputs_needed_by_the_workflow(self):
        self.assertEqual(H3ServeConnection.RETURN_NAMES, ("连接",))
        self.assertEqual(H3ServeConnection.RETURN_TYPES, ("H3_SERVE_CONNECTION",))
        self.assertEqual(
            H3ServeFL2VAPresetGenerate.RETURN_NAMES,
            ("最终视频", "预览视频"),
        )
        self.assertEqual(
            H3ServeFL2VAPresetGenerate.RETURN_TYPES,
            ("VIDEO", "VIDEO"),
        )
        self.assertEqual(
            H3ServeRef2VAPresetGenerate.RETURN_NAMES,
            ("最终视频", "预览视频"),
        )
        self.assertEqual(
            H3ServeRef2VAPresetGenerate.RETURN_TYPES,
            ("VIDEO", "VIDEO"),
        )
        self.assertEqual(
            H3ServeAdvancedGenerate.RETURN_NAMES,
            ("视频", "本地路径", "任务ID", "任务详情"),
        )

    def test_example_workflows_do_not_resave_service_video(self):
        workflow_dir = Path(__file__).parents[1] / "example_workflows"
        for workflow_path in workflow_dir.glob("H3_Serve_*.json"):
            workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
            node_types = {node["type"] for node in workflow["nodes"]}
            self.assertNotIn("SaveVideo", node_types)
            english = workflow_path.stem.endswith("_EN")
            second_type = (
                "H3ServeSecondSamplingEnglish" if english
                else "H3ServeSecondSampling"
            )
            creator_types = (
                {
                    "H3ServeFL2VAPresetGenerateEnglish",
                    "H3ServeRef2VAPresetGenerateEnglish",
                }
                if english else
                {
                    "H3ServeFL2VAPresetGenerate",
                    "H3ServeRef2VAPresetGenerate",
                }
            )
            self.assertIn(second_type, node_types)
            second = next(
                node for node in workflow["nodes"]
                if node["type"] == second_type
            )
            self.assertIsNotNone(second["inputs"][0]["link"])
            self.assertIsNotNone(second["inputs"][1]["link"])
            creator = next(
                node for node in workflow["nodes"]
                if node["type"] in creator_types
            )
            # OUTPUT_NODE inserts control_after_generate at index 9. Keep an
            # explicit placeholder before the two hidden checkpoint widgets.
            sentinel = "New task" if english else "新建任务"
            self.assertEqual(creator["widgets_values"][9:], ["", "", sentinel])
            if english:
                self.assertEqual(creator["widgets_values"][5:7], ["Base weights", "Off"])
                self.assertEqual(
                    second["widgets_values"][2], "Standard · Denoise 0.20",
                )


if __name__ == "__main__":
    unittest.main()
