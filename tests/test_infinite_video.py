from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from h3serve.contract import ContractError
from h3serve.infinite_video import (
    InfiniteContinuationSpec,
    InfiniteProjectStore,
    compile_infinite_prompt,
    context_frames_for_seconds,
    memory_capacity,
    visible_frames_for_seconds,
)
from h3serve.openapi import document as openapi_document


class InfiniteVideoContractTest(unittest.TestCase):
    def test_window_prompt_does_not_require_project_overview(self) -> None:
        prompt = compile_infinite_prompt(
            overview="",
            window_description="[Shot 1] Establish one complete opening window.",
            overall_soundscape="N/A",
            non_diegetic_music="N/A",
            continuation=False,
        )
        self.assertNotIn("[Overall continuity]", prompt)
        self.assertIn("complete current-window instructions", prompt)
        self.assertIn("[Shot 1] Establish one complete opening window.", prompt)

    def test_complete_h3_window_prompt_is_not_nested_or_duplicated(self) -> None:
        prompt = compile_infinite_prompt(
            overview="",
            window_description=(
                "integrated_multimodal_description: [Shot 1] One complete window.\n\n"
                "overall_soundscape: Continuous room tone.\n\n"
                "non_diegetic_music: N/A"
            ),
            overall_soundscape="N/A",
            non_diegetic_music="N/A",
            continuation=True,
            context_frames=39,
        )
        self.assertEqual(prompt.count("integrated_multimodal_description:"), 1)
        self.assertEqual(prompt.count("overall_soundscape:"), 1)
        self.assertEqual(prompt.count("non_diegetic_music:"), 1)
        self.assertIn("Continuous room tone.", prompt)
        self.assertIn("strict continuation", prompt)

    def test_physical_clocks_stay_on_h3_grid(self) -> None:
        context = context_frames_for_seconds(1.625)
        visible = visible_frames_for_seconds(
            8.0, maximum_physical_frames=362, context_frames=context
        )
        self.assertEqual(context, 39)
        self.assertEqual(visible % 17, 0)
        self.assertLessEqual(context + visible, 362)

    def test_prompt_makes_boundary_continuous_and_cut_window_local(self) -> None:
        prompt = compile_infinite_prompt(
            overview="Same actor and same cafe layout.",
            window_description=(
                "Continue the counter shot. At 4.0 seconds, hard cut inside "
                "this window to the reverse angle."
            ),
            overall_soundscape="Continuous cafe room tone.",
            non_diegetic_music="N/A",
            continuation=True,
            context_frames=39,
        )
        self.assertIn("strict continuation", prompt)
        self.assertIn("without a reset, transition or accidental cut", prompt)
        self.assertIn("only if the current-window instructions", prompt)
        self.assertIn("story clock starts at 00:00.000 immediately after", prompt)
        self.assertIn("At 4.0 seconds, hard cut inside this window", prompt)
        self.assertNotIn("window_boundary_contract:", prompt)
        self.assertNotIn("current_window:", prompt)
        self.assertEqual(prompt.count("integrated_multimodal_description:"), 1)
        self.assertLess(
            prompt.index("integrated_multimodal_description:"),
            prompt.index("overall_soundscape:"),
        )

    def test_reference_prompt_uses_native_six_section_layout(self) -> None:
        prompt = compile_infinite_prompt(
            overview="Keep <Picture 1> as the owner identity.",
            window_description=(
                "<Picture 1> owner (S1) says: <d>[Chinese] 你好。</d>"
            ),
            overall_soundscape="Quiet room tone.",
            non_diegetic_music="N/A",
            continuation=False,
            reference_image_count=1,
            reference_audio_count=1,
        )
        fields = [
            "subject_definitions:", "summary:", "retention_analysis:",
            "detailed_description:", "overall_soundscape:",
            "non_diegetic_music:",
        ]
        self.assertEqual(fields, sorted(fields, key=prompt.index))
        self.assertIn("<Picture 1>", prompt)
        self.assertIn("<Audio 1>", prompt)

    def test_whole_project_refinement_does_not_reestablish_the_opening(self) -> None:
        prompt = compile_infinite_prompt(
            overview="One accepted cafe film.",
            window_description="Preserve the accepted source latent.",
            overall_soundscape="Preserve source audio.",
            non_diegetic_music="N/A",
            continuation=False,
            refinement=True,
        )
        self.assertIn("low-noise refinement", prompt)
        self.assertIn("Complete accepted source timeline", prompt)
        self.assertNotIn("Establish the world", prompt)

    def test_memory_slider_is_bounded_and_monotonic(self) -> None:
        self.assertFalse(memory_capacity(0)["enabled"])
        self.assertEqual(memory_capacity(1)["video_slots"], 1)
        self.assertEqual(memory_capacity(60)["video_slots"], 6)
        self.assertEqual(memory_capacity(100)["video_slots"], 9)
        self.assertEqual(memory_capacity(60)["audio_ticks_per_clip"], 79)
        self.assertEqual(memory_capacity(100)["audio_ticks_per_clip"], 120)
        self.assertEqual(
            memory_capacity(100, service_family="reference")[
                "audio_ticks_per_clip"
            ],
            240,
        )

    def test_split_memory_controls_have_independent_capacities(self) -> None:
        capacity = memory_capacity(
            visual_capacity=24,
            audio_capacity=3,
            visual_resolution="360p",
        )
        self.assertTrue(capacity["enabled"])
        self.assertEqual(capacity["video_slots"], 24)
        self.assertEqual(capacity["audio_slots"], 3)
        self.assertEqual(capacity["audio_ticks_per_clip"], 80)
        self.assertEqual(capacity["visual_resolution"], "360p")
        audio_off = memory_capacity(
            visual_capacity=6,
            audio_capacity=0,
            visual_resolution="480p",
        )
        self.assertTrue(audio_off["enabled"])
        self.assertEqual(audio_off["audio_seconds"], 0.0)
        self.assertFalse(memory_capacity(
            visual_capacity=0,
            audio_capacity=0,
            visual_resolution="original",
        )["enabled"])

    def test_project_store_round_trips_tail_editable_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = InfiniteProjectStore(root)
            project = store.create({
                "title": "Cafe",
                "overview": "One stable cafe.",
                "overall_soundscape": "Room tone.",
                "memory": 70,
            })
            project.windows.append({"index": 0, "job_id": "job-1"})
            store.persist(project)
            restored = InfiniteProjectStore(root).require(project.id)
            self.assertEqual(restored.title, "Cafe")
            self.assertEqual(restored.memory, 70)
            self.assertEqual(restored.visual_memory_capacity, 7)
            self.assertEqual(restored.audio_memory_capacity, 1)
            self.assertEqual(restored.visual_memory_resolution, "original")
            self.assertEqual(restored.windows[0]["job_id"], "job-1")
            document = json.loads(
                (root / "infinite_projects" / f"{project.id}.json").read_text()
            )
            self.assertEqual(document["overview"], "One stable cafe.")

    def test_project_container_can_be_created_from_title_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = InfiniteProjectStore(Path(temporary)).create({
                "title": "雨停之前",
            })
            self.assertEqual(project.title, "雨停之前")
            self.assertEqual(project.overview, "")
            self.assertEqual(project.overall_soundscape, "N/A")
            self.assertEqual(project.non_diegetic_music, "N/A")
            self.assertEqual(project.overlap_seconds, 1.625)
            self.assertEqual(project.memory, 60)
            self.assertEqual(project.visual_memory_capacity, 6)
            self.assertEqual(project.audio_memory_capacity, 1)
            self.assertEqual(project.visual_memory_resolution, "360p")

    def test_zero_previous_tail_context_uses_only_hidden_h3_preroll(self) -> None:
        self.assertEqual(context_frames_for_seconds(0), 0)
        continuation = InfiniteContinuationSpec(
            project_id="hard-cut",
            window_index=1,
            source_job_id="opening",
            source_frames=56,
            context_frames=0,
            visible_frames=34,
            audio_bridge_ticks=0,
            memory=0,
        )
        self.assertEqual(continuation.hidden_prefix_frames, 5)
        self.assertEqual(continuation.physical_frames, 39)
        self.assertEqual(continuation.output_frames, 90)
        self.assertEqual(continuation.audio_trim_ticks, 8)

    def test_v2_project_locks_preview_trajectory_and_persists_batch_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = InfiniteProjectStore(root)
            project = store.create({
                "workflow_version": 2,
                "title": "Locked preview film",
                "preview_resolution": "540p",
                "aspect_ratio": "16:9",
                "model_variant": "lora",
                "sampling_steps": 8,
                "acceleration": 50,
                "window_duration_seconds": 5,
                "overlap_seconds": 0.75,
                "service_family": "first_last",
            })
            project.batch_plan = [{"window_description": "Window one", "seed": 7}]
            project.batch_status = "running"
            store.persist(project)

            restored = InfiniteProjectStore(root).require(project.id)
            self.assertEqual(restored.workflow_version, 2)
            self.assertEqual(restored.resolution, "540p")
            self.assertEqual(restored.aspect_ratio, "16:9")
            self.assertEqual(restored.model_variant, "lora")
            self.assertEqual(restored.sampling_steps, 8)
            self.assertEqual(restored.acceleration, 50)
            self.assertEqual(restored.window_duration_seconds, 5)
            self.assertEqual(restored.batch_status, "running")
            self.assertEqual(restored.batch_plan[0]["window_description"], "Window one")

    def test_v2_project_rejects_invalid_preview_canvas(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = InfiniteProjectStore(Path(temporary))
            with self.assertRaisesRegex(ContractError, "between 360p and 1080p"):
                store.create({
                    "workflow_version": 2,
                    "title": "Too large preview",
                    "preview_resolution": "1440p",
                })

    def test_v3_project_round_trips_creation_mode_and_two_pass_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = InfiniteProjectStore(root).create({
                "workflow_version": 3,
                "creation_mode": "json",
                "title": "One-click film",
                "preview_resolution": "540p",
                "final_resolution": "1080p",
                "model_variant": "lora",
                "sampling_steps": 8,
                "final_sampling_steps": 5,
                "preview_branch_steps": 3,
                "acceleration": 40,
                "second_pass_acceleration": 65,
            })
            restored = InfiniteProjectStore(root).require(project.id)
            self.assertEqual(restored.creation_mode, "json")
            self.assertFalse(restored.preview_enabled)
            self.assertEqual(restored.final_resolution, "1080p")
            self.assertEqual(restored.final_sampling_steps, 5)
            self.assertEqual(restored.preview_branch_steps, 3)
            self.assertEqual(restored.second_pass_acceleration, 65)
            public = restored.public({})
            self.assertFalse(public["window_controls_editable"])
            self.assertEqual(public["creation_mode"], "json")
            self.assertFalse(public["second_sampling_available"])
            self.assertEqual(
                public["final_generation_method"],
                "global_sliding_selflift",
            )

    def test_v3_project_allows_equal_resolution_and_rejects_lower_final(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = InfiniteProjectStore(Path(temporary))
            project = store.create({
                "workflow_version": 3,
                "title": "Same-resolution trajectory",
                "preview_resolution": "1080p",
                "final_resolution": "1080p",
            })
            self.assertEqual(project.resolution, project.final_resolution)
            with self.assertRaisesRegex(
                ContractError, "greater than or equal to preview_resolution"
            ):
                store.create({
                    "workflow_version": 3,
                    "title": "Invalid two-pass ladder",
                    "preview_resolution": "1080p",
                    "final_resolution": "720p",
                })

    def test_v3_project_accepts_continuous_final_resolution_and_caps_preview_branch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = InfiniteProjectStore(Path(temporary))
            project = store.create({
                "workflow_version": 3,
                "title": "Continuous final canvas",
                "preview_resolution": "540p",
                "final_resolution": "1016p",
                "preview_branch_steps": 4,
            })
            self.assertEqual(project.final_resolution, "1016p")
            self.assertEqual(project.preview_branch_steps, 4)
            with self.assertRaisesRegex(
                ContractError, "preview_branch_steps must be between 1 and 4"
            ):
                store.create({
                    "workflow_version": 3,
                    "title": "Too many preview steps",
                    "preview_resolution": "540p",
                    "final_resolution": "1016p",
                    "preview_branch_steps": 5,
                })

    def test_public_project_locks_timeline_while_final_sampling_is_active(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = InfiniteProjectStore(Path(temporary)).create({
                "workflow_version": 2,
                "title": "Locked during final",
                "preview_resolution": "540p",
            })
            project.windows.append({
                "index": 0,
                "job_id": "preview",
                "total_frames": 124,
                "model_variant": "lora",
                "sampling_steps": 8,
                "acceleration": 50,
            })
            project.final_job_id = "final"
            jobs = {
                "preview": SimpleNamespace(
                    id="preview", status="succeeded", output_path=Path("preview.mp4"),
                    final_latents_path=Path("preview.pt"), error=None,
                    progress_percent=100, progress_stage="complete", progress_detail="",
                ),
                "final": SimpleNamespace(
                    id="final", status="running", output_path=None,
                ),
            }
            public = project.public(jobs)
            self.assertTrue(public["final_sampling"]["active"])
            self.assertFalse(public["can_append"])
            self.assertFalse(public["second_sampling_available"])

    def test_legacy_project_public_settings_follow_its_latest_window(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = InfiniteProjectStore(Path(temporary)).create({"title": "Legacy"})
            project.windows.append({
                "index": 0,
                "job_id": "preview",
                "total_frames": 124,
                "requested_duration_seconds": 5,
                "model_variant": "lora",
                "sampling_steps": 7,
                "acceleration": 42,
            })
            jobs = {
                "preview": SimpleNamespace(
                    id="preview", status="succeeded", output_path=Path("preview.mp4"),
                    final_latents_path=Path("preview.pt"), error=None,
                    progress_percent=100, progress_stage="complete", progress_detail="",
                )
            }
            public = project.public(jobs)
            self.assertEqual(public["model_variant"], "lora")
            self.assertEqual(public["sampling_steps"], 7)
            self.assertEqual(public["acceleration"], 42)
            self.assertEqual(public["window_duration_seconds"], 5)

    def test_project_container_can_be_deleted_without_deleting_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = InfiniteProjectStore(root)
            project = store.create({"title": "Disposable project"})
            project.windows.append({"index": 0, "job_id": "retained-job"})
            store.persist(project)

            deleted = store.delete(project.id)

            self.assertEqual(deleted.windows[0]["job_id"], "retained-job")
            self.assertFalse(
                (root / "infinite_projects" / f"{project.id}.json").exists()
            )
            with self.assertRaisesRegex(ContractError, "not found"):
                store.require(project.id)

    def test_old_continuation_record_defaults_source_dialogue_to_false(self) -> None:
        restored = InfiniteContinuationSpec.from_dict({
            "project_id": "p", "window_index": 1, "source_job_id": "j",
            "source_frames": 124, "context_frames": 39,
            "visible_frames": 85, "audio_bridge_ticks": 65, "memory": 60,
        })
        self.assertFalse(restored.source_dialogue)

    def test_openapi_exposes_project_window_tail_and_whole_second_pass(self) -> None:
        document = openapi_document("test")
        paths = document["paths"]
        self.assertIn("/api/v1/infinite-projects", paths)
        self.assertIn("delete", paths["/api/v1/infinite-projects/{project_id}"])
        self.assertIn("/api/v1/infinite-projects/{project_id}/windows", paths)
        self.assertIn("/api/v1/infinite-projects/{project_id}/windows/last", paths)
        self.assertIn(
            "/api/v1/infinite-projects/{project_id}/second-sampling", paths
        )
        self.assertIn("post", paths["/api/v1/infinite-projects/{project_id}/batch"])
        self.assertIn("delete", paths["/api/v1/infinite-projects/{project_id}/batch"])
        self.assertIn(
            "/api/v1/infinite-projects/{project_id}/final-sampling", paths
        )
        window = document["components"]["schemas"]["InfiniteWindowAppend"]
        project = document["components"]["schemas"]["InfiniteProjectCreate"]
        batch = document["components"]["schemas"]["InfiniteBatchRequest"]
        self.assertEqual(project["properties"]["workflow_version"]["default"], 3)
        self.assertEqual(
            project["properties"]["creation_mode"]["enum"], ["online", "json"]
        )
        self.assertIn("final_resolution", project["properties"])
        self.assertIn("preview_branch_steps", project["properties"])
        self.assertEqual(
            project["properties"]["preview_branch_steps"]["maximum"], 4
        )
        self.assertIn("pattern", project["properties"]["final_resolution"])
        self.assertIn("references", batch["properties"])
        self.assertEqual(
            window["properties"]["execution_mode"]["enum"],
            ["complete", "checkpoint"],
        )
        self.assertIn("checkpoint_step", window["properties"])
        self.assertIn("checkpoint_preview_resolution", window["properties"])


if __name__ == "__main__":
    unittest.main()
