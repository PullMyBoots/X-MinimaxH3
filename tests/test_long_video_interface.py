import copy
import json
from pathlib import Path
import unittest

from h3serve.contract import ContractError, GenerationSpec
from h3serve.long_video import (
    compile_window_story,
    memory_budget,
    normalize_long_video,
    validate_reference_inputs,
)
from h3serve.openapi import document as openapi_document


EXAMPLES = Path(__file__).parents[1] / "static" / "long-video-examples"


def example(name: str) -> dict:
    return json.loads((EXAMPLES / f"{name}.json").read_text(encoding="utf-8"))


class LongVideoInterfaceTests(unittest.TestCase):
    def test_v3_owns_each_cut_inside_one_window_and_keeps_seams_in_one_shot(self) -> None:
        request = example("cafe30")
        request["long_video"]["version"] = 3
        request["long_video"]["overlap_seconds"] = 1.625
        canonical = normalize_long_video(request["long_video"])
        plan, preview = compile_window_story(
            canonical, seed=request["seed"], maximum_frames=345,
        )

        self.assertEqual(plan.planning_policy,
                         "strict_continuation_internal_cut_event_aware_v1")
        self.assertEqual(
            [segment.window_frames for segment in plan.segments],
            [328, 277, 192],
        )
        self.assertEqual(
            [segment.transition for segment in plan.segments],
            ["opening", "continue", "continue"],
        )
        self.assertEqual(
            [segment.video_prefix_frames for segment in plan.segments],
            [None, 39, 39],
        )
        self.assertEqual(
            [[item["global_seconds"] for item in window["semantic_cuts"]]
             for window in preview["windows"]],
            [[10.0], [20.0], []],
        )
        self.assertIn("hard camera cut", preview["windows"][0]["prompt"])
        self.assertIn("hard camera cut", preview["windows"][1]["prompt"])
        self.assertNotIn("hard camera cut", preview["windows"][2]["prompt"])
        for window in preview["windows"]:
            prompt = window["prompt"]
            self.assertIn("Stable entity identities", prompt)
            self.assertIn("Authoritative world state at the first newly generated frame", prompt)
            self.assertIn("Required authoritative world state at this physical window end", prompt)
            self.assertIn("overall_soundscape:", prompt)

        cuts = [round(value * 24) for value in preview["semantic_cut_seconds"]]
        for segment in plan.segments[1:]:
            overlap_start = segment.visible_start_frame - segment.context_frames
            self.assertFalse(any(
                overlap_start <= cut <= segment.visible_start_frame
                for cut in cuts
            ))

    def test_v3_same_shot_segment_boundary_is_a_continuous_action_phase(self) -> None:
        request = example("cafe30")
        request["long_video"]["version"] = 3
        request["long_video"]["overlap_seconds"] = 1.625
        shot = request["long_video"]["story"]["shots"][2]
        original = shot["segments"][0]
        first = copy.deepcopy(original)
        first["id"] = "pick_up_cup"
        first["duration_seconds"] = 5
        first["beats"] = original["beats"][:1]
        first["dialogue"] = []
        first["camera"]["end"] = (
            "The same side view with owner holding the cup."
        )
        second = copy.deepcopy(original)
        second["id"] = "carry_cup"
        second["duration_seconds"] = 5
        second["transition"] = "continue"
        second["establish_seconds"] = 0
        second["camera"].pop("composition", None)
        second["beats"] = [{
            "start_seconds": 0,
            "end_seconds": 4,
            "text": "{{owner}} carries {{cup}} to {{window_table}}.",
            "state_updates": {
                "owner": {"location": "beside {{window_table}}"},
                "cup": {"holder": "{{owner}}", "location": "{{owner}} right hand"},
            },
        }]
        second["dialogue"] = []
        shot["segments"] = [first, second]

        plan, preview = compile_window_story(
            normalize_long_video(request["long_video"]),
            seed=request["seed"],
            maximum_frames=345,
        )

        self.assertEqual(
            [[item["global_seconds"] for item in window["semantic_cuts"]]
             for window in preview["windows"]],
            [[10.0], [20.0], []],
        )
        final_prompt = preview["windows"][2]["prompt"]
        self.assertNotIn("instantaneous hard camera cut", final_prompt)
        self.assertIn(
            "continue the same semantic shot window_side into its next action phase",
            final_prompt,
        )
        self.assertEqual(
            [segment.transition for segment in plan.segments],
            ["opening", "continue", "continue"],
        )

    def test_explicit_camera_anchor_resolves_to_earlier_segment_frame_band(self) -> None:
        request = example("cafe30")["long_video"]
        request["story"]["shots"][2]["camera_anchor"] = {
            "shot_id": "opening_wide",
            "segment_id": "counter_cleanup",
        }
        canonical = normalize_long_video(request)
        plan, preview = compile_window_story(canonical, seed=94001)

        anchored = plan.segments[2]
        self.assertEqual(anchored.video_prefix_frames, 0)
        self.assertEqual(anchored.visual_memory_layout_anchor_start_frame, 0)
        self.assertEqual(anchored.visual_memory_layout_anchor_stop_frame, 243)
        self.assertTrue(anchored.visual_memory_progressive_layout_state)
        self.assertEqual(
            preview["windows"][2]["visual_memory_policy"],
            "cut_explicit_camera_anchor_then_latest_state",
        )
        self.assertEqual(
            preview["windows"][2]["camera_anchor"],
            {"shot_id": "opening_wide", "segment_id": "counter_cleanup"},
        )
        self.assertEqual(
            preview["windows"][2]["camera_anchor_actual_seconds"],
            [0.0, 243 / 24],
        )
        self.assertIn(
            "declared camera-anchor memory band supplies only this camera position",
            preview["windows"][2]["prompt"],
        )
        self.assertIn(
            "newest terminal memory view supplies only the same-instant character pose",
            preview["windows"][2]["prompt"],
        )
        self.assertNotRegex(
            preview["windows"][2]["prompt"].lower(),
            r"\b(?:cut|edit|transition)\b",
        )

    def test_camera_anchor_must_name_an_earlier_segment_and_requires_memory(self) -> None:
        request = example("cafe30")["long_video"]
        request["story"]["shots"][1]["camera_anchor"] = {
            "shot_id": "window_side",
            "segment_id": "move_cup",
        }
        with self.assertRaisesRegex(ValueError, "must name an earlier shot"):
            normalize_long_video(request)

        request = example("cafe30")["long_video"]
        request["memory"] = 0
        request["story"]["shots"][2]["camera_anchor"] = {
            "shot_id": "opening_wide",
            "segment_id": "counter_cleanup",
        }
        with self.assertRaisesRegex(ValueError, "requires long-video memory above zero"):
            normalize_long_video(request)

    def test_multi_shot_compiles_one_complete_local_prompt_per_segment(self) -> None:
        request = example("cafe30")
        canonical = normalize_long_video(request["long_video"])
        plan, preview = compile_window_story(canonical, seed=request["seed"])

        self.assertEqual(plan.output_frames, 719)
        self.assertEqual(
            [item["transition"] for item in preview["windows"]],
            ["opening", "cut", "cut"],
        )
        self.assertEqual(
            len(plan.segments),
            sum(len(shot["segments"]) for shot in request["long_video"]["story"]["shots"]),
        )
        self.assertEqual(
            [segment.preserve_latest_visual for segment in plan.segments],
            [True, True, False],
        )
        self.assertEqual(
            [segment.video_prefix_frames for segment in plan.segments],
            [None, 0, 0],
        )
        self.assertEqual(
            [segment.novel_camera_cut for segment in plan.segments],
            [False, True, True],
        )
        self.assertEqual(
            [segment.terminal_video_seed for segment in plan.segments],
            [False, False, False],
        )
        self.assertEqual(
            [segment.novel_camera_layout_probe for segment in plan.segments],
            [False, False, False],
        )
        self.assertEqual(
            [segment.transition for segment in plan.segments],
            ["opening", "cut", "cut"],
        )
        self.assertEqual(
            [segment.continuation_bridge_prompt for segment in plan.segments],
            [None, None, None],
        )

        detail = preview["windows"][1]["prompt"]
        self.assertIn("complete local target is one uninterrupted shot", detail)
        self.assertNotIn("[Shot 2]", detail)
        self.assertIn("same camera projection throughout", detail)
        self.assertNotRegex(detail.lower(), r"\b(?:cut|edit|transition)\b")
        self.assertIn(
            "Action phase 2/3 from 00:04.225 through 00:07.225: Her right hand turns",
            detail,
        )
        self.assertIn(
            "continue the identical locked camera position, lens",
            detail,
        )
        self.assertIn(
            "Any camera or view wording inside the authored action below",
            detail,
        )
        self.assertNotIn("looks toward the rainy window", detail)

        returned = preview["windows"][2]["prompt"]
        self.assertIn("radio: audio=quiet narrow-band instrumental jazz", returned)
        self.assertIn("cloth: folded=true; holder=null", returned)
        self.assertNotIn("scene-memory observation", returned)
        self.assertNotIn("{{", returned)
        self.assertFalse(preview["windows"][2]["canonical_memory_active"])
        self.assertFalse(preview["windows"][2]["progressive_layout_state"])
        self.assertTrue(preview["windows"][2]["novel_camera_cut"])
        self.assertFalse(preview["windows"][2]["terminal_video_seed"])
        self.assertFalse(preview["windows"][2]["novel_camera_layout_probe"])
        self.assertEqual(
            preview["windows"][2]["visual_memory_policy"],
            "cut_text_world_state_no_visual_refs",
        )
        self.assertEqual(plan.segments[0].opening_preroll_frames, 17)
        self.assertEqual(preview["windows"][0]["opening_preroll_seconds"], 17 / 24)

    def test_single_take_continues_camera_and_retimes_each_local_clock(self) -> None:
        request = example("theater30")
        canonical = normalize_long_video(request["long_video"])
        plan, preview = compile_window_story(canonical, seed=request["seed"])

        self.assertEqual(
            [item["transition"] for item in preview["windows"]],
            ["opening", "continue", "continue"],
        )
        continuation = preview["windows"][1]["prompt"]
        self.assertIn("protected carried video is the sole starting camera boundary", continuation)
        self.assertIn("every adjacent frame must differ only", continuation)
        self.assertIn("Never let a wall, curtain, doorway edge", continuation)
        self.assertIn(
            "from 00:01.625 through 00:03.825: Continue the turn",
            continuation,
        )
        self.assertNotIn("三号灯位确认", continuation)
        self.assertNotIn("A medium-wide frontal view", continuation)
        self.assertNotIn("location=corridor corner", continuation)
        self.assertNotIn("Previous required camera state", continuation)
        opening = preview["windows"][0]["prompt"]
        self.assertIn("Time-ranged beats for this interval only", opening)
        self.assertNotIn("Continuous trajectory checkpoints", opening)
        self.assertIn("hold the required segment-end character, object and camera states", opening)
        self.assertIn("Do not advance into the next segment", opening)
        self.assertIn("Do not replay the shot opening or any earlier camera path", continuation)
        self.assertEqual(
            [segment.preserve_latest_visual for segment in plan.segments],
            [True, True, False],
        )
        self.assertEqual(
            [segment.video_prefix_frames for segment in plan.segments],
            [None, 39, 39],
        )
        self.assertEqual(
            [segment.transition for segment in plan.segments],
            ["opening", "continue", "continue"],
        )
        self.assertEqual(
            [bool(segment.continuation_bridge_prompt) for segment in plan.segments],
            [False, False, False],
        )
        boundary = preview["windows"][1]["continuation_boundary_prompt"]
        self.assertIsNone(boundary)
        self.assertEqual(
            [item["visual_memory_policy"] for item in preview["windows"]],
            ["collect_opening_state", "continuous_latest_state_only", "continuous_latest_state_only"],
        )
        self.assertEqual(
            [segment.visual_memory_include_canonical for segment in plan.segments],
            [False, False, False],
        )

    def test_long_overlap_uses_exact_anchor_then_hidden_video_repaint(self) -> None:
        request = example("theater30")
        request["long_video"]["overlap_seconds"] = 3.75
        canonical = normalize_long_video(request["long_video"])
        plan, preview = compile_window_story(canonical, seed=request["seed"])

        self.assertEqual(plan.context_frames, 90)
        self.assertEqual(
            [segment.video_prefix_frames for segment in plan.segments],
            [None, 73, 73],
        )
        self.assertEqual(
            [item["hidden_video_repaint_seconds"] for item in preview["windows"]],
            [0.0, 17 / 24, 17 / 24],
        )
        continuation = preview["windows"][1]["prompt"]
        self.assertIn("exact protected history anchor", continuation)
        self.assertIn("writable continuation lead-in", continuation)
        self.assertIn("replaces the matching provisional predecessor tail", continuation)
        self.assertIn("Do not begin the authored camera path", continuation)
        boundary = preview["windows"][1]["continuation_boundary_prompt"]
        self.assertIsNone(boundary)

    def test_multi_shot_continuation_keeps_complete_overlap_exact(self) -> None:
        request = example("cafe30")
        request["long_video"]["overlap_seconds"] = 3.75
        request["long_video"]["story"]["shots"][-1]["segments"].append({
            "id": "hold_window_view",
            "duration_seconds": 5,
            "transition": "continue",
            "camera": {
                "motion": "Continue the exact carried side view without moving the camera.",
                "end": "The identical side view with the same room geometry.",
            },
            "beats": [{
                "start_seconds": 0,
                "end_seconds": 4.5,
                "text": "The owner remains beside the same window table and does not move any prop.",
                "state_updates": {},
            }],
            "dialogue": [],
        })
        canonical = normalize_long_video(request["long_video"])
        plan, preview = compile_window_story(canonical, seed=request["seed"])

        continuation = plan.segments[-1]
        self.assertEqual(continuation.transition, "continue")
        self.assertEqual(continuation.context_frames, 90)
        self.assertEqual(continuation.video_prefix_frames, 90)
        self.assertEqual(
            preview["windows"][-1]["hidden_video_repaint_seconds"], 0.0
        )
        self.assertIn(
            "No accepted predecessor frame is repainted or replaced",
            preview["windows"][-1]["prompt"],
        )
        self.assertNotIn(
            "writable continuation lead-in",
            preview["windows"][-1]["prompt"],
        )

    def test_each_intra_window_phase_has_a_physical_frame_handshake(self) -> None:
        request = example("theater30")
        request["long_video"]["overlap_seconds"] = 3.75
        canonical = normalize_long_video(request["long_video"])
        _, preview = compile_window_story(canonical, seed=request["seed"])

        continuation = preview["windows"][1]["prompt"]
        self.assertIn(
            "Continuous trajectory checkpoints inside this same local shot",
            continuation,
        )
        self.assertIn(
            "Every entry below is a timing checkpoint along one uninterrupted camera trajectory",
            continuation,
        )
        self.assertIn(
            "At 00:03.750, there is no edit, time skip, camera reset or new composition",
            continuation,
        )
        self.assertIn(
            "The boundary at 00:05.950 is only an action checkpoint inside the same uninterrupted shot",
            continuation,
        )
        self.assertIn(
            "At 00:05.950, there is no edit, time skip, camera reset or new composition",
            continuation,
        )
        self.assertIn(
            "Any entity needed in a later phase must enter the existing view only through visible continuous",
            continuation,
        )
        self.assertEqual(continuation.count("[Shot 1]"), 1)
        self.assertNotIn("[Shot 2]", continuation)

    def test_fl2va_endpoint_instructions_follow_each_physical_window(self) -> None:
        request = example("theater30")
        canonical = normalize_long_video(request["long_video"])
        _, preview = compile_window_story(
            canonical, seed=request["seed"], first_frame=True, last_frame=True,
        )
        opening, middle, ending = [item["prompt"] for item in preview["windows"]]
        self.assertTrue(opening.startswith(
            "For the target video, at 0.00 seconds into the target video, <Picture 1>"
        ))
        self.assertFalse(middle.startswith("How the reference pictures align"))
        self.assertTrue(ending.startswith("How the reference pictures align"))
        self.assertIn("11.54-second mark", ending)

    def test_closed_entity_glossary_and_state_are_enforced(self) -> None:
        request = example("cafe30")["long_video"]
        undefined = copy.deepcopy(request)
        undefined["story"]["shots"][0]["segments"][0]["beats"][0]["text"] += " {{ghost}} appears."
        with self.assertRaisesRegex(ValueError, "undefined entity"):
            normalize_long_video(undefined)

        missing_state = copy.deepcopy(request)
        missing_state["story"]["entities"]["door"] = {
            "kind": "prop", "identity": "One green entrance door."
        }
        with self.assertRaisesRegex(ValueError, "missing entities.*door"):
            normalize_long_video(missing_state)

    def test_bare_defined_prop_is_still_closed_into_its_local_window(self) -> None:
        authored = copy.deepcopy(example("cafe30")["long_video"])
        authored["story"]["entities"]["clock"] = {
            "kind": "prop",
            "identity": "One round brass wall clock.",
        }
        authored["story"]["initial_state"]["clock"] = {
            "location": "rear wall",
            "visibility": "visible",
        }
        authored["story"]["shots"][0]["segments"][0]["camera"][
            "composition"
        ] += " The clock is visible above the counter."
        canonical = normalize_long_video(authored)
        _, preview = compile_window_story(canonical, seed=1)
        prompt = preview["windows"][0]["prompt"]
        self.assertIn("clock [prop]: One round brass wall clock.", prompt)
        self.assertIn("clock: location=rear wall; visibility=visible", prompt)

    def test_window_limit_includes_overlap_and_reports_the_segment(self) -> None:
        request = example("theater30")["long_video"]
        request["max_window_seconds"] = 10
        with self.assertRaisesRegex(ValueError, "continuous_take/corridor.*maximum"):
            compile_window_story(normalize_long_video(request), seed=1)

    def test_memory_slider_is_a_monotone_capacity_budget(self) -> None:
        request = example("cafe30")["long_video"]
        values = []
        for amount in range(101):
            request["memory"] = amount
            budget = memory_budget(
                normalize_long_video(request),
                service_family="first_last",
                width=864,
                height=480,
            )
            values.append(budget["maximum_memory_tokens"])
        self.assertEqual(values, sorted(values))
        self.assertEqual(values[0], 0)

        request["memory"] = 60
        budget = memory_budget(
            normalize_long_video(request),
            service_family="first_last",
            width=864,
            height=480,
        )
        self.assertEqual(budget, {
            "memory": 60,
            "requested_video_frames": 6,
            "video_frames": 6,
            "audio_clips": 1,
            "audio_ticks_per_clip": 79,
            "audio_seconds": 1.975,
            "maximum_memory_tokens": 2588,
            "automatic_audio_reason": "enabled",
            "meaning": "capacity_budget_not_quality_strength",
        })

    def test_user_reference_audio_disables_automatic_voice_memory(self) -> None:
        authored = example("cafe30")["long_video"]
        authored["story"]["entities"]["owner"]["identity"] += (
            " Match <Picture 1>; use <Audio 1> as her voice-timbre reference."
        )
        canonical = normalize_long_video(authored)
        budget = memory_budget(
            canonical,
            service_family="reference",
            user_images=1,
            user_audios=1,
        )
        self.assertEqual(budget["audio_clips"], 0)
        self.assertEqual(budget["video_frames"], 6)
        self.assertEqual(budget["automatic_audio_reason"], "user_reference_priority")
        validate_reference_inputs(
            canonical,
            service_family="reference",
            reference_images=1,
            reference_audios=1,
        )

    def test_ref2va_compiler_uses_the_six_section_reference_contract(self) -> None:
        authored = example("cafe30")["long_video"]
        authored["story"]["entities"]["owner"]["identity"] += (
            " Match <Picture 1>; use <Audio 1> as her voice-timbre reference."
        )
        canonical = normalize_long_video(authored)
        _, preview = compile_window_story(
            canonical, seed=1, service_family="reference",
        )
        prompt = preview["windows"][0]["prompt"]
        fields = [
            "subject_definitions:", "summary:", "retention_analysis:",
            "detailed_description:", "overall_soundscape:",
            "non_diegetic_music:",
        ]
        self.assertEqual([prompt.index(field) for field in fields], sorted(
            prompt.index(field) for field in fields
        ))
        self.assertIn("<Subject 1> (S1) says once", prompt)
        self.assertIn(
            "<Audio 1> is the voice-timbre reference for <Subject 1> (S1)",
            prompt,
        )
        silent_prompt = preview["windows"][1]["prompt"]
        self.assertNotIn("<Audio 1>", silent_prompt)
        self.assertIn("the established voice identity", silent_prompt)
        self.assertIn("<Audio 1>", preview["windows"][2]["prompt"])
        validate_reference_inputs(
            canonical, service_family="reference",
            reference_images=1, reference_audios=1,
        )
        with self.assertRaisesRegex(ValueError, "unused pictures"):
            validate_reference_inputs(
                canonical, service_family="reference",
                reference_images=2, reference_audios=1,
            )

    def test_v2_cut_guard_and_transactional_state_are_enforced(self) -> None:
        authored = example("cafe30")["long_video"]
        invalid = copy.deepcopy(authored)
        invalid["story"]["shots"][1]["segments"][0]["beats"][0]["start_seconds"] = 0
        with self.assertRaisesRegex(ValueError, "starts before the cut establishing interval"):
            normalize_long_video(invalid)

        canonical = normalize_long_video(authored)
        _, preview = compile_window_story(canonical, seed=1)
        third = preview["windows"][2]["prompt"]
        self.assertIn("cup: holder=null; location=walnut counter beside radio", third)
        self.assertIn("Required authoritative world state", third)
        self.assertIn("cup: holder=null; location=centre of window_table", third)
        identity_line = next(
            line for line in third.splitlines() if line.startswith("cup [prop]:")
        )
        self.assertNotIn("counter", identity_line)

    def test_v1_contract_remains_accepted_for_persisted_jobs(self) -> None:
        authored = {
            "version": 1,
            "overlap_seconds": 1.625,
            "max_window_seconds": 15,
            "memory": 0,
            "story": {
                "overview": "One continuous realistic view.",
                "entities": {"actor": "One adult wearing a blue coat."},
                "initial_state": {"actor": "Standing still."},
                "shots": [{
                    "id": "shot",
                    "camera": "One locked medium view of {{actor}}.",
                    "overall_soundscape": "Quiet room tone.",
                    "non_diegetic_music": "N/A",
                    "segments": [{
                        "id": "A",
                        "duration_seconds": 5,
                        "actions": [{"at_seconds": 0, "text": "{{actor}} looks up."}],
                        "dialogue": [],
                        "end_state": {"actor": "Looking up."},
                    }],
                }],
            },
        }
        canonical = normalize_long_video(authored)
        plan, preview = compile_window_story(canonical, seed=1)
        self.assertEqual(plan.mechanism, "authored_window_script_joint_av_v1")
        self.assertEqual(preview["version"], 1)

    def test_generation_spec_round_trip_preserves_authored_contract(self) -> None:
        request = example("cafe30")
        spec = GenerationSpec.from_mapping(request)
        restored = GenerationSpec.from_mapping(spec.to_dict())
        self.assertEqual(restored, spec)
        self.assertIsNotNone(spec.long_video)
        self.assertEqual(spec.output_frames, 719)
        self.assertEqual(spec.frames, 345)
        self.assertTrue(spec.prompt.startswith("integrated_multimodal_description:"))

        conflict = copy.deepcopy(request)
        conflict["duration_seconds"] = 45
        with self.assertRaisesRegex(ContractError, "must equal the sum"):
            GenerationSpec.from_mapping(conflict)

    def test_openapi_exposes_explicit_long_video_alternative_and_preview(self) -> None:
        api = openapi_document("test")
        generation = api["components"]["schemas"]["GenerationRequest"]
        self.assertEqual(generation["anyOf"], [
            {"required": ["prompt"]}, {"required": ["long_video"]},
        ])
        self.assertIn("LongVideoRequest", api["components"]["schemas"])
        self.assertIn("/api/v1/long-video/preview", api["paths"])


if __name__ == "__main__":
    unittest.main()
