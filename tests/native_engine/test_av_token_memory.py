from __future__ import annotations

import unittest

import torch

from h3serve.native_engine.av_token_memory import (
    empty_av_token_memory,
    memory_conditioning,
    route_audio_memory_authority,
    route_visual_memory_authority,
    route_visual_memory_interval,
    token_memory_telemetry,
    update_av_token_memory,
    validate_av_token_memory,
)


class AVTokenMemoryTests(unittest.TestCase):
    def test_latest_only_reference_excludes_canonical_and_preserves_audio(self):
        memory = update_av_token_memory(
            empty_av_token_memory(audio_slots=1, audio_block_ticks=120),
            self._segment(73, 405), context_frames=0,
            visible_start_frame=0, visible_frames=243,
            preserve_latest_visual=True, audio_focus_frames=(108,),
        )
        routed = route_visual_memory_authority(memory, active=True, latest_only=True)
        self.assertEqual(len(routed['video_entries']), 1)
        self.assertEqual(routed['video_entries'][0]['position'], 242)
        self.assertEqual(len(memory['video_entries']), 6)
        torch.testing.assert_close(routed['audio_entries'][0]['latent'],
            memory['audio_entries'][0]['latent'], rtol=0, atol=0)
        self.assertEqual(routed['visual_route']['policy'], 'latest_generated_visual_authority_v1')
        self.assertEqual(len(route_visual_memory_authority(memory, active=True)['video_entries']), 6)
        self.assertEqual(route_visual_memory_authority(memory, active=False, latest_only=True)['video_entries'], [])

    def test_visual_ablation_preserves_voice_bank_and_source_state(self) -> None:
        memory = update_av_token_memory(
            empty_av_token_memory(audio_slots=1, audio_block_ticks=120),
            self._segment(73, 405),
            context_frames=0, visible_start_frame=0, visible_frames=243,
            audio_focus_frames=(108,),
        )
        routed = route_visual_memory_authority(memory, active=False)
        video, shapes, kinds, audio, ticks = memory_conditioning(routed)
        self.assertEqual((video, shapes, kinds), ((), (), ()))
        self.assertEqual(ticks, (120,))
        torch.testing.assert_close(audio[0], memory["audio_entries"][0]["latent"], rtol=0, atol=0)
        self.assertEqual(len(memory["video_entries"]), 6)
        self.assertEqual(routed["updates"], memory["updates"])
        silent = route_audio_memory_authority(routed, active=False)
        self.assertEqual(memory_conditioning(silent), ((), (), (), (), ()))

    @staticmethod
    def _segment(video_tokens: int, audio_ticks: int, offset: float = 0.0):
        video = torch.arange(
            24 * video_tokens * 4 * 6, dtype=torch.float32
        ).reshape(1, 24, video_tokens, 4, 6)
        audio = torch.arange(
            32 * 2 * audio_ticks, dtype=torch.float32
        ).reshape(1, 32, 2, audio_ticks)
        return {"video": video + offset, "audio": audio + offset}

    def test_update_keeps_fixed_multimodal_capacity(self) -> None:
        memory = empty_av_token_memory(
            video_slots=4,
            audio_slots=2,
            audio_block_ticks=20,
        )
        memory = update_av_token_memory(
            memory,
            self._segment(22, 120),
            context_frames=0,
            visible_start_frame=0,
            visible_frames=73,
        )
        memory = update_av_token_memory(
            memory,
            # Twelve video tokens and 65 audio ticks are the exact 39-frame
            # protected prefix; only the remaining suffix may enter memory.
            self._segment(32, 145, offset=1_000_000.0),
            context_frames=39,
            visible_start_frame=73,
            visible_frames=68,
        )
        validated = validate_av_token_memory(memory)
        self.assertEqual(len(validated["video_entries"]), 4)
        self.assertEqual(len(validated["audio_entries"]), 2)
        self.assertEqual(validated["updates"], 2)
        self.assertEqual(validated["video_entries"][0]["position"], 0)
        self.assertTrue(all(
            item["latent"].shape == (1, 24, 1, 4, 6)
            for item in validated["video_entries"]
        ))
        self.assertTrue(all(
            item["latent"].shape[-1] == 20
            for item in validated["audio_entries"]
        ))

    def test_visual_memory_resolution_reduces_only_spatial_latent_axes(self) -> None:
        video = torch.randn(1, 24, 7, 46, 80)
        audio = torch.randn(1, 32, 2, 160)
        memory = update_av_token_memory(
            empty_av_token_memory(
                video_slots=24,
                audio_slots=0,
                audio_block_ticks=80,
                visual_resolution="360p",
            ),
            {"video": video, "audio": audio},
            context_frames=0,
            visible_start_frame=0,
            visible_frames=22,
            preserve_latest_visual=True,
            collect_audio=False,
        )
        self.assertEqual(len(memory["video_entries"]), 7)
        self.assertEqual(memory["audio_entries"], [])
        self.assertEqual(memory["visual_resolution"], "360p")
        self.assertTrue(all(
            item["latent"].shape == (1, 24, 1, 22, 38)
            for item in memory["video_entries"]
        ))

    def test_audio_memory_can_be_enabled_without_visual_memory(self) -> None:
        memory = update_av_token_memory(
            empty_av_token_memory(
                video_slots=0,
                audio_slots=3,
                audio_block_ticks=20,
                visual_resolution="original",
            ),
            self._segment(22, 180),
            context_frames=0,
            visible_start_frame=0,
            visible_frames=73,
        )
        self.assertEqual(memory["video_entries"], [])
        self.assertEqual(len(memory["audio_entries"]), 3)

    def test_opening_preroll_never_enters_visual_memory(self) -> None:
        segment = self._segment(57, 320)
        memory = update_av_token_memory(
            empty_av_token_memory(video_slots=6, audio_slots=1),
            segment,
            context_frames=0,
            visible_start_frame=0,
            visible_frames=175,
            collect_audio=False,
            leading_preroll_frames=17,
        )
        first = min(memory["video_entries"], key=lambda item: item["position"])
        self.assertEqual(first["position"], 0)
        torch.testing.assert_close(
            first["latent"],
            segment["video"][:, :, 5:6],
            rtol=0,
            atol=0,
        )

    def test_conditioning_uses_native_reference_latent_geometry(self) -> None:
        memory = update_av_token_memory(
            empty_av_token_memory(
                video_slots=3,
                audio_slots=2,
                audio_block_ticks=40,
            ),
            self._segment(17, 175),
            context_frames=0,
            visible_start_frame=0,
            visible_frames=56,
        )
        video, shapes, kinds, audio, audio_frames = memory_conditioning(memory)
        self.assertEqual(len(video), 3)
        self.assertEqual(shapes, ((1, 4, 6),) * 3)
        self.assertEqual(kinds, ("image",) * 3)
        self.assertEqual(len(audio), 2)
        self.assertEqual(audio_frames, (40, 40))
        telemetry = token_memory_telemetry(memory)
        self.assertTrue(telemetry["bounded_active_context"])
        self.assertFalse(telemetry["text_summary"])
        self.assertEqual(
            telemetry["audio_representation"],
            "dialogue_gated_native_short_voice_excerpt",
        )
        self.assertTrue(telemetry["audio_temporal_order_preserved"])
        self.assertTrue(telemetry["audio_content_replayable"])
        self.assertEqual(
            telemetry["audio_denoise_exposure"],
            "runtime_authority_routed",
        )

    def test_audio_memory_keeps_one_native_short_voice_excerpt(self) -> None:
        memory = update_av_token_memory(
            empty_av_token_memory(
                video_slots=2,
                audio_slots=1,
                audio_block_ticks=40,
            ),
            self._segment(37, 207),
            context_frames=0,
            visible_start_frame=0,
            visible_frames=124,
        )
        entry = memory["audio_entries"][0]
        self.assertEqual(entry["latent"].shape, (1, 32, 2, 40))
        self.assertEqual(token_memory_telemetry(memory)["audio_reference_ticks_per_slot"], 40)

    def test_audio_collection_can_be_disabled_without_disabling_visual_memory(self) -> None:
        memory = update_av_token_memory(
            empty_av_token_memory(
                video_slots=2,
                audio_slots=1,
                audio_block_ticks=40,
            ),
            self._segment(37, 207),
            context_frames=0,
            visible_start_frame=0,
            visible_frames=124,
            collect_audio=False,
        )
        self.assertEqual(len(memory["video_entries"]), 2)
        self.assertEqual(memory["audio_entries"], [])

    def test_audio_long_memory_excludes_the_exact_recent_context_band(self) -> None:
        memory = update_av_token_memory(
            empty_av_token_memory(
                video_slots=3,
                audio_slots=6,
                audio_block_ticks=40,
            ),
            self._segment(37, 207),
            context_frames=0,
            visible_start_frame=0,
            visible_frames=124,
        )
        receipt = token_memory_telemetry(memory)
        self.assertEqual(receipt["audio_recent_guard_ticks"], 65)
        self.assertTrue(all(
            position + 40 <= 207 - 65
            for position in receipt["audio_positions"]
        ))

    def test_timed_dialogue_focus_excludes_stronger_ambient_block(self) -> None:
        segment = self._segment(37, 207)
        segment["audio"].zero_()
        # An unrelated loud source occurs at the opening; the authored
        # dialogue clock is two seconds into the visible interval.
        segment["audio"][..., :20] = 100.0
        segment["audio"][..., 80:100] = 10.0
        memory = update_av_token_memory(
            empty_av_token_memory(
                video_slots=2,
                audio_slots=1,
                audio_block_ticks=20,
            ),
            segment,
            context_frames=0,
            visible_start_frame=0,
            visible_frames=124,
            audio_focus_frames=(48,),
        )
        self.assertGreaterEqual(memory["audio_entries"][0]["position"], 40)
        self.assertEqual(
            token_memory_telemetry(memory)["audio_selection_policy"],
            "structured_dialogue_clock_local_energy_v1",
        )

    def test_dialogue_inside_recent_guard_does_not_fill_voice_slot_with_ambience(self) -> None:
        memory = update_av_token_memory(
            empty_av_token_memory(
                video_slots=2,
                audio_slots=1,
                audio_block_ticks=40,
            ),
            self._segment(37, 207),
            context_frames=0,
            visible_start_frame=0,
            visible_frames=124,
            # 4.17 seconds is beyond the eligible 142-tick long-memory band.
            audio_focus_frames=(100,),
        )
        self.assertEqual(memory["audio_entries"], [])
        self.assertEqual(
            token_memory_telemetry(memory)["audio_selection_policy"],
            "structured_dialogue_clock_not_eligible_v1",
        )

    def test_update_is_deterministic(self) -> None:
        source = self._segment(25, 123)
        left = update_av_token_memory(
            None,
            source,
            context_frames=0,
            visible_start_frame=10,
            visible_frames=85,
        )
        right = update_av_token_memory(
            None,
            source,
            context_frames=0,
            visible_start_frame=10,
            visible_frames=85,
        )
        self.assertEqual(token_memory_telemetry(left), token_memory_telemetry(right))
        for left_entry, right_entry in zip(
            left["video_entries"] + left["audio_entries"],
            right["video_entries"] + right["audio_entries"],
        ):
            self.assertTrue(torch.equal(left_entry["latent"], right_entry["latent"]))

    def test_structured_policy_reserves_canonical_and_latest_visual_state(self) -> None:
        memory = update_av_token_memory(
            empty_av_token_memory(
                video_slots=4,
                audio_slots=1,
                audio_block_ticks=20,
            ),
            self._segment(37, 207),
            context_frames=0,
            visible_start_frame=0,
            visible_frames=124,
            preserve_latest_visual=True,
        )
        positions = [item["position"] for item in memory["video_entries"]]
        self.assertEqual(positions[0], 0)
        self.assertEqual(positions[-1], 123)
        self.assertEqual(
            token_memory_telemetry(memory)["visual_selection_policy"],
            "canonical_latest_diverse_v3",
        )

    def test_director_route_filters_only_visual_positions(self) -> None:
        memory = empty_av_token_memory(video_slots=6, audio_slots=3)
        memory["video_entries"] = [
            {
                "position": position,
                "latent": torch.full((1, 24, 1, 4, 6), float(position)),
            }
            for position in (0, 337, 581, 587)
        ]
        memory["audio_entries"] = [
            {
                "position": position,
                "latent": torch.full((1, 32, 2, 20), float(position)),
            }
            for position in (0, 215, 915)
        ]
        routed = route_visual_memory_interval(
            memory,
            minimum_position=576,
            maximum_position=634,
        )
        self.assertEqual(
            [item["position"] for item in routed["video_entries"]],
            [581, 587],
        )
        self.assertEqual(
            [item["position"] for item in routed["audio_entries"]],
            [0, 215, 915],
        )
        receipt = token_memory_telemetry(routed)
        self.assertEqual(
            receipt["visual_route"]["policy"],
            "structured_director_active_shot_only_v1",
        )
        self.assertEqual(receipt["visual_route"]["routed_video_entries"], 2)

    def test_single_take_route_keeps_one_canonical_plus_recent_state(self) -> None:
        memory = empty_av_token_memory(video_slots=6, audio_slots=1)
        memory["video_entries"] = [
            {
                "position": position,
                "latent": torch.full((1, 24, 1, 4, 6), float(position)),
            }
            for position in (0, 120, 337, 581, 587)
        ]
        routed = route_visual_memory_interval(
            memory,
            minimum_position=576,
            maximum_position=634,
            include_canonical=True,
        )
        self.assertEqual(
            [item["position"] for item in routed["video_entries"]],
            [0, 581, 587],
        )
        receipt = token_memory_telemetry(routed)["visual_route"]
        self.assertEqual(
            receipt["policy"],
            "structured_director_canonical_plus_active_state_v2",
        )
        self.assertTrue(receipt["canonical_added"])

    def test_cut_route_labels_canonical_layout_separately_from_recent_state(self) -> None:
        memory = empty_av_token_memory(video_slots=6, audio_slots=1)
        memory["video_entries"] = [
            {
                "position": position,
                "latent": torch.full((1, 24, 1, 4, 6), float(position)),
            }
            for position in (0, 120, 337, 581, 587)
        ]
        routed = route_visual_memory_interval(
            memory,
            minimum_position=576,
            maximum_position=634,
            include_canonical=True,
            progressive_layout_state=True,
        )
        self.assertEqual(
            [item["position"] for item in routed["video_entries"]],
            [0, 581, 587],
        )
        receipt = token_memory_telemetry(routed)["visual_route"]
        self.assertEqual(
            receipt["policy"],
            "structured_director_progressive_layout_state_v1",
        )
        self.assertEqual(receipt["canonical_position"], 0)
        self.assertTrue(receipt["progressive_layout_state"])

    def test_explicit_camera_anchor_routes_terminal_camera_consensus(self) -> None:
        memory = empty_av_token_memory(video_slots=7, audio_slots=1)
        memory["video_entries"] = [
            {
                "position": position,
                "latent": torch.full((1, 24, 1, 4, 6), float(position)),
            }
            for position in (0, 14, 51, 251, 624, 645, 718)
        ]
        routed = route_visual_memory_interval(
            memory,
            minimum_position=629,
            maximum_position=719,
            progressive_layout_state=True,
            layout_minimum_position=0,
            layout_maximum_position=175,
        )
        self.assertEqual(
            [item["position"] for item in routed["video_entries"]],
            [14, 51, 645, 718],
        )
        receipt = token_memory_telemetry(routed)["visual_route"]
        self.assertEqual(
            receipt["policy"],
            "structured_director_terminal_camera_band_state_v3",
        )
        self.assertEqual(receipt["layout_positions"], [14, 51])
        self.assertEqual(receipt["layout_source_positions"], [0, 14, 51])
        self.assertEqual(
            receipt["layout_selection_policy"],
            "latest_two_retained_terminal_camera_consensus_v1",
        )
        self.assertEqual(receipt["layout_minimum_position"], 0)
        self.assertEqual(receipt["layout_maximum_position"], 175)

    def test_novel_camera_cut_withholds_all_full_frame_visual_memory(self) -> None:
        memory = empty_av_token_memory(video_slots=6, audio_slots=1)
        memory["video_entries"] = [
            {
                "position": position,
                "latent": torch.full((1, 24, 1, 4, 6), float(position)),
            }
            for position in (0, 51, 140, 157, 174)
        ]
        routed = route_visual_memory_interval(
            memory,
            minimum_position=85,
            maximum_position=175,
            novel_camera_cut=True,
        )
        self.assertEqual(routed["video_entries"], [])
        receipt = token_memory_telemetry(routed)["visual_route"]
        self.assertEqual(
            receipt["policy"],
            "structured_director_novel_camera_no_visual_rows_v1",
        )
        self.assertTrue(receipt["novel_camera_cut"])
        self.assertFalse(receipt["include_canonical"])
        self.assertEqual(receipt["source_video_entries"], 5)
        self.assertEqual(receipt["routed_video_entries"], 0)

    def test_novel_camera_probe_routes_only_one_canonical_scene_observation(self) -> None:
        memory = empty_av_token_memory(video_slots=6, audio_slots=1)
        memory["video_entries"] = [
            {
                "position": position,
                "latent": torch.full((1, 24, 1, 4, 6), float(position)),
            }
            for position in (0, 51, 140, 157, 174)
        ]
        routed = route_visual_memory_interval(
            memory,
            minimum_position=85,
            maximum_position=175,
            novel_camera_cut=True,
            novel_camera_layout_probe=True,
        )
        self.assertEqual(
            [item["position"] for item in routed["video_entries"]],
            [0],
        )
        receipt = token_memory_telemetry(routed)["visual_route"]
        self.assertEqual(
            receipt["policy"],
            "structured_director_novel_camera_layout_probe_v1",
        )
        self.assertTrue(receipt["progressive_layout_state"])
        self.assertTrue(receipt["novel_camera_layout_probe"])
        self.assertEqual(receipt["layout_positions"], [0])
        self.assertEqual(
            receipt["layout_selection_policy"],
            "canonical_scene_single_step_probe_v1",
        )

    def test_explicit_camera_anchor_fails_if_coreset_lost_the_band(self) -> None:
        memory = empty_av_token_memory(video_slots=2, audio_slots=1)
        memory["video_entries"] = [
            {
                "position": position,
                "latent": torch.full((1, 24, 1, 4, 6), float(position)),
            }
            for position in (645, 718)
        ]
        with self.assertRaisesRegex(RuntimeError, "has no retained visual-memory frame"):
            route_visual_memory_interval(
                memory,
                minimum_position=629,
                maximum_position=719,
                progressive_layout_state=True,
                layout_minimum_position=0,
                layout_maximum_position=175,
            )

    def test_dialogue_authority_route_removes_only_audio_memory(self) -> None:
        memory = empty_av_token_memory(video_slots=6, audio_slots=3)
        memory["video_entries"] = [
            {
                "position": 10,
                "latent": torch.ones((1, 24, 1, 4, 6)),
            }
        ]
        memory["audio_entries"] = [
            {
                "position": 20,
                "latent": torch.ones((1, 32, 2, 20)),
            },
            {
                "position": 40,
                "latent": torch.full((1, 32, 2, 20), 2.0),
            },
        ]
        routed = route_audio_memory_authority(memory, active=False)
        self.assertEqual(len(routed["video_entries"]), 1)
        self.assertEqual(routed["audio_entries"], [])
        self.assertTrue(torch.equal(
            routed["video_entries"][0]["latent"],
            memory["video_entries"][0]["latent"],
        ))
        receipt = token_memory_telemetry(routed)
        self.assertEqual(
            receipt["audio_route"],
            {
                "policy": "structured_dialogue_authority_v1",
                "active": False,
                "source_audio_entries": 2,
                "routed_audio_entries": 0,
                "video_entries_unchanged": 1,
                "prompt_content_inspected": False,
            },
        )


if __name__ == "__main__":
    unittest.main()
