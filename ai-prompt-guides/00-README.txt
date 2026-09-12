X-MinimaxH3 | Prompt-Writing Guides for AI Assistants

Purpose
=======
Upload the single TXT file that matches your current task to ChatGPT, Claude,
or another AI assistant that can read files. Then describe the story,
characters, reference assets, duration, and dialogue in natural language. The
assistant should follow the interaction flow in that file, ask only for
information that is truly required, and return content ready to paste into
X-MinimaxH3.

Six Task Modes
==============
1. 01-FL2VA-Single-Video.txt
   For text-only, first-frame, last-frame, and first-and-last-frame jobs in
   Single Video Creation.

2. 02-Ref2VA-Single-Video.txt
   For single-video jobs using multiple reference images, videos, or audio
   clips.

3. 03-FL2VA-Long-Video-Online.txt
   For interactive long-video creation, producing only the current window in
   each turn.

4. 04-Ref2VA-Long-Video-Online.txt
   For interactive long-video creation with reference images or audio,
   producing only the current window in each turn.

5. 05-FL2VA-Long-Video-JSON.txt
   For producing one complete FL2VA long-video JSON object ready to paste into
   JSON One-Click Creation.

6. 06-Ref2VA-Long-Video-JSON.txt
   For Ref2VA long-video JSON in which each window may inherit or replace its
   image and audio reference set.

Shared Rules
============
- These guides implement the official h3-prompt-writing structures from
  references/base-en.txt and references/ref-en.txt. Interface-specific limits
  in a selected guide still apply.
- The user may communicate with the assistant in any language. The final H3
  descriptive prose must be in English.
- Preserve dialogue, lyrics, and text that must visibly appear in the scene in
  the original language requested by the user.
- Dialogue format: <d>[English] I will be back soon.</d>. Use the matching
  language tag when dialogue is in another language.
- When dialogue crosses an authored cut, use <scenetrans> in both connected
  parts and state that the audio continues across the cut. Use <cutoff> only
  when speech is truncated by the end of the video.
- Unless subtitles are requested, explicitly prohibit visible subtitles,
  captions, dialogue transcription, lower-thirds, speech bubbles, and floating
  text.
- Do not place runtime settings such as inference steps, resolution, LoRA,
  acceleration, or Sigma inside an ordinary H3 prompt.
- Never invent reference file paths that the user did not provide.
- In long-video creation, each window is submitted as an independent H3
  request. When Previous-tail context is above zero, the runtime supplies the
  accepted preceding audiovisual tail as the physical continuation state; it
  does not supply earlier window prompt text.
- Therefore, make each window prompt semantically complete about the new action,
  dialogue, sound progression, and any internal camera cut that must occur in
  that window. Never rely on phrases such as "as described in Window 1" or
  "the earlier prompt" to carry required story instructions.
- Do not guess or restate an unobserved preceding ending. Let the supplied tail
  control the incoming camera, positions, poses, motion, and sound. Repeat only
  stable invariants that remain useful, such as identity, clothing, fixed layout,
  and prop identity; describe transient opening state only when the user has
  verified it from the generated result.
- A physical window boundary is always a continuation, never an authored cut.
  A window does not need a camera cut. If no viewpoint change is required, keep
  the whole window in one continuous shot.
- When a later window does require a cut, let one ongoing physical action span
  the boundary: begin the new window by continuing the action in the supplied
  incoming camera. Do not assign that opening continuation a ``00:00-X`` time
  range and do not replay the completed part of the action.
- After that continuation is clearly visible, declare the cut at one exact
  local point, for example ``[Shot 2] At 00:04.800, make one instantaneous hard
  cut to...``. Define the new camera position, framing and screen direction,
  then describe what occurs after the cut. Never put a cut at local 00:00.000
  or at the physical boundary. Avoid choosing a cut timestamp that coincides
  with the configured Previous-tail context duration.
- Strict physical continuity requires nonzero Previous-tail context. At zero,
  the next window is an independent clip and prompt wording alone cannot recover
  an unknown preceding final frame exactly.
- Each guide is self-contained. Upload only the guide needed for the current
  task.

Model Families and Input Support
================================
- FL2VA: single-video creation supports text-only, first-frame, last-frame, and
  first-and-last-frame input. Long-video windows may optionally use first or
  last keyframes.
- Ref2VA: single-video creation supports up to 9 reference images, 3 reference
  videos, and 3 reference audio clips.
- Ref2VA long video: each window currently supports up to 9 reference images
  and 3 reference audio clips. Reference video is not accepted.
