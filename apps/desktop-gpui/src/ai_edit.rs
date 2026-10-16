//! AI auto-edit: hand a recording's word-timed transcript to Claude Opus 5.5
//! through the user's own Claude Code install (`claude -p`, so it runs on
//! their Claude login rather than an API key) and apply the returned plan --
//! cuts, speed-ups, chapters and transitions, zooms, 3D camera moves, scene
//! layouts, animated titles, background music and sound effects -- through the
//! editor's own primitives and presets, so every track stays aligned and the
//! whole pass is one undo entry.

use std::{
    collections::HashMap,
    io::Write,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{LazyLock, Mutex},
};

use cap_project::{
    CaptionSegment, ClipTransition, ClipTransitionType, ProjectConfiguration, SceneMode,
    TimelineConfiguration, XY,
};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{editor_audio, editor_edits as edits, editor_panels, editor_timeline, store};

pub const MODEL: &str = "claude-opus-5-5";
const MIN_CUT_SECONDS: f64 = 0.3;
const MIN_SPEEDUP_SECONDS: f64 = 2.0;
const MIN_ZOOM_SECONDS: f64 = 1.0;
const MIN_MOTION_SECONDS: f64 = 2.0;
const MIN_LAYOUT_SECONDS: f64 = 2.0;
const MIN_TITLE_SECONDS: f64 = 1.0;
const TRANSITION_SECONDS: f64 = 0.5;
const SILENCE_MARKER_SECONDS: f32 = 0.6;

const AI_TEXT_PRESETS: &[&str] = &[
    "title",
    "headline",
    "cinematic",
    "lower-third",
    "kicker",
    "highlight",
    "neon",
    "stat",
    "quote",
];
const AI_MOTION_PRESETS: &[&str] = &[
    "spotlight",
    "perspective",
    "close-up",
    "low-angle",
    "glide-across",
    "pull-back",
    "rising-sweep",
    "drift-down",
    "slide-by",
    "tilt-away",
];
/// Bundled in `assets/sfx` (synthesized for Nex X Spark, so no licensing),
/// with their lengths as a fallback when ffprobe cannot read them.
const SOUND_EFFECTS: &[(&str, f64)] = &[
    ("whoosh", 0.7),
    ("swoosh-out", 0.55),
    ("pop", 0.16),
    ("chime", 1.1),
    ("click", 0.05),
];

#[derive(Clone, Debug, PartialEq)]
pub enum Status {
    Running,
    Done(String),
    Failed(String),
}

static STATUS: LazyLock<Mutex<HashMap<PathBuf, Status>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

fn statuses() -> std::sync::MutexGuard<'static, HashMap<PathBuf, Status>> {
    STATUS.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

pub fn status(project_path: &Path) -> Option<Status> {
    statuses().get(project_path).cloned()
}

pub fn begin(project_path: &Path) -> bool {
    let mut statuses = statuses();
    if statuses.get(project_path) == Some(&Status::Running) {
        return false;
    }
    statuses.insert(project_path.to_path_buf(), Status::Running);
    true
}

pub fn finish(project_path: &Path, result: Result<String, String>) {
    let status = match result {
        Ok(summary) => Status::Done(summary),
        Err(error) => Status::Failed(error),
    };
    statuses().insert(project_path.to_path_buf(), status);
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct EditPlan {
    pub title: String,
    pub summary: String,
    pub cuts: Vec<TimeRange>,
    #[serde(default)]
    pub speedups: Vec<Speedup>,
    pub chapters: Vec<Chapter>,
    #[serde(default)]
    pub transition: String,
    pub zooms: Vec<TimeRange>,
    #[serde(default)]
    pub titles: Vec<Title>,
    #[serde(default)]
    pub motion: Vec<Motion>,
    #[serde(default)]
    pub layouts: Vec<Layout>,
    #[serde(default)]
    pub music: Music,
    #[serde(default)]
    pub sound_effects: Vec<SoundEffect>,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct TimeRange {
    pub start: f64,
    pub end: f64,
    pub reason: String,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct Chapter {
    pub start: f64,
    pub title: String,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct Speedup {
    pub start: f64,
    pub end: f64,
    pub speed: f64,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct Title {
    pub start: f64,
    pub end: f64,
    pub text: String,
    pub preset: String,
    pub position: String,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct Motion {
    pub start: f64,
    pub end: f64,
    pub preset: String,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct Layout {
    pub start: f64,
    pub end: f64,
    pub mode: String,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct Music {
    pub track: String,
    pub volume: String,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct SoundEffect {
    pub time: f64,
    pub effect: String,
}

/// Audio files the plan needs, already copied into the project's
/// `assets/audio` so the project stays self-contained.
#[derive(Clone, Debug, Default)]
pub struct AudioAsset {
    pub path: String,
    pub name: String,
    pub duration: f64,
}

#[derive(Clone, Debug, Default)]
pub struct PreparedAssets {
    pub music: Option<AudioAsset>,
    pub sound_effects: HashMap<String, AudioAsset>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ApplyReport {
    pub cuts: usize,
    pub speedups: usize,
    pub chapters: usize,
    pub transitions: usize,
    pub zooms: usize,
    pub titles: usize,
    pub motion: usize,
    pub layouts: usize,
    pub music: bool,
    pub sound_effects: usize,
}

fn counted(count: usize, singular: &str, plural: &str) -> String {
    format!("{count} {}", if count == 1 { singular } else { plural })
}

impl ApplyReport {
    pub fn describe(&self, removed_seconds: f64) -> String {
        let mut parts = Vec::new();
        if self.cuts > 0 {
            parts.push(format!(
                "cut {} ({removed_seconds:.1}s)",
                counted(self.cuts, "pause", "pauses")
            ));
        }
        if self.speedups > 0 {
            parts.push(format!(
                "sped up {}",
                counted(self.speedups, "slow part", "slow parts")
            ));
        }
        if self.chapters > 0 {
            parts.push(format!(
                "named {}",
                counted(self.chapters, "chapter", "chapters")
            ));
        }
        if self.transitions > 0 {
            parts.push(counted(self.transitions, "transition", "transitions"));
        }
        if self.zooms > 0 {
            parts.push(counted(self.zooms, "zoom", "zooms"));
        }
        if self.motion > 0 {
            parts.push(counted(self.motion, "3D camera move", "3D camera moves"));
        }
        if self.layouts > 0 {
            parts.push(counted(self.layouts, "layout change", "layout changes"));
        }
        if self.titles > 0 {
            parts.push(counted(self.titles, "animated title", "animated titles"));
        }
        if self.music {
            parts.push("background music".to_string());
        }
        if self.sound_effects > 0 {
            parts.push(counted(self.sound_effects, "sound effect", "sound effects"));
        }
        if parts.is_empty() {
            "The recording already looks tight; nothing was changed.".to_string()
        } else {
            format!("Applied {}. Undo reverts all of it.", parts.join(", "))
        }
    }
}

/// One line per caption segment in source time, with explicit markers for the
/// gaps between words so the model can see silences it cannot hear.
pub fn build_transcript(segments: &[CaptionSegment]) -> String {
    let mut lines = Vec::new();
    let mut previous_end: Option<f32> = None;
    for segment in segments {
        let mut text = String::new();
        if segment.words.is_empty() {
            if let Some(end) = previous_end
                && segment.start - end >= SILENCE_MARKER_SECONDS
            {
                lines.push(format!("[{end:.2}-{:.2}] <silence>", segment.start));
            }
            text.push_str(segment.text.trim());
            previous_end = Some(segment.end);
        } else {
            for word in &segment.words {
                if let Some(end) = previous_end
                    && word.start - end >= SILENCE_MARKER_SECONDS
                {
                    if !text.is_empty() {
                        text.push(' ');
                    }
                    text.push_str(&format!("<silence {end:.2}-{:.2}>", word.start));
                }
                if !text.is_empty() {
                    text.push(' ');
                }
                text.push_str(&format!("{}@{:.2}", word.text.trim(), word.start));
                previous_end = Some(word.end);
            }
        }
        lines.push(format!("[{:.2}-{:.2}] {text}", segment.start, segment.end));
    }
    lines.join("\n")
}

pub fn edit_context(duration: f64, has_camera: bool) -> String {
    format!(
        "Recording length: {duration:.1}s. Webcam track: {}. Captions are shown along the \
         bottom of the frame.",
        if has_camera { "yes" } else { "no" }
    )
}

const SYSTEM_PROMPT: &str = "You are the auto-editor inside Nex X Spark, a screen recorder. You \
receive the word-timed transcript of one screen recording and some context about it. Times are \
seconds on the recording's source clock. Each word is written as word@start_time, and \
<silence a-b> marks a pause between words.

Plan a complete, polished edit without changing what the speaker says. Use restraint: a clean, \
professional edit beats a busy one.
- cuts: remove long pauses, false starts, repeated takes, and filler words (um, uh, like, you \
know) that add nothing. Keep about 0.15s of breathing room on both sides of speech, never cut \
inside a word that carries meaning, and do not cut pauses shorter than 0.4s.
- speedups: for long stretches where the speaker is waiting or doing something slow on screen \
(\"let me type this\", \"waiting for it to load\"), speed the stretch up instead of cutting it. \
speed is 1.5 to 4. Do not overlap cuts.
- chapters: 2 to 8 topic sections with short title-case names (max 5 words). The first chapter \
starts at the beginning of the first speech.
- transition: how chapters join: \"crossfade\", \"fade_to_black\", or \"none\".
- zooms: 2 to 6 second zoom-ins where the speaker points at something specific on screen \
(\"click here\", \"this button\").
- motion: at most 3 cinematic 3D camera moves (2 to 8 seconds) for big moments such as the \
intro, a reveal or the final result. Never overlap a zoom.
- layouts: only when the recording has a webcam. camera_only for a talking intro or outro, \
split_screen or floating while explaining, hide_camera for detailed screen work. Otherwise \
return an empty list.
- titles: an intro title with the recording's title in the first seconds (preset title, \
cinematic or headline), optional section titles at chapter starts (kicker or lower-third), and \
short callouts for key numbers or terms (highlight, stat, neon or quote). Text of at most 6 \
words, shown for 1.5 to 5 seconds. Keep titles away from the bottom of the frame when captions \
are shown.
- music: a background track that fits the tone, or \"none\" for serious or very short \
recordings. volume low unless the recording has little speech.
- sound_effects: sparse accents, at most one every 10 seconds: whoosh or swoosh-out on chapter \
changes and camera moves, pop when a title appears, chime for the intro or the final result, \
click when the speaker clicks something.
- title: a short title for the whole recording. summary: one or two sentences.

Return only times that exist in the transcript.";

fn plan_schema() -> Value {
    let string_enum = |values: &[&str]| json!({ "type": "string", "enum": values });
    let range = json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["start", "end", "reason"],
        "properties": {
            "start": { "type": "number" },
            "end": { "type": "number" },
            "reason": { "type": "string" }
        }
    });
    let music_tracks: Vec<&str> = std::iter::once("none")
        .chain(editor_audio::AUDIO_LIBRARY.iter().map(|(id, _)| *id))
        .collect();
    let effects: Vec<&str> = SOUND_EFFECTS.iter().map(|(id, _)| *id).collect();
    json!({
        "type": "object",
        "additionalProperties": false,
        "required": [
            "title", "summary", "cuts", "speedups", "chapters", "transition", "zooms",
            "titles", "motion", "layouts", "music", "sound_effects"
        ],
        "properties": {
            "title": { "type": "string" },
            "summary": { "type": "string" },
            "cuts": { "type": "array", "items": range },
            "speedups": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["start", "end", "speed"],
                    "properties": {
                        "start": { "type": "number" },
                        "end": { "type": "number" },
                        "speed": { "type": "number" }
                    }
                }
            },
            "chapters": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["start", "title"],
                    "properties": {
                        "start": { "type": "number" },
                        "title": { "type": "string" }
                    }
                }
            },
            "transition": string_enum(&["none", "crossfade", "fade_to_black"]),
            "zooms": { "type": "array", "items": range },
            "titles": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["start", "end", "text", "preset", "position"],
                    "properties": {
                        "start": { "type": "number" },
                        "end": { "type": "number" },
                        "text": { "type": "string" },
                        "preset": string_enum(AI_TEXT_PRESETS),
                        "position": string_enum(&["top", "center", "bottom"])
                    }
                }
            },
            "motion": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["start", "end", "preset"],
                    "properties": {
                        "start": { "type": "number" },
                        "end": { "type": "number" },
                        "preset": string_enum(AI_MOTION_PRESETS)
                    }
                }
            },
            "layouts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["start", "end", "mode"],
                    "properties": {
                        "start": { "type": "number" },
                        "end": { "type": "number" },
                        "mode": string_enum(&["camera_only", "hide_camera", "split_screen", "floating"])
                    }
                }
            },
            "music": {
                "type": "object",
                "additionalProperties": false,
                "required": ["track", "volume"],
                "properties": {
                    "track": string_enum(&music_tracks),
                    "volume": string_enum(&["low", "medium"])
                }
            },
            "sound_effects": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["time", "effect"],
                    "properties": {
                        "time": { "type": "number" },
                        "effect": string_enum(&effects)
                    }
                }
            }
        }
    })
}

const CLAUDE_NOT_FOUND: &str = "Claude Code was not found. Install it from \
claude.com/claude-code, run `claude` once to sign in, then try again.";

fn claude_binary_names() -> &'static [&'static str] {
    if cfg!(windows) {
        &["claude.exe"]
    } else {
        &["claude"]
    }
}

/// `claude` on PATH, then the places the installers put it. GUI apps on macOS
/// do not inherit the shell PATH, and on Windows the npm `claude.cmd` shim
/// only forwards to a native `claude.exe` in its package, which is run
/// directly because `Command` cannot launch a `.cmd` without a shell.
pub fn find_claude_cli() -> Option<PathBuf> {
    let path_dirs: Vec<PathBuf> = std::env::var_os("PATH")
        .map(|value| std::env::split_paths(&value).collect())
        .unwrap_or_default();
    let mut candidates = Vec::new();
    for dir in &path_dirs {
        for name in claude_binary_names() {
            candidates.push(dir.join(name));
        }
        candidates.push(
            dir.join("node_modules")
                .join("@anthropic-ai")
                .join("claude-code")
                .join("bin")
                .join(claude_binary_names()[0]),
        );
    }
    if let Some(home) = dirs::home_dir() {
        for name in claude_binary_names() {
            candidates.push(home.join(".local").join("bin").join(name));
            candidates.push(home.join(".claude").join("local").join(name));
        }
    }
    if let Some(app_data) = std::env::var_os("APPDATA") {
        candidates.push(
            PathBuf::from(app_data)
                .join("npm")
                .join("node_modules")
                .join("@anthropic-ai")
                .join("claude-code")
                .join("bin")
                .join("claude.exe"),
        );
    }
    if !cfg!(windows) {
        candidates.push(PathBuf::from("/opt/homebrew/bin/claude"));
        candidates.push(PathBuf::from("/usr/local/bin/claude"));
    }
    candidates.into_iter().find(|candidate| candidate.is_file())
}

/// Blocking: runs one non-interactive Claude Code turn with no tools, no MCP
/// servers and no saved session, and reads the plan from the structured
/// output. Run it off the UI thread.
pub fn request_plan_blocking(transcript: &str, context: &str) -> Result<EditPlan, String> {
    let claude = find_claude_cli().ok_or_else(|| CLAUDE_NOT_FOUND.to_string())?;
    let mut command = Command::new(claude);
    command
        .args(["-p", "--model", MODEL, "--output-format", "json", "--json-schema"])
        .arg(plan_schema().to_string())
        .args(["--tools", "", "--strict-mcp-config", "--no-session-persistence"])
        .args(["--system-prompt", SYSTEM_PROMPT])
        .current_dir(std::env::temp_dir())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        command.creation_flags(CREATE_NO_WINDOW);
    }
    let mut child = command
        .spawn()
        .map_err(|error| format!("Could not start Claude Code: {error}"))?;
    if let Some(mut stdin) = child.stdin.take() {
        let prompt =
            format!("<context>\n{context}\n</context>\n<transcript>\n{transcript}\n</transcript>");
        stdin
            .write_all(prompt.as_bytes())
            .map_err(|error| format!("Could not send the transcript to Claude Code: {error}"))?;
    }
    let output = child
        .wait_with_output()
        .map_err(|error| format!("Claude Code did not finish: {error}"))?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let parsed = serde_json::from_str::<Value>(stdout.trim()).ok();
    match parsed {
        Some(body) => parse_cli_output(&body),
        None => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let detail = stderr.trim().lines().last().unwrap_or("no output");
            Err(format!("Claude Code failed: {detail}"))
        }
    }
}

pub fn parse_cli_output(body: &Value) -> Result<EditPlan, String> {
    if body.get("is_error").and_then(Value::as_bool) == Some(true) {
        let message = body
            .get("result")
            .and_then(Value::as_str)
            .unwrap_or("unknown error");
        return Err(format!("Claude Code: {message}"));
    }
    let plan = body
        .get("structured_output")
        .filter(|value| value.is_object())
        .cloned()
        .or_else(|| {
            body.get("result")
                .and_then(Value::as_str)
                .and_then(|text| serde_json::from_str(text).ok())
        })
        .ok_or_else(|| "Claude Code returned no edit plan.".to_string())?;
    serde_json::from_value(plan).map_err(|error| format!("The edit plan was malformed: {error}"))
}

fn bundled_sound_effect(id: &str) -> Option<PathBuf> {
    if !SOUND_EFFECTS.iter().any(|(known, _)| *known == id) {
        return None;
    }
    let file = format!("{id}.wav");
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    store::bundled_resource_dirs()
        .into_iter()
        .map(|directory| directory.join("assets/sfx").join(&file))
        .chain(std::iter::once(manifest.join("assets/sfx").join(&file)))
        .find(|path| path.is_file())
}

fn copy_sound_effect(project_path: &Path, id: &str) -> Option<AudioAsset> {
    let source = bundled_sound_effect(id)?;
    let audio_dir = project_path.join("assets").join("audio");
    std::fs::create_dir_all(&audio_dir).ok()?;
    let file_name = format!("sfx-{id}.wav");
    let destination = audio_dir.join(&file_name);
    if !destination.exists() {
        std::fs::copy(&source, &destination).ok()?;
    }
    let fallback = SOUND_EFFECTS
        .iter()
        .find(|(known, _)| *known == id)
        .map_or(0.5, |(_, duration)| *duration);
    let probed = editor_audio::probe_audio_duration(&destination);
    Some(AudioAsset {
        path: format!("assets/audio/{file_name}"),
        name: format!("SFX {id}"),
        duration: if probed > 0.0 { probed } else { fallback },
    })
}

/// Blocking file copies for the music and sound effects the plan picked.
pub fn prepare_assets(project_path: &Path, plan: &EditPlan) -> PreparedAssets {
    let mut assets = PreparedAssets::default();
    if let Some((id, label)) = editor_audio::AUDIO_LIBRARY
        .iter()
        .find(|(id, _)| *id == plan.music.track)
    {
        match editor_audio::copy_library_track(project_path, id, label) {
            Ok((path, name, duration)) => {
                assets.music = Some(AudioAsset {
                    path,
                    name,
                    duration,
                });
            }
            Err(error) => tracing::warn!("AI edit could not add music: {error}"),
        }
    }
    for effect in &plan.sound_effects {
        if assets.sound_effects.contains_key(&effect.effect) {
            continue;
        }
        if let Some(asset) = copy_sound_effect(project_path, &effect.effect) {
            assets.sound_effects.insert(effect.effect.clone(), asset);
        }
    }
    assets
}

fn recording_offsets(recording_durations: &[f64]) -> Vec<f64> {
    let mut offsets = Vec::with_capacity(recording_durations.len());
    let mut cumulative = 0.0;
    for duration in recording_durations {
        offsets.push(cumulative);
        cumulative += duration;
    }
    offsets
}

/// Where a source-clock time currently plays on the output timeline, plus the
/// clip it falls in and that clip's end on the source clock.
fn locate_source_time(
    timeline: &TimelineConfiguration,
    recording_durations: &[f64],
    source_time: f64,
) -> Option<(f64, usize, f64)> {
    let recording_offsets = recording_offsets(recording_durations);
    let clip_offsets = editor_timeline::clip_timeline_offsets(timeline);
    let holds = timeline.hold_windows();
    timeline
        .segments
        .iter()
        .enumerate()
        .find_map(|(index, segment)| {
            let base = recording_offsets
                .get(segment.recording_clip as usize)
                .copied()
                .unwrap_or(0.0);
            let source_start = base + segment.start;
            let source_end = base + segment.end;
            if source_time < source_start || source_time >= source_end {
                return None;
            }
            let effective = clip_offsets.get(index).copied().unwrap_or(0.0)
                + (source_time - source_start) / segment.timescale;
            Some((
                edits::effective_to_output(&holds, effective),
                index,
                source_end,
            ))
        })
}

fn clip_index_starting_at(
    timeline: &TimelineConfiguration,
    recording_durations: &[f64],
    source_time: f64,
) -> Option<usize> {
    let recording_offsets = recording_offsets(recording_durations);
    timeline.segments.iter().position(|segment| {
        let base = recording_offsets
            .get(segment.recording_clip as usize)
            .copied()
            .unwrap_or(0.0);
        (base + segment.start - source_time).abs() < 0.01
    })
}

/// Splits the clips so `start..end` (clamped to the clip `start` falls in) is
/// a clip of its own, and returns its index.
fn isolate_source_range(
    timeline: &mut TimelineConfiguration,
    recording_durations: &[f64],
    start: f64,
    end: f64,
    min_length: f64,
) -> Option<usize> {
    let (_, _, clip_source_end) = locate_source_time(timeline, recording_durations, start)?;
    let end = end.min(clip_source_end);
    if end - start < min_length {
        return None;
    }
    if clip_source_end - end >= 0.01 {
        let (output_end, index, _) = locate_source_time(timeline, recording_durations, end)?;
        if !edits::split_clip_segment(timeline, output_end, Some(index)) {
            return None;
        }
    }
    let (output_start, index, _) = locate_source_time(timeline, recording_durations, start)?;
    if clip_index_starting_at(timeline, recording_durations, start).is_none()
        && !edits::split_clip_segment(timeline, output_start, Some(index))
    {
        return None;
    }
    clip_index_starting_at(timeline, recording_durations, start)
}

fn output_range(
    timeline: &TimelineConfiguration,
    recording_durations: &[f64],
    start: f64,
    end: f64,
) -> Option<(f64, f64)> {
    if !start.is_finite() || !end.is_finite() || end <= start {
        return None;
    }
    let (output_start, _, _) = locate_source_time(timeline, recording_durations, start)?;
    let (output_end, _, _) = locate_source_time(timeline, recording_durations, end)?;
    (output_end > output_start).then_some((output_start, output_end))
}

fn overlaps_any(ranges: impl IntoIterator<Item = (f64, f64)>, start: f64, end: f64) -> bool {
    ranges
        .into_iter()
        .any(|(other_start, other_end)| other_start < end && start < other_end)
}

fn free_lane(existing: impl Iterator<Item = (u32, f64, f64)>, start: f64, end: f64) -> u32 {
    let busy: Vec<u32> = existing
        .filter(|(_, other_start, other_end)| *other_start < end && start < *other_end)
        .map(|(lane, _, _)| lane)
        .collect();
    (0..).find(|lane| !busy.contains(lane)).unwrap_or(0)
}

fn sorted_non_overlapping(mut ranges: Vec<(f64, f64)>) -> Vec<(f64, f64)> {
    ranges.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut merged: Vec<(f64, f64)> = Vec::with_capacity(ranges.len());
    for range in ranges {
        match merged.last_mut() {
            Some(last) if range.0 <= last.1 => last.1 = last.1.max(range.1),
            _ => merged.push(range),
        }
    }
    merged
}

fn sanitized_cuts(plan: &EditPlan) -> Vec<(f64, f64)> {
    sorted_non_overlapping(
        plan.cuts
            .iter()
            .filter(|cut| cut.start.is_finite() && cut.end.is_finite())
            .map(|cut| (cut.start.max(0.0), cut.end))
            .filter(|(start, end)| end - start >= MIN_CUT_SECONDS)
            .collect(),
    )
}

fn sanitized_speedups(plan: &EditPlan) -> Vec<(f64, f64, f64)> {
    let mut speedups: Vec<(f64, f64, f64)> = plan
        .speedups
        .iter()
        .filter(|speedup| {
            speedup.start.is_finite() && speedup.end.is_finite() && speedup.speed.is_finite()
        })
        .map(|speedup| {
            (
                speedup.start.max(0.0),
                speedup.end,
                speedup.speed.clamp(1.25, 4.0),
            )
        })
        .filter(|(start, end, _)| end - start >= MIN_SPEEDUP_SECONDS)
        .collect();
    speedups.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut kept: Vec<(f64, f64, f64)> = Vec::with_capacity(speedups.len());
    for speedup in speedups {
        if kept.last().is_none_or(|last| speedup.0 >= last.1) {
            kept.push(speedup);
        }
    }
    kept
}

fn motion_template(id: &str) -> Option<editor_panels::MotionTemplate> {
    if !AI_MOTION_PRESETS.contains(&id) {
        return None;
    }
    if let Some(template) = editor_panels::MOTION_TEMPLATES
        .iter()
        .find(|template| template.id == id)
    {
        return Some(editor_panels::MotionTemplate {
            id: template.id,
            name: template.name,
            from: template.from,
            to: template.to,
            blur: template.blur,
        });
    }
    editor_panels::ANGLE_PRESETS
        .iter()
        .find(|preset| preset.id == id)
        .map(editor_panels::angle_preset_motion)
}

fn has_output_overlays(timeline: &TimelineConfiguration) -> bool {
    !timeline.transitions.is_empty()
        || !timeline.zoom_segments.is_empty()
        || !timeline.scene_segments.is_empty()
        || !timeline.mask_segments.is_empty()
        || !timeline.text_segments.is_empty()
        || !timeline.keyboard_segments.is_empty()
        || !timeline.audio_segments.is_empty()
        || !timeline.style_segments.is_empty()
        || !timeline.image_segments.is_empty()
        || !timeline.camera3d_segments.is_empty()
}

/// Applies the plan in source time. Clip edits go latest-first so the output
/// positions of the earlier ones are not disturbed before they are made, and
/// every overlay is placed afterwards against the final timeline. Clip
/// transitions shift everything after them, so they are only added when the
/// timeline has no overlays yet. Returns what was applied and how many output
/// seconds the cuts removed.
pub fn apply_plan(
    project: &mut ProjectConfiguration,
    plan: &EditPlan,
    recording_durations: &[f64],
    has_camera: bool,
    assets: &PreparedAssets,
) -> (ApplyReport, f64) {
    let mut report = ApplyReport::default();
    let Some(timeline) = project.timeline.as_mut() else {
        return (report, 0.0);
    };
    let can_add_transitions = !has_output_overlays(timeline);

    let duration_before_cuts = edits::clip_timeline_duration(timeline);
    for (start, end) in sanitized_cuts(plan).into_iter().rev() {
        if let Some(index) =
            isolate_source_range(timeline, recording_durations, start, end, MIN_CUT_SECONDS)
            && edits::delete_clip_segments(timeline, &[index])
        {
            report.cuts += 1;
        }
    }
    let removed = (duration_before_cuts - edits::clip_timeline_duration(timeline)).max(0.0);

    for (start, end, speed) in sanitized_speedups(plan).into_iter().rev() {
        if let Some(index) =
            isolate_source_range(timeline, recording_durations, start, end, MIN_SPEEDUP_SECONDS)
            && edits::set_clip_segment_timescale(timeline, index, speed)
        {
            report.speedups += 1;
        }
    }

    let mut chapters: Vec<&Chapter> = plan
        .chapters
        .iter()
        .filter(|chapter| chapter.start.is_finite() && !chapter.title.trim().is_empty())
        .collect();
    chapters.sort_by(|a, b| a.start.total_cmp(&b.start));
    for chapter in &chapters {
        let Some((output_time, index, _)) =
            locate_source_time(timeline, recording_durations, chapter.start)
        else {
            continue;
        };
        let target = match clip_index_starting_at(timeline, recording_durations, chapter.start) {
            Some(existing) => existing,
            None if output_time <= 0.05 => 0,
            None if edits::split_clip_segment(timeline, output_time, Some(index)) => index + 1,
            None => index,
        };
        if edits::set_clip_name(timeline, target, &chapter.title) {
            report.chapters += 1;
        }
    }

    let transition_kind = match plan.transition.as_str() {
        "crossfade" => Some(ClipTransitionType::CrossFade),
        "fade_to_black" => Some(ClipTransitionType::FadeThroughBlack),
        _ => None,
    };
    if can_add_transitions && let Some(kind) = transition_kind {
        let mut boundaries: Vec<usize> = chapters
            .iter()
            .filter_map(|chapter| {
                clip_index_starting_at(timeline, recording_durations, chapter.start)
            })
            .filter(|index| *index > 0)
            .collect();
        boundaries.sort_unstable();
        boundaries.dedup();
        for index in boundaries {
            timeline.transitions.push(ClipTransition {
                segment_index: index as u32,
                kind,
                duration: TRANSITION_SECONDS,
            });
            report.transitions += 1;
        }
        timeline
            .transitions
            .sort_by_key(|transition| transition.segment_index);
    }

    for zoom in &plan.zooms {
        let Some((start, end)) = output_range(timeline, recording_durations, zoom.start, zoom.end)
        else {
            continue;
        };
        if end - start < MIN_ZOOM_SECONDS
            || overlaps_any(
                timeline.zoom_segments.iter().map(|zoom| (zoom.start, zoom.end)),
                start,
                end,
            )
        {
            continue;
        }
        edits::insert_zoom_segment(timeline, start, end, edits::DEFAULT_ZOOM_AMOUNT);
        report.zooms += 1;
    }

    for motion in &plan.motion {
        let Some((start, end)) =
            output_range(timeline, recording_durations, motion.start, motion.end)
        else {
            continue;
        };
        let Some(template) = motion_template(&motion.preset) else {
            continue;
        };
        let busy = timeline
            .camera3d_segments
            .iter()
            .map(|shot| (shot.start, shot.end))
            .chain(timeline.zoom_segments.iter().map(|zoom| (zoom.start, zoom.end)));
        if end - start < MIN_MOTION_SECONDS || overlaps_any(busy, start, end) {
            continue;
        }
        let mut segment = edits::default_camera3d_segment(start, end);
        editor_panels::apply_motion_template(&mut segment, &template);
        edits::insert_camera3d_segment(timeline, segment);
        report.motion += 1;
    }

    if has_camera {
        for layout in &plan.layouts {
            let mode = match layout.mode.as_str() {
                "camera_only" => SceneMode::CameraOnly,
                "hide_camera" => SceneMode::HideCamera,
                "split_screen" => SceneMode::SplitScreen,
                "floating" => SceneMode::Floating,
                _ => continue,
            };
            let Some((start, end)) =
                output_range(timeline, recording_durations, layout.start, layout.end)
            else {
                continue;
            };
            if end - start < MIN_LAYOUT_SECONDS
                || overlaps_any(
                    timeline
                        .scene_segments
                        .iter()
                        .map(|scene| (scene.start, scene.end)),
                    start,
                    end,
                )
            {
                continue;
            }
            let mut segment = edits::default_scene_segment(start, end);
            segment.mode = mode;
            edits::insert_scene_segment(timeline, segment);
            report.layouts += 1;
        }
    }

    let installed = editor_panels::installed_fonts();
    for title in &plan.titles {
        let text = title.text.trim();
        if text.is_empty() {
            continue;
        }
        let Some(preset) = editor_panels::TEXT_PRESETS
            .iter()
            .find(|preset| preset.id == title.preset && AI_TEXT_PRESETS.contains(&preset.id))
        else {
            continue;
        };
        let Some((start, end)) =
            output_range(timeline, recording_durations, title.start, title.end)
        else {
            continue;
        };
        if end - start < MIN_TITLE_SECONDS {
            continue;
        }
        let lane = free_lane(
            timeline
                .text_segments
                .iter()
                .map(|segment| (segment.track, segment.start, segment.end)),
            start,
            end,
        );
        let mut segment = edits::default_text_segment(start, end, lane);
        segment.content = text.to_string();
        segment.size = XY::new(0.5, 0.09);
        editor_panels::apply_text_preset(&mut segment, preset, installed);
        if preset.center.is_none() {
            segment.center = match title.position.as_str() {
                "top" => XY::new(0.5, 0.16),
                "bottom" => XY::new(0.5, 0.8),
                _ => XY::new(0.5, 0.5),
            };
        }
        edits::insert_text_segment(timeline, segment);
        report.titles += 1;
    }

    let total = edits::clip_timeline_duration(timeline);
    if let Some(music) = &assets.music
        && total > 1.0
    {
        let lane = edits::used_lane_count(&timeline.audio_segments);
        let end = if music.duration > 0.0 {
            total.min(music.duration)
        } else {
            total
        };
        let mut segment = edits::default_audio_segment(
            0.0,
            end,
            lane,
            music.path.clone(),
            music.name.clone(),
            (music.duration > 0.0).then_some(music.duration),
        );
        segment.volume_db = if plan.music.volume == "medium" {
            -14.0
        } else {
            -20.0
        };
        segment.fade_in = 1.5_f64.min(end / 4.0);
        segment.fade_out = 2.5_f64.min(end / 4.0);
        edits::insert_audio_segment(timeline, segment);
        report.music = true;
    }

    let effects_lane = edits::used_lane_count(&timeline.audio_segments);
    for effect in &plan.sound_effects {
        let Some(asset) = assets.sound_effects.get(&effect.effect) else {
            continue;
        };
        if !effect.time.is_finite() {
            continue;
        }
        let Some((start, _, _)) = locate_source_time(timeline, recording_durations, effect.time)
        else {
            continue;
        };
        let end = (start + asset.duration).min(total);
        if end - start < 0.03
            || overlaps_any(
                timeline
                    .audio_segments
                    .iter()
                    .filter(|segment| segment.track == effects_lane)
                    .map(|segment| (segment.start, segment.end)),
                start,
                end,
            )
        {
            continue;
        }
        let mut segment = edits::default_audio_segment(
            start,
            end,
            effects_lane,
            asset.path.clone(),
            asset.name.clone(),
            Some(asset.duration),
        );
        segment.volume_db = -8.0;
        edits::insert_audio_segment(timeline, segment);
        report.sound_effects += 1;
    }

    (report, removed)
}

/// The transcript in source time. Older projects keep captions in output time
/// until their first sync, so the sync runs on a copy first.
pub fn source_timed_segments(
    project: &ProjectConfiguration,
    recording_durations: &[f64],
) -> Vec<CaptionSegment> {
    let mut project = project.clone();
    cap_project::synchronize_captions(&mut project, recording_durations);
    project
        .captions
        .map(|captions| captions.segments)
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use cap_project::{CaptionWord, TimelineSegment};

    fn project(duration: f64) -> ProjectConfiguration {
        let mut project = ProjectConfiguration::default();
        project.timeline = Some(TimelineConfiguration {
            segments: vec![TimelineSegment {
                recording_clip: 0,
                timescale: 1.0,
                start: 0.0,
                end: duration,
                name: None,
                speed_audio_mode: None,
                hide_cursor: None,
                volume: None,
            }],
            transitions: Vec::new(),
            zoom_segments: Vec::new(),
            scene_segments: Vec::new(),
            mask_segments: Vec::new(),
            text_segments: Vec::new(),
            caption_segments: Vec::new(),
            keyboard_segments: Vec::new(),
            audio_segments: Vec::new(),
            camera3d_segments: Vec::new(),
            style_segments: Vec::new(),
            image_segments: Vec::new(),
        });
        project
    }

    fn range(start: f64, end: f64) -> TimeRange {
        TimeRange {
            start,
            end,
            reason: String::new(),
        }
    }

    fn chapter(start: f64, title: &str) -> Chapter {
        Chapter {
            start,
            title: title.into(),
        }
    }

    #[test]
    fn transcript_marks_silences_between_words() {
        let segments = vec![CaptionSegment {
            id: "a".into(),
            start: 0.0,
            end: 3.0,
            text: "hello world".into(),
            words: vec![
                CaptionWord {
                    text: "hello".into(),
                    start: 0.0,
                    end: 0.5,
                },
                CaptionWord {
                    text: "world".into(),
                    start: 2.0,
                    end: 2.5,
                },
            ],
        }];
        assert_eq!(
            build_transcript(&segments),
            "[0.00-3.00] hello@0.00 <silence 0.50-2.00> world@2.00"
        );
    }

    #[test]
    fn cuts_remove_time_and_chapters_name_clips() {
        let mut project = project(30.0);
        let plan = EditPlan {
            cuts: vec![range(5.0, 7.0), range(20.0, 21.0), range(10.0, 10.1)],
            chapters: vec![chapter(0.0, "Intro"), chapter(15.0, "Demo")],
            zooms: vec![range(16.0, 19.0)],
            ..EditPlan::default()
        };
        let (report, removed) =
            apply_plan(&mut project, &plan, &[30.0], false, &PreparedAssets::default());
        let timeline = project.timeline.as_ref().unwrap();
        assert_eq!(report.cuts, 2);
        assert!((removed - 3.0).abs() < 1e-6);
        assert!((edits::clip_timeline_duration(timeline) - 27.0).abs() < 1e-6);
        assert_eq!(report.chapters, 2);
        assert_eq!(timeline.segments[0].name.as_deref(), Some("Intro"));
        assert!(
            timeline
                .segments
                .iter()
                .any(|segment| segment.name.as_deref() == Some("Demo"))
        );
        assert_eq!(report.zooms, 1);
        assert!((timeline.zoom_segments[0].start - 14.0).abs() < 1e-6);
        assert!((timeline.zoom_segments[0].end - 17.0).abs() < 1e-6);
    }

    #[test]
    fn speedups_shorten_the_timeline_and_transitions_join_chapters() {
        let mut project = project(40.0);
        let plan = EditPlan {
            speedups: vec![Speedup {
                start: 10.0,
                end: 20.0,
                speed: 2.0,
            }],
            chapters: vec![chapter(0.0, "Intro"), chapter(25.0, "Result")],
            transition: "crossfade".into(),
            ..EditPlan::default()
        };
        let (report, _) =
            apply_plan(&mut project, &plan, &[40.0], false, &PreparedAssets::default());
        let timeline = project.timeline.as_ref().unwrap();
        assert_eq!(report.speedups, 1);
        assert_eq!(report.transitions, 1);
        assert_eq!(timeline.transitions.len(), 1);
        let expected = 40.0 - 5.0 - TRANSITION_SECONDS;
        assert!((edits::clip_timeline_duration(timeline) - expected).abs() < 1e-6);
    }

    #[test]
    fn titles_layouts_and_sound_are_placed_on_the_final_timeline() {
        let mut project = project(30.0);
        let mut assets = PreparedAssets::default();
        assets.sound_effects.insert(
            "pop".into(),
            AudioAsset {
                path: "assets/audio/sfx-pop.wav".into(),
                name: "SFX pop".into(),
                duration: 0.16,
            },
        );
        assets.music = Some(AudioAsset {
            path: "assets/audio/library-lofi-beats-mirostar.mp3".into(),
            name: "Lofi Beats".into(),
            duration: 120.0,
        });
        let plan = EditPlan {
            cuts: vec![range(2.0, 4.0)],
            titles: vec![Title {
                start: 5.0,
                end: 8.0,
                text: "Getting Started".into(),
                preset: "title".into(),
                position: "top".into(),
            }],
            layouts: vec![Layout {
                start: 10.0,
                end: 15.0,
                mode: "split_screen".into(),
            }],
            music: Music {
                track: "lofi-beats-mirostar".into(),
                volume: "low".into(),
            },
            sound_effects: vec![SoundEffect {
                time: 5.0,
                effect: "pop".into(),
            }],
            ..EditPlan::default()
        };
        let (report, _) = apply_plan(&mut project, &plan, &[30.0], true, &assets);
        let timeline = project.timeline.as_ref().unwrap();
        assert_eq!(report.titles, 1);
        assert_eq!(timeline.text_segments[0].content, "Getting Started");
        assert!((timeline.text_segments[0].start - 3.0).abs() < 1e-6);
        assert_eq!(report.layouts, 1);
        assert!(matches!(timeline.scene_segments[0].mode, SceneMode::SplitScreen));
        assert!(report.music);
        assert_eq!(report.sound_effects, 1);
        assert_eq!(timeline.audio_segments.len(), 2);
        let effect = timeline
            .audio_segments
            .iter()
            .find(|segment| segment.path.ends_with("sfx-pop.wav"))
            .unwrap();
        assert!((effect.start - 3.0).abs() < 1e-6);
        let music = timeline
            .audio_segments
            .iter()
            .find(|segment| segment.path.ends_with(".mp3"))
            .unwrap();
        assert_ne!(music.track, effect.track);
        assert!((music.end - 28.0).abs() < 1e-6);
    }

    #[test]
    fn layouts_need_a_camera_and_unknown_presets_are_ignored() {
        let mut project = project(20.0);
        let plan = EditPlan {
            layouts: vec![Layout {
                start: 2.0,
                end: 8.0,
                mode: "camera_only".into(),
            }],
            motion: vec![Motion {
                start: 2.0,
                end: 8.0,
                preset: "not-a-preset".into(),
            }],
            ..EditPlan::default()
        };
        let (report, _) =
            apply_plan(&mut project, &plan, &[20.0], false, &PreparedAssets::default());
        assert_eq!(report, ApplyReport::default());
    }

    #[test]
    fn parses_claude_code_output_and_surfaces_errors() {
        let body = json!({
            "type": "result",
            "is_error": false,
            "result": "",
            "structured_output": {
                "title": "T", "summary": "S", "cuts": [], "chapters": [], "zooms": []
            }
        });
        assert_eq!(parse_cli_output(&body).unwrap().title, "T");
        let failed = json!({
            "type": "result",
            "is_error": true,
            "result": "API Error: 400 version 2.1.280 or newer is required."
        });
        let error = parse_cli_output(&failed).unwrap_err();
        assert!(error.contains("2.1.280"));
    }
}
