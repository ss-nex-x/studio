//! AI auto-edit: hand a recording's word-timed transcript to Claude Opus 5.5
//! through the user's own Claude Code install (`claude -p`, so it runs on
//! their Claude login rather than an API key) and apply the returned plan (filler/silence cuts, named chapters, zooms) to
//! the timeline through the same split/delete/ripple primitives the timeline
//! itself uses, so every other track stays aligned and the whole pass is one
//! undo entry.

use std::{
    collections::HashMap,
    io::Write,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{LazyLock, Mutex},
};

use cap_project::{CaptionSegment, ProjectConfiguration, TimelineConfiguration};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{editor_edits as edits, editor_timeline};

pub const MODEL: &str = "claude-opus-5-5";
const MIN_CUT_SECONDS: f64 = 0.3;
const MIN_ZOOM_SECONDS: f64 = 1.0;
const SILENCE_MARKER_SECONDS: f32 = 0.6;

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

#[derive(Clone, Debug, Deserialize)]
pub struct EditPlan {
    pub title: String,
    pub summary: String,
    pub cuts: Vec<TimeRange>,
    pub chapters: Vec<Chapter>,
    pub zooms: Vec<TimeRange>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct TimeRange {
    pub start: f64,
    pub end: f64,
    pub reason: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Chapter {
    pub start: f64,
    pub title: String,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ApplyReport {
    pub cuts: usize,
    pub chapters: usize,
    pub zooms: usize,
}

impl ApplyReport {
    pub fn describe(&self, removed_seconds: f64) -> String {
        format!(
            "Removed {} pause{} and filler ({:.1}s), named {} chapter{}, added {} zoom{}. Undo reverts all of it.",
            self.cuts,
            if self.cuts == 1 { "" } else { "s" },
            removed_seconds,
            self.chapters,
            if self.chapters == 1 { "" } else { "s" },
            self.zooms,
            if self.zooms == 1 { "" } else { "s" },
        )
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

const SYSTEM_PROMPT: &str = "You are the auto-editor inside Nex X Spark, a screen recorder. You \
receive the word-timed transcript of one screen recording. Times are seconds on the recording's \
source clock. Each word is written as word@start_time, and <silence a-b> marks a pause between \
words.

Plan an edit that makes the recording tighter and easier to follow without changing what the \
speaker says:
- cuts: remove long pauses, false starts, repeated takes, and filler words (um, uh, like, you \
know) that add nothing. Keep about 0.15s of breathing room on both sides of speech, never cut \
inside a word that carries meaning, and do not cut pauses shorter than 0.4s. If the recording is \
already tight, return few or no cuts.
- chapters: split the recording into 2 to 8 topic sections with short title-case names (max 5 \
words). The first chapter starts at the beginning of the first speech.
- zooms: suggest moments where the speaker points at something specific on screen (\"click \
here\", \"this button\", \"look at this\"). Each zoom lasts 2 to 6 seconds; zero zooms is fine.
- title: a short title for the whole recording. summary: one or two sentences.

Return only times that exist in the transcript.";

fn plan_schema() -> Value {
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
    json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["title", "summary", "cuts", "chapters", "zooms"],
        "properties": {
            "title": { "type": "string" },
            "summary": { "type": "string" },
            "cuts": { "type": "array", "items": range },
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
            "zooms": { "type": "array", "items": range }
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
pub fn request_plan_blocking(transcript: &str) -> Result<EditPlan, String> {
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
        let prompt = format!("<transcript>\n{transcript}\n</transcript>");
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

fn sanitized_cuts(plan: &EditPlan) -> Vec<(f64, f64)> {
    let mut cuts: Vec<(f64, f64)> = plan
        .cuts
        .iter()
        .filter(|cut| cut.start.is_finite() && cut.end.is_finite())
        .map(|cut| (cut.start.max(0.0), cut.end))
        .filter(|(start, end)| end - start >= MIN_CUT_SECONDS)
        .collect();
    cuts.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut merged: Vec<(f64, f64)> = Vec::with_capacity(cuts.len());
    for cut in cuts {
        match merged.last_mut() {
            Some(last) if cut.0 <= last.1 => last.1 = last.1.max(cut.1),
            _ => merged.push(cut),
        }
    }
    merged
}

/// Applies the plan in source time. Cuts go latest-first so the output
/// positions of the earlier ones are not disturbed before they are made.
/// Returns what was applied and how many output seconds were removed.
pub fn apply_plan(
    project: &mut ProjectConfiguration,
    plan: &EditPlan,
    recording_durations: &[f64],
) -> (ApplyReport, f64) {
    let mut report = ApplyReport::default();
    let Some(timeline) = project.timeline.as_mut() else {
        return (report, 0.0);
    };
    let duration_before = edits::clip_timeline_duration(timeline);

    for (cut_start, cut_end) in sanitized_cuts(plan).into_iter().rev() {
        let Some((_, _, clip_source_end)) =
            locate_source_time(timeline, recording_durations, cut_start)
        else {
            continue;
        };
        let cut_end = cut_end.min(clip_source_end);
        if cut_end - cut_start < MIN_CUT_SECONDS {
            continue;
        }
        let ends_clip = clip_source_end - cut_end < 0.01;
        if !ends_clip {
            let Some((output_end, index, _)) =
                locate_source_time(timeline, recording_durations, cut_end)
            else {
                continue;
            };
            if !edits::split_clip_segment(timeline, output_end, Some(index)) {
                continue;
            }
        }
        let Some((output_start, index, _)) =
            locate_source_time(timeline, recording_durations, cut_start)
        else {
            continue;
        };
        if clip_index_starting_at(timeline, recording_durations, cut_start).is_none()
            && !edits::split_clip_segment(timeline, output_start, Some(index))
        {
            continue;
        }
        let Some(cut_index) = clip_index_starting_at(timeline, recording_durations, cut_start)
        else {
            continue;
        };
        if edits::delete_clip_segments(timeline, &[cut_index]) {
            report.cuts += 1;
        }
    }

    let mut chapters: Vec<&Chapter> = plan
        .chapters
        .iter()
        .filter(|chapter| chapter.start.is_finite() && !chapter.title.trim().is_empty())
        .collect();
    chapters.sort_by(|a, b| a.start.total_cmp(&b.start));
    for chapter in chapters {
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

    for zoom in &plan.zooms {
        if !zoom.start.is_finite() || !zoom.end.is_finite() {
            continue;
        }
        let Some((output_start, _, _)) =
            locate_source_time(timeline, recording_durations, zoom.start)
        else {
            continue;
        };
        let Some((output_end, _, _)) = locate_source_time(timeline, recording_durations, zoom.end)
        else {
            continue;
        };
        if output_end - output_start < MIN_ZOOM_SECONDS {
            continue;
        }
        let overlaps = timeline
            .zoom_segments
            .iter()
            .any(|existing| existing.start < output_end && output_start < existing.end);
        if overlaps {
            continue;
        }
        edits::insert_zoom_segment(timeline, output_start, output_end, edits::DEFAULT_ZOOM_AMOUNT);
        report.zooms += 1;
    }

    let removed = (duration_before - edits::clip_timeline_duration(timeline)).max(0.0);
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
            title: String::new(),
            summary: String::new(),
            cuts: vec![range(5.0, 7.0), range(20.0, 21.0), range(10.0, 10.1)],
            chapters: vec![
                Chapter {
                    start: 0.0,
                    title: "Intro".into(),
                },
                Chapter {
                    start: 15.0,
                    title: "Demo".into(),
                },
            ],
            zooms: vec![range(16.0, 19.0)],
        };
        let (report, removed) = apply_plan(&mut project, &plan, &[30.0]);
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
