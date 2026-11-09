//! The eight layer mixer (DESIGN.md sections 3.2 and 5.2).
//!
//! Most layer fields can be patched with a single `Set` request, which keeps a
//! dragged slider responsive. Two of them cannot: `enabled` and `name` are not
//! part of `SetArgs`, so they take the read-modify-write path through
//! [`LayerIntent`] and the command thread's `GetPreset` / `Load` pair.
//!
//! Every editable field is read from a [`LayerDraft`] rather than straight from
//! the last status frame. The daemon echoes changes asynchronously, so reading
//! status directly would make a checkbox flick back to its old value for a frame
//! or two while the round trip completes.

use egui::{Color32, ComboBox, Rect, RichText, Sense, Ui, Vec2};
use noise_daemon::protocol::LayerStatus;
use noise_dsp::{Character, Color};

use crate::app::App;

/// Layers the daemon supports (DESIGN.md section 3.2).
pub const MAX_LAYERS: usize = noise_dsp::MAX_LAYERS;
/// Size of the little colour swatch drawn next to a layer.
const SWATCH_SIZE: f32 = 14.0;

/// Editable mirror of one layer.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerDraft {
    /// Whether the layer contributes to the mix.
    pub enabled: bool,
    /// Display name.
    pub name: String,
    /// Generation source.
    pub color: Color,
    /// Post colour character.
    pub character: Character,
    /// Layer gain in dB.
    pub gain_db: f64,
    /// Stereo correlation, -1.0 to 1.0.
    pub stereo: f64,
}

impl LayerDraft {
    /// Mirror a status entry.
    ///
    /// A colour or character string the daemon sends always parses back, so the
    /// fallbacks only cover a protocol from the future.
    pub fn from_status(layer: &LayerStatus) -> Self {
        Self {
            enabled: layer.enabled,
            name: layer.name.clone(),
            color: Color::parse(&layer.color).unwrap_or(Color::Custom),
            character: Character::parse(&layer.character).unwrap_or(Character::Neutral),
            gain_db: layer.gain_db,
            stereo: layer.stereo_correlation,
        }
    }
}

/// Resize and refill the drafts from a status snapshot.
///
/// The mixer iterates the status list, so the drafts must follow its length;
/// a layer the daemon added appears at once, and one it removed disappears.
pub fn sync_drafts(drafts: &mut Vec<LayerDraft>, layers: &[LayerStatus]) {
    drafts.truncate(layers.len());
    for (index, layer) in layers.iter().enumerate() {
        let fresh = LayerDraft::from_status(layer);
        match drafts.get_mut(index) {
            Some(slot) => *slot = fresh,
            None => drafts.push(fresh),
        }
    }
}

/// A mixer edit that needs a request of its own.
#[derive(Clone, Debug, PartialEq)]
pub enum LayerIntent {
    /// Enable or disable a layer.
    Enable(usize, bool),
    /// Rename a layer.
    Rename(usize, String),
    /// Switch a layer's colour.
    SetColor(usize, Color),
    /// Switch a layer's character.
    SetCharacter(usize, Character),
    /// Set a layer's gain in dB.
    Gain(usize, f64),
    /// Set a layer's stereo correlation.
    Stereo(usize, f64),
    /// Append a new layer.
    Add,
    /// Copy a layer.
    Duplicate(usize),
    /// Delete a layer.
    Remove(usize),
}

/// Colour used to hint at a colour's spectrum in the mixer.
pub fn swatch(color: Color) -> Color32 {
    match color {
        Color::White => Color32::from_rgb(235, 235, 235),
        Color::Pink => Color32::from_rgb(240, 150, 180),
        Color::Brown => Color32::from_rgb(165, 110, 70),
        Color::Blue => Color32::from_rgb(110, 150, 240),
        Color::Violet => Color32::from_rgb(170, 120, 240),
        Color::Custom => Color32::from_rgb(180, 180, 190),
    }
}

/// Theoretical PSD slope of a colour, with its sign, for the mixer caption.
pub fn slope_hint(color: Color, custom_alpha: f64) -> String {
    let alpha = color.resolve_alpha(custom_alpha);
    let slope = noise_dsp::colors::psd_slope_db_per_octave(alpha);
    // A zero slope multiplies out to `-0.0`, which formats as "-0.00" and reads
    // like a downward tilt. Normalise it to a real zero first.
    let slope = if slope == 0.0 { 0.0 } else { slope };
    format!("{slope:+.2} dB/oct")
}

/// Draw the mixer.
///
/// `layers` is the status snapshot for this frame; the editable values live in
/// `app.layer_drafts` and are written back as the widgets report changes.
pub fn ui(ui: &mut Ui, app: &mut App, layers: &[LayerStatus]) {
    let mut intents: Vec<LayerIntent> = Vec::new();
    ui.heading("Layer mixer");
    ui.label(
        RichText::new(format!(
            "{} of {MAX_LAYERS} layers used. The daemon refuses more than {MAX_LAYERS}, and \
             refuses to remove the last one.",
            layers.len()
        ))
        .small()
        .weak(),
    );

    ui.spacing_mut().slider_width = 140.0;

    for (index, status) in layers.iter().enumerate() {
        let Some(draft) = app.layer_drafts.get(index).cloned() else {
            continue;
        };
        let mut enabled = draft.enabled;
        let mut name = draft.name.clone();
        let mut color = draft.color;
        let mut character = draft.character;
        let mut gain_db = draft.gain_db;
        let mut stereo = draft.stereo;

        ui.group(|ui| {
            ui.horizontal(|ui| {
                if ui.checkbox(&mut enabled, "").changed() {
                    intents.push(LayerIntent::Enable(index, enabled));
                }
                ui.label(RichText::new(format!("#{index}")).monospace());
                let name_response = ui.add(
                    egui::TextEdit::singleline(&mut name)
                        .desired_width(150.0)
                        .hint_text("name"),
                );
                if name_response.lost_focus() && name != status.name {
                    intents.push(LayerIntent::Rename(index, name.clone()));
                }

                ui.label("colour");
                ComboBox::from_id_salt(("nf-layer-color", index))
                    .selected_text(color.label())
                    .width(90.0)
                    .show_ui(ui, |ui| {
                        for c in Color::ALL {
                            ui.selectable_value(&mut color, c, c.label());
                        }
                    });

                ui.label("character");
                ComboBox::from_id_salt(("nf-layer-character", index))
                    .selected_text(character.label())
                    .width(90.0)
                    .show_ui(ui, |ui| {
                        for c in Character::ALL {
                            ui.selectable_value(&mut character, c, c.label());
                        }
                    });

                if ui.small_button("Duplicate").clicked() {
                    intents.push(LayerIntent::Duplicate(index));
                }
                if ui.small_button("Remove").clicked() {
                    intents.push(LayerIntent::Remove(index));
                }
            });

            ui.horizontal(|ui| {
                colour_chip(ui, swatch(color));
                ui.label(
                    RichText::new(format!(
                        "{}, alpha {:.2}, {}, {}",
                        color.as_str(),
                        color.resolve_alpha(status.alpha),
                        character.as_str(),
                        slope_hint(color, status.alpha)
                    ))
                    .small()
                    .weak(),
                );
            });

            ui.horizontal(|ui| {
                if ui
                    .add(
                        egui::Slider::new(&mut gain_db, -60.0..=24.0)
                            .text("gain")
                            .suffix(" dB")
                            .fixed_decimals(1),
                    )
                    .changed()
                {
                    intents.push(LayerIntent::Gain(index, gain_db));
                }
                if ui
                    .add(
                        egui::Slider::new(&mut stereo, -1.0..=1.0)
                            .text("stereo")
                            .fixed_decimals(2),
                    )
                    .changed()
                {
                    intents.push(LayerIntent::Stereo(index, stereo));
                }
                ui.label(
                    RichText::new(format!(
                        "HP {:.0} Hz  LP {:.0} Hz  {} sos",
                        status.high_pass_hz, status.low_pass_hz, status.sections
                    ))
                    .small()
                    .weak(),
                );
            });

            crate::advanced::accuracy_line(ui, status);
        });
        ui.add_space(4.0);

        if color != draft.color {
            intents.push(LayerIntent::SetColor(index, color));
        }
        if character != draft.character {
            intents.push(LayerIntent::SetCharacter(index, character));
        }
        if let Some(slot) = app.layer_drafts.get_mut(index) {
            slot.enabled = enabled;
            slot.name = name;
            slot.color = color;
            slot.character = character;
            slot.gain_db = gain_db;
            slot.stereo = stereo;
        }
    }

    ui.horizontal(|ui| {
        if ui.button("Add layer").clicked() {
            intents.push(LayerIntent::Add);
        }
        ui.label(
            RichText::new("Duplicate copies a row, Remove deletes it.")
                .small()
                .weak(),
        );
    });

    for intent in intents {
        app.apply_layer_intent(intent);
    }
}

/// A small filled square, used as a colour chip.
fn colour_chip(ui: &mut Ui, color: Color32) {
    let (response, painter) = ui.allocate_painter(Vec2::splat(SWATCH_SIZE), Sense::hover());
    let rect: Rect = response.rect;
    painter.rect_filled(rect, egui::CornerRadius::same(2), color);
    painter.rect_stroke(
        rect,
        egui::CornerRadius::same(2),
        egui::Stroke::new(1.0, Color32::from_gray(90)),
        egui::StrokeKind::Inside,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn status_layer(color: &str, character: &str, alpha: f64) -> LayerStatus {
        LayerStatus {
            index: 0,
            name: "Main".into(),
            enabled: true,
            color: color.into(),
            alpha,
            character: character.into(),
            character_amount: 0.7,
            high_pass_hz: 35.0,
            low_pass_hz: 16_000.0,
            stereo_correlation: 0.35,
            gain_db: -1.5,
            seed: 42,
            sections: 5,
            shape_error_db: Some(0.25),
            slope_error_db_per_octave: Some(-0.01),
            verified: true,
            predicted_rms_dbfs: -37.0,
        }
    }

    #[test]
    fn every_colour_has_a_distinct_swatch() {
        let mut seen: Vec<Color32> = Vec::new();
        for c in Color::ALL {
            let s = swatch(c);
            assert!(!seen.contains(&s), "duplicate swatch for {}", c.as_str());
            seen.push(s);
        }
        assert_eq!(seen.len(), 6);
    }

    #[test]
    fn slope_hints_match_the_theory() {
        // Pink is -3.01 dB/oct, brown is twice that, blue is the mirror image.
        assert_eq!(slope_hint(Color::Pink, 1.0), "-3.01 dB/oct");
        assert_eq!(slope_hint(Color::Brown, 1.0), "-6.02 dB/oct");
        assert_eq!(slope_hint(Color::Blue, 1.0), "+3.01 dB/oct");
        assert_eq!(slope_hint(Color::White, 1.0), "+0.00 dB/oct");
        // A named colour ignores the custom value and uses its nominal alpha.
        assert_eq!(slope_hint(Color::Pink, 3.0), "-3.01 dB/oct");
        // Custom honours it.
        assert_eq!(slope_hint(Color::Custom, 2.0), "-6.02 dB/oct");
    }

    #[test]
    fn drafts_mirror_a_status_entry() {
        let s = status_layer("brown", "smooth", 2.0);
        let d = LayerDraft::from_status(&s);
        assert_eq!(d.color, Color::Brown);
        assert_eq!(d.character, Character::Smooth);
        assert_eq!(d.name, "Main");
        assert!(d.enabled);
        assert_eq!(d.gain_db, -1.5);
        assert_eq!(d.stereo, 0.35);
    }

    #[test]
    fn unknown_names_fall_back_instead_of_panicking() {
        let s = status_layer("chartreuse", "warm", 1.0);
        let d = LayerDraft::from_status(&s);
        assert_eq!(d.color, Color::Custom);
        assert_eq!(d.character, Character::Neutral);
    }

    #[test]
    fn drafts_follow_the_length_of_the_status_list() {
        let mut drafts = Vec::new();
        sync_drafts(&mut drafts, &[]);
        assert!(drafts.is_empty());

        let one = status_layer("pink", "neutral", 1.0);
        let mut second = status_layer("brown", "smooth", 2.0);
        second.index = 1;
        second.gain_db = -6.0;
        sync_drafts(&mut drafts, &[one.clone(), second.clone()]);
        assert_eq!(drafts.len(), 2);
        assert_eq!(drafts[1].color, Color::Brown);
        assert_eq!(drafts[1].gain_db, -6.0);

        // A removed layer drops its draft, and a stale value cannot survive.
        sync_drafts(&mut drafts, std::slice::from_ref(&one));
        assert_eq!(drafts.len(), 1);
        assert_eq!(drafts[0].color, Color::Pink);
        assert_eq!(drafts[0].gain_db, -1.5);
    }
}
