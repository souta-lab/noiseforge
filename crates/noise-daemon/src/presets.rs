//! Preset and configuration storage (DESIGN.md section 7).
//!
//! TOML plus Serde under `$XDG_CONFIG_HOME/noiseforge/`, written atomically so a
//! crash mid-save cannot leave a truncated file behind. Every file carries a
//! schema version and a graph version.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use noise_dsp::params::{PresetParams, SCHEMA_VERSION};

use crate::protocol::preset_file_name;

/// Storage failures.
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("toml error: {0}")]
    Toml(#[from] toml::de::Error),
    #[error("could not serialise the preset: {0}")]
    Serialise(#[from] toml::ser::Error),
    #[error("preset '{0}' does not exist")]
    NotFound(String),
    #[error("no configuration directory is available")]
    NoConfigDir,
}

/// Where presets live.
#[derive(Clone, Debug)]
pub struct Store {
    root: PathBuf,
}

impl Store {
    /// Use `$XDG_CONFIG_HOME/noiseforge`, falling back to `~/.config/noiseforge`.
    pub fn discover() -> Result<Self, StoreError> {
        let base = std::env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .filter(|p| !p.as_os_str().is_empty())
            .or_else(|| dirs::home_dir().map(|h| h.join(".config")))
            .ok_or(StoreError::NoConfigDir)?;
        Ok(Self {
            root: base.join("noiseforge"),
        })
    }

    /// Use an explicit root; used by tests.
    pub fn at(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Root directory.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Directory holding user presets.
    pub fn preset_dir(&self) -> PathBuf {
        self.root.join("presets")
    }

    /// Path of the live configuration file.
    pub fn config_path(&self) -> PathBuf {
        self.root.join("config.toml")
    }

    /// Path of a named preset.
    pub fn preset_path(&self, name: &str) -> PathBuf {
        self.preset_dir().join(preset_file_name(name))
    }

    /// Create the directories with owner-only permissions.
    pub fn ensure_dirs(&self) -> Result<(), StoreError> {
        for dir in [self.root.clone(), self.preset_dir()] {
            fs::create_dir_all(&dir)?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let mut perms = fs::metadata(&dir)?.permissions();
                perms.set_mode(0o700);
                fs::set_permissions(&dir, perms)?;
            }
        }
        Ok(())
    }

    /// Write a preset atomically.
    pub fn save_preset(&self, name: &str, preset: &PresetParams) -> Result<PathBuf, StoreError> {
        self.ensure_dirs()?;
        let path = self.preset_path(name);
        let mut copy = preset.clone();
        copy.name = name.to_string();
        copy.schema_version = SCHEMA_VERSION;
        write_atomic(&path, &toml::to_string_pretty(&copy)?)?;
        Ok(path)
    }

    /// Read a preset by name.
    pub fn load_preset(&self, name: &str) -> Result<PresetParams, StoreError> {
        let path = self.preset_path(name);
        if !path.exists() {
            return Err(StoreError::NotFound(name.to_string()));
        }
        let text = fs::read_to_string(&path)?;
        let mut preset: PresetParams = toml::from_str(&text)?;
        // A corrupt preset must never take the daemon down (DESIGN.md 9.3).
        preset.sanitize();
        Ok(preset)
    }

    /// Delete a preset.
    pub fn delete_preset(&self, name: &str) -> Result<(), StoreError> {
        let path = self.preset_path(name);
        if !path.exists() {
            return Err(StoreError::NotFound(name.to_string()));
        }
        fs::remove_file(path)?;
        Ok(())
    }

    /// List preset names, sorted.
    pub fn list_presets(&self) -> Result<Vec<String>, StoreError> {
        let dir = self.preset_dir();
        if !dir.exists() {
            return Ok(Vec::new());
        }
        let mut out = Vec::new();
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("toml") {
                continue;
            }
            if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                out.push(stem.to_string());
            }
        }
        out.sort();
        Ok(out)
    }

    /// Save the live configuration.
    pub fn save_config(&self, preset: &PresetParams) -> Result<PathBuf, StoreError> {
        self.ensure_dirs()?;
        let path = self.config_path();
        write_atomic(&path, &toml::to_string_pretty(preset)?)?;
        Ok(path)
    }

    /// Load the live configuration if it exists.
    ///
    /// A damaged file is reported but never fatal: the caller keeps its defaults
    /// and the daemon still starts.
    pub fn load_config(&self) -> Result<Option<PresetParams>, StoreError> {
        let path = self.config_path();
        if !path.exists() {
            return Ok(None);
        }
        let text = fs::read_to_string(&path)?;
        let mut preset: PresetParams = toml::from_str(&text)?;
        preset.sanitize();
        Ok(Some(preset))
    }
}

/// Write `contents` to `path` via a temporary file and a rename.
fn write_atomic(path: &Path, contents: &str) -> Result<(), StoreError> {
    let dir = path.parent().unwrap_or(Path::new("."));
    fs::create_dir_all(dir)?;
    let tmp = dir.join(format!(
        ".{}.tmp-{}",
        path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("noiseforge"),
        std::process::id()
    ));
    {
        let mut file = fs::File::create(&tmp)?;
        file.write_all(contents.as_bytes())?;
        file.sync_all()?;
    }
    fs::rename(&tmp, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use noise_dsp::params::{LayerParams, MasterParams};

    fn temp_store() -> (Store, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        (Store::at(dir.path().join("noiseforge")), dir)
    }

    #[test]
    fn preset_round_trip() {
        let (store, _guard) = temp_store();
        let mut preset = PresetParams::default();
        preset.layers.push(LayerParams {
            name: "Second".into(),
            color: noise_dsp::Color::Brown,
            character: noise_dsp::Character::Intense,
            character_amount: 0.4,
            ..LayerParams::default()
        });
        preset.master = MasterParams {
            target_rms_dbfs: -42.0,
            ..MasterParams::default()
        };
        let path = store.save_preset("Soft Pink", &preset).unwrap();
        assert!(path.exists());
        let back = store.load_preset("Soft Pink").unwrap();
        assert_eq!(back.layers.len(), 2);
        assert_eq!(back.layers[1].color, noise_dsp::Color::Brown);
        assert_eq!(back.master.target_rms_dbfs, -42.0);
        assert_eq!(back.name, "Soft Pink");
        assert_eq!(store.list_presets().unwrap(), vec!["Soft Pink".to_string()]);
    }

    #[test]
    fn saving_is_atomic_and_leaves_no_temporary_files() {
        let (store, _guard) = temp_store();
        store.save_preset("A", &PresetParams::default()).unwrap();
        store.save_preset("A", &PresetParams::default()).unwrap();
        let leftovers: Vec<_> = std::fs::read_dir(store.preset_dir())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().contains(".tmp-"))
            .collect();
        assert!(leftovers.is_empty(), "{leftovers:?}");
    }

    #[test]
    fn a_corrupt_preset_is_repaired_not_fatal() {
        let (store, _guard) = temp_store();
        store.ensure_dirs().unwrap();
        std::fs::write(
            store.preset_path("broken"),
            "schema_version = 1\n[[layers]]\nalpha = -99.0\ncharacter_amount = 12.0\n",
        )
        .unwrap();
        let preset = store.load_preset("broken").unwrap();
        assert_eq!(preset.layers[0].alpha, -4.0);
        assert_eq!(preset.layers[0].character_amount, 1.0);
    }

    #[test]
    fn invalid_toml_is_an_error_not_a_panic() {
        let (store, _guard) = temp_store();
        store.ensure_dirs().unwrap();
        std::fs::write(store.preset_path("garbage"), "this is not = = toml").unwrap();
        assert!(matches!(
            store.load_preset("garbage"),
            Err(StoreError::Toml(_))
        ));
    }

    #[test]
    fn missing_presets_are_reported() {
        let (store, _guard) = temp_store();
        assert!(matches!(
            store.load_preset("nope"),
            Err(StoreError::NotFound(_))
        ));
        assert!(matches!(
            store.delete_preset("nope"),
            Err(StoreError::NotFound(_))
        ));
        assert!(store.list_presets().unwrap().is_empty());
    }

    #[test]
    fn names_cannot_escape_the_preset_directory() {
        let (store, _guard) = temp_store();
        let path = store.save_preset("../../../etc/evil", &PresetParams::default());
        let path = path.unwrap();
        assert!(
            path.starts_with(store.preset_dir()),
            "{} escaped {}",
            path.display(),
            store.preset_dir().display()
        );
        assert!(!path.to_string_lossy().contains(".."));
    }

    #[test]
    fn config_round_trip() {
        let (store, _guard) = temp_store();
        assert!(store.load_config().unwrap().is_none());
        store.save_config(&PresetParams::default()).unwrap();
        let back = store.load_config().unwrap().unwrap();
        assert_eq!(back.schema_version, SCHEMA_VERSION);
    }
}
