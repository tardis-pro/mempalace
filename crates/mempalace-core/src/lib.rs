#![forbid(unsafe_code)]
#![doc = "Core primitives for mempalace: version, config, sanitization, path helpers."]

pub mod config;
pub mod error;
pub mod paths;
pub mod sanitize;

pub const VERSION: &str = "3.1.0";

pub use config::{
    default_hall_keywords, default_topic_wings, Config, FileConfig, DEFAULT_COLLECTION_NAME,
    DEFAULT_TOPIC_WINGS,
};
pub use error::{CoreError, Result, ValidationError};
pub use paths::{
    default_config_dir, default_kg_path, default_palace_path, ensure_private_dir,
    ensure_private_file_perms, home_dir,
};
pub use sanitize::{sanitize_content, sanitize_name, DEFAULT_MAX_CONTENT_LENGTH, MAX_NAME_LENGTH};

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::VERSION;

    #[test]
    fn version_matches_workspace() {
        assert_eq!(VERSION, "3.1.0");
    }

    #[test]
    fn version_matches_workspace_cargo_toml() {
        let body = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../Cargo.toml"),
        )
        .unwrap();
        let version_line = body
            .lines()
            .find(|l| l.trim().starts_with("version") && l.contains('"'))
            .expect("workspace Cargo.toml must have a quoted version line");
        assert!(
            version_line.contains(VERSION),
            "workspace Cargo.toml version `{version_line}` does not contain VERSION `{VERSION}`"
        );
    }
}
