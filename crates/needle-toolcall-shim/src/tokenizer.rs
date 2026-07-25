use anyhow::{Context, Result};
use sentencepiece_rs::SentencePieceProcessor;
use std::path::Path;

pub const PAD_ID: u32 = 0;
pub const EOS_ID: u32 = 1;
pub const TOOL_CALL_ID: u32 = 4;
pub const TOOLS_ID: u32 = 5;

#[derive(Clone, Debug)]
pub struct NeedleTokenizer {
    sp: SentencePieceProcessor,
}

impl NeedleTokenizer {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let sp = SentencePieceProcessor::open(path.as_ref())
            .with_context(|| format!("loading tokenizer {}", path.as_ref().display()))?;
        Ok(Self { sp })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.sp
            .encode_to_ids(text)
            .context("encoding with sentencepiece")
            .map(|ids| ids.into_iter().map(|id| id as u32).collect())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        let ids = ids.iter().map(|id| *id as usize).collect::<Vec<_>>();
        self.sp
            .decode_ids(&ids)
            .context("decoding with sentencepiece")
    }

    pub fn vocab_size(&self) -> usize {
        self.sp.model().vocab_size()
    }

    pub fn token_piece(&self, id: usize) -> Result<&str> {
        self.sp
            .model()
            .id_to_piece(id)
            .map_err(|e| anyhow::anyhow!(e.to_string()))
    }

    pub fn token_text(&self, id: usize) -> Result<String> {
        let model = self.sp.model();
        if model.is_control(id) || model.is_unknown(id) {
            return Ok(String::new());
        }
        let piece = model
            .id_to_piece(id)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        if model.is_byte(id) {
            return decode_byte_piece(piece);
        }
        Ok(piece.replace('\u{2581}', " "))
    }

    pub fn decode_single_token_for_display(&self, id: u32) -> Result<String> {
        self.decode(&[id])
    }
}

fn decode_byte_piece(piece: &str) -> Result<String> {
    let hex = piece
        .strip_prefix("<0x")
        .and_then(|s| s.strip_suffix('>'))
        .ok_or_else(|| anyhow::anyhow!("invalid byte piece {piece}"))?;
    let byte =
        u8::from_str_radix(hex, 16).with_context(|| format!("invalid byte piece {piece}"))?;
    Ok(String::from_utf8_lossy(&[byte]).to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizer_special_ids_match_needle() -> Result<()> {
        let path = "needle-weights/tokenizer/needle.model";
        if !std::path::Path::new(path).exists() {
            eprintln!("skipping tokenizer test; local tokenizer is missing");
            return Ok(());
        }
        let tok = NeedleTokenizer::load(path)?;
        assert_eq!(tok.vocab_size(), 8192);
        assert_eq!(
            tok.encode(r#"[{"name":"get_weather"}]"#)?[..5],
            [356, 294, 264, 358, 8062]
        );
        assert_eq!(tok.token_text(356)?, " [{\"");
        assert_eq!(tok.token_text(4)?, "<tool_call>");
        assert_eq!(PAD_ID, 0);
        assert_eq!(EOS_ID, 1);
        assert_eq!(TOOL_CALL_ID, 4);
        assert_eq!(TOOLS_ID, 5);
        Ok(())
    }
}
