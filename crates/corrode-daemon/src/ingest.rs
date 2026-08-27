//! Reference-doc ingestion for the doc GraphRAG (`AgentCommand::DocIngest`).
//!
//! docling.rs (vendored, `third_party/docling.rs`, MIT) converts a document
//! (PDF text layer, DOCX/HTML/MD/XLSX/PPTX/…) into a `DoclingDocument`; the
//! tokenizer-free `HierarchicalChunker` splits it into heading-contextualized
//! chunks. The caller (daemon) embeds the chunks and writes doc/chunk nodes
//! into the graph store — this module only converts + chunks, so it stays
//! sync/CPU and free of store/network concerns.
//!
//! ponytail: `pdf-text` feature only — pure-Rust text-layer PDF, no pdfium/ONNX.
//! Scanned-PDF layout/OCR is deliberately absent: the plan is hipfire-served
//! vision models (docling's remote-VLM path), not an in-process ort runtime.

use docling::chunker::{contextualize, HierarchicalChunker};
use docling::{DocumentConverter, SourceDocument};

/// One converted document: a stable doc id, its title, and the chunk texts
/// (already contextualized with their heading path) with stable chunk ids.
pub struct IngestedDoc {
    pub doc_id: String,
    pub title: String,
    pub chunks: Vec<(String, String)>, // (chunk_id, text)
}

/// Convert + chunk one file. Ids are derived from the path so re-ingesting the
/// same file upserts rather than duplicates.
pub fn ingest(path: &str) -> anyhow::Result<IngestedDoc> {
    let source =
        SourceDocument::from_file(path).map_err(|e| anyhow::anyhow!("load {path}: {e}"))?;
    let result = DocumentConverter::new()
        .convert(source)
        .map_err(|e| anyhow::anyhow!("convert {path}: {e}"))?;

    let doc_id = format!("doc:{path}");
    let title = std::path::Path::new(path)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_string());

    let chunks = HierarchicalChunker
        .chunk(&result.document)
        .iter()
        .map(contextualize)
        .filter(|text| !text.trim().is_empty())
        .enumerate()
        .map(|(i, text)| (format!("chunk:{path}#{i}"), text))
        .collect();

    Ok(IngestedDoc { doc_id, title, chunks })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn markdown_file_converts_to_contextualized_chunks() {
        let dir = std::env::temp_dir().join(format!("corrode-ingest-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("guide.md");
        std::fs::write(
            &file,
            "# CPU Reference\n\n## Registers\n\nR0 through R15 are general purpose.\n\n## Interrupts\n\nVector table lives at 0x0000.\n",
        )
        .unwrap();

        let doc = ingest(file.to_str().unwrap()).expect("markdown ingests");
        assert_eq!(doc.title, "guide.md");
        assert!(doc.doc_id.starts_with("doc:"));
        assert!(!doc.chunks.is_empty(), "chunker produced nothing");
        let all = doc
            .chunks
            .iter()
            .map(|(_, t)| t.as_str())
            .collect::<Vec<_>>()
            .join("\n---\n");
        // content survives, and the heading path is folded into the chunk text
        assert!(all.contains("general purpose"), "chunk text lost: {all}");
        assert!(all.contains("Registers"), "heading context missing: {all}");
        // ids are stable + unique
        let mut ids: Vec<_> = doc.chunks.iter().map(|(id, _)| id.clone()).collect();
        ids.dedup();
        assert_eq!(ids.len(), doc.chunks.len());

        std::fs::remove_dir_all(&dir).ok();
    }
}
