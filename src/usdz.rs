use anyhow::{anyhow, bail, Context, Result};
use std::path::{Component, Path, PathBuf};

#[derive(Debug, Clone)]
pub struct ExtractedTexture {
    pub package_path: String,
    pub path: PathBuf,
}

/// Extract image payloads from a USDZ package into `out_dir`.
///
/// USDZ packages are ZIP containers whose payload entries are stored
/// uncompressed. We parse the central directory directly to avoid adding a
/// runtime dependency just for this import path.
pub fn extract_embedded_textures(
    usdz_path: &Path,
    out_dir: &Path,
) -> Result<Vec<ExtractedTexture>> {
    let bytes = std::fs::read(usdz_path)
        .with_context(|| format!("read USDZ package {}", usdz_path.display()))?;
    let entries = zip_entries(&bytes)?;
    std::fs::create_dir_all(out_dir)
        .with_context(|| format!("create USDZ texture cache {}", out_dir.display()))?;

    let mut extracted = Vec::new();
    for entry in entries {
        if !is_texture_name(&entry.name) {
            continue;
        }
        if entry.method != 0 {
            log::warn!(
                "USDZ texture entry {} uses unsupported ZIP method {}; skipping",
                entry.name,
                entry.method
            );
            continue;
        }
        let Some(rel) = safe_relative_path(&entry.name) else {
            log::warn!(
                "USDZ texture entry {} has an unsafe path; skipping",
                entry.name
            );
            continue;
        };
        let out_path = out_dir.join(rel);
        if let Some(parent) = out_path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create {}", parent.display()))?;
        }
        std::fs::write(&out_path, entry.data)
            .with_context(|| format!("extract {}", out_path.display()))?;
        extracted.push(ExtractedTexture {
            package_path: normalize_package_path(&entry.name),
            path: out_path,
        });
    }
    Ok(extracted)
}

pub fn normalize_package_path(path: &str) -> String {
    path.replace('\\', "/").trim_start_matches("./").to_string()
}

fn is_texture_name(name: &str) -> bool {
    let Some(ext) = Path::new(name).extension().and_then(|e| e.to_str()) else {
        return false;
    };
    matches!(
        ext.to_ascii_lowercase().as_str(),
        "png" | "jpg" | "jpeg" | "exr" | "hdr" | "tif" | "tiff"
    )
}

fn safe_relative_path(name: &str) -> Option<PathBuf> {
    let mut out = PathBuf::new();
    for comp in Path::new(name).components() {
        match comp {
            Component::Normal(part) => out.push(part),
            Component::CurDir => {}
            _ => return None,
        }
    }
    if out.as_os_str().is_empty() {
        None
    } else {
        Some(out)
    }
}

struct ZipEntry<'a> {
    name: String,
    method: u16,
    data: &'a [u8],
}

fn zip_entries(bytes: &[u8]) -> Result<Vec<ZipEntry<'_>>> {
    let eocd = find_eocd(bytes).ok_or_else(|| anyhow!("USDZ missing ZIP end-of-directory"))?;
    let entry_count = read_u16(bytes, eocd + 10)? as usize;
    let cd_size = read_u32(bytes, eocd + 12)? as usize;
    let cd_offset = read_u32(bytes, eocd + 16)? as usize;
    let cd_end = cd_offset
        .checked_add(cd_size)
        .filter(|&end| end <= bytes.len())
        .ok_or_else(|| anyhow!("USDZ central directory is out of range"))?;

    let mut entries = Vec::with_capacity(entry_count);
    let mut pos = cd_offset;
    while pos < cd_end {
        if read_u32(bytes, pos)? != 0x0201_4b50 {
            bail!("USDZ central directory has an invalid header");
        }
        let method = read_u16(bytes, pos + 10)?;
        let compressed_size = read_u32(bytes, pos + 20)? as usize;
        let name_len = read_u16(bytes, pos + 28)? as usize;
        let extra_len = read_u16(bytes, pos + 30)? as usize;
        let comment_len = read_u16(bytes, pos + 32)? as usize;
        let local_offset = read_u32(bytes, pos + 42)? as usize;
        let name_start = pos + 46;
        let name_end = name_start
            .checked_add(name_len)
            .filter(|&end| end <= bytes.len())
            .ok_or_else(|| anyhow!("USDZ entry name is out of range"))?;
        let name = String::from_utf8_lossy(&bytes[name_start..name_end]).into_owned();

        if read_u32(bytes, local_offset)? != 0x0403_4b50 {
            bail!("USDZ local file header for {name} is invalid");
        }
        let local_name_len = read_u16(bytes, local_offset + 26)? as usize;
        let local_extra_len = read_u16(bytes, local_offset + 28)? as usize;
        let data_start = local_offset
            .checked_add(30)
            .and_then(|v| v.checked_add(local_name_len))
            .and_then(|v| v.checked_add(local_extra_len))
            .ok_or_else(|| anyhow!("USDZ local header for {name} is out of range"))?;
        let data_end = data_start
            .checked_add(compressed_size)
            .filter(|&end| end <= bytes.len())
            .ok_or_else(|| anyhow!("USDZ entry {name} data is out of range"))?;
        entries.push(ZipEntry {
            name,
            method,
            data: &bytes[data_start..data_end],
        });

        pos = name_end
            .checked_add(extra_len)
            .and_then(|v| v.checked_add(comment_len))
            .ok_or_else(|| anyhow!("USDZ central directory entry is out of range"))?;
    }
    Ok(entries)
}

fn find_eocd(bytes: &[u8]) -> Option<usize> {
    if bytes.len() < 22 {
        return None;
    }
    let min = bytes.len().saturating_sub(22 + 65_535);
    (min..=bytes.len() - 22)
        .rev()
        .find(|&i| bytes.get(i..i + 4) == Some(&[0x50, 0x4b, 0x05, 0x06]))
}

fn read_u16(bytes: &[u8], offset: usize) -> Result<u16> {
    let raw = bytes
        .get(offset..offset + 2)
        .ok_or_else(|| anyhow!("unexpected end of USDZ while reading u16"))?;
    Ok(u16::from_le_bytes([raw[0], raw[1]]))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32> {
    let raw = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| anyhow!("unexpected end of USDZ while reading u32"))?;
    Ok(u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]))
}
