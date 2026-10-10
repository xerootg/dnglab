use std::collections::{HashMap, HashSet};

use crate::{bits::*, rawsource::RawSource};

/// Upper bound for directory entries in one CIFF heap; real files have a few dozen.
const MAX_CIFF_ENTRIES: usize = 4096;

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum CiffTag {
  //Null         = 0x0000,
  ColorInfo1 = 0x0032,
  MakeModel = 0x080a,
  //ShotInfo     = 0x102a,
  ColorInfo2 = 0x102c,
  WhiteBalance = 0x10a9,
  SensorInfo = 0x1031,
  //ImageInfo    = 0x1810,
  DecoderTable = 0x1835,
  //RawData      = 0x2005,
  //SubIFD       = 0x300a,
  //Exif         = 0x300b,
}

fn ct(tag: CiffTag) -> u16 {
  tag as u16
}

#[derive(Debug, Clone)]
pub struct CiffEntry {
  pub tag: u16,
  pub typ: u16,
  pub count: usize,
  pub bytesize: usize,
  pub data_offset: usize,
  pub data: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct CiffIFD {
  entries: HashMap<u16, CiffEntry>,
  subifds: Vec<CiffIFD>,
}

pub fn is_ciff(file: &RawSource) -> bool {
  match file.subview(0, 14) {
    Ok(buf) => buf[6..14] == b"HEAPCCDR"[..],
    Err(_) => false,
  }
}

impl CiffIFD {
  pub fn new_file(file: &RawSource) -> Result<CiffIFD, String> {
    let data = file.as_vec().map_err(|err| format!("Failed to read whole file: {:?}", err))?;

    CiffIFD::new(&data, LEu32(&data, 2) as usize, data.len(), 1)
  }

  pub fn new(buf: &[u8], start: usize, end: usize, depth: u32) -> Result<CiffIFD, String> {
    let mut visited = HashSet::new();
    // Entries copy their data; bound the total so that many entries (or
    // sub-IFDs) pointing at the same large region cannot multiply memory use.
    let mut budget = buf.len().saturating_mul(2).max(1 << 20);
    Self::new_inner(buf, start, end, depth, &mut visited, &mut budget)
  }

  fn new_inner(buf: &[u8], start: usize, end: usize, depth: u32, visited: &mut HashSet<(usize, usize)>, budget: &mut usize) -> Result<CiffIFD, String> {
    let mut entries = HashMap::new();
    let mut subifds = Vec::new();

    if end > buf.len() || end < start.saturating_add(4) {
      return Err(format!("CIFF: heap range {}..{} is outside the file ({} bytes)", start, end, buf.len()));
    }
    if !visited.insert((start, end)) {
      return Err(format!("CIFF: heap {}..{} references itself", start, end));
    }

    let valuedata_size = LEu32(buf, end - 4) as usize;
    let dir_start = start
      .checked_add(valuedata_size)
      .filter(|pos| pos + 2 <= end)
      .ok_or_else(|| format!("CIFF: value data size {} does not fit into heap {}..{}", valuedata_size, start, end))?;
    let dircount = LEu16(buf, dir_start) as usize;
    if dircount > MAX_CIFF_ENTRIES || dir_start + 2 + dircount * 10 > end {
      return Err(format!("CIFF: directory with {} entries does not fit into heap {}..{}", dircount, start, end));
    }

    for i in 0..dircount {
      let entry_offset: usize = dir_start + 2 + i * 10;
      let e = CiffEntry::new(buf, start, entry_offset, budget)?;
      if e.typ == 0x2800 || e.typ == 0x3000 {
        // SubIFDs
        if depth < 10 {
          // Avoid infinite looping IFDs
          let ifd = CiffIFD::new_inner(buf, e.data_offset, e.data_offset + e.bytesize, depth + 1, visited, budget);
          match ifd {
            Ok(val) => {
              subifds.push(val);
            }
            Err(_) => {
              entries.insert(e.tag, e);
            } // Ignore unparsable IFDs
          }
        }
      } else {
        entries.insert(e.tag, e);
      }
    }

    Ok(CiffIFD { entries, subifds })
  }

  pub fn find_entry(&self, tag: CiffTag) -> Option<&CiffEntry> {
    if self.entries.contains_key(&ct(tag)) {
      self.entries.get(&ct(tag))
    } else {
      for ifd in &self.subifds {
        if let Some(x) = ifd.find_entry(tag) {
          return Some(x);
        }
      }
      None
    }
  }
}

impl CiffEntry {
  pub fn new(buf: &[u8], value_data: usize, offset: usize, budget: &mut usize) -> Result<CiffEntry, String> {
    let p = LEu16(buf, offset);
    let tag = p & 0x3fff;
    let datalocation = (p & 0xc000) as usize;
    let typ = p & 0x3800;

    let (bytesize, data_offset) = match datalocation {
      // Data is offset in value_data
      0x0000 => (
        LEu32(buf, offset + 2) as usize,
        (LEu32(buf, offset + 6) as usize)
          .checked_add(value_data)
          .ok_or_else(|| "CIFF: entry data offset overflows".to_string())?,
      ),
      // Data is stored directly in entry
      0x4000 => (8, offset + 2),
      val => return Err(format!("CIFF: Don't know about data location {:x}", val)),
    };
    let data_end = data_offset
      .checked_add(bytesize)
      .filter(|end| *end <= buf.len())
      .ok_or_else(|| format!("CIFF: entry 0x{:x} data {}+{} is outside the file ({} bytes)", tag, data_offset, bytesize, buf.len()))?;
    *budget = budget
      .checked_sub(bytesize)
      .ok_or_else(|| format!("CIFF: entries reference more data than the file can plausibly hold (entry 0x{:x})", tag))?;
    let data = &buf[data_offset..data_end];
    let count = bytesize >> CiffEntry::element_shift(typ);

    Ok(CiffEntry {
      tag,
      typ,
      count,
      bytesize,
      data_offset,
      data: Vec::from(data),
    })
  }

  pub fn element_shift(typ: u16) -> usize {
    match typ {
      // Byte and ASCII
      0x0000 | 0x8000 => 0,
      // Short
      0x1000 => 1,
      // Long, Mix, Sub1 and Sub2
      0x1800 | 0x2000 | 0x2800 | 0x3000 => 2,
      // Default to 0
      _ => 0,
    }
  }

  pub fn get_strings(&self) -> Vec<String> {
    String::from_utf8_lossy(&self.data).split_terminator('\0').map(|x| x.to_string()).collect()
  }

  pub fn get_u32(&self, idx: usize) -> u32 {
    match self.typ {
      0x0000 | 0x8000 => self.data[idx] as u32,
      0x1000 => LEu16(&self.data, idx * 2) as u32,
      0x1800 | 0x2000 | 0x2800 | 0x3000 => LEu32(&self.data, idx * 4),
      _ => panic!("{}", format!("Trying to read typ {} for a u32", self.typ)),
    }
  }

  pub fn get_usize(&self, idx: usize) -> usize {
    self.get_u32(idx) as usize
  }
  pub fn get_f32(&self, idx: usize) -> f32 {
    self.get_u32(idx) as f32
  }
  pub fn get_force_u16(&self, idx: usize) -> u16 {
    LEu16(&self.data, idx * 2)
  }
}

#[cfg(test)]
mod hardening_tests {
  use super::*;

  #[test]
  fn ciff_heap_with_bogus_value_size_is_an_error() {
    // value data size (last 4 bytes) points far outside the buffer.
    let buf = [0u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xFF, 0xFF, 0xFF, 0x7F];
    assert!(CiffIFD::new(&buf, 0, buf.len(), 1).is_err());
  }

  #[test]
  fn ciff_heap_outside_buffer_is_an_error() {
    let buf = [0u8; 16];
    assert!(CiffIFD::new(&buf, 0, 64, 1).is_err());
  }

  #[test]
  fn ciff_entry_pointing_outside_buffer_is_an_error() {
    // Heap: value data = 0 bytes, directory with 1 entry (10 bytes), then value size (4 bytes).
    let mut buf = Vec::new();
    buf.extend_from_slice(&1u16.to_le_bytes()); // dircount
    buf.extend_from_slice(&0x1031u16.to_le_bytes()); // tag, location 0x0000 (in value data)
    buf.extend_from_slice(&0x1000_0000u32.to_le_bytes()); // bytesize
    buf.extend_from_slice(&0u32.to_le_bytes()); // offset
    buf.extend_from_slice(&0u32.to_le_bytes()); // value data size
    assert!(CiffIFD::new(&buf, 0, buf.len(), 1).is_err());
  }
}
