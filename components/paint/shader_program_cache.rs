/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! Disposable diagnostic program-binary cache. All filesystem work stays on one
//! bounded, exclusively locked IO worker. This is not a cold-start solution.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender, TryRecvError};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use gleam::gl;
use serde::de::{self, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use sha2::{Digest, Sha256};
use webrender::{ProgramBinary, ProgramCache, ProgramCacheObserver};

const SCHEMA: u32 = 1;
const WR_REVISION: &str = "82affc7e81013b6aec09ad43328d53710a77facd";
const MAX_FILE: usize = 16 * 1024 * 1024;
const MAX_BINARY: usize = 1024 * 1024;
const MAX_TOTAL_BINARY: usize = 4 * 1024 * 1024;
const MAX_RECORDS: usize = 64;
const MAX_TEXT: usize = 2048;
const QUEUE_CAPACITY: usize = 8;
static SETTINGS: OnceLock<Option<PathBuf>> = OnceLock::new();
// This prototype owns one cache/worker per fresh process, not one per Play.
static CLAIMED: AtomicBool = AtomicBool::new(false);

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(transparent)]
struct Text(String);
impl<'de> Deserialize<'de> for Text {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct TextVisitor;
        impl Visitor<'_> for TextVisitor {
            type Value = Text;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a bounded cache string")
            }
            fn visit_str<E: de::Error>(self, s: &str) -> Result<Text, E> {
                if s.len() > MAX_TEXT {
                    return Err(E::custom("cache string limit"));
                }
                Ok(Text(s.to_owned()))
            }
        }
        d.deserialize_str(TextVisitor)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Fingerprint {
    schema: u32,
    wr: Text,
    // Required compile-time hash of actual host/WR/source/feature layout.
    build: Text,
    vendor: Text,
    renderer: Text,
    version: Text,
    shading_language: Text,
}

#[derive(Clone, Serialize)]
#[serde(transparent)]
struct Bytes(Vec<u8>);
impl<'de> Deserialize<'de> for Bytes {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct BytesVisitor;
        impl<'de> Visitor<'de> for BytesVisitor {
            type Value = Bytes;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("at most MAX_BINARY byte values")
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Bytes, A::Error> {
                if seq.size_hint().is_some_and(|n| n > MAX_BINARY) {
                    return Err(de::Error::custom("binary limit"));
                }
                let mut bytes = Vec::new();
                while let Some(byte) = seq.next_element::<u8>()? {
                    if bytes.len() == MAX_BINARY {
                        return Err(de::Error::custom("binary limit"));
                    }
                    bytes.push(byte);
                }
                Ok(Bytes(bytes))
            }
        }
        d.deserialize_seq(BytesVisitor)
    }
}

// Exact serde representation of the pinned WR ProgramBinary. Unknown/missing
// fields or a different layout fail closed; no C/private-map access is used.
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireBinary {
    bytes: Bytes,
    format: u32,
    source_digest: u64,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Record {
    binary: WireBinary,
    sha256: Text,
}
#[derive(Default, Serialize)]
#[serde(transparent)]
struct Records(Vec<Record>);
impl<'de> Deserialize<'de> for Records {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct RecordsVisitor;
        impl<'de> Visitor<'de> for RecordsVisitor {
            type Value = Records;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("bounded cache records")
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Records, A::Error> {
                if seq.size_hint().is_some_and(|n| n > MAX_RECORDS) {
                    return Err(de::Error::custom("record count limit"));
                }
                let mut out = Vec::new();
                let mut total = 0usize;
                while let Some(record) = seq.next_element::<Record>()? {
                    total = total
                        .checked_add(record.binary.bytes.0.len())
                        .ok_or_else(|| de::Error::custom("binary total overflow"))?;
                    if out.len() == MAX_RECORDS || total > MAX_TOTAL_BINARY {
                        return Err(de::Error::custom("cache total limit"));
                    }
                    out.push(record);
                }
                Ok(Records(out))
            }
        }
        d.deserialize_seq(RecordsVisitor)
    }
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Store {
    fingerprint: Fingerprint,
    records: Records,
}

struct CappedBuffer(Vec<u8>, usize);
impl Write for CappedBuffer {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() > self.1.saturating_sub(self.0.len()) {
            return Err(io::Error::other("cache serialization limit"));
        }
        self.0.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
fn encode<T: Serialize>(value: &T, cap: usize) -> Option<Vec<u8>> {
    let mut out = CappedBuffer(Vec::new(), cap);
    serde_json::to_writer(&mut out, value).ok()?;
    Some(out.0)
}
fn checksum(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
fn key(binary: &WireBinary) -> String {
    format!("{:02x}", binary.source_digest)
}
fn make_record(binary: &ProgramBinary) -> Option<Record> {
    // Both JSON passes are bounded and occur only on IO worker.
    let encoded = encode(binary, MAX_BINARY * 4 + 128)?;
    let wire: WireBinary = serde_json::from_slice(&encoded).ok()?;
    if wire.bytes.0.is_empty() || key(&wire) != binary.source_digest().to_string() {
        return None;
    }
    Some(Record {
        sha256: Text(checksum(&encode(&wire, MAX_BINARY * 4 + 128)?)),
        binary: wire,
    })
}
fn validate(store: Store, expected: &Fingerprint, formats: &[u32]) -> Option<Vec<Record>> {
    if store.fingerprint != *expected {
        return None;
    }
    let mut seen = BTreeSet::new();
    for record in &store.records.0 {
        if record.binary.bytes.0.is_empty()
            || !formats.contains(&record.binary.format)
            || !seen.insert(key(&record.binary))
            || checksum(&encode(&record.binary, MAX_BINARY * 4 + 128)?) != record.sha256.0
        {
            return None;
        }
    }
    Some(store.records.0)
}

#[derive(Default)]
struct Shared {
    quarantine: AtomicBool,
    failed_digest: AtomicU64,
    closed: AtomicBool,
}
impl Shared {
    fn quarantined(&self) -> bool {
        self.quarantine.load(Ordering::Acquire)
    }
}
enum Command {
    Save(Vec<Arc<ProgramBinary>>),
    Wake,
}
struct Loaded {
    programs: Vec<Arc<ProgramBinary>>,
}
struct Observer {
    tx: SyncSender<Command>,
    shared: Arc<Shared>,
    imported: Rc<RefCell<BTreeSet<String>>>,
}
impl ProgramCacheObserver for Observer {
    fn save_shaders_to_disk(&self, mut entries: Vec<Arc<ProgramBinary>>) {
        // Owner-side bounded bookkeeping only; no bytes, IO, wait or map borrow.
        if self.shared.quarantined() {
            return;
        }
        entries.truncate(MAX_RECORDS);
        let _ = self.tx.try_send(Command::Save(entries));
    }
    fn set_startup_shaders(&self, entries: Vec<Arc<ProgramBinary>>) {
        if self.shared.quarantined() {
            return;
        }
        // WR reports only linked cache entries at its frame10. Because import
        // preceded ALL Program creation, an imported-and-linked entry proves a
        // successful binary link (rejection quarantines this entire observer).
        // This is distinct startup shaders, not paints or exhaustive later hits.
        let linked = entries
            .iter()
            .filter(|binary| {
                self.imported
                    .borrow()
                    .contains(&binary.source_digest().to_string())
            })
            .count();
        log::info!("[Loom shader cache] startup_linked_imports={linked}");
    }
    fn try_load_shader_from_disk(
        &self,
        _digest: &webrender::ProgramSourceDigest,
        _cache: &Rc<ProgramCache>,
    ) {
        // Misses are always nonblocking. Completed imports happen outside all
        // WR map/observer borrows, before a later owner renderer entry.
    }
    fn notify_program_binary_failed(&self, binary: &Arc<ProgramBinary>) {
        // WR holds entries.borrow_mut here. Do not access that map or any IO lock.
        // Digest recording is bounded; any diagnostic log happens on IO worker.
        if !self.shared.quarantined() {
            let digest = u64::from_str_radix(&binary.source_digest().to_string(), 16).unwrap_or(0);
            self.shared.failed_digest.store(digest, Ordering::Relaxed);
        }
        self.shared.quarantine.store(true, Ordering::Release);
        let _ = self.tx.try_send(Command::Wake);
    }
}

// Initial diagnostic prototype is GLES3-only. Native also supports desktop
// GL3.2, where binary enums need extra capability proof; never query them there.
fn supports_binary_queries(kind: gl::GlType, version: &str) -> bool {
    matches!(kind, gl::GlType::Gles)
        && version
            .strip_prefix("OpenGL ES ")
            .is_some_and(|v| v.starts_with("3."))
}

/// Consuming diagnostic observation, deliberately independent of cache enablement
/// so disabled/empty/hit/rejection arms can use the same probe. Any observed
/// nonzero error is fatal evidence, never a rendering/FIFO decision or a fix.
pub(crate) fn gl_error_probe_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("LOOM_PROGRAM_CACHE_GL_DIAGNOSTICS")
            .ok()
            .as_deref()
            == Some("1")
            && std::env::var("LOOM_PACING_ORACLE").ok().as_deref() == Some("1")
            && std::env::var("LOOM_PACING_PROFILE").ok().is_some_and(|p| {
                matches!(
                    p.as_str(),
                    "app"
                        | "oracle-retained-backgrounds"
                        | "oracle-selected-highlight"
                        | "oracle-certified-backgrounds"
                )
            })
    })
}

// Called only after new() identifies the current owner context as GLES 3+.
// GL capability queries synchronously fill caller-provided buffers. Gleam marks
// the generic query unsafe because its output length depends on the enum.
#[allow(unsafe_code)]
fn query_binary_formats(gl: &dyn gl::Gl) -> Option<Vec<u32>> {
    let mut count = [0];
    // SAFETY: NUM_PROGRAM_BINARY_FORMATS writes exactly one GLint in GLES 3.
    unsafe {
        gl.get_integer_v(gl::NUM_PROGRAM_BINARY_FORMATS, &mut count);
    }
    if !(1..=32).contains(&count[0]) {
        return None;
    }
    let mut formats = vec![0; count[0] as usize];
    // SAFETY: PROGRAM_BINARY_FORMATS writes exactly the count just queried on
    // this same owner/current context. The bounded vector has that many slots.
    unsafe {
        gl.get_integer_v(gl::PROGRAM_BINARY_FORMATS, &mut formats);
    }
    Some(formats.into_iter().map(|value| value as u32).collect())
}

/// Owner-thread half. Default disabled: no clocks, filesystem calls or GL query.
pub(crate) struct ShaderProgramCache {
    cache: Rc<ProgramCache>,
    rx: Option<Receiver<Loaded>>,
    shared: Arc<Shared>,
    imported: Rc<RefCell<BTreeSet<String>>>,
}
impl ShaderProgramCache {
    pub(crate) fn new(gl: &dyn gl::Gl) -> Option<Self> {
        let directory = SETTINGS
            .get_or_init(|| {
                let profile = std::env::var("LOOM_PACING_PROFILE").ok()?;
                if std::env::var("LOOM_PROGRAM_BINARY_CACHE").ok().as_deref() != Some("1")
                    || std::env::var("LOOM_PACING_ORACLE").ok().as_deref() != Some("1")
                    || !matches!(
                        profile.as_str(),
                        "app"
                            | "oracle-retained-backgrounds"
                            | "oracle-selected-highlight"
                            | "oracle-certified-backgrounds"
                    )
                {
                    return None;
                }
                let path = PathBuf::from(std::env::var_os("LOOM_PROGRAM_BINARY_CACHE_DIR")?);
                path.is_absolute().then_some(path)
            })
            .as_ref()?
            .clone();
        let build = option_env!("LOOM_PROGRAM_CACHE_BUILD_SHA256")?;
        if build.len() != 64 || !build.bytes().all(|b| b.is_ascii_hexdigit()) {
            return None;
        }
        let fingerprint = Fingerprint {
            schema: SCHEMA,
            wr: Text(WR_REVISION.into()),
            build: Text(build.into()),
            vendor: Text(gl.get_string(gl::VENDOR)),
            renderer: Text(gl.get_string(gl::RENDERER)),
            version: Text(gl.get_string(gl::VERSION)),
            shading_language: Text(gl.get_string(gl::SHADING_LANGUAGE_VERSION)),
        };
        if [
            &fingerprint.vendor,
            &fingerprint.renderer,
            &fingerprint.version,
            &fingerprint.shading_language,
        ]
        .iter()
        .any(|v| v.0.is_empty() || v.0.len() > MAX_TEXT)
        {
            return None;
        }
        if !supports_binary_queries(gl.get_type(), &fingerprint.version.0) {
            return None;
        }
        let mut formats = query_binary_formats(gl)?;
        formats.sort_unstable();
        formats.dedup();
        if CLAIMED.swap(true, Ordering::AcqRel) {
            return None;
        }
        let shared = Arc::new(Shared::default());
        let (tx, commands) = mpsc::sync_channel(QUEUE_CAPACITY);
        let (loaded, rx) = mpsc::sync_channel(1);
        let worker_shared = shared.clone();
        std::thread::Builder::new()
            .name("WRProgramCacheIO".into())
            .spawn(move || {
                run_worker(
                    directory,
                    fingerprint,
                    formats,
                    commands,
                    loaded,
                    worker_shared,
                );
            })
            .ok()?;
        let imported = Rc::new(RefCell::new(BTreeSet::new()));
        let cache = ProgramCache::new(Some(Box::new(Observer {
            tx,
            shared: shared.clone(),
            imported: imported.clone(),
        })));
        Some(Self {
            cache,
            rx: Some(rx),
            shared,
            imported,
        })
    }
    pub(crate) fn cache(&self) -> Rc<ProgramCache> {
        self.cache.clone()
    }
    pub(crate) fn import_before_first_program(&mut self) {
        // Exactly one opportunity, before first update/render. A Program created
        // before import would not have received glProgramBinary at creation;
        // late insertion is therefore unsafe with the pinned WR public API.
        let Some(rx) = self.rx.take() else {
            return;
        };
        if self.shared.quarantined() {
            return;
        }
        match rx.try_recv() {
            Ok(loaded) => {
                for binary in loaded.programs {
                    if self.shared.quarantined() {
                        break;
                    }
                    self.imported
                        .borrow_mut()
                        .insert(binary.source_digest().to_string());
                    self.cache.load_program_binary(binary);
                }
            },
            Err(TryRecvError::Empty | TryRecvError::Disconnected) => {},
        }
        log::info!(
            "[Loom shader cache] imported_before_first_program={}",
            self.imported.borrow().len()
        );
        // Empty means cold/no import this renderer; drop receiver and never wait.
    }
}
impl Drop for ShaderProgramCache {
    fn drop(&mut self) {
        // No join/flush/disk wait on the rendering owner. Worker owns its data.
        self.shared.closed.store(true, Ordering::Release);
    }
}

fn run_worker(
    directory: PathBuf,
    fingerprint: Fingerprint,
    formats: Vec<u32>,
    commands: Receiver<Command>,
    loaded: SyncSender<Loaded>,
    shared: Arc<Shared>,
) {
    let Some(identity) = encode(&fingerprint, 16 * 1024).map(|v| checksum(&v)) else {
        return;
    };
    if fs::create_dir_all(&directory).is_err() {
        return;
    }
    // Dedicated diagnostic directory. Bound even namespace growth; do not delete
    // unrelated files or recursively scan arbitrary user directories.
    let Ok(entries) = fs::read_dir(&directory) else {
        return;
    };
    if entries.take(33).count() > 32 {
        return;
    }
    let Ok(lock) = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(directory.join(format!("{identity}.lock")))
    else {
        return;
    };
    // OS-owned advisory lock releases on crash; no create_new stale sentinel.
    // Another process disables this IO cache, never waits or competes with saves.
    if lock.try_lock().is_err() {
        log::info!("[Loom shader cache] IO disabled: namespace already locked");
        return;
    }
    let path = directory.join(format!("{identity}.json"));
    let temporary = directory.join(format!("{identity}.tmp"));
    let mut records = BTreeMap::<String, Record>::new();
    if let Ok(mut file) = File::open(&path) {
        if file
            .metadata()
            .is_ok_and(|m| m.is_file() && m.len() <= MAX_FILE as u64)
        {
            let mut bytes = Vec::new();
            if (&mut file)
                .take(MAX_FILE as u64 + 1)
                .read_to_end(&mut bytes)
                .is_ok()
                && bytes.len() <= MAX_FILE
            {
                if let Ok(store) = serde_json::from_slice::<Store>(&bytes) {
                    if let Some(valid) = validate(store, &fingerprint, &formats) {
                        for record in valid {
                            records.insert(key(&record.binary), record);
                        }
                    }
                }
            }
        }
    }
    let mut programs = Vec::new();
    for record in records.values() {
        if let Some(bytes) = encode(&record.binary, MAX_BINARY * 4 + 128) {
            if let Ok(binary) = serde_json::from_slice::<ProgramBinary>(&bytes) {
                programs.push(Arc::new(binary));
            }
        }
    }
    log::info!(
        "[Loom shader cache] validated_disk_programs={}",
        programs.len()
    );
    let _ = loaded.try_send(Loaded { programs });
    let mut dirty = false;
    let mut export_ordinal = 0usize;
    loop {
        if shared.quarantined() {
            // Terminal for this worker. No queued/delayed Save is handled after
            // this invalidation; the exclusive lock remains held through exit.
            log::info!(
                "[Loom shader cache] quarantined_driver_digest={:02x}",
                shared.failed_digest.load(Ordering::Relaxed)
            );
            remove_quarantined_file(&path);
            remove_quarantined_file(&temporary);
            return;
        }
        match commands.recv_timeout(Duration::from_millis(100)) {
            Ok(Command::Save(entries)) => {
                for entry in entries {
                    let Some(record) = make_record(&entry) else {
                        continue;
                    };
                    if !formats.contains(&record.binary.format) {
                        continue;
                    }
                    let digest = key(&record.binary);
                    if records.len() >= MAX_RECORDS && !records.contains_key(&digest) {
                        continue;
                    }
                    let total: usize = records.values().map(|r| r.binary.bytes.0.len()).sum();
                    let old = records.get(&digest).map_or(0, |r| r.binary.bytes.0.len());
                    if total - old + record.binary.bytes.0.len() > MAX_TOTAL_BINARY {
                        continue;
                    }
                    if export_ordinal < MAX_RECORDS {
                        export_ordinal += 1;
                        log::info!(
                            "[Loom shader cache] collected_ordinal={export_ordinal} collected_digest={digest} format={}",
                            record.binary.format
                        );
                    }
                    records.insert(digest, record);
                    dirty = true;
                }
            },
            Ok(Command::Wake) => {},
            Err(mpsc::RecvTimeoutError::Timeout) => {},
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
        if shared.quarantined() {
            continue;
        }
        // Flush one bounded Save bundle, never a synchronous owner operation.
        if dirty {
            let store = Store {
                fingerprint: fingerprint.clone(),
                records: Records(records.values().cloned().collect()),
            };
            if let Some(bytes) = encode(&store, MAX_FILE) {
                if let Ok(mut out) = OpenOptions::new()
                    .write(true)
                    .create(true)
                    .truncate(true)
                    .open(&temporary)
                {
                    if out.write_all(&bytes).is_ok()
                        && out.sync_all().is_ok()
                        && !shared.quarantined()
                    {
                        drop(out);
                        // Windows rename cannot replace an existing file. Removing
                        // old first may lose the cache on crash, never valid pixels.
                        let _ = fs::remove_file(&path);
                        let _ = fs::rename(&temporary, &path);
                    }
                }
            }
            dirty = false;
        }
        // Quarantine is rechecked after any in-flight write, before exit or the
        // next command, so a save cannot resurrect data AFTER invalidation.
        if shared.quarantined() {
            continue;
        }
        if shared.closed.load(Ordering::Acquire) {
            break;
        }
    }
    if shared.quarantined() {
        remove_quarantined_file(&path);
        remove_quarantined_file(&temporary);
    }
}
fn remove_quarantined_file(path: &std::path::Path) {
    if let Err(error) = fs::remove_file(path) {
        if error.kind() != io::ErrorKind::NotFound {
            log::warn!("[Loom shader cache] quarantine invalidation failed: {error}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;

    fn fingerprint() -> Fingerprint {
        Fingerprint {
            schema: SCHEMA,
            wr: Text(WR_REVISION.into()),
            build: Text("a".repeat(64)),
            vendor: Text("vendor".into()),
            renderer: Text("gpu-driver".into()),
            version: Text("ANGLE-exact-version".into()),
            shading_language: Text("ESSL3".into()),
        }
    }
    fn program(digest: u64) -> Arc<ProgramBinary> {
        let wire = WireBinary {
            bytes: Bytes(vec![1, 2, 3]),
            format: 7,
            source_digest: digest,
        };
        Arc::new(serde_json::from_slice(&encode(&wire, 1024).unwrap()).unwrap())
    }
    struct Directory(PathBuf);
    impl Directory {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(1);
            let path = std::env::temp_dir().join(format!(
                "loom-program-cache-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
        fn store(&self) -> PathBuf {
            self.0.join(format!(
                "{}.json",
                checksum(&encode(&fingerprint(), 16384).unwrap())
            ))
        }
    }
    impl Drop for Directory {
        fn drop(&mut self) {
            // Nonrecursive; this newly created test directory owns only files.
            for file in fs::read_dir(&self.0).unwrap().flatten() {
                if file.file_type().unwrap().is_file() {
                    let _ = fs::remove_file(file.path());
                }
            }
            let _ = fs::remove_dir(&self.0);
        }
    }
    fn save_one(directory: &Directory) {
        let (tx, rx) = mpsc::sync_channel(1);
        tx.send(Command::Save(vec![program(11)])).unwrap();
        let (loaded, _result) = mpsc::sync_channel(1);
        let shared = Arc::new(Shared::default());
        shared.closed.store(true, Ordering::Release);
        run_worker(
            directory.0.clone(),
            fingerprint(),
            vec![7],
            rx,
            loaded,
            shared,
        );
    }

    #[test]
    fn binary_query_admission_rejects_desktop_and_old_gles() {
        assert!(supports_binary_queries(
            gl::GlType::Gles,
            "OpenGL ES 3.0.0 (ANGLE exact-build)"
        ));
        assert!(!supports_binary_queries(gl::GlType::Gl, "3.2"));
        assert!(!supports_binary_queries(gl::GlType::Gl, "4.6"));
        assert!(!supports_binary_queries(gl::GlType::Gles, "OpenGL ES 2.0"));
        assert!(!supports_binary_queries(gl::GlType::Gles, "unknown"));
    }

    #[test]
    fn actual_program_binary_serde_roundtrip_and_checksum_are_pin_locked() {
        let source = program(11);
        let record = make_record(&source).unwrap();
        assert_eq!(record.binary.format, 7);
        assert_eq!(record.binary.bytes.0, [1, 2, 3]);
        assert_eq!(key(&record.binary), source.source_digest().to_string());
        let store = Store {
            fingerprint: fingerprint(),
            records: Records(vec![record.clone()]),
        };
        assert_eq!(validate(store, &fingerprint(), &[7]).unwrap().len(), 1);
        let mut bad = record;
        bad.binary.bytes.0[0] ^= 1;
        assert!(
            validate(
                Store {
                    fingerprint: fingerprint(),
                    records: Records(vec![bad])
                },
                &fingerprint(),
                &[7]
            )
            .is_none()
        );
    }

    #[test]
    fn decoding_rejects_byte_record_total_and_output_limits() {
        let bytes = format!("[{}]", "0,".repeat(MAX_BINARY) + "0");
        assert!(serde_json::from_str::<Bytes>(&bytes).is_err());
        let record = make_record(&program(11)).unwrap();
        let records = Records(vec![record.clone(); MAX_RECORDS + 1]);
        assert!(serde_json::from_slice::<Records>(&encode(&records, MAX_FILE).unwrap()).is_err());
        let mut large = record;
        large.binary.bytes = Bytes(vec![0; MAX_BINARY]);
        let records = Records(vec![large; MAX_TOTAL_BINARY / MAX_BINARY + 1]);
        let input = encode(&records, MAX_FILE).unwrap();
        assert!(serde_json::from_slice::<Records>(&input).is_err());
        assert!(encode(&program(11), 2).is_none());
    }

    #[test]
    fn fingerprint_and_supported_format_mismatch_cannot_import() {
        for changed in 0..4 {
            let mut wrong = fingerprint();
            match changed {
                0 => wrong.build = Text("b".repeat(64)),
                1 => wrong.renderer = Text("other-driver".into()),
                2 => wrong.version = Text("other-ANGLE".into()),
                _ => wrong.schema += 1,
            }
            let store = Store {
                fingerprint: wrong,
                records: Records(vec![make_record(&program(11)).unwrap()]),
            };
            assert!(validate(store, &fingerprint(), &[7]).is_none());
        }
        let store = Store {
            fingerprint: fingerprint(),
            records: Records(vec![make_record(&program(11)).unwrap()]),
        };
        assert!(validate(store, &fingerprint(), &[8]).is_none());
    }

    #[test]
    fn real_worker_saves_then_loads_the_actual_program_binary() {
        let directory = Directory::new();
        save_one(&directory);
        assert!(directory.store().is_file());
        let (tx, commands) = mpsc::sync_channel(1);
        tx.send(Command::Wake).unwrap();
        let (loaded, result) = mpsc::sync_channel(1);
        let shared = Arc::new(Shared::default());
        shared.closed.store(true, Ordering::Release);
        run_worker(
            directory.0.clone(),
            fingerprint(),
            vec![7],
            commands,
            loaded,
            shared,
        );
        let programs = result.try_recv().unwrap().programs;
        assert_eq!(programs.len(), 1);
        assert_eq!(programs[0].source_digest().to_string(), "0b");
    }

    #[test]
    fn real_failure_observer_quarantines_full_queue_before_stale_save() {
        let directory = Directory::new();
        save_one(&directory);
        let (tx, commands) = mpsc::sync_channel(1);
        tx.send(Command::Save(vec![program(12)])).unwrap(); // Queue deliberately full.
        let shared = Arc::new(Shared::default());
        let observer = Observer {
            tx,
            shared: shared.clone(),
            imported: Rc::default(),
        };
        observer.notify_program_binary_failed(&program(11)); // Wake cannot enqueue.
        assert!(shared.quarantine.load(Ordering::Acquire));
        shared.closed.store(true, Ordering::Release);
        let (loaded, _result) = mpsc::sync_channel(1);
        run_worker(
            directory.0.clone(),
            fingerprint(),
            vec![7],
            commands,
            loaded,
            shared,
        );
        assert!(
            !directory.store().exists(),
            "quarantine must win over queued stale save"
        );
    }

    #[test]
    fn owner_discards_late_load_instead_of_overwriting_created_programs() {
        let (loaded, rx) = mpsc::sync_channel(1);
        let mut owner = ShaderProgramCache {
            cache: ProgramCache::new(None),
            rx: Some(rx),
            shared: Arc::new(Shared::default()),
            imported: Rc::default(),
        };
        owner.import_before_first_program(); // Empty, never waits.
        assert!(
            loaded
                .send(Loaded {
                    programs: vec![program(11)]
                })
                .is_err()
        );
        owner.import_before_first_program();
        assert!(owner.imported.borrow().is_empty());
    }

    #[test]
    fn second_worker_does_not_compete_with_locked_cache_namespace() {
        let directory = Directory::new();
        let identity = checksum(&encode(&fingerprint(), 16384).unwrap());
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(directory.0.join(format!("{identity}.lock")))
            .unwrap();
        lock.try_lock().unwrap();
        let (tx, commands) = mpsc::sync_channel(1);
        tx.send(Command::Save(vec![program(11)])).unwrap();
        let (loaded, result) = mpsc::sync_channel(1);
        run_worker(
            directory.0.clone(),
            fingerprint(),
            vec![7],
            commands,
            loaded,
            Arc::new(Shared::default()),
        );
        assert!(result.try_recv().is_err());
        assert!(!directory.store().exists());
    }
}
