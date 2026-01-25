import io, os, zipfile, time, threading
from functools import partial, lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict
import gc
import hashlib

import streamlit as st
from PIL import Image, ImageOps, ImageFilter
import fitz  # PyMuPDF

# ===== HEIC/HEIF =====
HEIF_OK = False
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIF_OK = True
except Exception:
    HEIF_OK = False

# ==========================
# SESSION & RESOURCE CONFIG
# ==========================
def get_session_id():
    if 'session_id' not in st.session_state:
        st.session_state.session_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
    return st.session_state.session_id

# 🚀 OPTIMIZED FOR 8 CORES / 23GB RAM
TOTAL_CORES = 4
THREADS_PER_SESSION = 7  # 75% utilization
BATCH_SIZE = 20  # Optimal batch size
MAX_FILE_SIZE_MB = 6144  # 5GB max

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="Smart Image Compressor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk tampilan modern
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    .main-header p {
        color: rgba(255,255,255,0.9);
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
    }
    .stats-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin: 1rem 0;
    }
    .success-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# Header with Hero Image
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    <div class="main-header">
        <h1>⚡ Smart Image Compressor</h1>
        <p>Intelligent auto-compression • q/w/e files: ≤198KB • Other files: ≤138KB</p>
    </div>
    """, unsafe_allow_html=True)

session_id = get_session_id()

# ==========================
# SIDEBAR SETTINGS
# ==========================
with st.sidebar:
    st.markdown("### ⚙️ Pengaturan Kompresi")
    
    with st.expander("🔧 Advanced Settings", expanded=False):
        SPEED_PRESET = st.selectbox("Mode Kecepatan", ["fast", "balanced"], index=0)
        MIN_SIDE_PX = st.number_input("Min Side (px)", 64, 2048, 256, 32)
        SCALE_MIN = st.slider("Min Scale", 0.10, 0.75, 0.35, 0.05)
        SHARPEN_ON_RESIZE = st.checkbox("Enable Sharpening", True)
        SHARPEN_AMOUNT = st.slider("Sharpen Amount", 0.0, 2.0, 1.0, 0.1)
        PDF_DPI = 180 if SPEED_PRESET == "fast" else 220
    
    st.markdown("---")
    st.markdown("### 📊 Session Info")
    st.info(f"**Session:** `{session_id}`\n\n**Threads:** {THREADS_PER_SESSION}\n\n**Max File:** {MAX_FILE_SIZE_MB}MB")
    
    st.markdown("---")
    st.markdown("### 🎯 Target Sizes")
    st.success("**q, w, e** → ≤198 KB")
    st.info("**Others** → ≤138 KB")

# ==========================
# CONSTANTS
# ==========================
MAX_QUALITY = 95
MIN_QUALITY = 15
BG_FOR_ALPHA = (255, 255, 255)
ZIP_COMP_ALGO = zipfile.ZIP_STORED if SPEED_PRESET == "fast" else zipfile.ZIP_DEFLATED

TARGET_KB_HIGH = 198
TARGET_KB_LOW = 138

IMG_EXT = {".jpg", ".jpeg", ".jfif", ".png", ".webp", ".tif", ".tiff", ".bmp", ".gif", ".heic", ".heif", ".icloud"}
PDF_EXT = {".pdf"}
ALLOW_ZIP = True

# ==========================
# CORE FUNCTIONS
# ==========================
@lru_cache(maxsize=256)
def get_target_size_for_path_cached(filename_lower: str) -> int:
    return TARGET_KB_HIGH if filename_lower in ['q', 'w', 'e'] else TARGET_KB_LOW

def get_target_size_for_path(relpath: Path) -> int:
    return get_target_size_for_path_cached(relpath.stem.lower())

def maybe_sharpen(img: Image.Image, do_it=True, amount=1.0) -> Image.Image:
    if not do_it or amount <= 0:
        return img
    return img.filter(ImageFilter.UnsharpMask(radius=1.0, percent=int(150*amount), threshold=2))

def to_rgb_flat(img: Image.Image, bg=BG_FOR_ALPHA) -> Image.Image:
    mode = img.mode
    if mode == "RGB":
        return img
    if mode in ("RGBA", "LA") or (mode == "P" and "transparency" in img.info):
        base = Image.new("RGB", img.size, bg)
        base.paste(img, mask=img.convert("RGBA").split()[-1])
        return base
    return img.convert("RGB")

def get_fresh_buffer() -> io.BytesIO:
    return io.BytesIO()

def save_jpg_bytes(img: Image.Image, quality: int) -> bytes:
    buf = get_fresh_buffer()
    if SPEED_PRESET == "fast":
        img.save(buf, format="JPEG", quality=quality, optimize=False, progressive=False, subsampling=2)
    else:
        img.save(buf, format="JPEG", quality=quality, optimize=True, progressive=True, subsampling=2)
    return buf.getvalue()

def try_quality_bs(img: Image.Image, target_kb: int, q_min=MIN_QUALITY, q_max=MAX_QUALITY):
    lo, hi = q_min, q_max
    best_bytes = None
    best_q = None
    
    data = save_jpg_bytes(img, hi)
    if len(data) <= target_kb * 1024:
        return data, hi
    
    while lo <= hi:
        mid = (lo + hi) // 2
        data = save_jpg_bytes(img, mid)
        size = len(data)
        
        if size <= target_kb * 1024:
            best_bytes, best_q = data, mid
            if size >= target_kb * 1024 * 0.95:
                return best_bytes, best_q
            lo = mid + 1
        else:
            hi = mid - 1
    
    return best_bytes, best_q

def resize_to_scale(img: Image.Image, scale: float, do_sharpen=True, amount=1.0) -> Image.Image:
    w, h = img.size
    nw, nh = max(int(w*scale), 1), max(int(h*scale), 1)
    
    if abs(scale - 1.0) < 0.01:
        return img
    
    out = img.resize((nw, nh), Image.LANCZOS)
    return maybe_sharpen(out, do_sharpen, amount)

def ensure_min_side(img: Image.Image, min_side_px: int, do_sharpen=True, amount=1.0) -> Image.Image:
    w, h = img.size
    min_side = min(w, h)
    
    if min_side >= min_side_px:
        return img
    
    scale = min_side_px / max(min_side, 1)
    if scale > 1.0:
        return img
    
    return resize_to_scale(img, scale, do_sharpen, amount)

def load_image_from_bytes(name: str, raw: bytes) -> Image.Image:
    if name.lower().endswith('.icloud'):
        try:
            im = Image.open(io.BytesIO(raw))
            return ImageOps.exif_transpose(im)
        except Exception:
            raise ValueError(f"iCloud file {name} is a placeholder. Download actual file first.")
    
    im = Image.open(io.BytesIO(raw))
    return ImageOps.exif_transpose(im)

def gif_first_frame(im: Image.Image) -> Image.Image:
    try:
        im.seek(0)
    except Exception:
        pass
    return im.convert("RGBA") if im.mode == "P" else im

def compress_into_range(base_img: Image.Image, max_kb: int, min_side_px: int, scale_min: float, do_sharpen: bool, sharpen_amount: float):
    base = to_rgb_flat(base_img)
    
    data, q = try_quality_bs(base, max_kb)
    if data is not None and len(data) <= max_kb * 1024:
        return (data, 1.0, q, len(data))
    
    lo, hi = scale_min, 1.0
    best_pack = None
    max_steps = 8 if SPEED_PRESET == "fast" else 12
    
    for step in range(max_steps):
        mid = (lo + hi) / 2
        candidate = resize_to_scale(base, mid, do_sharpen, sharpen_amount)
        candidate = ensure_min_side(candidate, min_side_px, do_sharpen, sharpen_amount)
        
        d, q2 = try_quality_bs(candidate, max_kb)
        
        if d is not None and len(d) <= max_kb * 1024:
            best_pack = (d, mid, q2, len(d))
            if len(d) >= max_kb * 1024 * 0.90:
                break
            lo = mid + (hi - mid) * 0.35
        else:
            hi = mid - (mid - lo) * 0.35
        
        if hi - lo < 5e-3:
            break
    
    if best_pack is None:
        smallest = resize_to_scale(base, scale_min, do_sharpen, sharpen_amount)
        smallest = ensure_min_side(smallest, min_side_px, do_sharpen, sharpen_amount)
        d = save_jpg_bytes(smallest, MIN_QUALITY)
        result = (d, scale_min, MIN_QUALITY, len(d))
    else:
        result = best_pack
    
    data, scale_used, q_used, size_b = result
    
    if size_b > max_kb * 1024:
        for q_try in range(q_used - 10, MIN_QUALITY - 1, -10):
            if q_try < MIN_QUALITY:
                q_try = MIN_QUALITY
            img_final = resize_to_scale(base, scale_used, do_sharpen, sharpen_amount)
            img_final = ensure_min_side(img_final, min_side_px, do_sharpen, sharpen_amount)
            d = save_jpg_bytes(img_final, q_try)
            if len(d) <= max_kb * 1024:
                return d, scale_used, q_try, len(d)
            if q_try == MIN_QUALITY:
                break
    
    if size_b > max_kb * 1024:
        try:
            img_recompress = Image.open(io.BytesIO(data))
            for scale_try in [0.9, 0.8, 0.7, 0.6, 0.5]:
                candidate = resize_to_scale(img_recompress, scale_try, do_sharpen, sharpen_amount)
                d, q2 = try_quality_bs(candidate, max_kb)
                if d is not None and len(d) <= max_kb * 1024:
                    return d, scale_used * scale_try, q2, len(d)
            
            smallest = resize_to_scale(img_recompress, 0.5, do_sharpen, sharpen_amount)
            d = save_jpg_bytes(smallest, MIN_QUALITY)
            return d, scale_used * 0.5, MIN_QUALITY, len(d)
        except Exception:
            pass
    
    return data, scale_used, q_used, size_b

def pdf_bytes_to_images(pdf_bytes: bytes, dpi: int) -> List[Image.Image]:
    images = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            rect = page.rect
            long_inch = max(rect.width, rect.height) / 72.0
            target_long_px = 2000
            dpi_eff = int(min(max(dpi, 72), max(72, target_long_px / max(long_inch, 1e-6))))
            zoom = dpi_eff / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(ImageOps.exif_transpose(img))
    return images

def extract_zip_to_memory(zf_bytes: bytes) -> List[Tuple[Path, bytes]]:
    out = []
    with zipfile.ZipFile(io.BytesIO(zf_bytes), 'r') as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            with zf.open(info, 'r') as f:
                data = f.read()
            out.append((Path(info.filename), data))
    return out

def guess_base_name_from_zip(zipname: str) -> str:
    base = Path(zipname).stem
    return base or "output"

# ✅ THREAD-SAFE PROCESSING
def process_one_file_entry(task_data: Tuple[str, Path, bytes]):
    input_root_label, relpath, raw_bytes = task_data
    
    processed: List[Tuple[str, int, float, int, bool, int]] = []
    outputs: Dict[str, bytes] = {}
    skipped: List[Tuple[str, str]] = []
    
    ext = relpath.suffix.lower()
    raw_bytes_isolated = bytes(raw_bytes)
    
    relpath = Path(relpath.parent, relpath.stem + ext)
    target_kb = get_target_size_for_path(relpath)
    
    try:
        if ext in PDF_EXT:
            pages = pdf_bytes_to_images(raw_bytes_isolated, dpi=PDF_DPI)
            for idx, pil_img in enumerate(pages, start=1):
                try:
                    img_isolated = pil_img.copy()
                    
                    data, scale, q, size_b = compress_into_range(
                        img_isolated, target_kb, MIN_SIDE_PX, SCALE_MIN, SHARPEN_ON_RESIZE, SHARPEN_AMOUNT
                    )
                    
                    out_rel = relpath.with_suffix("").as_posix() + f"_p{idx}.jpg"
                    
                    if data and len(data) > 0:
                        outputs[out_rel] = bytes(data)
                        processed.append((out_rel, size_b, scale, q, size_b <= target_kb*1024, target_kb))
                    
                    del img_isolated, data, pil_img
                    
                except Exception as e:
                    skipped.append((f"{relpath} (page {idx})", str(e)))
            
            del pages
            
        elif ext in IMG_EXT:
            if ext == ".icloud":
                try:
                    im = load_image_from_bytes(relpath.name, raw_bytes_isolated)
                except ValueError as e:
                    skipped.append((str(relpath), str(e)))
                    return input_root_label, processed, skipped, outputs
            elif ext in {".heic", ".heif"} and not HEIF_OK:
                skipped.append((str(relpath), "Butuh pillow-heif"))
                return input_root_label, processed, skipped, outputs
            else:
                im = load_image_from_bytes(relpath.name, raw_bytes_isolated)
            
            if ext == ".gif":
                im = gif_first_frame(im)
            
            img_isolated = im.copy()
            del im
            
            data, scale, q, size_b = compress_into_range(
                img_isolated, target_kb, MIN_SIDE_PX, SCALE_MIN, SHARPEN_ON_RESIZE, SHARPEN_AMOUNT
            )
            
            if ext == ".icloud":
                out_rel = relpath.with_suffix("").with_suffix(".jpg").as_posix()
            else:
                out_rel = relpath.with_suffix(".jpg").as_posix()
            
            if data and len(data) > 0:
                outputs[out_rel] = bytes(data)
                processed.append((out_rel, size_b, scale, q, size_b <= target_kb*1024, target_kb))
            
            del img_isolated, data
            
    except Exception as e:
        skipped.append((str(relpath), str(e)))
    finally:
        del raw_bytes_isolated
    
    return input_root_label, processed, skipped, outputs

# ==========================
# MAIN UPLOAD & AUTO-PROCESS
# ==========================
allowed_exts = sorted({e.lstrip('.') for e in IMG_EXT.union(PDF_EXT)} | ({"zip"} if ALLOW_ZIP else set()))

if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0
if 'processing_done' not in st.session_state:
    st.session_state.processing_done = False
if 'master_zip_data' not in st.session_state:
    st.session_state.master_zip_data = None

# Upload Zone
st.markdown('<div class="upload-zone">', unsafe_allow_html=True)
uploaded_files = st.file_uploader(
    "📁 Drag & Drop atau Klik untuk Upload",
    type=allowed_exts,
    accept_multiple_files=True,
    key=f"uploader_{st.session_state.uploader_key}",
    help=f"Supported: ZIP, Images, PDF (Max {MAX_FILE_SIZE_MB}MB per file)"
)
st.markdown('</div>', unsafe_allow_html=True)

# ✅ AUTO-PROCESS when files uploaded
if uploaded_files and not st.session_state.processing_done:
    
    # Validate file sizes
    oversized = []
    total_size_mb = 0
    for f in uploaded_files:
        f.seek(0, 2)
        size_mb = f.tell() / (1024 * 1024)
        f.seek(0)
        total_size_mb += size_mb
        if size_mb > MAX_FILE_SIZE_MB:
            oversized.append((f.name, size_mb))
    
    if oversized:
        st.error("❌ Files too large:")
        for name, size in oversized:
            st.write(f"- `{name}`: {size:.1f}MB")
        st.stop()
    
    # Show stats
    st.markdown(f"""
    <div class="stats-card">
        <h3>📊 Upload Summary</h3>
        <p><strong>{len(uploaded_files)}</strong> files • <strong>{total_size_mb:.1f}MB</strong> total</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Prepare jobs
    jobs = []
    used_labels = set()
    
    def unique_name(base: str, used: set) -> str:
        name = base
        idx = 2
        while name in used:
            name = f"{base}_{idx}"
            idx += 1
        used.add(name)
        return name
    
    zip_inputs, loose_inputs = [], []
    for f in uploaded_files:
        name, raw = f.name, f.read()
        if name.lower().endswith(".zip"):
            zip_inputs.append((name, raw))
        else:
            loose_inputs.append((name, raw))
    
    allowed = IMG_EXT.union(PDF_EXT)
    
    for zname, zbytes in zip_inputs:
        try:
            pairs = extract_zip_to_memory(zbytes)
            base_label = unique_name(guess_base_name_from_zip(zname), used_labels)
            items = [(relp, data) for (relp, data) in pairs if relp.suffix.lower() in allowed]
            if items:
                jobs.append({"label": base_label, "items": items})
        except Exception as e:
            st.error(f"Failed to open ZIP {zname}: {e}")
    
    if loose_inputs:
        ts = time.strftime("%Y%m%d_%H%M%S")
        base_label = unique_name(f"compressed_{ts}", used_labels)
        items = [(Path(name), data) for (name, data) in loose_inputs if Path(name).suffix.lower() in allowed]
        if items:
            jobs.append({"label": base_label, "items": items})
    
    if not jobs:
        st.error("❌ No valid files found")
        st.stop()
    
    total_files = sum(len(j['items']) for j in jobs)
    
    # Processing UI
    st.markdown("### ⚡ Processing...")
    progress_bar = st.progress(0.0)
    status_text = st.empty()
    stats_placeholder = st.empty()
    
    summary: Dict[str, List[Tuple[str, int, float, int, bool, int]]] = defaultdict(list)
    skipped_all: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    
    master_buf = io.BytesIO()
    zip_write_lock = threading.Lock()
    written_paths = set()
    
    with zipfile.ZipFile(master_buf, "w", compression=ZIP_COMP_ALGO) as master:
        top_folders: Dict[str, str] = {}
        for job in jobs:
            top = f"{job['label']}_compressed"
            top_folders[job['label']] = top
            master.writestr(f"{top}/", "")
        
        def add_to_master_zip_threadsafe(top_folder: str, rel_path: str, data: bytes):
            with zip_write_lock:
                try:
                    if not data or len(data) == 0:
                        return False
                    
                    try:
                        test_img = Image.open(io.BytesIO(data))
                        test_img.verify()
                        del test_img
                    except Exception:
                        return False
                    
                    full_path = f"{top_folder}/{rel_path}"
                    if full_path in written_paths:
                        base, ext = os.path.splitext(rel_path)
                        counter = 2
                        while f"{top_folder}/{base}_{counter}{ext}" in written_paths:
                            counter += 1
                        full_path = f"{top_folder}/{base}_{counter}{ext}"
                    
                    master.writestr(full_path, data)
                    written_paths.add(full_path)
                    return True
                    
                except Exception:
                    return False
        
        all_tasks = [
            (job["label"], relp, bytes(data))
            for job in jobs 
            for (relp, data) in job["items"]
        ]
        
        total = len(all_tasks)
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=THREADS_PER_SESSION) as ex:
            done = 0
            
            futures = {ex.submit(process_one_file_entry, task): task for task in all_tasks}
            
            for fut in as_completed(futures):
                try:
                    label, prc, skp, outs = fut.result(timeout=300)
                    summary[label].extend(prc)
                    skipped_all[label].extend(skp)
                    
                    if outs:
                        top = top_folders[label]
                        for rel_path, data in outs.items():
                            success = add_to_master_zip_threadsafe(top, rel_path, data)
                            if not success:
                                skipped_all[label].append((rel_path, "Failed ZIP write"))
                    
                    done += 1
                    
                    elapsed = time.time() - start_time
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (total - done) / rate if rate > 0 else 0
                    
                    progress_bar.progress(min(done / total, 1.0))
                    status_text.text(f"⚡ {done}/{total} files • {rate:.1f} files/s • ETA: {eta:.0f}s")
                    
                except Exception as e:
                    done += 1
            
            if done % (BATCH_SIZE * 2) == 0:
                gc.collect()
        
        elapsed_total = time.time() - start_time
    
    master_buf.seek(0)
    
    # Store result
    st.session_state.master_zip_data = master_buf.getvalue()
    st.session_state.processing_done = True
    
    # Success message
    grand_ok = sum(sum(1 for _, _, _, _, ok, _ in summary[job["label"]] if ok) for job in jobs)
    grand_cnt = sum(len(summary[job["label"]]) for job in jobs)
    
    st.markdown(f"""
    <div class="success-box">
        <h2>✅ Compression Complete!</h2>
        <p><strong>{grand_ok}/{grand_cnt}</strong> files successfully compressed in <strong>{elapsed_total:.1f}s</strong></p>
        <p>Average speed: <strong>{total/elapsed_total:.1f} files/second</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show details
    with st.expander("📊 View Details", expanded=False):
        for job in jobs:
            base = job["label"]
            items = summary[base]
            skipped = skipped_all[base]
            
            st.markdown(f"#### 📦 {base}")
            st.write(f"Processed: {len(items)} • Skipped: {len(skipped)}")
            
            if skipped:
                with st.expander("❌ Skipped Files"):
                    for n, reason in skipped[:20]:
                        st.write(f"- {n}: {reason}")

# ✅ AUTO-DOWNLOAD when processing done
if st.session_state.processing_done and st.session_state.master_zip_data:
    
    st.markdown("### 📥 Download Ready")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.download_button(
            label="⬇️ Download Compressed ZIP",
            data=st.session_state.master_zip_data,
            file_name=f"compressed_{time.strftime('%Y%m%d_%H%M%S')}.zip",
            mime="application/zip",
            type="primary",
            use_container_width=True
        )
    
    with col2:
        if st.button("🔄 Process New Files", type="secondary", use_container_width=True):
            st.session_state.uploader_key += 1
            st.session_state.processing_done = False
            st.session_state.master_zip_data = None
            gc.collect()
            st.rerun()

# Footer with Credits
st.markdown("---")
st.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 10px; margin-top: 2rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
    <div style="text-align: center; color: white;">
        <h3 style="margin: 0 0 1rem 0; font-weight: 600;">⚡Smart Compression Engine</h3>
        <p style="margin: 0.5rem 0; opacity: 0.9;">Session: <code style="background: rgba(255,255,255,0.2); padding: 0.2rem 0.5rem; border-radius: 4px;">{}</code></p>
        <div style="margin: 1.5rem 0; padding: 1rem; background: rgba(255,255,255,0.1); border-radius: 8px; backdrop-filter: blur(10px);">
            <p style="margin: 0 0 0.5rem 0; font-size: 0.9rem; font-weight: 600;">© 2025 All Rights Reserved</p>
            <p style="margin: 0.3rem 0; font-size: 0.85rem; opacity: 0.95;">
                <strong>Original Design:</strong><br>
                Matthew Artur Panahatan Sitorus • Muhammad Ardiansyah Pangestu • Kevin Deniswara Harvian
            </p>
            <p style="margin: 0.8rem 0 0 0; font-size: 0.85rem; opacity: 0.95;">
                <strong>Developed by:</strong> Aditya Fathan Santoso
            </p>
        </div>
    </div>
</div>
""".format(session_id), unsafe_allow_html=True)
