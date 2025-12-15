# Final Projek PCV — Realtime Filtering + HSV Color Detection (OpenCV)

Program Python (OpenCV) untuk **real-time webcam processing** yang mencakup:
- **Tugas 1:** pemilihan filter citra (Normal, Average Blur 5×5, Average Blur 9×9, Gaussian 2D custom via `filter2D`, Sharpen)
- **Tugas 2:** **deteksi objek berwarna** (Biru / Hijau) menggunakan **HSV + Morphology (Opening/Closing) + Contour**, lalu memicu **event/action** bila objek cukup besar terdeteksi.

> Cocok untuk demo praktikum Pengolahan Citra Video: konsep **konvolusi/kernel**, **low-pass / high-pass filtering**, **segmentasi HSV**, **morfologi**, dan **kontur**.

---

## Demo Singkat

- Pilih mode filter dengan tombol `0..4`
- Toggle deteksi warna dengan `H`
- Pilih warna target `B` (blue) / `G` (green)
- Jika objek terdeteksi dengan area cukup besar → muncul kotak + tulisan + banner **ACTION**

> Tambahkan screenshot/gif demo di sini:
- `docs/demo.gif`
- `docs/screenshot.png`

---

## Fitur

### Tugas 1 — Filtering (Spatial Domain)
- `Normal` (tanpa filter)
- `Average Blur` 5×5 dan 9×9 (`cv2.blur`)
- `Gaussian Blur 2D` **custom kernel** (dibangun dari `getGaussianKernel` lalu outer product) dan diterapkan via **`cv2.filter2D`**
- `Sharpen` (kernel 3×3) via **`cv2.filter2D`**

### Tugas 2 — HSV Detection + Morphology + Contour
- Konversi BGR → HSV (`cv2.cvtColor`)
- Threshold warna (`cv2.inRange`)
- Morphology:
  - **Opening**: menghapus noise kecil
  - **Closing**: menutup lubang kecil pada objek
- Deteksi kontur terbesar dan cek **luas (area)** untuk menghindari false positive
- Overlay bounding box + label + preview mask

### HUD & FPS
- Menampilkan mode aktif, status HSV, warna target, dan FPS realtime.

---

## Requirements

- Python **3.9+** (disarankan 3.10/3.11)
- Paket:
  - `opencv-python`
  - `numpy`

---

## Instalasi

### 1) Clone repo
```bash
git clone <repo-url-kamu>
cd <nama-folder-repo>
````

### 2) Buat virtual environment (opsional tapi disarankan)

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux/Mac:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3) Install dependency

```bash
pip install -r requirements.txt
```

Jika belum ada `requirements.txt`, bisa install manual:

```bash
pip install opencv-python numpy
```

---

## Menjalankan Program

```bash
python main.py
```

> Jika file kamu masih satu script (misal `app.py`), jalankan:

```bash
python app.py
```

---

## Kontrol Keyboard

|      Tombol | Fungsi                                   |
| ----------: | ---------------------------------------- |
|         `0` | Normal                                   |
|         `1` | Average Blur 5×5                         |
|         `2` | Average Blur 9×9                         |
|         `3` | Gaussian 2D (custom kernel + `filter2D`) |
|         `4` | Sharpen                                  |
|         `H` | Toggle HSV Detection (ON/OFF)            |
|         `B` | Target warna Biru                        |
|         `G` | Target warna Hijau                       |
| `Q` / `ESC` | Keluar                                   |

---

## Parameter yang Bisa Dituning

Semua parameter ada di bagian **Config** pada source code:

### Kamera

* `CAM_INDEX` : ganti ke `1`, `2`, dst jika webcam tidak terbaca
* `CAP_W`, `CAP_H` : resolusi capture (kamera mungkin menyesuaikan otomatis)

### Gaussian Kernel (Tugas 1)

* `GAUSS_KSIZE` : ukuran kernel (ganjil: 3,5,7,9,…)
* `GAUSS_SIGMA` : tingkat blur (lebih besar → lebih halus)

### HSV Color Range (Tugas 2)

OpenCV Hue range: `0..179`

* Biru:

  * `BLUE_LO = [H, S, V]`
  * `BLUE_HI = [H, S, V]`
* Hijau:

  * `GREEN_LO`
  * `GREEN_HI`

> Jika objek sulit terdeteksi: biasanya perlu adjust **S** dan **V** (lighting), bukan hanya Hue.

### Morphology

* `MORPH_KERNEL` (default ellipse 5×5)
* `iterations` pada `morphologyEx` (default 1)

### Threshold Area (Deteksi Action)

* `area_thresh` pada `detect_color_and_action(..., area_thresh=3000)`

  * Naikkan jika terlalu sensitif (banyak false positive)
  * Turunkan jika objek kecil tidak terdeteksi

---

## Penjelasan Konsep (Ringkas untuk Dokumentasi)

* **Filtering/Kernel (Konvolusi):** Output piksel dihitung dari kombinasi berbobot tetangga (kernel).
* **Average Blur:** low-pass sederhana (meredam noise, mengurangi detail).
* **Gaussian Blur:** low-pass yang lebih “natural” (bobot mengikuti distribusi Gaussian).
* **Sharpen:** high-pass effect (menonjolkan tepi/detail, bisa memperkuat noise).
* **HSV Segmentation:** threshold warna lebih stabil dibanding RGB/BGR karena Hue memisahkan jenis warna dari intensitas.
* **Morphology (Open/Close):**

  * Opening: bersihkan bintik/noise kecil
  * Closing: isi lubang kecil pada objek
* **Contour + Area Threshold:** memilih objek utama dan memicu event hanya jika ukurannya memadai.

---

## Struktur Folder (Disarankan)

Contoh struktur repo yang rapi:

```
.
├─ main.py
├─ requirements.txt
├─ README.md
└─ docs/
   ├─ demo.gif
   └─ screenshot.png
```

---

## Troubleshooting

### Webcam tidak terbaca / frame gagal di-grab

* Ubah `CAM_INDEX` (misal 0 → 1)
* Tutup aplikasi lain yang sedang memakai kamera (Zoom/OBS/Meet)
* Coba turunkan resolusi (`CAP_W`, `CAP_H`)

### Deteksi warna tidak stabil

* Perbaiki pencahayaan (lebih rata, tidak terlalu gelap)
* Tuning range HSV (terutama `S` dan `V`)
* Perbesar `MORPH_KERNEL` atau tambah `iterations` jika mask terlalu noisy
* Naikkan `area_thresh` jika sering false positive

### FPS rendah

* Turunkan resolusi capture (misal 640×480)
* Hindari sharpen saat HSV ON (kadang bikin mask lebih “berisik”)
* Kurangi ukuran kernel (Gaussian ksize lebih kecil)

---


## Credit

Dibuat untuk Final Project / Praktikum **Pengolahan Citra Video (PCV)** menggunakan **OpenCV + NumPy**.

```

