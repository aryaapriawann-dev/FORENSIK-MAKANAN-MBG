# Design System & UI Style Guide
## NutriSafe AI — Food Forensic & Nutrition Web Platform

Dokumen ini adalah acuan resmi (*design token & style guide*) untuk menjaga konsistensi visual, tata warna, tipografi, komponen tabel, dan elemen antarmuka (*UI*) pada aplikasi web **NutriSafe AI** (Next.js + Tailwind CSS).

---

## 1. Brand Identity & Design Principles

* **Clean & Laboratory Grade:** Menghadirkan kesan higienis, terpercaya, dan akurat layaknya perangkat uji klinis.
* **Instant Clarity (Zero Latency Feel):** Informasi kelayakan pangan dan risiko bahaya langsung terlihat dalam 1 detik tanpa kebingungan.
* **High Semantic Contrast:** Status keamanan (*Safe, Caution, Danger*) harus sangat kontras terhadap warna latar putih-hijau utama.
* **Mobile-First Ergonomics:** Tombol aksi kamera berada dalam jangkauan satu jempol (*thumb zone*).

---

## 2. Color Palette & Token Reference

### 2.1 Primary Brand Colors (Putih & Hijau Medis)

| Token Name | Hex Code | Tailwind Class | Penggunaan Utama |
|---|---|---|---|
| `brand-bg` | `#F8FAFC` | `bg-slate-50` | Latar belakang halaman utama (luar card) |
| `brand-surface` | `#FFFFFF` | `bg-white` | Latar belakang Card, Modal, dan Container Tabel |
| `brand-primary` | `#059669` | `bg-emerald-600` | Tombol CTA utama (Capture & Scan), link aktif, logo mark |
| `brand-hover` | `#047857` | `bg-emerald-700` | Status hover / active pada tombol utama |
| `brand-accent-deep`| `#064E3B` | `bg-emerald-950` / `text-emerald-950` | Header tabel, background HUD scanner, teks penekanan |
| `brand-soft` | `#ECFDF5` | `bg-emerald-50` | Pill badge fitur, latar kartu sorotan gizi |
| `brand-border` | `#E2E8F0` | `border-slate-200` | Garis pemisah card, border tabel, input container |

---

### 2.2 Semantic Safety Status (Forensic Alert Colors)

Warna status penentu kelayakan makanan untuk mencegah keracunan:

```
🟢 SAFE (Layak Konsumsi)
   ├── Background Pill: #D1FAE5 (bg-emerald-100)
   ├── Text Color:      #065F46 (text-emerald-800)
   └── Border:          #A7F3D0 (border-emerald-200)

🟡 CAUTION (Perhatian / Mendekati Basi / Minyak Berlebih)
   ├── Background Pill: #FEF3C7 (bg-amber-100)
   ├── Text Color:      #92400E (text-amber-800)
   └── Border:          #FDE68A (border-amber-200)

🔴 DANGER (BAHAYA / Basi / Jamur / Kontaminasi)
   ├── Background Pill: #FFE4E6 (bg-rose-100)
   ├── Text Color:      #9F1239 (text-rose-800)
   ├── Border:          #FECDD3 (border-rose-300)
   └── Animation:       animate-pulse
```

---

### 2.3 Typography & Neutrals

| Token Name | Hex Code | Tailwind Class | Penggunaan |
|---|---|---|---|
| `text-headline` | `#0F172A` | `text-slate-900` | Judul utama (H1, H2), angka kalori besar |
| `text-body` | `#334155` | `text-slate-700` | Deskripsi umum, teks sel tabel |
| `text-muted` | `#64748B` | `text-slate-500` | Subtitle, satuan (kkal, gram), label timestamp |
| `text-inverse` | `#FFFFFF` | `text-white` | Teks di atas background tombol hijau / dark header |

---

## 3. Typography Scale & Hierarchy

| Elemen UI | Font Weight | Tailwind Size | Line Height | Tracking |
|---|---|---|---|---|
| **App Main Title (H1)** | 900 (Black) | `text-3xl md:text-5xl` | `leading-tight` | `tracking-tight` |
| **Section Header (H2)** | 700 (Bold) | `text-xl md:text-2xl` | `leading-snug` | `tracking-normal` |
| **Card / Table Title** | 700 (Bold) | `text-lg` | `leading-normal` | `tracking-normal` |
| **Body / Table Cell** | 400 (Regular) | `text-sm` (14px) | `leading-relaxed` | `tracking-normal` |
| **Badge / Meta Caption**| 600 (Semibold)| `text-xs` (12px) | `leading-none` | `tracking-wider uppercase` |
| **Big Nutrient Metric** | 900 (Black) | `text-2xl md:text-3xl`| `leading-none` | `tracking-tight` |

---

## 4. Component Design Specifications

### 4.1 Header & Hero Banner
* **Badge Pill:** `inline-flex items-center gap-2 bg-emerald-50 text-emerald-700 border border-emerald-200 text-xs font-semibold px-3 py-1 rounded-full`
* **Title:** Hitam tegas (`text-slate-900`) dengan kata kunci hijau (`text-emerald-600`).

### 4.2 Scanner Viewfinder Container
* **Card Outer:** `bg-white rounded-3xl p-4 shadow-xl border border-slate-200/80`
* **Video Frame:** `aspect-video rounded-2xl overflow-hidden bg-slate-950 relative`
* **HUD Overlay:** Garis bidik semi-transparan dengan aksen `border-emerald-500/40`.
* **Primary Scan Button:** 
  * Default: `bg-emerald-600 hover:bg-emerald-700 active:scale-95 text-white font-bold py-3.5 px-6 rounded-xl shadow-lg shadow-emerald-600/25 transition-all`
  * Loading state: `bg-slate-400 cursor-not-allowed opacity-80`

### 4.3 Forensic & Nutrition Data Table
* **Wrapper:** `w-full bg-white rounded-2xl shadow-xl border border-slate-100 overflow-hidden`
* **Table Header Bar:** `bg-slate-900 text-white p-5 flex justify-between items-center`
* **Column Headers:** `bg-slate-50 text-slate-500 text-xs font-bold uppercase tracking-wider py-3.5 px-4 border-b border-slate-200`
* **Row Interaction:** `hover:bg-slate-50/80 transition-colors border-b border-slate-100`
* **Table Footer (Summary Bar):** `bg-emerald-950 text-white p-4 font-bold flex justify-between items-center`

---

## 5. Tailwind Configuration Code (`tailwind.config.ts`)

Salin konfigurasi ini ke file `tailwind.config.ts` di proyek Next.js untuk memastikan seluruh token terdaftar secara otomatis:

```typescript
import type { Config } from 'tailwindcss';

const config: Config = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          50: '#ecfdf5',
          100: '#d1fae5',
          200: '#a7f3d0',
          500: '#10b981',
          600: '#059669', // Primary Brand Green
          700: '#047857', // Hover State
          800: '#065f46',
          900: '#064e3b',
          950: '#022c22', // Dark Emerald Container
        },
        surface: {
          base: '#f8fafc',
          card: '#ffffff',
          dark: '#0f172a',
        },
        forensic: {
          safeBg: '#d1fae5',
          safeText: '#065f46',
          safeBorder: '#a7f3d0',
          warnBg: '#fef3c7',
          warnText: '#92400e',
          warnBorder: '#fde68a',
          dangerBg: '#ffe4e6',
          dangerText: '#9f1239',
          dangerBorder: '#fecdd3',
        }
      },
      borderRadius: {
        '2xl': '1rem',
        '3xl': '1.5rem',
      },
      boxShadow: {
        'brand': '0 10px 25px -5px rgba(5, 150, 105, 0.15), 0 8px 10px -6px rgba(5, 150, 105, 0.1)',
      }
    },
  },
  plugins: [],
};

export default config;
```

---

## 6. Checklist Konsistensi Desain (Quality Assurance)

- [ ] Latar belakang web menggunakan `bg-slate-50` (bukan abu-abu pekat atau putih murni tanpa kontras).
- [ ] Semua kartu (*cards*) memiliki warna dasar putih (`#FFFFFF`) dengan sudut melengkung `rounded-2xl` atau `rounded-3xl` dan border `border-slate-200`.
- [ ] Tombol aksi utama selalu berwarna hijau emerald (`#059669`) dengan hover `#047857`.
- [ ] Status bahaya makanan basi/jamur wajib menggunakan badge warna `rose-100` / `rose-800` dengan efek `animate-pulse`.
- [ ] Angka kalori dan gizi ditampilkan dengan format angka yang jelas dan konsisten (menggunakan font tebal / bold).
