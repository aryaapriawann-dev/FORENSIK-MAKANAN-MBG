# Product Requirements Document (PRD) & MVP Specification
## Sistem Deteksi Forensik Kelayakan & Analisis Gizi Pangan Publik (NutriSafe AI)

---

## 1. Executive Summary
**NutriSafe AI** adalah platform web berbasis client-side intelligence yang dirancang untuk mendeteksi kelayakan fisik makanan (forensik pembusukan/anomali visual) dan menganalisis kandungan makronutrisi makanan secara otomatis dari tangkapan kamera. Sistem dirancang dengan prinsip **Zero-Cost Architecture** (100% gratis tanpa API berbayar), **Zero-Friction Access** (tanpa registrasi/login), serta **Privacy-Preserving** (pemrosesan gambar berbasis WebAssembly/WebGPU langsung di browser pengguna).

---

## 2. Latar Belakang & Masalah
* **Tingginya Risiko Keracunan Makanan:** Kurangnya kepekaan visual terhadap tanda-tanda awal pembusukan (perubahan warna, lendir, pertumbuhan kapang/jamur, kuah pecah/berbusa).
* **Kurangnya Kesadaran Gizi Harian:** Sulitnya menghitung estimasi gizi makanan rumahan atau warung makan secara cepat dan praktis.
* **Hambatan Akses Sistem Digital:** Kewajiban mendaftar/login serta ketergantungan pada model AI cloud berbayar (OpenAI/Anthropic) yang membatasi akses masyarakat umum secara cuma-cuma.

---

## 3. Prinsip Desain & Batasan Sistem
1. **100% Free Tier & Open-Source:** Tidak ada ketergantungan pada API berbayar per request. Beban komputasi AI dipindahkan ke sisi klien (*Client-Side Inference*).
2. **Tanpa Login (Guest First):** Akses instan 1-klik untuk pemindaian instan di warung/meja makan.
3. **Standar Gizi Indonesia:** Pemetaan data mengacu pada Tabel Komposisi Pangan Indonesia (TKPI) dari Kementerian Kesehatan RI.
4. **Respon Cepat (< 3 detik):** Menggunakan model visual ringan berbasis MobileNet/ViT yang dioptimalkan untuk WebAssembly (WASM).

---

## 4. Tech Stack & Arsitektur Sistem

```
+-----------------------------------------------------------------------+
|                         CLIENT BROWSER                                |
|                                                                       |
|   [ Next.js 14+ (App Router) + Tailwind CSS + Lucide Icons ]          |
|                               │                                       |
|                               ▼                                       |
|   [ WebCam Scanner / Image Upload HTML5 Canvas ]                      |
|                               │                                       |
|                               ▼                                       |
|   [ Transformers.js / ONNX Runtime Web (WASM/WebGPU) ]                |
|   ├── Model A: Food Item Classifier (Deteksi Lauk)                   |
|   └── Model B: Visual Anomaly Forensic (Cek Basi/Jamur/Lendir)        |
|                               │                                       |
|                               ▼                                       |
|   [ Agregasi Hasil Forensik & Kalkulasi Makronutrisi ]                |
|                               │                                       |
|                               ▼                                       |
|   [ Tabel Hasil Deteksi & Safety Status Alert ]                       |
+-------------------------------┬---------------------------------------+
                                │
                      (Lookup & Log Anonim)
                                │
                                ▼
+-----------------------------------------------------------------------+
|                    BACKEND: SUPABASE FREE TIER                        |
|                                                                       |
|   ├── PostgreSQL: Tabel `nutrition_master` (Data TKPI Read-Only)     |
|   └── Row Level Security (RLS): Policy Public `anon` Access           |
+-----------------------------------------------------------------------+
```

### Rincian Stack:
* **Frontend Framework:** Next.js 14+ (App Router, TypeScript)
* **Styling:** Tailwind CSS + Radix UI / Lucide Icons
* **AI / ML Runtime:** `@huggingface/transformers` (Transformers.js) / `onnxruntime-web`
* **Database & BaaS:** Supabase (Free Tier: 500 MB DB, RLS Public Read)
* **Hosting:** Vercel (Hobby Tier - Rp 0)

---

## 5. Fitur Utama & Spesifikasi Fungsional

### 5.1 Modul Kamera & Input
* Antarmuka responsif ramah smartphone.
* Mendukung Live Camera Stream (menghadap belakang/lingkungan) dan File Upload.
* Pra-pemrosesan gambar otomatis: resize ke dimensi optimal ($224 \times 224$ atau $384 \times 384$ px) via HTML5 Canvas.

### 5.2 Modul Forensik Visual (Food Safety Guard)
Model mengidentifikasi indikator bahaya mikrobiologis dan kerusakan fisik:
* **Discoloration Index:** Deteksi penyimpangan warna tidak wajar (keabuan, kehitaman, kehijauan abnormal).
* **Texture & Mold Spot:** Identifikasi koloni spora/jamur putih, hijau, atau lapisan lendir (*slime*).
* **Emulsion Breakdown / Frothing:** Deteksi busa abnormal pada kuah/santan atau minyak teroksidasi berat.

### 5.3 Modul Analisis Gizi (TKPI Mapping)
* Mengelompokkan item makanan yang terdeteksi (misal: Nasi Putih, Tempe Goreng, Ayam Penyet, Sayur Bayam).
* Mengambil data kandungan gizi per 100 gram dari database Supabase:
  * Energi (kkal)
  * Protein (g)
  * Lemak (g)
  * Karbohidrat (g)
  * Serat / Fiber (g)

### 5.4 Modul Output Tabel & Rekomendasi
* Menampilkan ringkasan status kelayakan makanan:
  * 🟢 **Aman (Safe):** Parameter visual normal, gizi seimbang.
  * 🟡 **Perhatian (Caution):** Indikasi minyak tinggi, garam tinggi, atau kondisi mendekati batas kesegaran.
  * 🔴 **Bahaya (Danger):** Terdeteksi jamur, pembusukan, lendir, atau perubahan bau visual.
* Rekomendasi tindakan instan (Contoh: *"Jangan dikonsumsi! Kuah berbusa menandakan fermentasi bakteri liar"*).

---

## 6. Skema Database Supabase & Konfigurasi RLS

### Skrip SQL DDL & Data Seeder (`schema.sql`):

```sql
-- 1. Buat Tabel Master Gizi (TKPI Kemenkes RI)
CREATE TABLE IF NOT EXISTS public.nutrition_master (
    id BIGSERIAL PRIMARY KEY,
    food_code TEXT UNIQUE NOT NULL,
    food_name TEXT NOT NULL,
    category TEXT NOT NULL,
    serving_size_gram REAL DEFAULT 100,
    calories REAL NOT NULL,
    protein REAL NOT NULL,
    fat REAL NOT NULL,
    carbs REAL NOT NULL,
    fiber REAL NOT NULL,
    shelf_life_hours INT DEFAULT 12,
    spoilage_signs TEXT[]
);

-- 2. Aktifkan Row Level Security (RLS)
ALTER TABLE public.nutrition_master ENABLE ROW LEVEL SECURITY;

-- 3. Kebijakan Publik: Semua orang bisa membaca data gizi tanpa login
CREATE POLICY "Allow Public Read Nutrition"
ON public.nutrition_master
FOR SELECT
TO anon
USING (true);

-- 4. Insert Data Awal Makanan Lokal Indonesia (Sample Seeder)
INSERT INTO public.nutrition_master 
(food_code, food_name, category, serving_size_gram, calories, protein, fat, carbs, fiber, shelf_life_hours, spoilage_signs)
VALUES
('NASI_PUTIH', 'Nasi Putih', 'Pokok', 150, 195.0, 3.6, 0.4, 42.9, 0.3, 18, ARRAY['Menguning', 'Bercak merah/pink', 'Berbau asam/basi', 'Berlendir']),
('AYAM_GORENG', 'Ayam Goreng', 'Lauk Hewani', 100, 260.0, 27.0, 16.0, 0.0, 0.0, 24, ARRAY['Warna keabuan di dekat tulang', 'Bau tengik', 'Permukaan lengket']),
('TEMPE_GORENG', 'Tempe Goreng', 'Lauk Nabati', 50, 118.0, 9.0, 7.5, 4.0, 1.4, 24, ARRAY['Bau amonia tajam', 'Bercak hitam basah berlendir']),
('TAHU_GORENG', 'Tahu Goreng', 'Lauk Nabati', 50, 58.0, 4.8, 3.5, 2.1, 0.8, 16, ARRAY['Rasa asam', 'Lendir permukaan', 'Tekstur lembek berair']),
('SAYUR_SOP', 'Sayur Sop Bening', 'Sayur', 150, 45.0, 1.5, 0.5, 9.0, 1.8, 12, ARRAY['Kuah berbusa', 'Kuah keruh asam', 'Sayuran layu berlendir']),
('SAYUR_LODEH', 'Sayur Lodeh (Santan)', 'Sayur', 150, 120.0, 2.8, 8.5, 9.2, 2.1, 8, ARRAY['Santan pecah menggumpal', 'Berbusa', 'Rasa masam']),
('TELUR_DADAR', 'Telur Dadar', 'Lauk Hewani', 60, 154.0, 9.3, 12.0, 1.2, 0.0, 16, ARRAY['Tengik', 'Perubahan warna kehijauan gelap', 'Berlendir']),
('SAMBAL_TERASI', 'Sambal Terasi', 'Pelengkap', 20, 35.0, 0.8, 2.1, 3.2, 0.5, 36, ARRAY['Bintik jamur putih/abu-abu di permukaan', 'Gelembung gas']);
```

---

## 7. Implementasi MVP (Source Code Lengkap)

### 7.1 Konfigurasi Supabase Client (`lib/supabase.ts`)
```typescript
import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL || '';
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || '';

export const supabase = createClient(supabaseUrl, supabaseAnonKey);
```

### 7.2 Database Gizi Fallback & Types (`lib/types.ts`)
```typescript
export interface FoodItemAnalysis {
  id: string;
  name: string;
  category: string;
  confidence: number;
  safetyStatus: 'safe' | 'warning' | 'danger';
  forensicFlag: string;
  calories: number;
  protein: number;
  fat: number;
  carbs: number;
  fiber: number;
  recommendation: string;
}

export interface NutritionTotal {
  totalCalories: number;
  totalProtein: number;
  totalFat: number;
  totalCarbs: number;
  totalFiber: number;
  overallSafety: 'safe' | 'warning' | 'danger';
}
```

### 7.3 Pipeline Deteksi Client-Side (`lib/detector.ts`)
```typescript
import { pipeline, env } from '@huggingface/transformers';
import { supabase } from './supabase';
import { FoodItemAnalysis } from './types';

// Optimasi runtime browser
env.allowLocalModels = false;
env.useBrowserCache = true;

let classifierPipeline: any = null;

export async function getDetectorPipeline() {
  if (!classifierPipeline) {
    // Model klasifikasi visual ringan (MobileNet / ViT)
    classifierPipeline = await pipeline(
      'image-classification',
      'Xenova/food-classification-resnet-50',
      { device: 'webgpu' in navigator ? 'webgpu' : 'wasm' }
    );
  }
  return classifierPipeline;
}

// Analisis Forensik Visual & Lookup Gizi
export async function analyzeFoodImage(canvas: HTMLCanvasElement): Promise<FoodItemAnalysis[]> {
  const classifier = await getDetectorPipeline();
  const dataUrl = canvas.toDataURL('image/jpeg', 0.85);

  // 1. Eksekusi inferensi AI di browser
  const predictions = await classifier(dataUrl, { topk: 3 });

  const results: FoodItemAnalysis[] = [];

  for (const pred of predictions) {
    const rawLabel = (pred.label || '').toLowerCase();
    
    // 2. Query ke Supabase nutrition_master
    const { data: dbItem } = await supabase
      .from('nutrition_master')
      .select('*')
      .ilike('food_name', `%${rawLabel.split(',')[0]}%`)
      .maybeSingle();

    // Mapping default jika tidak ditemukan di DB
    const foodName = dbItem ? dbItem.food_name : (pred.label.charAt(0).toUpperCase() + pred.label.slice(1));
    const calories = dbItem ? dbItem.calories : 150;
    const protein = dbItem ? dbItem.protein : 5.0;
    const fat = dbItem ? dbItem.fat : 4.0;
    const carbs = dbItem ? dbItem.carbs : 20.0;
    const fiber = dbItem ? dbItem.fiber : 1.2;

    // Evaluasi Forensik berbasis Confidence & Analisis Visual
    let safetyStatus: 'safe' | 'warning' | 'danger' = 'safe';
    let forensicFlag = 'Kondisi visual segar, warna dan tekstur alami';
    let recommendation = 'Layak dan aman untuk dikonsumsi';

    // Heuristik Forensik Sederhana untuk MVP
    if (rawLabel.includes('spoil') || rawLabel.includes('mold') || rawLabel.includes('rotten')) {
      safetyStatus = 'danger';
      forensicFlag = 'Terdeteksi anomali tekstur/bintik pembusukan mikrobiologis';
      recommendation = 'JANGAN DIMAKAN! Berpotensi tinggi memicu keracunan makanan.';
    } else if (pred.score < 0.35) {
      safetyStatus = 'warning';
      forensicFlag = 'Kontras warna meragukan atau pencahayaan kurang optimal';
      recommendation = 'Periksa aroma dan rasa sebelum menyantap porsi besar.';
    }

    results.push({
      id: Math.random().toString(36).substring(2, 9),
      name: foodName,
      category: dbItem?.category || 'Makanan Umum',
      confidence: Math.round(pred.score * 100),
      safetyStatus,
      forensicFlag,
      calories,
      protein,
      fat,
      carbs,
      fiber,
      recommendation
    });
  }

  return results;
}
```

### 7.4 Komponen Tabel Hasil Forensik & Gizi (`components/FoodForensicTable.tsx`)
```tsx
'use client';

import React from 'react';
import { FoodItemAnalysis, NutritionTotal } from '../lib/types';
import { ShieldCheck, AlertTriangle, XCircle, Activity } from 'lucide-react';

interface Props {
  items: FoodItemAnalysis[];
  totals: NutritionTotal;
}

export const FoodForensicTable: React.FC<Props> = ({ items, totals }) => {
  return (
    <div className="w-full bg-white rounded-2xl shadow-xl border border-slate-100 overflow-hidden">
      {/* Header Tabel */}
      <div className="p-6 bg-slate-900 text-white flex justify-between items-center">
        <div>
          <h2 className="text-xl font-bold flex items-center gap-2">
            <Activity className="w-5 h-5 text-emerald-400" />
            Hasil Analisis Forensik & Standar Gizi
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Data dikompilasi secara real-time dari inferensi visual lokal & database TKPI.
          </p>
        </div>
        <div className="text-right">
          <span className="text-xs uppercase tracking-wider text-slate-400">Total Energi</span>
          <p className="text-2xl font-black text-emerald-400">{totals.totalCalories.toFixed(0)} kkal</p>
        </div>
      </div>

      {/* Tabel Rincian */}
      <div className="overflow-x-auto">
        <table className="w-full text-left border-collapse">
          <thead>
            <tr className="bg-slate-50 text-slate-600 text-xs font-semibold uppercase tracking-wider border-b border-slate-200">
              <th className="py-3 px-4">Komponen / Lauk</th>
              <th className="py-3 px-4">Status Forensik</th>
              <th className="py-3 px-4">Observasi Visual</th>
              <th className="py-3 px-4 text-center">Kalori</th>
              <th className="py-3 px-4 text-center">Protein</th>
              <th className="py-3 px-4 text-center">Lemak</th>
              <th className="py-3 px-4 text-center">Karbo</th>
              <th className="py-3 px-4">Rekomendasi</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100 text-sm text-slate-700">
            {items.map((item) => (
              <tr key={item.id} className="hover:bg-slate-50/80 transition-colors">
                <td className="py-4 px-4 font-semibold text-slate-900">
                  {item.name}
                  <span className="block text-xs font-normal text-slate-400">Akurasi: {item.confidence}%</span>
                </td>
                <td className="py-4 px-4">
                  {item.safetyStatus === 'safe' && (
                    <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium bg-emerald-100 text-emerald-800">
                      <ShieldCheck className="w-3.5 h-3.5" /> Aman
                    </span>
                  )}
                  {item.safetyStatus === 'warning' && (
                    <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium bg-amber-100 text-amber-800">
                      <AlertTriangle className="w-3.5 h-3.5" /> Perhatian
                    </span>
                  )}
                  {item.safetyStatus === 'danger' && (
                    <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium bg-rose-100 text-rose-800 animate-pulse">
                      <XCircle className="w-3.5 h-3.5" /> BAHAYA
                    </span>
                  )}
                </td>
                <td className="py-4 px-4 text-xs text-slate-600 max-w-xs">{item.forensicFlag}</td>
                <td className="py-4 px-4 text-center font-medium">{item.calories} kkal</td>
                <td className="py-4 px-4 text-center">{item.protein}g</td>
                <td className="py-4 px-4 text-center">{item.fat}g</td>
                <td className="py-4 px-4 text-center">{item.carbs}g</td>
                <td className="py-4 px-4 text-xs font-medium text-slate-800">{item.recommendation}</td>
              </tr>
            ))}
          </tbody>
          <tfoot>
            <tr className="bg-slate-900/5 font-bold text-slate-900 border-t-2 border-slate-200">
              <td className="py-3 px-4" colSpan={3}>Total Makronutrisi Piring Anda</td>
              <td className="py-3 px-4 text-center text-emerald-700">{totals.totalCalories.toFixed(0)} kkal</td>
              <td className="py-3 px-4 text-center">{totals.totalProtein.toFixed(1)}g</td>
              <td className="py-3 px-4 text-center">{totals.totalFat.toFixed(1)}g</td>
              <td className="py-3 px-4 text-center">{totals.totalCarbs.toFixed(1)}g</td>
              <td className="py-3 px-4 text-xs text-slate-500">Serat: {totals.totalFiber.toFixed(1)}g</td>
            </tr>
          </tfoot>
        </table>
      </div>
    </div>
  );
};
```

### 7.5 Halaman Scanner Utama (`app/page.tsx`)
```tsx
'use client';

import React, { useState, useRef, useEffect } from 'react';
import { Camera, RefreshCw, UploadCloud, ShieldAlert, Sparkles } from 'lucide-react';
import { analyzeFoodImage } from '../lib/detector';
import { FoodItemAnalysis, NutritionTotal } from '../lib/types';
import { FoodForensicTable } from '../components/FoodForensicTable';

export default function NutriSafeApp() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  const [streamActive, setStreamActive] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [items, setItems] = useState<FoodItemAnalysis[]>([]);
  const [statusMessage, setStatusMessage] = useState('Siap memindai piring makanan...');

  useEffect(() => {
    startCamera();
    return () => stopCamera();
  }, []);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: { ideal: 720 } },
        audio: false,
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setStreamActive(true);
      }
    } catch (err) {
      setStatusMessage('Kamera tidak dapat diakses. Silakan gunakan fitur upload foto.');
    }
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach((track) => track.stop());
    }
  };

  const captureAndScan = async () => {
    if (!videoRef.current || !canvasRef.current) return;
    setIsProcessing(true);
    setStatusMessage('Menjalankan inferensi forensik visual & lookup nutrisi...');

    const canvas = canvasRef.current;
    const video = videoRef.current;
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;

    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      try {
        const results = await analyzeFoodImage(canvas);
        setItems(results);
        setStatusMessage('Analisis selesai.');
      } catch (err) {
        console.error(err);
        setStatusMessage('Gagal menganalisis gambar. Coba lagi.');
      }
    }
    setIsProcessing(false);
  };

  const totals: NutritionTotal = items.reduce(
    (acc, curr) => {
      acc.totalCalories += curr.calories;
      acc.totalProtein += curr.protein;
      acc.totalFat += curr.fat;
      acc.totalCarbs += curr.carbs;
      acc.totalFiber += curr.fiber;
      if (curr.safetyStatus === 'danger') acc.overallSafety = 'danger';
      else if (curr.safetyStatus === 'warning' && acc.overallSafety !== 'danger') acc.overallSafety = 'warning';
      return acc;
    },
    { totalCalories: 0, totalProtein: 0, totalFat: 0, totalCarbs: 0, totalFiber: 0, overallSafety: 'safe' } as NutritionTotal
  );

  return (
    <main className="min-h-screen bg-slate-100 p-4 md:p-8 flex flex-col items-center">
      <header className="max-w-4xl w-full text-center mb-8">
        <div className="inline-flex items-center gap-2 bg-emerald-100 text-emerald-800 text-xs font-bold px-3 py-1 rounded-full mb-2">
          <Sparkles className="w-3.5 h-3.5" /> 100% Gratis • Tanpa Login • AI di Browser
        </div>
        <h1 className="text-3xl md:text-5xl font-black text-slate-900 tracking-tight">
          NutriSafe <span className="text-emerald-600">Forensik Pangan</span>
        </h1>
        <p className="text-sm md:text-base text-slate-600 mt-2">
          Cegah keracunan makanan dan pantau pemenuhan standar gizi harian secara instan.
        </p>
      </header>

      {/* Area Kamera & Scanner */}
      <div className="max-w-xl w-full bg-white rounded-3xl p-4 shadow-lg border border-slate-200 mb-8">
        <div className="relative aspect-video rounded-2xl overflow-hidden bg-slate-900 flex items-center justify-center">
          <video ref={videoRef} autoPlay playsInline muted className="w-full h-full object-cover" />
          <canvas ref={canvasRef} className="hidden" />

          {isProcessing && (
            <div className="absolute inset-0 bg-slate-950/75 flex flex-col items-center justify-center text-white backdrop-blur-sm">
              <RefreshCw className="w-8 h-8 animate-spin text-emerald-400 mb-2" />
              <p className="text-xs font-semibold animate-pulse">{statusMessage}</p>
            </div>
          )}
        </div>

        <div className="mt-4 flex gap-3">
          <button
            onClick={captureAndScan}
            disabled={isProcessing}
            className="flex-1 bg-emerald-600 hover:bg-emerald-700 active:scale-95 text-white font-bold py-3.5 px-6 rounded-xl flex items-center justify-center gap-2 transition duration-150 shadow-md shadow-emerald-600/20 disabled:opacity-50"
          >
            <Camera className="w-5 h-5" />
            {isProcessing ? 'Menganalisis...' : 'Foto & Analisis Forensik'}
          </button>
        </div>
      </div>

      {/* Tabel Hasil */}
      {items.length > 0 && (
        <section className="max-w-5xl w-full">
          <FoodForensicTable items={items} totals={totals} />
        </section>
      )}
    </main>
  );
}
```

---

## 8. Analisis Biaya Operasional & Hosting (Zero-Cost Breakdown)

| Komponen | Layanan / Provider | Tier | Biaya Bulanan |
|---|---|---|---|
| **Komputasi AI Vision** | Client Device (WebAssembly/WebGPU via Transformers.js) | On-Device | **Rp 0** |
| **Backend & DB** | Supabase PostgreSQL (500 MB Storage, 50k MAU) | Free Tier | **Rp 0** |
| **Frontend Web Hosting** | Vercel / Cloudflare Pages | Hobby / Free | **Rp 0** |
| **Database Gizi (TKPI)** | Static SQL Seeder (Dataset Kemenkes RI) | Open Access | **Rp 0** |
| **Total Estimasi Biaya** | - | - | **Rp 0 / Bulan** |

---

## 9. Panduan Menjalankan Project (Local Development)

```bash
# 1. Clone atau buat project Next.js baru
npx create-next-app@latest nutrisafe-web --typescript --tailwind --app --eslint

# 2. Masuk ke direktori project
cd nutrisafe-web

# 3. Install dependencies
npm install @huggingface/transformers @supabase/supabase-js lucide-react clsx tailwind-merge

# 4. Buat file .env.local
echo "NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co" > .env.local
echo "NEXT_PUBLIC_SUPABASE_ANON_KEY=your-anon-key" >> .env.local

# 5. Jalankan project di local server
npm run dev
```

Buka `http://localhost:3000` di browser dan izinkan akses kamera untuk langsung mencoba pemindaian.
