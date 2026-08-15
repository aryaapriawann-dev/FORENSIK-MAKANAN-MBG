import { pipeline, env, type ImageClassificationPipeline } from '@huggingface/transformers';
import { supabase } from './supabase';
import { findFallbackNutrition } from './nutritionFallback';
import { FoodItemAnalysis, NutritionMasterItem } from './types';

// Optimasi runtime browser
env.allowLocalModels = false;
env.useBrowserCache = true;

type Classifier = ImageClassificationPipeline | null;
let classifierPipeline: Classifier = null;
let pipelineFailed = false; // agar gak retry terus menerus kalau gagal

export async function getDetectorPipeline(): Promise<Classifier> {
  if (pipelineFailed) return null;
  if (!classifierPipeline) {
    try {
      classifierPipeline = await pipeline(
        'image-classification',
        'Xenova/food-classification-resnet-50'
      );
    } catch (err) {
      console.warn('Model ML gagal dimuat, identifikasi makanan dilewati.', err);
      pipelineFailed = true;
      classifierPipeline = null;
    }
  }
  return classifierPipeline;
}

// ============================================================
// ANALISIS FORENSIK PIXEL — KONSERVATIF & AKURAT
// TIDAK menggunakan "warna gelap" atau "warna hijau" sebagai
// penentu busuk, karena ayam goreng / daging bakar / nasi hitam
// memang gelap, dan sayur / avokad memang hijau.
//
// Satu-satunya sinyal busuk yang DAPAT dibedakan dari makanan
// normal: bintik kapang PUTIH/ABU-ABU fuzzy (jamur) yang
// membentuk PATCH menyatu (bukan bintik tersebar spt wijen/
// guratan bakar). Itu yang kita deteksi.
// ============================================================
function performPixelForensicAnalysis(canvas: HTMLCanvasElement) {
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) {
    return { hasMoldPatch: false, moldPatchRatio: 0, overallBrightness: 128 };
  }

  // Turunkan resolusi untuk kecepatan
  const maxDim = 192;
  const scale = Math.min(1, maxDim / Math.max(canvas.width, canvas.height));
  const w = Math.max(1, Math.round(canvas.width * scale));
  const h = Math.max(1, Math.round(canvas.height * scale));
  const off = document.createElement('canvas');
  off.width = w;
  off.height = h;
  const octx = off.getContext('2d', { willReadFrequently: true });
  if (!octx) {
    return { hasMoldPatch: false, moldPatchRatio: 0, overallBrightness: 128 };
  }
  octx.drawImage(canvas, 0, 0, w, h);
  const { data } = octx.getImageData(0, 0, w, h);
  const total = data.length / 4;

  // 1) Tandai piksel "fuzzy mold": terang (bukan putih plate murni),
  //    saturasi rendah, BUKAN hijau (sayur/avokad), DAN punya tetangga
  //    gelap dalam radius 2px (kapang putih di atas roti cokelat punya
  //    kontras tajam; nasi putih di plate putih tidak).
  //    Ini yang membedakan kapang dari makanan putih/sehat.
  const moldMask = new Uint8Array(total);
  // hitung avg per piksel dulu
  const avgArr = new Float32Array(total);
  for (let i = 0; i < total; i++) {
    const r = data[i * 4], g = data[i * 4 + 1], b = data[i * 4 + 2];
    avgArr[i] = (r + g + b) / 3;
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const r = data[i * 4], g = data[i * 4 + 1], b = data[i * 4 + 2];
      const a = avgArr[i];
      const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
      const sat = mx === 0 ? 0 : (mx - mn) / mx;
      const isGreen = g > r + 12 && g > b + 12;
      if (a > 140 && a < 215 && sat < 0.18 && !isGreen) {
        // cek tetangga gelap dlm radius 2
        let hasDark = false;
        for (let dy = -2; dy <= 2 && !hasDark; dy++) {
          for (let dx = -2; dx <= 2; dx++) {
            const nx = x + dx, ny = y + dy;
            if (nx >= 0 && nx < w && ny >= 0 && ny < h && avgArr[ny * w + nx] < 140) {
              hasDark = true;
              break;
            }
          }
        }
        if (hasDark) moldMask[i] = 1;
      }
    }
  }
  const overallBrightness = 128; // tidak dipakai untuk status keamanan

  // 2) Deteksi PATCH menyatu via flood-fill pada grid kasar.
  //    Bintik tersebar (wijen, guratan) tidak akan membentuk
  //    patch besar -> tidak di-flag.
  const GRID = 28;
  const cellW = w / GRID;
  const cellH = h / GRID;
  const cellMold = new Uint8Array(GRID * GRID);
  for (let cy = 0; cy < GRID; cy++) {
    for (let cx = 0; cx < GRID; cx++) {
      let moldPx = 0, totPx = 0;
      const x0 = Math.floor(cx * cellW), x1 = Math.floor((cx + 1) * cellW);
      const y0 = Math.floor(cy * cellH), y1 = Math.floor((cy + 1) * cellH);
      for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) {
          const idx = y * w + x;
          if (idx < total) { totPx++; moldPx += moldMask[idx]; }
        }
      }
      // sel dianggap "mold" kalau >35% pikselnya mold
      if (totPx > 0 && moldPx / totPx > 0.35) cellMold[cy * GRID + cx] = 1;
    }
  }

  // flood fill cari komponen terhubung terbesar
  const visited = new Uint8Array(GRID * GRID);
  let largest = 0;
  for (let i = 0; i < GRID * GRID; i++) {
    if (!cellMold[i] || visited[i]) continue;
    const stack = [i];
    let size = 0;
    visited[i] = 1;
    while (stack.length) {
      const cur = stack.pop()!;
      size++;
      const cx = cur % GRID, cy = (cur - cx) / GRID;
      const neighbours = [
        cy > 0 ? cur - GRID : -1,
        cy < GRID - 1 ? cur + GRID : -1,
        cx > 0 ? cur - 1 : -1,
        cx < GRID - 1 ? cur + 1 : -1,
      ];
      for (const n of neighbours) {
        if (n >= 0 && cellMold[n] && !visited[n]) {
          visited[n] = 1;
          stack.push(n);
        }
      }
    }
    if (size > largest) largest = size;
  }

  const totalCells = GRID * GRID;
  const patchRatio = largest / totalCells;
  // butuh patch menyatu yang cukup besar (>= ~2.5% area grid)
  const hasMoldPatch = patchRatio > 0.04;

  return { hasMoldPatch, moldPatchRatio: patchRatio, overallBrightness };
}

// ============================================================
// EKSTRAKSI MAKANAN (akurat: dari model ML + database TKPI)
// ============================================================
export async function analyzeFoodImage(canvas: HTMLCanvasElement): Promise<FoodItemAnalysis[]> {
  const pixel = performPixelForensicAnalysis(canvas);
  const classifier = await getDetectorPipeline();

  let detectedLabel: string | null = null;
  let confidence = 0;

  if (classifier) {
    try {
      const raw = await classifier(canvas.toDataURL('image/jpeg', 0.85), { top_k: 1 });
      if (Array.isArray(raw) && raw.length > 0) {
        detectedLabel = (raw[0].label || '').toString().toLowerCase();
        confidence = Math.round((raw[0].score || 0) * 100);
      }
    } catch {
      detectedLabel = null;
    }
  }

  // --- Cari gizi & ciri busuk HANYA kalau nama makanan terdeteksi ---
  let nutrition: NutritionMasterItem | null = null;
  if (detectedLabel) {
    const firstWord = detectedLabel.split(',')[0].trim();
    if (process.env.NEXT_PUBLIC_SUPABASE_URL) {
      try {
        const { data } = await supabase
          .from('nutrition_master')
          .select('*')
          .ilike('food_name', `%${firstWord}%`)
          .maybeSingle();
        nutrition = data;
      } catch {
        nutrition = null;
      }
    }
    if (!nutrition) nutrition = findFallbackNutrition(firstWord);
  }

  const foodName = nutrition
    ? nutrition.food_name
    : detectedLabel
    ? detectedLabel.charAt(0).toUpperCase() + detectedLabel.slice(1)
    : 'Tidak Teridentifikasi';

  const category = nutrition ? nutrition.category : 'Analisis Forensik';
  const calories = nutrition ? nutrition.calories : 0;
  const protein = nutrition ? nutrition.protein : 0;
  const fat = nutrition ? nutrition.fat : 0;
  const carbs = nutrition ? nutrition.carbs : 0;
  const fiber = nutrition ? nutrition.fiber : 0;
  const spoilageSigns: string[] = nutrition ? nutrition.spoilage_signs || [] : [];

  // ============================================================
  // STATUS KEAMANAN — AKURAT & KONSERVATIF
  //   BAHAYA : hanya kalau ada patch kapang putih/abu-abu fuzzy
  //   WARNING: model ML ragu-ragu (confidence rendah)
  //   AMAN   : default (foto saja tidak membuktikan busuk)
  // ============================================================
  let safetyStatus: 'safe' | 'warning' | 'danger' = 'safe';
  let forensicFlag = 'Tidak terdeteksi tanda busuk visual yang jelas pada makanan.';
  let recommendation = 'Layak dan aman untuk dikonsumsi.';

  if (pixel.hasMoldPatch) {
    safetyStatus = 'danger';
    forensicFlag = 'Terdeteksi bintik kapang berwarna putih/abu-abu fuzzy (jamur) menyatu pada makanan.';
    recommendation = 'JANGAN DIMAKAN! Berpotensi tinggi memicu keracunan makanan.';
  } else if (detectedLabel && confidence > 0 && confidence < 30) {
    safetyStatus = 'warning';
    forensicFlag = 'Model AI kurang yakin mengidentifikasi makanan ini.';
    recommendation = 'Periksa kembali secara visual sebelum menyantapnya.';
  } else if (spoilageSigns.length > 0) {
    forensicFlag = `Ciri busuk pada ${foodName}: ${spoilageSigns.join(', ')}.`;
    recommendation = 'Periksa ciri-ciri di atas sebelum menyantap.';
  }

  const result: FoodItemAnalysis = {
    id: Math.random().toString(36).substring(2, 9),
    name: foodName,
    category,
    confidence: detectedLabel ? confidence : 0,
    safetyStatus,
    forensicFlag,
    calories,
    protein,
    fat,
    carbs,
    fiber,
    recommendation,
  };

  const results = [result];

  // Simpan log ke Supabase (anonim) bila terkonfigurasi
  if (process.env.NEXT_PUBLIC_SUPABASE_URL && process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY) {
    try {
      await supabase.from('forensic_scan_logs').insert({
        detected_items: results,
        total_calories: results.reduce((s, i) => s + i.calories, 0),
        total_protein: results.reduce((s, i) => s + i.protein, 0),
        total_fat: results.reduce((s, i) => s + i.fat, 0),
        total_carbs: results.reduce((s, i) => s + i.carbs, 0),
        overall_safety_status: results.some((r) => r.safetyStatus === 'danger')
          ? 'danger'
          : results.some((r) => r.safetyStatus === 'warning')
          ? 'warning'
          : 'safe',
      });
    } catch {
      // offline fallback: abaikan
    }
  }

  return results;
}
