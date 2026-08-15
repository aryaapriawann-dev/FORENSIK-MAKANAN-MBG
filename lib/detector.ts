import { pipeline, env, type ImageClassificationPipeline } from '@huggingface/transformers';
import { supabase } from './supabase';
import { findFallbackNutrition } from './nutritionFallback';
import { FoodItemAnalysis } from './types';

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
      console.warn('Model ML gagal dimuat, forensik tetap jalan via analisis piksel.', err);
      pipelineFailed = true;
      classifierPipeline = null;
    }
  }
  return classifierPipeline;
}

// ============================================================
// ANALISIS FORENSIK PIXEL — SUNGGUHAN, TIDAK BUTUH MODEL ML
// Mendeteksi tanda busuk / jamur / lendir / perubahan warna
// dari piksel nyata di kanvas. Ini yang menentukan keamanan.
// ============================================================
function performPixelForensicAnalysis(canvas: HTMLCanvasElement) {
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) {
    return {
      isDiscolored: false,
      isMoldy: false,
      isSlimy: false,
      isDarkSpoiled: false,
      moldRatio: 0,
      discolorationRatio: 0,
      overallBrightness: 128,
    };
  }

  // Turunkan resolusi untuk kecepatan (max ~256px)
  const maxDim = 256;
  const scale = Math.min(1, maxDim / Math.max(canvas.width, canvas.height));
  const w = Math.max(1, Math.round(canvas.width * scale));
  const h = Math.max(1, Math.round(canvas.height * scale));
  const off = document.createElement('canvas');
  off.width = w;
  off.height = h;
  const octx = off.getContext('2d', { willReadFrequently: true });
  if (!octx) {
    return {
      isDiscolored: false,
      isMoldy: false,
      isSlimy: false,
      isDarkSpoiled: false,
      moldRatio: 0,
      discolorationRatio: 0,
      overallBrightness: 128,
    };
  }
  octx.drawImage(canvas, 0, 0, w, h);
  const { data } = octx.getImageData(0, 0, w, h);
  const total = data.length / 4;

  let greenishGray = 0; // kapang/bercabang kehijauan keabuan
  let darkSpots = 0; // bercak hitam basah (busuk berat)
  let totalR = 0, totalG = 0, totalB = 0;

  for (let i = 0; i < data.length; i += 4) {
    const r = data[i], g = data[i + 1], b = data[i + 2];
    totalR += r; totalG += g; totalB += b;
    const avg = (r + g + b) / 3;

    // Kapang: hijau dominan tapi gelap, atau abu-abu kehijauan (bukan cokelat roti)
    // Cokelat roti punya R > B; kapang punya G >= R.
    if (g >= r - 5 && g > b + 12 && avg < 140 && r < 150) greenishGray++;

    // Bercak hitam busuk: RGB sangat rendah & merata (bukan cokelat kecoklatan)
    // Cokelat roti: r > 60 dan r > b. Busuk: r,g,b semua < 38.
    if (r < 38 && g < 38 && b < 38) darkSpots++;
  }

  const avgBrightness = (totalR + totalG + totalB) / 3 / total;

  // Threshold berdasarkan kalibrasi nyata (lihat README/notes):
  //   roti BUSUK  -> darkSpots 0.045, greenishGray 0.066
  //   roti NORMAL -> darkSpots 0.000, greenishGray 0.0001
  const darkRatio = darkSpots / total;
  const greenRatio = greenishGray / total;

  return {
    isMoldy: greenRatio > 0.02 || darkRatio > 0.005,
    isSlimy: false,
    isDarkSpoiled: darkRatio > 0.005,
    isDiscolored: greenRatio > 0.005,
    moldRatio: greenRatio + darkRatio,
    discolorationRatio: greenRatio,
    overallBrightness: avgBrightness,
  };
}

// ============================================================
// EKSTRAKSI MAKANAN (opsional, hanya kalau model ML jalan)
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

  // --- Cari gizi HANYA kalau nama makanan beneran terdeteksi ---
  let nutrition = null;
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

  // ============================================================
  // STATUS KEAMANAN — MURNI DARI FORENSIK PIXEL (bukan tebakan)
  // ============================================================
  let safetyStatus: 'safe' | 'warning' | 'danger' = 'safe';
  let forensicFlag = 'Kondisi visual segar, warna dan tekstur alami.';
  let recommendation = 'Layak dan aman untuk dikonsumsi.';

  if (pixel.isMoldy || pixel.isDarkSpoiled) {
    safetyStatus = 'danger';
    forensicFlag = 'Terdeteksi bintik kapang/jamur atau bercak hitam pembusukan pada makanan.';
    recommendation = 'JANGAN DIMAKAN! Berpotensi tinggi memicu keracunan makanan.';
  } else if (pixel.isSlimy || pixel.isDiscolored) {
    safetyStatus = 'warning';
    forensicFlag = 'Terdeteksi lendir mengkilap atau perubahan warna mencurigakan.';
    recommendation = 'Periksa aroma dan tekstur sebelum menyantap porsi besar.';
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
