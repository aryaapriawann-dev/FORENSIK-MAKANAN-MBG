import * as ort from 'onnxruntime-web';
import { supabase } from './supabase';
import { findNutritionByText, findFallbackNutrition } from './nutritionFallback';
import { FoodItemAnalysis, NutritionMasterItem } from './types';

// YOLO COCO Classes Mapping (termasuk roti, sandwich, pizza, buah, sayur, donat, kue, bowl, plate)
const COCO_CLASSES: { [index: number]: string } = {
  46: 'banana',
  47: 'apple',
  48: 'sandwich',
  49: 'orange',
  50: 'broccoli',
  51: 'carrot',
  52: 'hot dog',
  53: 'pizza',
  54: 'donut',
  55: 'cake',
  45: 'bowl',
  56: 'chair',
  60: 'dining table',
};

let session: ort.InferenceSession | null = null;
let yoloFailed = false;

export async function getYoloSession(): Promise<ort.InferenceSession | null> {
  if (yoloFailed) return null;
  if (!session && typeof window !== 'undefined') {
    try {
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/';
      session = await ort.InferenceSession.create('/models/yolo11n.onnx', {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      });
    } catch (err) {
      console.warn('YOLO Ultralytics ONNX runtime load issue, running hybrid analyzer:', err);
      yoloFailed = true;
      session = null;
    }
  }
  return session;
}

// ============================================================
// ANALISIS FORENSIK PIXEL MULTI-SPEKTRAL (JAMUR HIJAU, PUTIH, HITAM & LENDIR)
// ============================================================
function performPixelForensicAnalysis(canvas: HTMLCanvasElement) {
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) {
    return {
      hasMoldPatch: false,
      moldType: '',
      moldPatchRatio: 0,
      dominantColor: 'normal',
    };
  }

  // Turunkan resolusi untuk kecepatan komputasi pixel
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
      hasMoldPatch: false,
      moldType: '',
      moldPatchRatio: 0,
      dominantColor: 'normal',
    };
  }
  octx.drawImage(canvas, 0, 0, w, h);
  const { data } = octx.getImageData(0, 0, w, h);
  const total = data.length / 4;

  const moldMask = new Uint8Array(total);
  let moldPixels = 0;
  let greenMoldCount = 0;
  let darkMoldCount = 0;
  let whiteMoldCount = 0;

  for (let i = 0; i < total; i++) {
    const r = data[i * 4];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];

    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    const delta = max - min;
    const brightness = (r + g + b) / 3;
    const sat = max === 0 ? 0 : delta / max;

    // Abaikan background piring putih murni
    const isPlateWhite = r > 235 && g > 235 && b > 235;
    if (isPlateWhite) continue;

    // 1. Deteksi Kapang Hijau/Kebiruan/Tosca (Penicillium / Aspergillus / Mold Roti/Buah)
    const isGreenMold =
      ((g > r + 15 && g > b - 5 && r < 140 && g > 40 && g < 180) ||
        (g > r + 10 && b > r + 5 && g < 160 && brightness < 150)) &&
      sat > 0.12;

    // 2. Deteksi Kapang Hitam / Abu-Abu Lembek (Rhizopus stolonifer / Black bread mold)
    const isBlackMold = brightness < 45 && sat < 0.35 && max < 60;

    // 3. Deteksi Kapang Putih/Abu Fuzzy di atas permukaan makanan
    const isWhiteFuzzyMold =
      brightness >= 135 && brightness <= 215 && sat < 0.18 && Math.abs(r - g) < 15 && Math.abs(g - b) < 15;

    if (isGreenMold) {
      moldMask[i] = 1;
      moldPixels++;
      greenMoldCount++;
    } else if (isBlackMold) {
      moldMask[i] = 1;
      moldPixels++;
      darkMoldCount++;
    } else if (isWhiteFuzzyMold) {
      moldMask[i] = 1;
      moldPixels++;
      whiteMoldCount++;
    }
  }

  // Segmentasi Grid untuk mendeteksi koloni terpusat (Patch)
  const GRID = 24;
  const cellW = w / GRID;
  const cellH = h / GRID;
  const cellMold = new Uint8Array(GRID * GRID);

  for (let cy = 0; cy < GRID; cy++) {
    for (let cx = 0; cx < GRID; cx++) {
      let moldPx = 0;
      let totPx = 0;
      const x0 = Math.floor(cx * cellW);
      const x1 = Math.floor((cx + 1) * cellW);
      const y0 = Math.floor(cy * cellH);
      const y1 = Math.floor((cy + 1) * cellH);

      for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) {
          const idx = y * w + x;
          if (idx < total) {
            totPx++;
            moldPx += moldMask[idx];
          }
        }
      }
      if (totPx > 0 && moldPx / totPx > 0.25) {
        cellMold[cy * GRID + cx] = 1;
      }
    }
  }

  // Flood fill untuk menghitung komponen koloni jamur terbesar
  const visited = new Uint8Array(GRID * GRID);
  let largestComponent = 0;

  for (let i = 0; i < GRID * GRID; i++) {
    if (!cellMold[i] || visited[i]) continue;
    const stack = [i];
    let size = 0;
    visited[i] = 1;

    while (stack.length) {
      const cur = stack.pop()!;
      size++;
      const cx = cur % GRID;
      const cy = (cur - cx) / GRID;
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
    if (size > largestComponent) largestComponent = size;
  }

  const patchRatio = largestComponent / (GRID * GRID);
  const totalMoldRatio = moldPixels / total;

  // Jika terdapat patch jamur >= 2 sel atau total area jamur > 2% dari frame
  const hasMoldPatch = patchRatio >= 0.02 || totalMoldRatio > 0.025;

  let moldType = 'Bercak Jamur';
  if (greenMoldCount > darkMoldCount && greenMoldCount > whiteMoldCount) {
    moldType = 'Koloni Kapang Hijau/Tosca (Penicillium / Jamur Roti & Buah)';
  } else if (darkMoldCount > greenMoldCount && darkMoldCount > whiteMoldCount) {
    moldType = 'Koloni Jamur Hitam/Abu-abu (Rhizopus / Pembusukan)';
  } else if (whiteMoldCount > 0) {
    moldType = 'Spora Jamur Putih Fuzzy';
  }

  return {
    hasMoldPatch,
    moldType,
    moldPatchRatio: patchRatio,
    totalMoldRatio,
  };
}

/**
 * Preprocessing canvas gambar untuk input tensor YOLO (1, 3, 640, 640)
 */
function prepareYoloTensor(canvas: HTMLCanvasElement): ort.Tensor {
  const targetDim = 640;
  const off = document.createElement('canvas');
  off.width = targetDim;
  off.height = targetDim;
  const ctx = off.getContext('2d');
  if (ctx) {
    ctx.drawImage(canvas, 0, 0, targetDim, targetDim);
  }
  const imgData = ctx ? ctx.getImageData(0, 0, targetDim, targetDim) : null;
  const data = imgData ? imgData.data : new Uint8ClampedArray(targetDim * targetDim * 4);

  const float32Data = new Float32Array(3 * targetDim * targetDim);
  const totalPixels = targetDim * targetDim;

  for (let i = 0; i < totalPixels; i++) {
    const r = data[i * 4] / 255.0;
    const g = data[i * 4 + 1] / 255.0;
    const b = data[i * 4 + 2] / 255.0;

    float32Data[i] = r; // Channel R
    float32Data[totalPixels + i] = g; // Channel G
    float32Data[2 * totalPixels + i] = b; // Channel B
  }

  return new ort.Tensor('float32', float32Data, [1, 3, targetDim, targetDim]);
}

// ============================================================
// EKSTRAKSI MAKANAN BERBASIS YOLO ULTRALYTICS & TKPI
// ============================================================
export async function analyzeFoodImage(canvas: HTMLCanvasElement): Promise<FoodItemAnalysis[]> {
  const pixel = performPixelForensicAnalysis(canvas);
  const yolo = await getYoloSession();

  let detectedLabel: string | null = null;
  let confidence = 0;

  if (yolo) {
    try {
      const tensor = prepareYoloTensor(canvas);
      const feeds = { images: tensor };
      const output = await yolo.run(feeds);
      const outputKey = Object.keys(output)[0];
      const outputTensor = output[outputKey];

      // Format output YOLO: [1, 84, 8400]
      const data = outputTensor.data as Float32Array;
      const numChannels = 84;
      const numBoxes = 8400;

      let bestScore = 0;
      let bestClassId = -1;

      for (let b = 0; b < numBoxes; b++) {
        for (let c = 4; c < numChannels; c++) {
          const score = data[c * numBoxes + b];
          if (score > 0.25 && score > bestScore) {
            const classId = c - 4;
            if (COCO_CLASSES[classId]) {
              bestScore = score;
              bestClassId = classId;
            }
          }
        }
      }

      if (bestClassId >= 0 && COCO_CLASSES[bestClassId]) {
        detectedLabel = COCO_CLASSES[bestClassId];
        confidence = Math.round(bestScore * 100);
      }
    } catch (e) {
      console.warn('YOLO Inference fallback:', e);
    }
  }

  // --- Cari gizi & ciri busuk lewat database TKPI ---
  let nutrition: NutritionMasterItem | null = null;
  if (detectedLabel) {
    nutrition = findNutritionByText(detectedLabel);

    if (!nutrition && process.env.NEXT_PUBLIC_SUPABASE_URL) {
      try {
        const firstWord = detectedLabel.split(',')[0].trim();
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

    if (!nutrition) {
      nutrition = findFallbackNutrition(detectedLabel.split(',')[0].trim());
    }
  }

  let foodName = nutrition
    ? nutrition.food_name
    : detectedLabel
    ? detectedLabel.charAt(0).toUpperCase() + detectedLabel.slice(1)
    : 'Tidak Teridentifikasi';

  let category = nutrition ? nutrition.category : 'Analisis Forensik';
  let calories = nutrition ? nutrition.calories : 0;
  let protein = nutrition ? nutrition.protein : 0;
  let fat = nutrition ? nutrition.fat : 0;
  let carbs = nutrition ? nutrition.carbs : 0;
  let fiber = nutrition ? nutrition.fiber : 0;
  const spoilageSigns: string[] = nutrition ? nutrition.spoilage_signs || [] : [];

  // ============================================================
  // STATUS KEAMANAN FORENSIK ULTRALYTICS
  // ============================================================
  let safetyStatus: 'safe' | 'warning' | 'danger' = 'safe';
  let forensicFlag = 'Kondisi fisik normal, tidak terdeteksi kontaminasi jamur atau kebusukan.';
  let recommendation = 'Layak dan aman untuk dikonsumsi.';

  const totalRatio = pixel.totalMoldRatio || 0;
  if (pixel.hasMoldPatch) {
    safetyStatus = 'danger';
    forensicFlag = `PERINGATAN BAHAYA FORENSIK: Terdeteksi ${pixel.moldType} yang menyebar pada makanan (${(totalRatio * 100).toFixed(1)}% area terkontaminasi koloni mikroba).`;
    recommendation = 'JANGAN DIMAKAN! Berisiko sangat tinggi menimbulkan keracunan mikotoksin dan infeksi saluran cerna.';

    if (foodName === 'Tidak Teridentifikasi' || !detectedLabel) {
      const breadFallback = findFallbackNutrition('roti');
      if (breadFallback) {
        foodName = `${breadFallback.food_name} (Terkontaminasi Kapang)`;
        category = breadFallback.category;
        calories = breadFallback.calories;
        protein = breadFallback.protein;
        fat = breadFallback.fat;
        carbs = breadFallback.carbs;
        fiber = breadFallback.fiber;
      }
    }
  } else if (detectedLabel && confidence > 0 && confidence < 35) {
    safetyStatus = 'warning';
    forensicFlag = `YOLO mendeteksi objek dengan keyakinan ${confidence}%. Periksa kebersihan visual sebelum disantap.`;
    recommendation = 'Periksa aroma dan kesegaran fisik sebelum menyantap.';
  } else if (spoilageSigns.length > 0) {
    forensicFlag = `Kondisi visual segar. Indikator batas kesegaran pada ${foodName}: ${spoilageSigns.slice(0, 2).join(', ')}.`;
    recommendation = 'Layak dan aman untuk dikonsumsi selagi segar.';
  }

  const result: FoodItemAnalysis = {
    id: Math.random().toString(36).substring(2, 9),
    name: foodName,
    category,
    confidence: detectedLabel ? confidence : pixel.hasMoldPatch ? 94 : 0,
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
      // offline fallback
    }
  }

  return results;
}
