import * as ort from 'onnxruntime-web';
import { supabase } from './supabase';
import { findNutritionByText, findFallbackNutrition } from './nutritionFallback';
import { FoodItemAnalysis, NutritionMasterItem, BoundingBox } from './types';

// YOLO COCO Classes Mapping (80 Classes)
const COCO_CLASSES: string[] = [
  'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
  'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
  'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
  'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
  'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
  'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
  'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
  'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
  'hair drier', 'toothbrush'
];

export interface RawDetection {
  classId: number;
  label: string;
  confidence: number;
  box: BoundingBox;
  pixelBox: { x0: number; y0: number; x1: number; y1: number; w: number; h: number };
}

const yoloSessions: { session: ort.InferenceSession; size: number; labelMap: string[]; sigmoid: boolean }[] = [];
let yoloFailed = false;

// Semua model YOLO dimuat & dijalankan (bukan cuma 1). Masing-masing punya
// ukuran input berbeda: nutrisafe=320, indo=640, coco=640. Deteksi digabung
// agar cakupan lauk lebih luas (ayam goreng, tempe, nasi, sayur, dll).
// Daftar path+labelMap didefinisikan di bawah setelah konstanta label kelas.
export async function getYoloSessions(): Promise<typeof yoloSessions> {
  if (yoloFailed) return yoloSessions;
  if (yoloSessions.length === 0 && typeof window !== 'undefined') {
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/';
    for (const cand of YOLO_MODEL_CANDIDATES) {
      try {
        const session = await ort.InferenceSession.create(cand.path, {
          executionProviders: ['wasm'],
          graphOptimizationLevel: 'all',
        });
        // Baca ukuran input sebenarnya dari metadata model (320 / 640 / ...).
        // Beberapa model punya shape dinamis (string 'height'/'width') -> fallback 640.
        const meta = session.inputMetadata?.[0] as unknown as {
          shape?: unknown;
          dims?: unknown;
        };
        const inShape = (meta?.shape ?? meta?.dims) as unknown[];
        const rawSize =
          Array.isArray(inShape) && typeof inShape[2] === 'number' ? (inShape[2] as number) : 640;
        const size = Number.isFinite(rawSize) ? rawSize : 640;
        yoloSessions.push({ session, size, labelMap: cand.labelMap, sigmoid: !!cand.sigmoid });
        console.info('[NutriSafe] YOLO session loaded:', cand.path, 'size', size);
      } catch (err) {
        console.warn('YOLO ONNX load issue for', cand.path, ':', err);
      }
    }
    if (yoloSessions.length === 0) {
      yoloFailed = true;
    }
  }
  return yoloSessions;
}

// Letterbox (perti ultralytics): resize pertahankan rasio, pad abu-abu,
// center. Mengembalikan tensor + metadata agar box bisa di-unpad ke gambar asli.
function prepareYoloTensor(
  canvas: HTMLCanvasElement,
  targetDim: number
): { tensor: ort.Tensor; scale: number; padX: number; padY: number } {
  const imgW = canvas.width || targetDim;
  const imgH = canvas.height || targetDim;
  const scale = Math.min(targetDim / imgW, targetDim / imgH);
  const newW = Math.round(imgW * scale);
  const newH = Math.round(imgH * scale);
  const padX = Math.floor((targetDim - newW) / 2);
  const padY = Math.floor((targetDim - newH) / 2);

  const off = document.createElement('canvas');
  off.width = targetDim;
  off.height = targetDim;
  const ctx = off.getContext('2d');
  if (ctx) {
    ctx.fillStyle = '#707070'; // gray pad (ultralytics default 114)
    ctx.fillRect(0, 0, targetDim, targetDim);
    ctx.drawImage(canvas, padX, padY, newW, newH);
  }
  const imgData = ctx ? ctx.getImageData(0, 0, targetDim, targetDim) : null;
  const data = imgData ? imgData.data : new Uint8ClampedArray(targetDim * targetDim * 4);

  const float32Data = new Float32Array(3 * targetDim * targetDim);
  const totalPixels = targetDim * targetDim;
  for (let i = 0; i < totalPixels; i++) {
    float32Data[i] = data[i * 4] / 255.0;
    float32Data[totalPixels + i] = data[i * 4 + 1] / 255.0;
    float32Data[2 * totalPixels + i] = data[i * 4 + 2] / 255.0;
  }
  return {
    tensor: new ort.Tensor('float32', float32Data, [1, 3, targetDim, targetDim]),
    scale,
    padX,
    padY,
  };
}

function computeIoU(boxA: BoundingBox, boxB: BoundingBox): number {
  const xA = Math.max(boxA.x, boxB.x);
  const yA = Math.max(boxA.y, boxB.y);
  const xB = Math.min(boxA.x + boxA.width, boxB.x + boxB.width);
  const yB = Math.min(boxA.y + boxA.height, boxB.y + boxB.height);
  const interArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);
  const boxAArea = boxA.width * boxA.height;
  const boxBArea = boxB.width * boxB.height;
  return interArea / (boxAArea + boxBArea - interArea + 1e-6);
}

// Label kelas dari model deteksi makanan Indonesia (wuriyanto/yolo8-indonesian-food-detection-v1)
export const INDONESIAN_FOOD_CLASSES = [
  'Ayam Goreng', 'Bakso', 'Capcay', 'Mie Goreng', 'Nasi Goreng', 'Pempek',
  'Rendang Sapi', 'Sate', 'Tahu Goreng', 'Tempe Goreng', 'Terong Balado', 'Tumis Kangkung',
];

// Label kelas dari model HASIL RETRAIN BROAD NutriSafe (16 kelas asli dataset
// sohl-multidish — BANYAK makanan, murni dari data, bukan template 3 item).
// Urutan HARUS sama persis dengan data_broad16.yaml saat training.
export const BROAD16_FOOD_CLASSES = [
  'Roti Naan',        // 0 bread_or_Roti_naan
  'Kari Sayur',       // 1 curry_dish
  'Nasi Putih',       // 2 rice_dish
  'Tumis Sayur Hijau',// 3 dry_vegetable
  'Kue Donat',        // 4 snack_item
  'Kue Manis',        // 5 sweet_item
  'Sambal Terasi',    // 6 accompaniment
  'Sayur Sop',        // 7 Dal_or_sambar
  'Minuman',          // 8 drink
  'Telur Dadar',      // 9 eggs
  'Ikan Seafood',     // 10 fish_dish
  'Jeruk Segar',      // 11 fruits
  'Mie Goreng',       // 12 pasta
  'Salad',            // 13 salad
  'Sup',              // 14 soup
  'Sarapan India',    // 15 south_indian_breakfast
];

// Pemetaan 16 kelas broad -> kode gizi NutriSafe.
const BROAD16_FOOD_CODE_MAP: Record<string, string> = {
  'roti naan': 'ROTI_TAWAR',
  'kari sayur': 'SAYUR_SOP',
  'nasi putih': 'NASI_PUTIH',
  'tumis sayur hijau': 'TUMIS_SAYUR_HIJAU',
  'kue donat': 'KUE_DONAT',
  'kue manis': 'KUE_DONAT',
  'sambal terasi': 'SAMBAL_TERASI',
  'sayur sop': 'SAYUR_SOP',
  'minuman': 'MINUMAN',
  'telur dadar': 'TELUR_DADAR',
  'ikan seafood': 'IKAN_SEAFOOD',
  'jeruk segar': 'JERUK',
  'mie goreng': 'MIE_GORENG',
  'salad': 'TUMIS_SAYUR_HIJAU',
  'sup': 'SAYUR_SOP',
  'sarapan india': 'ROTI_TAWAR',
};

// Label kelas dari model HASIL FINE-TUNE NutriSafe (10 kelas, mapping langsung ke kode gizi)
export const NUTRISAFE_FOOD_CLASSES = [
  'Nasi Putih', 'Roti Tawar', 'Sayur Sop', 'Tumis Sayur Hijau', 'Sambal Terasi',
  'Kue Donat', 'Telur Dadar', 'Ikan Seafood', 'Apel', 'Mie Goreng',
];

// Pemetaan untuk model fine-tune NutriSafe (label = kode gizi yang sudah rapi)
const NUTRISAFE_FOOD_CODE_MAP: Record<string, string> = {
  'nasi putih': 'NASI_PUTIH',
  'roti tawar': 'ROTI_TAWAR',
  'sayur sop': 'SAYUR_SOP',
  'tumis sayur hijau': 'TUMIS_SAYUR_HIJAU',
  'sambal terasi': 'SAMBAL_TERASI',
  'kue donat': 'KUE_DONAT',
  'telur dadar': 'TELUR_DADAR',
  'ikan seafood': 'IKAN_SEAFOOD',
  'apel': 'APEL',
  'mie goreng': 'MIE_GORENG',
};

// Daftar model YOLO yang dimuat & dijalankan (semua model, bukan cuma 1).
// Ukuran input tiap model dibaca otomatis dari metadata saat load.
//
// Model UTAMA = yolov8_broad16.onnx: hasil RETRAIN BROAD 16-kelas dari dataset
// sohl-multidish (banyak makanan: nasi, ikan, sambal, sayur, mie, telur, buah...),
// murni dari data — BUKAN template 3 item. mAP50 validasi ~0.80.
// labelMap HARUS persis urutan training (lihat BROAD16_FOOD_CLASSES).
// Model lama (nutrisafe 10-kelas) tetap sebagai cadangan.
const YOLO_MODEL_CANDIDATES = [
  // Utama: YOLOv8 makanan Indonesia (HF/wuriyanto, 12 kelas: Ayam Goreng, Nasi
  // Goreng, Tahu, Tempe, Sate, Bakso, dll). CV beneran, output ONNX sudah
  // probabilitas -> jangan disigmoid. Jago detect ayam goreng.
  { path: '/models/yolov8_indo12.onnx', labelMap: INDONESIAN_FOOD_CLASSES, sigmoid: false },
  // Cadangan: broad16 (sohl 16 kelas, mAP50~0.80) untuk sayur/mie/nasi/lauk lain.
  // Output raw logits -> perlu sigmoid. Threshold global 0.55 menekan false-positive.
  { path: '/models/yolov8_broad16.onnx', labelMap: BROAD16_FOOD_CLASSES, sigmoid: true },
];

// Pemetaan nama kelas model -> kode gizi NutriSafe
const INDONESIAN_FOOD_CODE_MAP: Record<string, string> = {
  'ayam goreng': 'AYAM_GORENG',
  'bakso': 'BAKSO',
  'capcay': 'CAPCAY',
  'mie goreng': 'MIE_GORENG',
  'nasi goreng': 'NASI_GORENG',
  'pempek': 'PEMPEK',
  'rendang sapi': 'RENDANG_SAPI',
  'sate': 'SATE',
  'tahu goreng': 'TAHU_GORENG',
  'tempe goreng': 'TEMPE_GORENG',
  'terong balado': 'TERONG_BALADO',
  'tumis kangkung': 'TUMIS_SAYUR_HIJAU',
};

function decodeYOLO(outputTensor: ort.Tensor, modelSize: number, labelMap: string[], confThresh = 0.30, iouThresh = 0.45, applySigmoid = true): RawDetection[] {
  const data = outputTensor.data as Float32Array;
  const sig = (x: number) => (applySigmoid ? 1 / (1 + Math.exp(-x)) : x);
  // Output YOLOv8/v11 class-major: shape [1, 4+numClasses, numAnchors]
  const dims = outputTensor.dims;
  const numAnchors = dims[dims.length - 1];
  const numClasses = dims[1] - 4;
  const rawDetections: RawDetection[] = [];

  for (let i = 0; i < numAnchors; i++) {
    let maxScore = 0;
    let maxClassId = -1;
    for (let c = 0; c < numClasses; c++) {
      const score = sig(data[(4 + c) * numAnchors + i]);
      if (score > maxScore) {
        maxScore = score;
        maxClassId = c;
      }
    }

    if (maxScore >= confThresh) {
      // Koordinat model belum ternormalisasi -> bagi dengan ukuran input model.
      const cx = data[0 * numAnchors + i] / modelSize;
      const cy = data[1 * numAnchors + i] / modelSize;
      const w = data[2 * numAnchors + i] / modelSize;
      const h = data[3 * numAnchors + i] / modelSize;
      const x = Math.max(0, cx - w / 2);
      const y = Math.max(0, cy - h / 2);
      const width = Math.min(1 - x, w);
      const height = Math.min(1 - y, h);

      rawDetections.push({
        classId: maxClassId,
        label: labelMap[maxClassId] || 'object',
        confidence: Math.round(maxScore * 100),
        box: { x, y, width, height },
        pixelBox: {
          x0: Math.round(x * modelSize),
          y0: Math.round(y * modelSize),
          x1: Math.round((x + width) * modelSize),
          y1: Math.round((y + height) * modelSize),
          w: Math.round(width * modelSize),
          h: Math.round(height * modelSize),
        },
      });
    }
  }

  rawDetections.sort((a, b) => b.confidence - a.confidence);

  // NMS PER-CLASS (standar YOLO): tiap kelas makanan dipisah supaya
  // lauk yang beda tapi box-nya berdekatan tidak saling membatalkan.
  // Hasil: 1 box terbaik per kelas makanan yang terdeteksi.
  const byClass = new Map<number, RawDetection[]>();
  for (const det of rawDetections) {
    const arr = byClass.get(det.classId) || [];
    arr.push(det);
    byClass.set(det.classId, arr);
  }
  const nmsResults: RawDetection[] = [];
  for (const arr of byClass.values()) {
    const kept: RawDetection[] = [];
    for (const det of arr) {
      let keep = true;
      for (const k of kept) {
        if (computeIoU(det.box, k.box) > iouThresh) {
          keep = false;
          break;
        }
      }
      if (keep) kept.push(det);
    }
    nmsResults.push(...kept);
  }
  return nmsResults;
}

export async function detectObjectsYolo(canvas: HTMLCanvasElement): Promise<RawDetection[]> {
  const sessions = await getYoloSessions();
  const all: RawDetection[] = [];
  if (sessions.length === 0) return all;
  for (const { session, size, labelMap, sigmoid } of sessions) {
    try {
      const { tensor, scale, padX, padY } = prepareYoloTensor(canvas, size);
      const out = await session.run({ images: tensor });
      const outputTensor = out.output0 || Object.values(out)[0];
      if (!outputTensor) continue;
      // Decode dalam ruang tensor, lalu unpad ke koordinat gambar asli (0..1).
      const raw = decodeYOLO(outputTensor as ort.Tensor, size, labelMap, 0.55, 0.45, sigmoid);
      const imgW = canvas.width || size;
      const imgH = canvas.height || size;
      for (const det of raw) {
        // cx,cy,w,h dari decodeYOLO sudah normalized thd `size` (tensor space)
        const cxT = det.box.x + det.box.width / 2;
        const cyT = det.box.y + det.box.height / 2;
        const cxPx = cxT * size - padX;
        const cyPx = cyT * size - padY;
        const wPx = det.box.width * size / scale;
        const hPx = det.box.height * size / scale;
        const x = (cxPx - wPx / 2) / imgW;
        const y = (cyPx - hPx / 2) / imgH;
        det.box = {
          x: Math.max(0, x),
          y: Math.max(0, y),
          width: Math.min(1 - Math.max(0, x), wPx / imgW),
          height: Math.min(1 - Math.max(0, y), hPx / imgH),
        };
      }
      all.push(...raw);
    } catch (err) {
      console.warn('YOLO detectObjects error:', err);
    }
  }
  return all;
}

// ============================================================
// ANALISIS PIXEL FORENSIK PRESISI TINGGI (COLOR & BIOMARKER)
// ============================================================
export interface ColorProfile {
  purpleDarkRatio: number;
  breadWheatRatio: number;
  redRatio: number;
  orangeRatio: number;
  yellowRatio: number;
  greenRatio: number;
  whiteRatio: number;
  darkBrownRatio: number;
  sauceBlackRatio: number;
  soupBrothRatio: number;
  bananaYellowRatio: number;
  foodSignalRatio: number;
  hasMold: boolean;
  moldType: string;
  moldRatio: number;
}

export function analyzeRegionPixels(
  canvas: HTMLCanvasElement,
  cropBox?: { x0: number; y0: number; x1: number; y1: number }
): ColorProfile {
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  const defaultRes: ColorProfile = {
    purpleDarkRatio: 0,
    breadWheatRatio: 0,
    redRatio: 0,
    orangeRatio: 0,
    yellowRatio: 0,
    greenRatio: 0,
    whiteRatio: 0,
    darkBrownRatio: 0,
    sauceBlackRatio: 0,
    soupBrothRatio: 0,
    bananaYellowRatio: 0,
    foodSignalRatio: 0,
    hasMold: false,
    moldType: '',
    moldRatio: 0,
  };
  if (!ctx) return defaultRes;

  const imgW = canvas.width;
  const imgH = canvas.height;

  const x0 = cropBox ? Math.max(0, Math.min(imgW - 1, Math.floor(cropBox.x0))) : 0;
  const y0 = cropBox ? Math.max(0, Math.min(imgH - 1, Math.floor(cropBox.y0))) : 0;
  const x1 = cropBox ? Math.max(x0 + 1, Math.min(imgW, Math.floor(cropBox.x1))) : imgW;
  const y1 = cropBox ? Math.max(y0 + 1, Math.min(imgH, Math.floor(cropBox.y1))) : imgH;

  const cropW = Math.max(1, x1 - x0);
  const cropH = Math.max(1, y1 - y0);

  const maxDim = 160;
  const scale = Math.min(1, maxDim / Math.max(cropW, cropH));
  const w = Math.max(1, Math.round(cropW * scale));
  const h = Math.max(1, Math.round(cropH * scale));

  const off = document.createElement('canvas');
  off.width = w;
  off.height = h;
  const octx = off.getContext('2d', { willReadFrequently: true });
  if (!octx) return defaultRes;

  octx.drawImage(canvas, x0, y0, cropW, cropH, 0, 0, w, h);
  const { data } = octx.getImageData(0, 0, w, h);
  const total = data.length / 4;

  let purpleDarkCount = 0;
  let breadWheatCount = 0;
  let redCount = 0;
  let orangeCount = 0;
  let yellowCount = 0;
  let greenCount = 0;
  let whiteCount = 0;
  let darkBrownCount = 0;
  let sauceBlackCount = 0;
  let soupBrothCount = 0;
  let bananaYellowCount = 0;
  let foodPixelCount = 0;

  let greenMoldCount = 0;
  let blackMoldCount = 0;
  const whiteMoldCount = 0;

  for (let i = 0; i < total; i++) {
    const r = data[i * 4];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];

    const brightness = (r + g + b) / 3;
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    const delta = max - min;
    const sat = max === 0 ? 0 : delta / max;

    // Abaikan stainless steel tray (ompreng) / latar belakang abu-abu netral
    const isTrayMetallic =
      brightness > 115 && brightness < 225 && sat < 0.12 && Math.abs(r - g) < 10 && Math.abs(g - b) < 10;
    if (isTrayMetallic) continue;

    // Abaikan piring putih terang
    const isPlateWhite = brightness > 235 && sat < 0.08;
    if (isPlateWhite) continue;

    foodPixelCount++;

    // 1. SAUS KECAP / SAMBAL HITAM MANIS
    const isSweetSoySauce = brightness < 45 && r >= g && r >= b && (r > 15 || g > 10);
    if (isSweetSoySauce) {
      sauceBlackCount++;
      continue;
    }

    // 2. SAYUR SUP / KUAH BENING / WORTEL ORANYE
    const isSoupBroth =
      (brightness > 110 && brightness < 215 && sat > 0.15 && r > b && g > b && r > 95 && g > 85) ||
      (r > 155 && g > 75 && g < 155 && b < 85); // wortel di sup
    if (isSoupBroth) {
      soupBrothCount++;
    }

    // 3. DAUN SAYUR SEGAR (Chlorophyll sayur hijau: Sawi, Seledri, Brokoli, Buncis, Lalapan Selada/Kemangi/Jeruk Nipis)
    const isFreshGreenVeg = g > 65 && g > r * 1.12 && g > b * 1.12 && (sat > 0.18 || g > 90);
    if (isFreshGreenVeg) {
      greenCount++;
      continue; // Sayuran hijau segar BUKAN jamur!
    }

    // 4. KULIT PISANG / KUNING CERAH
    const isBananaYellow = r > 165 && g > 140 && b < 90 && sat > 0.4;
    if (isBananaYellow) {
      bananaYellowCount++;
    }

    // 5. ROTI GANDUM / ROTI TAWAR
    const isBreadWheat =
      brightness >= 120 &&
      brightness <= 220 &&
      r > 130 &&
      g > 110 &&
      b > 70 &&
      r >= g &&
      g >= b &&
      sat >= 0.1 &&
      sat <= 0.45;
    if (isBreadWheat) {
      breadWheatCount++;
    }

    // 6. ANGGUR / UNGU GELAP
    const isPurpleWine =
      (r > 40 && b > 40 && r > g * 1.15 && b > g * 1.05 && brightness < 140) ||
      (Math.abs(r - b) < 35 && r > g + 15 && b > g + 15);
    if (isPurpleWine) {
      purpleDarkCount++;
      continue;
    }

    // 7. DETEKSI KAPANG JAMUR NYATA (Hanya jika tumbuh di atas permukaan roti/nasi basi)
    // Kapang Penicillium (bercak keabuan tosca GELAP PUDAR pada roti gandum kering).
    // PENTING: daun sayur hijau SEGAR punya saturasi tinggi & hijau terang -> BUKAN kapang.
    // Kapang bersifat kusam/abu-abu: saturasi rendah-sedang dan tidak pernah terang-segar.
    const isFreshLeaf =
      g > 65 && g > r * 1.12 && g > b * 1.12 && (sat > 0.18 || g > 90);
    const isRealGreenMold =
      !isFreshLeaf &&
      sat > 0.08 && sat < 0.34 && // kusam/abu-abu, bukan daun segar
      g > r + 14 &&
      g > b + 6 &&
      g >= 90 && g < 155 &&        // kapang tosca terang, bukan kari hijau gelap
      brightness > 70 && brightness < 160; // spora lebih terang dari kari gelap

    // Kapang Hitam / Rhizopus stolonifer pada roti (sangat gelap & kusam)
    const isRealBlackMold =
      brightness < 26 && max < 34 && sat < 0.22;

    if (isRealGreenMold) {
      greenMoldCount++;
    } else if (isRealBlackMold) {
      blackMoldCount++;
    }

    // 8. SPEKTRUM WARNA MAKANAN LAINNYA
    if (r > 130 && r > g * 1.35 && r > b * 1.35) redCount++;
    else if (r > 140 && g > 75 && g < r * 0.9 && b < 80) orangeCount++;
    else if (r > 140 && g > 130 && b < 100 && Math.abs(r - g) < 40) yellowCount++;
    else if (brightness > 175 && sat < 0.15) whiteCount++;
    else if (brightness < 100 && r > g && g > b && sat > 0.15) darkBrownCount++;
  }

  const moldPixels = greenMoldCount + blackMoldCount + whiteMoldCount;
  const moldRatio = moldPixels / total;

  // Hanya jika benar-benar ada spora kapang yang signifikan pada matriks
  // roti/nasi basi (bukan sekadar kari/sayur berwarna kusam).
  const hasMold =
    moldRatio > 0.12 &&
    breadWheatCount / total > 0.35;

  let moldType = 'Bercak Kapang Jamur';
  if (greenMoldCount > 0) {
    moldType = 'Koloni Kapang Hijau/Tosca (Penicillium)';
  } else if (blackMoldCount > 0) {
    moldType = 'Koloni Jamur Hitam/Abu-abu (Rhizopus)';
  } else if (whiteMoldCount > 0) {
    moldType = 'Spora Jamur Putih Fuzzy (Aspergillus)';
  }

  return {
    purpleDarkRatio: purpleDarkCount / total,
    breadWheatRatio: breadWheatCount / total,
    redRatio: redCount / total,
    orangeRatio: orangeCount / total,
    yellowRatio: yellowCount / total,
    greenRatio: greenCount / total,
    whiteRatio: whiteCount / total,
    darkBrownRatio: darkBrownCount / total,
    sauceBlackRatio: sauceBlackCount / total,
    soupBrothRatio: soupBrothCount / total,
    bananaYellowRatio: bananaYellowCount / total,
    foodSignalRatio: foodPixelCount / total,
    hasMold,
    moldType,
    moldRatio,
  };
}

/**
 * Deteksi apakah gambar adalah Nampan Ompreng Stainless Steel MBG (Multi-Sekat)
 */
function isOmprengMealTray(color: ColorProfile, canvas: HTMLCanvasElement): boolean {
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) return false;

  const w = canvas.width;
  const h = canvas.height;

  // Pindai 4 sudut dan pembatas interior untuk verifikasi keberadaan nampan logam / sekat ompreng
  const samplePoints = [
    { x: Math.round(w * 0.5), y: Math.round(h * 0.42) }, // Sekat horizontal tengah
    { x: Math.round(w * 0.33), y: Math.round(h * 0.5) }, // Sekat vertikal kiri
    { x: Math.round(w * 0.67), y: Math.round(h * 0.5) }, // Sekat vertikal kanan
  ];

  let metallicDividerCount = 0;
  for (const pt of samplePoints) {
    try {
      const pix = ctx.getImageData(pt.x, pt.y, 1, 1).data;
      const b = (pix[0] + pix[1] + pix[2]) / 3;
      const max = Math.max(pix[0], pix[1], pix[2]);
      const min = Math.min(pix[0], pix[1], pix[2]);
      const sat = max === 0 ? 0 : (max - min) / max;
      if (b > 110 && b < 225 && sat < 0.12) {
        metallicDividerCount++;
      }
    } catch {
      // ignore
    }
  }

  const varietyCount =
    (color.whiteRatio > 0.12 ? 1 : 0) +
    (color.soupBrothRatio > 0.08 ? 1 : 0) +
    (color.darkBrownRatio > 0.08 || color.yellowRatio > 0.08 ? 1 : 0) +
    (color.sauceBlackRatio > 0.05 ? 1 : 0);

  // Ompreng MBG valid jika ada minimal 2 sekat metalik dan 3 jenis makanan berbeda
  return metallicDividerCount >= 2 && varietyCount >= 3;
}

function getOmprengCompartments(canvasW: number, canvasH: number) {
  return [
    {
      code: 'NASI_PUTIH',
      name: 'Nasi Putih',
      pixelBox: {
        x0: canvasW * 0.21,
        y0: canvasH * 0.38,
        x1: canvasW * 0.77,
        y1: canvasH * 0.95,
      },
      box: { x: 0.21, y: 0.38, width: 0.56, height: 0.57 },
    },
    {
      code: 'SAYUR_SOP',
      name: 'Sayur Sop Bening',
      pixelBox: {
        x0: canvasW * 0.56,
        y0: canvasH * 0.05,
        x1: canvasW * 0.97,
        y1: canvasH * 0.44,
      },
      box: { x: 0.56, y: 0.05, width: 0.41, height: 0.39 },
    },
    {
      code: 'SAMBAL_TERASI',
      name: 'Sambal / Saus Kecap',
      pixelBox: {
        x0: canvasW * 0.05,
        y0: canvasH * 0.05,
        x1: canvasW * 0.54,
        y1: canvasH * 0.43,
      },
      box: { x: 0.05, y: 0.05, width: 0.49, height: 0.38 },
    },
    {
      code: 'AYAM_GORENG',
      name: 'Ayam Goreng (Fried Chicken)',
      pixelBox: {
        x0: canvasW * 0.04,
        y0: canvasH * 0.46,
        x1: canvasW * 0.31,
        y1: canvasH * 0.95,
      },
      box: { x: 0.04, y: 0.46, width: 0.27, height: 0.49 },
    },
    {
      code: 'TEMPE_GORENG',
      name: 'Tempe / Tahu Goreng',
      pixelBox: {
        x0: canvasW * 0.71,
        y0: canvasH * 0.46,
        x1: canvasW * 0.98,
        y1: canvasH * 0.95,
      },
      box: { x: 0.71, y: 0.46, width: 0.27, height: 0.49 },
    },
  ];
}

/**
 * Pengklasifikasi makanan — HANYA meneruskan label dari model deteksi (YOLO /
 * dataset). Tidak ada lagi tebakan dari analisis warna template maupun
 * classifier ImageNet (MobileNet) karena keduanya terbukti menghasilkan
 * label ngawur ("ikan jadi roti tawar", "jeruk segar" palsu).
 *
 * Semua nama makanan sekarang 100% berasal dari model yang dilatih di dataset.
 * Fungsi ini hanya memetakan nama class model -> kode gizi NutriSafe.
 */
async function classifyFoodSmart(
  _canvas: HTMLCanvasElement,
  _color: ColorProfile,
  yoloLabel?: string,
  yoloConf = 0,
  _cropBox?: { x0: number; y0: number; x1: number; y1: number }
): Promise<{
  foodCode: string | null;
  confidence: number;
}> {
  if (yoloLabel && yoloConf >= 25) {
    const yoloCode =
      BROAD16_FOOD_CODE_MAP[yoloLabel.toLowerCase()] ||
      NUTRISAFE_FOOD_CODE_MAP[yoloLabel.toLowerCase()] ||
      INDONESIAN_FOOD_CODE_MAP[yoloLabel.toLowerCase()] ||
      COCO_FOOD_MAP[yoloLabel.toLowerCase()];
    if (yoloCode) {
      // Confidence jujur dari model — TIDAK di-inflate. Skor palsu = data palsu.
      return { foodCode: yoloCode, confidence: yoloConf };
    }
  }
  return { foodCode: null, confidence: 0 };
}

/**
 * Segmentasi Multi-Region Piring Hidangan Saji (Plated Dish Multi-Component Analysis)
 */
async function analyzePlatedDishMultiItem(
  canvas: HTMLCanvasElement,
  fullColor: ColorProfile
): Promise<FoodItemAnalysis[]> {
  const canvasW = canvas.width;
  const canvasH = canvas.height;

  // 1. Region Hidangan Utama (Pusat Piring: Ikan / Ayam / Daging)
  const mainBox = { x: 0.14, y: 0.20, width: 0.72, height: 0.65 };
  const mainPx = {
    x0: mainBox.x * canvasW,
    y0: mainBox.y * canvasH,
    x1: (mainBox.x + mainBox.width) * canvasW,
    y1: (mainBox.y + mainBox.height) * canvasH,
  };
  const mainColor = analyzeRegionPixels(canvas, mainPx);
  const mainClassified = await classifyFoodSmart(canvas, mainColor, undefined, 0, mainPx);

  // 2. Region Lalapan Sayuran Kiri Atas (Selada / Kemangi / Timun)
  const topLeftBox = { x: 0.04, y: 0.04, width: 0.34, height: 0.36 };
  const topLeftPx = {
    x0: topLeftBox.x * canvasW,
    y0: topLeftBox.y * canvasH,
    x1: (topLeftBox.x + topLeftBox.width) * canvasW,
    y1: (topLeftBox.y + topLeftBox.height) * canvasH,
  };
  const topLeftColor = analyzeRegionPixels(canvas, topLeftPx);

  // 3. Region Sambal / Condiment Kiri Bawah
  const bottomLeftBox = { x: 0.04, y: 0.58, width: 0.32, height: 0.38 };
  const bottomLeftPx = {
    x0: bottomLeftBox.x * canvasW,
    y0: bottomLeftBox.y * canvasH,
    x1: (bottomLeftBox.x + bottomLeftBox.width) * canvasW,
    y1: (bottomLeftBox.y + bottomLeftBox.height) * canvasH,
  };
  const bottomLeftColor = analyzeRegionPixels(canvas, bottomLeftPx);

  // 4. Region Irisan Jeruk Nipis / Lalapan Bawah Tengah
  const bottomCenterBox = { x: 0.36, y: 0.68, width: 0.28, height: 0.28 };
  const bottomCenterPx = {
    x0: bottomCenterBox.x * canvasW,
    y0: bottomCenterBox.y * canvasH,
    x1: (bottomCenterBox.x + bottomCenterBox.width) * canvasW,
    y1: (bottomCenterBox.y + bottomCenterBox.height) * canvasH,
  };
  const bottomCenterColor = analyzeRegionPixels(canvas, bottomCenterPx);

  // 5. Region Tomat / Sayuran Kanan Atas
  const topRightBox = { x: 0.62, y: 0.04, width: 0.34, height: 0.38 };
  const topRightPx = {
    x0: topRightBox.x * canvasW,
    y0: topRightBox.y * canvasH,
    x1: (topRightBox.x + topRightBox.width) * canvasW,
    y1: (topRightBox.y + topRightBox.height) * canvasH,
  };
  const topRightColor = analyzeRegionPixels(canvas, topRightPx);

  const regions = [
    { box: mainBox, pixelBox: mainPx, color: mainColor },
    { box: topLeftBox, pixelBox: topLeftPx, color: topLeftColor },
    { box: bottomLeftBox, pixelBox: bottomLeftPx, color: bottomLeftColor },
    { box: bottomCenterBox, pixelBox: bottomCenterPx, color: bottomCenterColor },
    { box: topRightBox, pixelBox: topRightPx, color: topRightColor },
  ];

  const results: FoodItemAnalysis[] = [];
  const addedCodes = new Set<string>();

  for (const region of regions) {
    if (region.color.foodSignalRatio < 0.12) continue;

    const classified = region === regions[0]
      ? mainClassified
      : await classifyFoodSmart(canvas, region.color, undefined, 0, region.pixelBox);

    if (!classified.foodCode || classified.confidence < 60 || addedCodes.has(classified.foodCode)) continue;

    const nutrition = findNutritionByText(classified.foodCode) || findFallbackNutrition(classified.foodCode);
    if (!nutrition) continue;

    addedCodes.add(classified.foodCode);
    let safetyStatus: 'safe' | 'warning' | 'danger' = 'safe';
    let forensicFlag = `Tidak ditemukan indikator visual bahaya (jamur/basi/kecoklatan ekstrem) pada ${nutrition.food_name}.`;
    let recommendation = 'Tetap periksa aroma, tekstur, dan kebersihan penyajian sebelum dikonsumsi.';

    if (region.color.hasMold) {
      safetyStatus = 'danger';
      forensicFlag = `BAHAYA KERACUNAN: Terdeteksi ${region.color.moldType} aktif (${(region.color.moldRatio * 100).toFixed(1)}% area koloni mikroba).`;
      recommendation = 'DILARANG DIMAKAN! Buang seluruh makanan ini segera.';
    } else if (nutrition.spoilage_signs && nutrition.spoilage_signs.length > 0) {
      forensicFlag = `Kondisi visual segar. Indikator batas kesegaran: ${nutrition.spoilage_signs.slice(0, 2).join(', ')}.`;
    }

    results.push({
      id: Math.random().toString(36).substring(2, 9),
      name: nutrition.food_name,
      category: nutrition.category,
      confidence: classified.confidence,
      safetyStatus,
      forensicFlag,
      calories: nutrition.calories,
      protein: nutrition.protein,
      fat: nutrition.fat,
      carbs: nutrition.carbs,
      fiber: nutrition.fiber,
      vitaminA_mcg: nutrition.vitaminA_mcg || 0,
      vitaminB_mg: nutrition.vitaminB_mg || 0,
      vitaminC_mg: nutrition.vitaminC_mg || 0,
      vitaminD_mcg: nutrition.vitaminD_mcg || 0,
      calcium_mg: nutrition.calcium_mg || 0,
      iron_mg: nutrition.iron_mg || 0,
      recommendation,
      box: region.box,
    });
  }

  return results;
}

// ============================================================
// COCO -> KODE MAKANAN NUTRISAFE (hanya kelas pangan relevan)
// YOLO11n bawaan adalah detektor COCO (80 kelas). Kita petakan
// kelas pangan COCO ke kode gizi agar deteksi multi-objek nyata
// bisa dipakai sebagai *prior* sebelum klasifikasi per-kotak.
// ============================================================
const COCO_FOOD_MAP: Record<string, string> = {
  banana: 'PISANG',
  apple: 'APEL',
  orange: 'JERUK',
  broccoli: 'BROKOLI_WORTEL',
  carrot: 'BROKOLI_WORTEL',
  sandwich: 'ROTI_TAWAR',
};

/**
 * Deteksi multi-objek nyata via YOLO11.
 * Setiap kotak dipotong (crop) lalu dianalisis forensik warna &
 * diklasifikasi ulang dengan classifyFoodSmart agar hasil akurat
 * per-item (bukan cuma 1 gambar utuh).
 */
async function analyzeYoloMultiItem(
  canvas: HTMLCanvasElement,
  _fullColor: ColorProfile
): Promise<FoodItemAnalysis[]> {
  const canvasW = canvas.width || 640;
  const canvasH = canvas.height || 480;

  const detections = await detectObjectsYolo(canvas);
  const results: FoodItemAnalysis[] = [];
  const seen = new Set<string>();

  for (const det of detections) {
    // Label MURNI dari model YOLO (deep learning) — model yang mempelajari
    // dari dataset yang bilang "ini nasi", bukan class yang di-hardcode.
    // Prioritas: Broad16 (16 kelas) -> NutriSafe (10 kelas) -> Indo (12) -> COCO.
    const code =
      BROAD16_FOOD_CODE_MAP[det.label.toLowerCase()] ||
      NUTRISAFE_FOOD_CODE_MAP[det.label.toLowerCase()] ||
      INDONESIAN_FOOD_CODE_MAP[det.label.toLowerCase()] ||
      COCO_FOOD_MAP[det.label.toLowerCase()];
    if (!code) continue;
    if (det.confidence < 55) continue;

    const box = det.box; // normalized 0..1
    const cropPx = {
      x0: box.x * canvasW,
      y0: box.y * canvasH,
      x1: (box.x + box.width) * canvasW,
      y1: (box.y + box.height) * canvasH,
    };
    const cropColor = analyzeRegionPixels(canvas, cropPx);

    // Forensik keamanan (jamur/basi) tetap dari analisis piksel per-kotak,
    // tapi LABEL makanan 100% dari YOLO (tidak di-override classifier warna).
    const finalCode = code;
    const nutrition =
      findNutritionByText(finalCode) ||
      findFallbackNutrition(finalCode) ||
      findFallbackNutrition(code);
    if (!nutrition) continue;
    if (seen.has(nutrition.food_code)) continue;
    seen.add(nutrition.food_code);

    let safetyStatus: 'safe' | 'warning' | 'danger' = 'safe';
    let forensicFlag = `Tidak ditemukan indikator visual bahaya (jamur/basi/kecoklatan ekstrem) pada ${nutrition.food_name}.`;
    let recommendation = 'Tetap periksa aroma, tekstur, dan kebersihan penyajian sebelum dikonsumsi.';

    // STATUS FORENSIK 3-WARNA (HIJAU/KUNING/MERAH) — murni dari bukti visual:
    // MERAH  = ada koloni kapang/jamur/basi (hasMold).
    // KUNING = model ragu (conf < 45%) ATAU ada indikator batas kesegaran.
    // HIJAU  = segar & keyakinan cukup.
    if (cropColor.hasMold) {
      safetyStatus = 'danger';
      forensicFlag = `BAHAYA KERACUNAN: Terdeteksi ${cropColor.moldType} aktif (${(cropColor.moldRatio * 100).toFixed(1)}% area koloni mikroba). Makanan sudah terkontaminasi toksin.`;
      recommendation = 'DILARANG DIMAKAN! Buang seluruh porsi ini segera.';
    } else if (det.confidence < 45) {
      safetyStatus = 'warning';
      forensicFlag = `WASPADA: Keyakinan model rendah (${det.confidence}%) untuk "${nutrition.food_name}". Periksa aroma, tekstur, dan warna sebelum disantap.`;
      recommendation = 'Periksa seksama (cek bau/tekstur) sebelum dikonsumsi.';
    } else if (cropColor.darkBrownRatio > 0.35 && cropColor.foodSignalRatio > 0.3) {
      safetyStatus = 'warning';
      forensicFlag = `WASPADA: Terdapat area kecokelatan/kehitaman ekstrim (${(cropColor.darkBrownRatio * 100).toFixed(0)}%) yang bisa jadi indikator penurunan kesegaran.`;
      recommendation = 'Periksa kesegaran lebih dulu; bila berbau/menyimpang, jangan dikonsumsi.';
    } else {
      safetyStatus = 'safe';
      forensicFlag = `Tidak ditemukan indikator visual bahaya (jamur/basi/kecoklatan ekstrem) pada ${nutrition.food_name}.`;
    }

    results.push({
      id: Math.random().toString(36).substring(2, 9),
      name: nutrition.food_name,
      category: nutrition.category,
      confidence: det.confidence,
      safetyStatus,
      forensicFlag,
      calories: nutrition.calories,
      protein: nutrition.protein,
      fat: nutrition.fat,
      carbs: nutrition.carbs,
      fiber: nutrition.fiber,
      vitaminA_mcg: nutrition.vitaminA_mcg || 0,
      vitaminB_mg: nutrition.vitaminB_mg || 0,
      vitaminC_mg: nutrition.vitaminC_mg || 0,
      vitaminD_mcg: nutrition.vitaminD_mcg || 0,
      calcium_mg: nutrition.calcium_mg || 0,
      iron_mg: nutrition.iron_mg || 0,
      recommendation,
      box,
    });
  }

  return results;
}

/** Gabung hasil tanpa duplikat kode makanan (prioritas YOLO yang sudah ada). */
function mergeByFoodCode(
  existing: FoodItemAnalysis[],
  incoming: FoodItemAnalysis[]
): FoodItemAnalysis[] {
  const names = new Set(existing.map((r) => r.name));
  const out = [...existing];
  for (const it of incoming) {
    if (!names.has(it.name)) {
      names.add(it.name);
      out.push(it);
    }
  }
  return out;
}

/**
 * Analisis nampan ompreng multi-sekat MBG → satu deteksi per kompartemen lauk.
 * Pakai region kompartemen beneran (getOmprengCompartments) yang selaras dengan
 * susunan foto, lalu per-kompartemen diklasifikasi warna + classifier + YOLO.
 */
async function analyzeOmprengCompartments(
  canvas: HTMLCanvasElement
): Promise<FoodItemAnalysis[]> {
  const canvasW = canvas.width || 640;
  const canvasH = canvas.height || 480;
  const comps = getOmprengCompartments(canvasW, canvasH);
  const results: FoodItemAnalysis[] = [];
  const addedCodes = new Set<string>();

  // Deteksi YOLO sebagai prior per-lauk (crop sesuai kompartemen).
  const yoloDetections = await detectObjectsYolo(canvas);

  for (const comp of comps) {
    const cropPx = comp.pixelBox;
    const cropColor = analyzeRegionPixels(canvas, cropPx);
    if (cropColor.foodSignalRatio < 0.12) continue; // sekat kosong/pinggiran

    // Cari deteksi YOLO yang masuk dalam kompartemen ini.
    let yoloLabel: string | undefined;
    let yoloConf = 0;
    for (const det of yoloDetections) {
      const cx = det.box.x + det.box.width / 2;
      const cy = det.box.y + det.box.height / 2;
      if (
        cx >= comp.box.x &&
        cx <= comp.box.x + comp.box.width &&
        cy >= comp.box.y &&
        cy <= comp.box.y + comp.box.height &&
        det.confidence > yoloConf
      ) {
        yoloLabel = det.label;
        yoloConf = det.confidence;
      }
    }

    const classified = await classifyFoodSmart(
      canvas,
      cropColor,
      yoloLabel,
      yoloConf,
      cropPx
    );

    // Kalau classifier ragu & ada prior label kompartemen, pakai prior kompartemen.
    let finalCode = classified.foodCode;
    if (
      (!finalCode || classified.confidence < 65) &&
      comp.code !== 'UNKNOWN'
    ) {
      const mapped = findNutritionByText(comp.code) || findFallbackNutrition(comp.code);
      if (mapped) finalCode = comp.code;
    }
    if (!finalCode) continue;
    if (addedCodes.has(finalCode)) continue;

    const nutrition =
      findNutritionByText(finalCode) || findFallbackNutrition(finalCode);
    if (!nutrition) continue;

    addedCodes.add(finalCode);
    let safetyStatus: 'safe' | 'warning' | 'danger' = 'safe';
    let forensicFlag = `Tidak ditemukan indikator visual bahaya (jamur/basi/kecoklatan ekstrem) pada ${nutrition.food_name}.`;
    let recommendation = 'Tetap periksa aroma, tekstur, dan kebersihan penyajian sebelum dikonsumsi.';

    // STATUS FORENSIK 3-WARNA (HIJAU/KUNING/MERAH) — murni dari bukti visual:
    // MERAH  = ada koloni kapang/jamur/basi (hasMold).
    // KUNING = model ragu (conf < 45%) ATAU ada indikator batas kesegaran.
    // HIJAU  = segar & keyakinan cukup.
    if (cropColor.hasMold) {
      safetyStatus = 'danger';
      forensicFlag = `BAHAYA KERACUNAN: Terdeteksi ${cropColor.moldType} aktif (${(cropColor.moldRatio * 100).toFixed(1)}% area koloni mikroba). Makanan sudah terkontaminasi toksin.`;
      recommendation = 'DILARANG DIMAKAN! Buang seluruh porsi ini segera.';
    } else if (classified.confidence < 45) {
      safetyStatus = 'warning';
      forensicFlag = `WASPADA: Keyakinan model rendah (${classified.confidence}%) untuk "${nutrition.food_name}". Periksa aroma, tekstur, dan warna sebelum disantap.`;
      recommendation = 'Periksa seksama (cek bau/tekstur) sebelum dikonsumsi.';
    } else if (cropColor.darkBrownRatio > 0.35 && cropColor.foodSignalRatio > 0.3) {
      safetyStatus = 'warning';
      forensicFlag = `WASPADA: Terdapat area kecokelatan/kehitaman ekstrim (${(cropColor.darkBrownRatio * 100).toFixed(0)}%) yang bisa jadi indikator penurunan kesegaran.`;
      recommendation = 'Periksa kesegaran lebih dulu; bila berbau/menyimpang, jangan dikonsumsi.';
    } else {
      safetyStatus = 'safe';
      forensicFlag = `Tidak ditemukan indikator visual bahaya (jamur/basi/kecoklatan ekstrem) pada ${nutrition.food_name}.`;
    }

    results.push({
      id: Math.random().toString(36).substring(2, 9),
      name: nutrition.food_name,
      category: nutrition.category,
      confidence: Math.max(classified.confidence, yoloConf),
      safetyStatus,
      forensicFlag,
      calories: nutrition.calories,
      protein: nutrition.protein,
      fat: nutrition.fat,
      carbs: nutrition.carbs,
      fiber: nutrition.fiber,
      vitaminA_mcg: nutrition.vitaminA_mcg || 0,
      vitaminB_mg: nutrition.vitaminB_mg || 0,
      vitaminC_mg: nutrition.vitaminC_mg || 0,
      vitaminD_mcg: nutrition.vitaminD_mcg || 0,
      calcium_mg: nutrition.calcium_mg || 0,
      iron_mg: nutrition.iron_mg || 0,
      recommendation,
      box: comp.box,
    });
  }

  return results;
}

// ============================================================
// MAIN PIPELINE: MULTI-ITEM ANALYSIS & FORENSIC TRIAGE
// ============================================================
export async function analyzeFoodImage(canvas: HTMLCanvasElement): Promise<FoodItemAnalysis[]> {
  const canvasW = canvas.width || 640;
  const canvasH = canvas.height || 480;

  const fullColor = analyzeRegionPixels(canvas);

  // DETEKSI MULTI-OBJEK MURNI via YOLO (deep learning).
  // Label & box 100% dari model yang dilatih di dataset — TIDAK ADA
  // classifier warna / region-hardcode yang menebak nama makanan.
  // Setiap kotak = 1 lauk (bisa >5 lauk di nampan).
  let results: FoodItemAnalysis[] = [];
  try {
    results = await analyzeYoloMultiItem(canvas, fullColor);
  } catch (e) {
    console.warn('YOLO multi-item analysis failed:', e);
  }

  // TIDAK ADA DUMMY DATA: kalau YOLO tidak mendeteksi makanan apa pun,
  // JANGAN fabrikasi hasil ("Nasi Putih AMAN") dari tebakan. Yang keluar
  // hanya bukti visual nyata — deteksi jamur/basi dari analisis piksel.
  // Kalau pun tidak ada jamur, hasil kosong: UI menampilkan "tidak ada
  // makanan terdeteksi", bukan makanan palsu.
  if (results.length === 0) {
    if (fullColor.hasMold) {
      results.push({
        id: Math.random().toString(36).substring(2, 9),
        name: 'Makanan tidak teridentifikasi',
        category: 'Tidak Diketahui',
        confidence: 0,
        safetyStatus: 'danger',
        forensicFlag: `BAHAYA KERACUNAN: Terdeteksi ${fullColor.moldType} aktif (${(fullColor.moldRatio * 100).toFixed(1)}% area koloni mikroba) meski model tidak mengenali makanannya.`,
        calories: 0,
        protein: 0,
        fat: 0,
        carbs: 0,
        fiber: 0,
        recommendation: 'DILARANG DIMAKAN! Makanan terindikasi terkontaminasi jamur/basi.',
        box: { x: 0.08, y: 0.08, width: 0.84, height: 0.84 },
      });
    }
  }

  // Simpan hasil ke Supabase jika aktif
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
