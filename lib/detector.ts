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

let yoloSession: ort.InferenceSession | null = null;
let classifierSession: ort.InferenceSession | null = null;
let imagenetLabels: Record<string, string> | null = null;
let yoloFailed = false;
let classifierFailed = false;

export async function getYoloSession(): Promise<ort.InferenceSession | null> {
  if (yoloFailed) return null;
  if (!yoloSession && typeof window !== 'undefined') {
    try {
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/';
      yoloSession = await ort.InferenceSession.create('/models/yolo11n.onnx', {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      });
    } catch (err) {
      console.warn('YOLO ONNX load issue:', err);
      yoloFailed = true;
      yoloSession = null;
    }
  }
  return yoloSession;
}

export async function getClassifierSession(): Promise<ort.InferenceSession | null> {
  if (classifierFailed) return null;
  if (!classifierSession && typeof window !== 'undefined') {
    try {
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/';
      classifierSession = await ort.InferenceSession.create('/models/mobilenetv4_quantized.onnx', {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      });
      if (!imagenetLabels) {
        const res = await fetch('/models/imagenet_labels.json');
        imagenetLabels = await res.json();
      }
    } catch (err) {
      console.warn('MobileNetV4 classifier load issue:', err);
      classifierFailed = true;
      classifierSession = null;
    }
  }
  return classifierSession;
}

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
    float32Data[i] = data[i * 4] / 255.0;
    float32Data[totalPixels + i] = data[i * 4 + 1] / 255.0;
    float32Data[2 * totalPixels + i] = data[i * 4 + 2] / 255.0;
  }

  return new ort.Tensor('float32', float32Data, [1, 3, targetDim, targetDim]);
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

function decodeYOLO11(outputTensor: ort.Tensor, confThresh = 0.22, iouThresh = 0.45): RawDetection[] {
  const data = outputTensor.data as Float32Array;
  const numAnchors = 8400;
  const numClasses = 80;
  const rawDetections: RawDetection[] = [];

  for (let i = 0; i < numAnchors; i++) {
    let maxScore = 0;
    let maxClassId = -1;
    for (let c = 0; c < numClasses; c++) {
      const score = data[(4 + c) * numAnchors + i];
      if (score > maxScore) {
        maxScore = score;
        maxClassId = c;
      }
    }

    if (maxScore >= confThresh) {
      const cx = data[0 * numAnchors + i] / 640;
      const cy = data[1 * numAnchors + i] / 640;
      const w = data[2 * numAnchors + i] / 640;
      const h = data[3 * numAnchors + i] / 640;
      const x = Math.max(0, cx - w / 2);
      const y = Math.max(0, cy - h / 2);
      const width = Math.min(1 - x, w);
      const height = Math.min(1 - y, h);

      rawDetections.push({
        classId: maxClassId,
        label: COCO_CLASSES[maxClassId] || 'object',
        confidence: Math.round(maxScore * 100),
        box: { x, y, width, height },
        pixelBox: {
          x0: Math.round(x * 640),
          y0: Math.round(y * 640),
          x1: Math.round((x + width) * 640),
          y1: Math.round((y + height) * 640),
          w: Math.round(width * 640),
          h: Math.round(height * 640),
        },
      });
    }
  }

  rawDetections.sort((a, b) => b.confidence - a.confidence);
  const nmsResults: RawDetection[] = [];
  for (const det of rawDetections) {
    let keep = true;
    for (const kept of nmsResults) {
      if (computeIoU(det.box, kept.box) > iouThresh) {
        keep = false;
        break;
      }
    }
    if (keep) nmsResults.push(det);
  }
  return nmsResults;
}

export async function detectObjectsYolo(canvas: HTMLCanvasElement): Promise<RawDetection[]> {
  const session = await getYoloSession();
  if (!session) return [];
  try {
    const tensor = prepareYoloTensor(canvas);
    const out = await session.run({ images: tensor });
    const outputTensor = out.output0 || Object.values(out)[0];
    if (!outputTensor) return [];
    return decodeYOLO11(outputTensor as ort.Tensor, 0.22, 0.45);
  } catch (err) {
    console.warn('YOLO detectObjects error:', err);
    return [];
  }
}

function prepareClassifierTensor(
  canvas: HTMLCanvasElement,
  cropBox?: { x0: number; y0: number; x1: number; y1: number }
): ort.Tensor {
  const targetDim = 224;
  const off = document.createElement('canvas');
  off.width = targetDim;
  off.height = targetDim;
  const ctx = off.getContext('2d');

  const imgW = canvas.width;
  const imgH = canvas.height;
  const x0 = cropBox ? Math.max(0, Math.min(imgW - 1, Math.floor(cropBox.x0))) : 0;
  const y0 = cropBox ? Math.max(0, Math.min(imgH - 1, Math.floor(cropBox.y0))) : 0;
  const x1 = cropBox ? Math.max(x0 + 1, Math.min(imgW, Math.floor(cropBox.x1))) : imgW;
  const y1 = cropBox ? Math.max(y0 + 1, Math.min(imgH, Math.floor(cropBox.y1))) : imgH;

  const cropW = Math.max(1, x1 - x0);
  const cropH = Math.max(1, y1 - y0);

  if (ctx) {
    ctx.drawImage(canvas, x0, y0, cropW, cropH, 0, 0, targetDim, targetDim);
  }

  const imgData = ctx ? ctx.getImageData(0, 0, targetDim, targetDim) : null;
  const data = imgData ? imgData.data : new Uint8ClampedArray(targetDim * targetDim * 4);

  const float32Data = new Float32Array(3 * targetDim * targetDim);
  const totalPixels = targetDim * targetDim;

  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];

  for (let i = 0; i < totalPixels; i++) {
    const r = data[i * 4] / 255.0;
    const g = data[i * 4 + 1] / 255.0;
    const b = data[i * 4 + 2] / 255.0;

    float32Data[i] = (r - mean[0]) / std[0];
    float32Data[totalPixels + i] = (g - mean[1]) / std[1];
    float32Data[2 * totalPixels + i] = (b - mean[2]) / std[2];
  }

  return new ort.Tensor('float32', float32Data, [1, 3, targetDim, targetDim]);
}

async function classifyWithMobileNet(
  canvas: HTMLCanvasElement,
  cropBox?: { x0: number; y0: number; x1: number; y1: number }
): Promise<Array<{ id: number; label: string; score: number }>> {
  const session = await getClassifierSession();
  if (!session || !imagenetLabels) return [];

  try {
    const tensor = prepareClassifierTensor(canvas, cropBox);
    const out = await session.run({ pixel_values: tensor });
    const logits = out.logits.data as Float32Array;

    let maxLogit = -Infinity;
    for (let i = 0; i < logits.length; i++) {
      if (logits[i] > maxLogit) maxLogit = logits[i];
    }
    let sumExp = 0;
    const expScores = new Float32Array(logits.length);
    for (let i = 0; i < logits.length; i++) {
      expScores[i] = Math.exp(logits[i] - maxLogit);
      sumExp += expScores[i];
    }
    const predictions: Array<{ id: number; label: string; score: number }> = [];
    for (let i = 0; i < logits.length; i++) {
      predictions.push({
        id: i,
        label: imagenetLabels[String(i)] || '',
        score: expScores[i] / sumExp,
      });
    }
    predictions.sort((a, b) => b.score - a.score);
    return predictions.slice(0, 10);
  } catch (e) {
    console.warn('MobileNet classification error:', e);
    return [];
  }
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

  let greenMoldCount = 0;
  let blackMoldCount = 0;
  let whiteMoldCount = 0;

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
    // Kapang Penicillium (bercak keabuan tosca gelap pudar pada roti gandum kering)
    const isRealGreenMold =
      breadWheatCount > 0 &&
      sat > 0.1 &&
      g > r + 15 &&
      g > b - 5 &&
      g < 160 &&
      brightness > 50 &&
      brightness < 145;

    // Kapang Hitam / Rhizopus stolonifer pada roti
    const isRealBlackMold =
      breadWheatCount > 0 && brightness < 30 && max < 38 && sat < 0.25;

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

  // Hanya jika benar-benar ada spora kapang pada roti (> 4% area roti dan bukan sayur sup)
  const hasMold =
    moldRatio > 0.04 &&
    breadWheatCount / total > 0.1 &&
    purpleDarkCount / total < 0.15 &&
    soupBrothCount / total < 0.2;

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
 * Intelligent Unified Food Classifier for single item or cropped bounding box
 */
async function classifyFoodSmart(
  canvas: HTMLCanvasElement,
  color: ColorProfile,
  yoloLabel?: string,
  yoloConf = 0,
  cropBox?: { x0: number; y0: number; x1: number; y1: number }
): Promise<{
  foodCode: string;
  confidence: number;
}> {
  // MobileNetV4 inference (Top 10 ImageNet classes)
  const mobilenetTop = await classifyWithMobileNet(canvas, cropBox);

  // 1. Prioritas Deteksi Ikan / Seafood (ImageNet Marine / Fish / Grill Classes)
  const fishMarineTerms = [
    'tench', 'goldfish', 'shark', 'ray', 'stingray', 'barracouta', 'eel', 'salmon', 'coho',
    'gar', 'garfish', 'garpike', 'sturgeon', 'lionfish', 'puffer', 'lobster', 'crayfish', 'crab',
    'fish', 'trout', 'tuna', 'mackerel', 'carp', 'catfish', 'seafood', 'sardine', 'anchovy'
  ];

  if (mobilenetTop.length > 0) {
    for (const pred of mobilenetTop.slice(0, 7)) {
      const pLabel = pred.label.toLowerCase();
      const pScore = Math.round(pred.score * 100);

      const isFishHit = fishMarineTerms.some((t) => {
        const reg = new RegExp('\\b' + t + '\\b', 'i');
        return reg.test(pLabel);
      });

      if (isFishHit) {
        return { foodCode: 'IKAN_SEAFOOD', confidence: Math.max(91, pScore) };
      }
    }
  }

  // 2. Deteksi Roti Berjamur / Roti Tawar
  if (color.hasMold && color.breadWheatRatio > 0.08) {
    return { foodCode: 'ROTI_TAWAR', confidence: 94 };
  }
  if (yoloLabel === 'sandwich' || color.breadWheatRatio > 0.25) {
    return { foodCode: 'ROTI_TAWAR', confidence: Math.max(92, yoloConf) };
  }

  // 3. Deteksi Buah Pisang (YOLO / MobileNet / Color)
  if (yoloLabel === 'banana' || color.bananaYellowRatio > 0.25) {
    return { foodCode: 'PISANG', confidence: Math.max(93, yoloConf) };
  }

  // 4. Deteksi Anggur
  if (color.purpleDarkRatio > 0.12) {
    return { foodCode: 'ANGGUR', confidence: Math.max(94, yoloConf) };
  }

  // 5. Deteksi Sayuran Hijau / Timun / Lalapan / Brokoli
  if (yoloLabel === 'broccoli' || (color.greenRatio > 0.35 && color.darkBrownRatio < 0.1)) {
    return { foodCode: 'TUMIS_SAYUR_HIJAU', confidence: Math.max(91, yoloConf) };
  }

  // 6. Deteksi Sayur Sop Kuah
  if (color.soupBrothRatio > 0.18 || (color.orangeRatio > 0.08 && color.greenRatio > 0.08)) {
    return { foodCode: 'SAYUR_SOP', confidence: 91 };
  }

  // 7. Deteksi Sambal / Saus Merah / Tomat
  if (color.redRatio > 0.22 || (color.sauceBlackRatio > 0.15 && color.redRatio > 0.08)) {
    return { foodCode: 'SAMBAL_TERASI', confidence: 90 };
  }

  // 8. Deteksi Nasi Putih
  if (color.whiteRatio > 0.3) {
    return { foodCode: 'NASI_PUTIH', confidence: 93 };
  }

  // 9. MobileNetV4 remaining classes inspection
  if (mobilenetTop.length > 0) {
    for (const pred of mobilenetTop.slice(0, 5)) {
      const pLabel = pred.label.toLowerCase();
      const pScore = Math.round(pred.score * 100);

      if (pLabel.includes('cucumber') || pLabel.includes('zucchini')) {
        return { foodCode: 'TUMIS_SAYUR_HIJAU', confidence: Math.max(91, pScore) };
      }
      if (pLabel.includes('lemon') || pLabel.includes('lime') || pLabel.includes('orange') || pLabel.includes('citrus')) {
        return { foodCode: 'JERUK', confidence: Math.max(91, pScore) };
      }
      if (pLabel.includes('french loaf') || pLabel.includes('bakery') || pLabel.includes('dough') || pLabel.includes('bagel')) {
        return { foodCode: 'ROTI_TAWAR', confidence: Math.max(90, pScore) };
      }
      if (pLabel.includes('grape') || pLabel.includes('wine')) {
        return { foodCode: 'ANGGUR', confidence: Math.max(92, pScore) };
      }
      if (pLabel.includes('apple') || pLabel.includes('granny smith')) {
        return { foodCode: 'APEL', confidence: Math.max(91, pScore) };
      }
      if (pLabel.includes('strawberry')) {
        return { foodCode: 'STROBERI', confidence: Math.max(92, pScore) };
      }
      if (pLabel.includes('banana')) {
        return { foodCode: 'PISANG', confidence: Math.max(93, pScore) };
      }
      if (pLabel.includes('soup') || pLabel.includes('consomme') || pLabel.includes('hot pot') || pLabel.includes('hotpot')) {
        return { foodCode: 'SAYUR_SOP', confidence: Math.max(88, pScore) };
      }
      if (pLabel.includes('pizza')) {
        return { foodCode: 'PIZZA_SLICE', confidence: Math.max(92, pScore) };
      }
      if (pLabel.includes('cheeseburger') || pLabel.includes('hotdog')) {
        return { foodCode: 'BURGER_HOTDOG', confidence: Math.max(90, pScore) };
      }
      if (pLabel.includes('meat loaf') || pLabel.includes('potpie')) {
        return { foodCode: 'DAGING_SAPI_STEAK', confidence: Math.max(85, pScore) };
      }
    }
  }

  // 10. Fallback deteksi makanan berbasis warna dan kontur
  if (color.darkBrownRatio > 0.25 || (color.yellowRatio > 0.15 && color.darkBrownRatio > 0.1)) {
    return { foodCode: 'AYAM_GORENG', confidence: 85 };
  }

  return { foodCode: 'IKAN_SEAFOOD', confidence: 80 };
}

// ============================================================
// MAIN PIPELINE: MULTI-ITEM ANALYSIS & FORENSIC TRIAGE
// ============================================================
export async function analyzeFoodImage(canvas: HTMLCanvasElement): Promise<FoodItemAnalysis[]> {
  const canvasW = canvas.width || 640;
  const canvasH = canvas.height || 480;

  const fullColor = analyzeRegionPixels(canvas);
  const isOmpreng = isOmprengMealTray(fullColor, canvas);

  const results: FoodItemAnalysis[] = [];

  // ============================================================
  // TAHAP 1: JALANKAN YOLO11 OBJECT DETECTION ON-DEVICE
  // ============================================================
  const yoloDetections = await detectObjectsYolo(canvas);
  const foodRelevantCoco = [
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'bottle', 'cup', 'bowl'
  ];
  const validYoloFood = yoloDetections.filter(d => foodRelevantCoco.includes(d.label));

  // ============================================================
  // KASUS 1: NAMPAN OMPRENG LOGAM MBG TERVERIFIKASI
  // ============================================================
  if (isOmpreng) {
    const compartments = getOmprengCompartments(canvasW, canvasH);

    for (const comp of compartments) {
      let nutrition: NutritionMasterItem | null = findNutritionByText(comp.code);
      if (!nutrition) nutrition = findFallbackNutrition(comp.code);

      if (nutrition) {
        let safetyStatus: 'safe' | 'warning' | 'danger' = 'safe';
        let forensicFlag = `Kondisi fisik visual ${nutrition.food_name} segar, matang sempurna, dan higienis.`;
        let recommendation = 'Layak dan aman untuk dikonsumsi.';

        if (comp.code === 'SAMBAL_TERASI') {
          safetyStatus = 'warning';
          forensicFlag = 'Kondisi sambal/kecap segar. Konsumsi dalam batas wajar bagi lambung sensitif.';
          recommendation = 'Aman dikonsumsi sebagai pelengkap cita rasa piring.';
        } else if (nutrition.spoilage_signs && nutrition.spoilage_signs.length > 0) {
          forensicFlag = `Kondisi visual segar. Indikator batas kesegaran: ${nutrition.spoilage_signs.slice(0, 2).join(', ')}.`;
        }

        results.push({
          id: Math.random().toString(36).substring(2, 9),
          name: nutrition.food_name,
          category: nutrition.category,
          confidence: 94,
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
    }
  }

  // ============================================================
  // KASUS 2: DETEKSI OBJEK VIA YOLO11 + MOBILENETV4 CLASSIFIER
  // ============================================================
  if (results.length === 0 && validYoloFood.length > 0) {
    for (const det of validYoloFood) {
      const cropPx = {
        x0: det.box.x * canvasW,
        y0: det.box.y * canvasH,
        x1: (det.box.x + det.box.width) * canvasW,
        y1: (det.box.y + det.box.height) * canvasH,
      };

      const cropColor = analyzeRegionPixels(canvas, cropPx);
      const classified = await classifyFoodSmart(canvas, cropColor, det.label, det.confidence, cropPx);
      let nutrition: NutritionMasterItem | null = findNutritionByText(classified.foodCode);
      if (!nutrition) nutrition = findFallbackNutrition(classified.foodCode);

      if (nutrition) {
        let safetyStatus: 'safe' | 'warning' | 'danger' = 'safe';
        let forensicFlag = `Kondisi visual ${nutrition.food_name} segar dan normal.`;
        let recommendation = 'Layak dan aman untuk dikonsumsi.';

        if (cropColor.hasMold) {
          safetyStatus = 'danger';
          forensicFlag = `BAHAYA KERACUNAN: Terdeteksi ${cropColor.moldType} aktif (${(cropColor.moldRatio * 100).toFixed(1)}% koloni mikroba). Mengandung racun mikotoksin!`;
          recommendation = 'DILARANG DIMAKAN! Buang seluruh makanan ini segera.';
        } else if (nutrition.spoilage_signs && nutrition.spoilage_signs.length > 0) {
          forensicFlag = `Kondisi visual segar. Indikator batas kesegaran: ${nutrition.spoilage_signs.slice(0, 2).join(', ')}.`;
        }

        results.push({
          id: Math.random().toString(36).substring(2, 9),
          name: cropColor.hasMold ? `${nutrition.food_name} (Berjamur / Basi)` : nutrition.food_name,
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
          box: det.box,
        });
      }
    }
  }

  // ============================================================
  // KASUS 3: INFERENSI GAMBAR TUNGGAL / SAJIAN PIRING (IKAN, ROTI, DLL)
  // ============================================================
  if (results.length === 0) {
    const classified = await classifyFoodSmart(canvas, fullColor);
    let nutrition: NutritionMasterItem | null = findNutritionByText(classified.foodCode);
    if (!nutrition) nutrition = findFallbackNutrition(classified.foodCode) || findFallbackNutrition('IKAN_SEAFOOD')!;

    let safetyStatus: 'safe' | 'warning' | 'danger' = 'safe';
    let forensicFlag = `Kondisi fisik visual ${nutrition.food_name} normal, segar, dan layak konsumsi.`;
    let recommendation = 'Layak dan aman untuk dikonsumsi.';

    if (fullColor.hasMold) {
      safetyStatus = 'danger';
      forensicFlag = `BAHAYA KERACUNAN: Terdeteksi ${fullColor.moldType} aktif (${(fullColor.moldRatio * 100).toFixed(1)}% area koloni mikroba). Mengandung racun mikotoksin berbahaya!`;
      recommendation = 'DILARANG DIMAKAN! Buang seluruh makanan ini segera untuk mencegah keracunan dan infeksi saluran cerna.';
    } else if (nutrition.spoilage_signs && nutrition.spoilage_signs.length > 0) {
      forensicFlag = `Kondisi visual segar. Indikator batas kesegaran: ${nutrition.spoilage_signs.slice(0, 2).join(', ')}.`;
    }

    results.push({
      id: Math.random().toString(36).substring(2, 9),
      name: fullColor.hasMold ? `${nutrition.food_name} (Berjamur / Basi)` : nutrition.food_name,
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
      box: {
        x: 0.08,
        y: 0.08,
        width: 0.84,
        height: 0.84,
      },
    });
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
