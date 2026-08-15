import { jsPDF } from 'jspdf';
import { FoodItemAnalysis, NutritionTotal } from './types';

const SAFE_LABEL: Record<FoodItemAnalysis['safetyStatus'], string> = {
  safe: 'AMAN',
  warning: 'PERHATIAN',
  danger: 'BAHAYA',
};

const STATUS_COLOR: Record<FoodItemAnalysis['safetyStatus'], [number, number, number]> = {
  safe: [0, 105, 72],
  warning: [154, 85, 0],
  danger: [186, 26, 26],
};

export function generateForensicReport(
  items: FoodItemAnalysis[],
  totals: NutritionTotal,
  imageDataUrl?: string | null
) {
  const doc = new jsPDF({ unit: 'pt', format: 'a4' });
  const pageW = doc.internal.pageSize.getWidth();
  const margin = 40;
  const contentW = pageW - margin * 2;
  let y = margin;

  // ---- Header ----
  doc.setFillColor(0, 105, 72);
  doc.rect(0, 0, pageW, 70, 'F');
  doc.setTextColor(255, 255, 255);
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(18);
  doc.text('NutriSafe AI', margin, 32);
  doc.setFont('helvetica', 'normal');
  doc.setFontSize(10);
  doc.text('Laporan Analisis Forensik Pangan & Gizi', margin, 50);

  const now = new Date();
  doc.setTextColor(255, 255, 255);
  doc.setFontSize(9);
  doc.text(
    `Tanggal: ${now.toLocaleString('id-ID')}`,
    pageW - margin,
    32,
    { align: 'right' }
  );
  doc.text('On-Device WASM Inference', pageW - margin, 50, { align: 'right' });

  y = 95;

  // ---- Foto piring (jika ada) ----
  if (imageDataUrl) {
    try {
      const imgW = contentW;
      const imgH = (imgW * 3) / 4; // asumsi rasio 4:3
      const maxH = 200;
      const drawH = Math.min(imgH, maxH);
      const drawW = (drawH * 4) / 3;
      doc.addImage(imageDataUrl, 'JPEG', margin, y, drawW, drawH);
      y += drawH + 16;
    } catch {
      // abaikan jika gambar gagal dimuat
    }
  }

  // ---- Ringkasan keseluruhan ----
  doc.setTextColor(23, 29, 25);
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(13);
  doc.text('Ringkasan Keamanan Pangan', margin, y);
  y += 8;

  const overallLabel =
    totals.overallSafety === 'danger'
      ? 'BAHAYA — jangan dikonsumsi'
      : totals.overallSafety === 'warning'
      ? 'PERHATIAN — periksa sebelum makan'
      : 'AMAN — layak dikonsumsi';
  const overallColor =
    STATUS_COLOR[
      totals.overallSafety as FoodItemAnalysis['safetyStatus']
    ];
  doc.setFillColor(overallColor[0], overallColor[1], overallColor[2]);
  doc.roundedRect(margin, y, contentW, 26, 4, 4, 'F');
  doc.setTextColor(255, 255, 255);
  doc.setFontSize(11);
  doc.text(overallLabel, margin + 12, y + 17);
  y += 40;

  // ---- Tabel hasil ----
  doc.setTextColor(23, 29, 25);
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(13);
  doc.text('Detail Komponen Lauk', margin, y);
  y += 14;

  const colName = margin;
  const colStatus = margin + 150;
  const colCal = margin + 260;
  const colMacro = margin + 320;

  doc.setFontSize(8);
  doc.setTextColor(109, 122, 114);
  doc.text('Komponen / Lauk', colName, y);
  doc.text('Status', colStatus, y);
  doc.text('Kalori', colCal, y);
  doc.text('Protein / Lemak / Karbo', colMacro, y);
  y += 4;
  doc.setDrawColor(222, 228, 222);
  doc.line(margin, y, pageW - margin, y);
  y += 14;

  doc.setFont('helvetica', 'normal');
  doc.setFontSize(9);

  for (const item of items) {
    if (y > 760) {
      doc.addPage();
      y = margin;
    }
    doc.setTextColor(23, 29, 25);
    doc.setFont('helvetica', 'bold');
    doc.text(item.name, colName, y);
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(7);
    doc.setTextColor(109, 122, 114);
    doc.text(`Akurasi: ${item.confidence}%`, colName, y + 10);

    // status pill
    const sc = STATUS_COLOR[item.safetyStatus];
    doc.setFillColor(sc[0], sc[1], sc[2]);
    doc.roundedRect(colStatus, y - 9, 70, 14, 3, 3, 'F');
    doc.setTextColor(255, 255, 255);
    doc.setFontSize(7.5);
    doc.text(SAFE_LABEL[item.safetyStatus], colStatus + 8, y + 1);

    // kalori
    doc.setTextColor(23, 29, 25);
    doc.setFontSize(9);
    doc.text(`${item.calories} kkal`, colCal, y);

    // makro
    doc.text(
      `${item.protein}g / ${item.fat}g / ${item.carbs}g`,
      colMacro,
      y
    );

    // observasi + rekomendasi
    doc.setFontSize(7.5);
    doc.setTextColor(61, 74, 66);
    doc.text(`Observasi: ${item.forensicFlag}`, colName, y + 22);
    doc.text(`Rekomendasi: ${item.recommendation}`, colName, y + 32);

    y += 46;
    doc.setDrawColor(238, 245, 239);
    doc.line(margin, y, pageW - margin, y);
    y += 10;
  }

  // ---- Total makronutrisi ----
  if (y > 720) {
    doc.addPage();
    y = margin;
  }
  doc.setFillColor(233, 239, 233);
  doc.roundedRect(margin, y, contentW, 30, 4, 4, 'F');
  doc.setTextColor(0, 105, 72);
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(10);
  doc.text(
    `Total: ${totals.totalCalories.toFixed(0)} kkal | Protein ${totals.totalProtein.toFixed(
      1
    )}g | Lemak ${totals.totalFat.toFixed(1)}g | Karbo ${totals.totalCarbs.toFixed(
      1
    )}g | Serat ${totals.totalFiber.toFixed(1)}g`,
    margin + 12,
    y + 19
  );
  y += 50;

  // ---- Tips keamanan ----
  doc.setTextColor(23, 29, 25);
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(11);
  doc.text('Indikator Fisik Forensik', margin, y);
  y += 14;
  doc.setFont('helvetica', 'normal');
  doc.setFontSize(8.5);
  doc.setTextColor(61, 74, 66);
  const tips = [
    '• Perubahan warna: keabuan pada daging atau bercak hijau/pink pada nasi matang.',
    '• Jamur & lendir: lapisan licin pada tahu/tempe atau spora jamur putih.',
    '• Gas fermentasi: busa abnormal pada kuah santan atau bau asam tajam.',
  ];
  for (const t of tips) {
    doc.text(t, margin, y);
    y += 12;
  }

  // ---- Footer ----
  const pageCount = doc.getNumberOfPages();
  for (let i = 1; i <= pageCount; i++) {
    doc.setPage(i);
    doc.setFontSize(7);
    doc.setTextColor(109, 122, 114);
    doc.text(
      `Dihasilkan oleh NutriSafe AI • Inferensi lokal di perangkat • Halaman ${i}/${pageCount}`,
      pageW / 2,
      doc.internal.pageSize.getHeight() - 20,
      { align: 'center' }
    );
  }

  const fileName = `laporan-forensik-pangan-${now
    .toISOString()
    .slice(0, 10)}.pdf`;
  doc.save(fileName);
}
