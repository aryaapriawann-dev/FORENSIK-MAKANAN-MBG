'use client';

import React from 'react';
import { Download, ShieldCheck, AlertTriangle, XCircle, Sparkles, CheckCircle2, Info, Activity } from 'lucide-react';
import { FoodItemAnalysis, NutritionTotal } from '../lib/types';
import { generateForensicReport } from '../lib/generateReport';

interface Props {
  items: FoodItemAnalysis[];
  totals: NutritionTotal;
  scannedImage?: string | null;
}

export const FoodForensicTable: React.FC<Props> = ({ items, totals, scannedImage }) => {
  // Evaluasi keseimbangan makronutrisi piring (Standar Isi Piringku Kemenkes RI)
  const isDangerous = totals.overallSafety === 'danger' || items.some((i) => i.safetyStatus === 'danger');
  const isWarning = totals.overallSafety === 'warning' || items.some((i) => i.safetyStatus === 'warning');

  const totalMacroGram = totals.totalProtein + totals.totalFat + totals.totalCarbs;
  const proteinPercent = totalMacroGram > 0 ? (totals.totalProtein / totalMacroGram) * 100 : 0;
  const fatPercent = totalMacroGram > 0 ? (totals.totalFat / totalMacroGram) * 100 : 0;
  const carbsPercent = totalMacroGram > 0 ? (totals.totalCarbs / totalMacroGram) * 100 : 0;

  let balanceVerdict = 'Seimbang';
  let balanceDesc = 'Komposisi karbohidrat, protein, dan lemak memenuhi proporsi gizi seimbang harian.';
  let balanceBadgeClass = 'bg-emerald-100 text-emerald-800 border-emerald-300';

  if (isDangerous) {
    balanceVerdict = 'TIDAK LAYAK MAKAN';
    balanceDesc = 'Terdeteksi bahaya mikrobiologis/pembusukan. Jangan dikonsumsi demi keselamatan.';
    balanceBadgeClass = 'bg-rose-100 text-rose-800 border-rose-300';
  } else if (carbsPercent > 65) {
    balanceVerdict = 'Tinggi Karbohidrat';
    balanceDesc = 'Porsi karbohidrat dominan. Disarankan menambah lauk berprotein atau sayuran berserat.';
    balanceBadgeClass = 'bg-amber-100 text-amber-800 border-amber-300';
  } else if (fatPercent > 45) {
    balanceVerdict = 'Tinggi Lemak';
    balanceDesc = 'Kadar lemak/minyak cukup tinggi. Kurangi konsumsi gorengan atau santan berlebih.';
    balanceBadgeClass = 'bg-amber-100 text-amber-800 border-amber-300';
  } else if (proteinPercent < 15) {
    balanceVerdict = 'Kurang Protein';
    balanceDesc = 'Asupan protein masih minim. Tambahkan telur, tempe, tahu, atau ikan.';
    balanceBadgeClass = 'bg-blue-100 text-blue-800 border-blue-300';
  }

  return (
    <div className="w-full flex flex-col gap-6">
      {/* Header & Quick Action */}
      <div className="flex flex-col sm:flex-row justify-between sm:items-center gap-3 bg-white p-5 rounded-2xl border border-slate-200 shadow-sm">
        <div>
          <div className="flex items-center gap-2">
            <Activity className="w-5 h-5 text-emerald-600" />
            <h3 className="font-extrabold text-xl text-slate-900">Laporan Forensik & Status Gizi</h3>
          </div>
          <p className="text-xs text-slate-500 mt-0.5">
            Hasil evaluasi laboratorium visual on-device & verifikasi standar pangan Kemenkes RI.
          </p>
        </div>

        <button
          onClick={() => generateForensicReport(items, totals, scannedImage)}
          className="flex items-center justify-center gap-2 bg-[#006948] hover:bg-[#005137] text-white font-bold text-xs px-4 py-2.5 rounded-xl shadow-sm active:scale-[0.98] transition-all cursor-pointer w-full sm:w-auto"
        >
          <Download className="w-4 h-4" />
          Unduh Laporan PDF
        </button>
      </div>

      {/* Grid: Preview Foto Pindai & Kartu Kelayakan Gizi */}
      <div className="grid grid-cols-1 md:grid-cols-12 gap-4">
        {/* Kolom 1: Foto Tangkapan Visual */}
        <div className="md:col-span-5 bg-white p-4 rounded-2xl border border-slate-200 shadow-sm flex flex-col items-center">
          <span className="text-xs font-bold text-slate-700 self-start mb-2 uppercase tracking-wider">
            Dokumentasi Visual Forensik
          </span>
          <div className="w-full aspect-[4/3] rounded-xl overflow-hidden bg-slate-950 relative border border-slate-200 flex items-center justify-center">
            {scannedImage ? (
              <img
                src={scannedImage}
                alt="Tangkapan Makanan"
                className="w-full h-full object-cover"
              />
            ) : (
              <div className="text-xs text-slate-400 font-mono text-center p-4">
                Tidak ada tangkapan gambar
              </div>
            )}
            <div className="absolute top-2 right-2">
              {isDangerous ? (
                <span className="inline-flex items-center gap-1 bg-rose-600 text-white font-bold text-[10px] px-2.5 py-1 rounded-full shadow-md animate-pulse">
                  <XCircle className="w-3 h-3" /> BAHAYA
                </span>
              ) : isWarning ? (
                <span className="inline-flex items-center gap-1 bg-amber-500 text-white font-bold text-[10px] px-2.5 py-1 rounded-full shadow-md">
                  <AlertTriangle className="w-3 h-3" /> PERIKSA
                </span>
              ) : (
                <span className="inline-flex items-center gap-1 bg-emerald-600 text-white font-bold text-[10px] px-2.5 py-1 rounded-full shadow-md">
                  <ShieldCheck className="w-3 h-3" /> AMAN
                </span>
              )}
            </div>
          </div>
          <p className="text-[11px] text-slate-500 mt-2 text-center">
            Analisis tekstur, spektrum warna, dan kontras spora mikrobiologis.
          </p>
        </div>

        {/* Kolom 2: Status Kelayakan Makan & Keseimbangan Gizi */}
        <div className="md:col-span-7 bg-white p-5 rounded-2xl border border-slate-200 shadow-sm flex flex-col justify-between gap-4">
          <div>
            <span className="text-xs font-bold text-slate-700 uppercase tracking-wider block mb-2">
              Evaluasi Kelayakan & Keseimbangan Porsi
            </span>

            <div className="flex flex-wrap items-center gap-2 mb-3">
              <span className={`text-xs font-black px-3 py-1 rounded-lg border ${balanceBadgeClass}`}>
                {balanceVerdict}
              </span>
              <span className="text-xs font-bold px-3 py-1 rounded-lg bg-slate-100 text-slate-800 border border-slate-200">
                Total Energi: {totals.totalCalories.toFixed(0)} kkal
              </span>
            </div>

            <p className="text-xs text-slate-600 leading-relaxed bg-slate-50 p-3 rounded-xl border border-slate-100">
              {balanceDesc}
            </p>
          </div>

          {/* Rincian Distribusi Makronutrisi */}
          <div>
            <span className="text-[11px] font-bold text-slate-500 uppercase tracking-wider block mb-1.5">
              Rasio Makronutrisi Piring:
            </span>
            <div className="grid grid-cols-4 gap-2 text-center font-mono">
              <div className="bg-emerald-50 border border-emerald-100 p-2 rounded-xl">
                <span className="text-[10px] text-emerald-800 font-bold block">Protein</span>
                <span className="text-xs md:text-sm font-black text-emerald-700">{totals.totalProtein.toFixed(1)}g</span>
                <span className="text-[9px] text-emerald-600 block">{proteinPercent.toFixed(0)}%</span>
              </div>
              <div className="bg-amber-50 border border-amber-100 p-2 rounded-xl">
                <span className="text-[10px] text-amber-800 font-bold block">Lemak</span>
                <span className="text-xs md:text-sm font-black text-amber-700">{totals.totalFat.toFixed(1)}g</span>
                <span className="text-[9px] text-amber-600 block">{fatPercent.toFixed(0)}%</span>
              </div>
              <div className="bg-blue-50 border border-blue-100 p-2 rounded-xl">
                <span className="text-[10px] text-blue-800 font-bold block">Karbo</span>
                <span className="text-xs md:text-sm font-black text-blue-700">{totals.totalCarbs.toFixed(1)}g</span>
                <span className="text-[9px] text-blue-600 block">{carbsPercent.toFixed(0)}%</span>
              </div>
              <div className="bg-purple-50 border border-purple-100 p-2 rounded-xl">
                <span className="text-[10px] text-purple-800 font-bold block">Serat</span>
                <span className="text-xs md:text-sm font-black text-purple-700">{totals.totalFiber.toFixed(1)}g</span>
                <span className="text-[9px] text-purple-600 block">TKPI</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Tabel Komprehensif Hasil Forensik Tiap Makanan */}
      <div className="w-full bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden">
        <div className="p-4 bg-slate-900 text-white flex justify-between items-center">
          <div>
            <h4 className="font-bold text-sm md:text-base flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-emerald-400" />
              Rincian Tabel Forensik & Komposisi Pangan (TKPI)
            </h4>
            <p className="text-[11px] text-slate-400">
              Analisis per item makanan terdeteksi beserta diagnosa visual fisiknya.
            </p>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="bg-slate-50 text-slate-600 text-[11px] font-bold uppercase tracking-wider border-b border-slate-200">
                <th className="py-3 px-4">Komponen Pangan</th>
                <th className="py-3 px-3 text-center">Status Kelayakan</th>
                <th className="py-3 px-4">Diagnosa Forensik Visual</th>
                <th className="py-3 px-3 text-center">Kalori</th>
                <th className="py-3 px-3 text-center">Protein</th>
                <th className="py-3 px-3 text-center">Lemak</th>
                <th className="py-3 px-3 text-center">Karbo</th>
                <th className="py-3 px-4">Rekomendasi Tindakan</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 text-xs text-slate-700">
              {items.map((item) => (
                <tr key={item.id} className="hover:bg-slate-50/80 transition-colors">
                  <td className="py-3.5 px-4 font-bold text-slate-900 whitespace-nowrap">
                    {item.name}
                    <span className="block text-[10px] font-normal text-slate-400">
                      Kategori: {item.category}
                    </span>
                  </td>
                  <td className="py-3.5 px-3 text-center whitespace-nowrap">
                    {item.safetyStatus === 'safe' && (
                      <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-[11px] font-bold bg-emerald-100 text-emerald-800 border border-emerald-200">
                        <CheckCircle2 className="w-3 h-3 text-emerald-600" /> Layak
                      </span>
                    )}
                    {item.safetyStatus === 'warning' && (
                      <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-[11px] font-bold bg-amber-100 text-amber-800 border border-amber-200">
                        <AlertTriangle className="w-3 h-3 text-amber-600" /> Periksa
                      </span>
                    )}
                    {item.safetyStatus === 'danger' && (
                      <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-[11px] font-bold bg-rose-100 text-rose-800 border border-rose-200 animate-pulse">
                        <XCircle className="w-3 h-3 text-rose-600" /> BAHAYA
                      </span>
                    )}
                  </td>
                  <td className="py-3.5 px-4 text-xs text-slate-600 min-w-[200px] leading-relaxed">
                    {item.forensicFlag}
                  </td>
                  <td className="py-3.5 px-3 text-center font-mono font-bold text-slate-900 whitespace-nowrap">
                    {item.calories} kkal
                  </td>
                  <td className="py-3.5 px-3 text-center font-mono text-slate-700 whitespace-nowrap">
                    {item.protein}g
                  </td>
                  <td className="py-3.5 px-3 text-center font-mono text-slate-700 whitespace-nowrap">
                    {item.fat}g
                  </td>
                  <td className="py-3.5 px-3 text-center font-mono text-slate-700 whitespace-nowrap">
                    {item.carbs}g
                  </td>
                  <td className="py-3.5 px-4 text-xs font-semibold text-slate-800 min-w-[180px] leading-relaxed">
                    {item.recommendation}
                  </td>
                </tr>
              ))}
            </tbody>
            <tfoot>
              <tr className="bg-slate-900 text-white font-bold text-xs border-t border-slate-800">
                <td className="py-3.5 px-4" colSpan={3}>
                  TOTAL GIZI PIRING KONSUMSI
                </td>
                <td className="py-3.5 px-3 text-center text-emerald-400 font-mono">
                  {totals.totalCalories.toFixed(0)} kkal
                </td>
                <td className="py-3.5 px-3 text-center font-mono">{totals.totalProtein.toFixed(1)}g</td>
                <td className="py-3.5 px-3 text-center font-mono">{totals.totalFat.toFixed(1)}g</td>
                <td className="py-3.5 px-3 text-center font-mono">{totals.totalCarbs.toFixed(1)}g</td>
                <td className="py-3.5 px-4 text-[11px] text-slate-300 font-normal">
                  Total Serat: {totals.totalFiber.toFixed(1)}g
                </td>
              </tr>
            </tfoot>
          </table>
        </div>
      </div>
    </div>
  );
};

