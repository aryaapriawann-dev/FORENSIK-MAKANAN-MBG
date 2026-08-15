'use client';

import React from 'react';
import { Download } from 'lucide-react';
import { FoodItemAnalysis, NutritionTotal } from '../lib/types';
import { generateForensicReport } from '../lib/generateReport';

interface Props {
  items: FoodItemAnalysis[];
  totals: NutritionTotal;
  scannedImage?: string | null;
}

export const FoodForensicTable: React.FC<Props> = ({ items, totals, scannedImage }) => {
  return (
    <div className="w-full flex flex-col gap-4">
      <div className="flex justify-between items-baseline px-1">
        <h3 className="font-bold text-xl text-[#171d19]">Hasil Diagnostik</h3>
        <div className="flex items-center gap-2">
          <button
            onClick={() => generateForensicReport(items, totals, scannedImage)}
            className="flex items-center gap-1.5 bg-[#006948] hover:bg-[#005137] text-white font-semibold text-xs px-3 py-2 rounded-lg shadow-sm active:scale-[0.98] transition-transform cursor-pointer"
          >
            <Download className="w-4 h-4" />
            Unduh Laporan PDF
          </button>
          <span className="text-xs font-mono text-[#6d7a72]">
            Total: <strong className="text-[#006948] font-bold">{totals.totalCalories.toFixed(0)} kcal</strong>
          </span>
        </div>
      </div>

      <div className="flex flex-col gap-[2px] rounded-2xl overflow-hidden shadow-xs bg-[#dee4de]">
        {items.map((item) => (
          <div key={item.id} className="bg-white text-[#171d19] p-4 flex flex-col gap-3 relative">
            <div className="flex justify-between items-start">
              <div className="flex items-center gap-2.5">
                {item.safetyStatus === 'safe' && (
                  <div className="w-3 h-3 rounded-full bg-[#006948] shadow-[0_0_6px_#006948]" />
                )}
                {item.safetyStatus === 'warning' && (
                  <div className="w-3 h-3 rounded-full bg-[#ba5551] shadow-[0_0_6px_#ba5551]" />
                )}
                {item.safetyStatus === 'danger' && (
                  <div className="w-3 h-3 rounded-full bg-[#ba1a1a] shadow-[0_0_6px_#ba1a1a] animate-pulse" />
                )}

                <span className="font-bold text-lg text-[#171d19]">{item.name}</span>
              </div>

              {item.safetyStatus === 'safe' && (
                <span className="bg-[#00855d] text-[#f5fff7] font-mono text-[11px] font-bold px-3 py-1 rounded-full uppercase tracking-wider">
                  SAFE
                </span>
              )}
              {item.safetyStatus === 'warning' && (
                <span className="bg-[#ffdad7] text-[#410004] font-mono text-[11px] font-bold px-3 py-1 rounded-full uppercase tracking-wider">
                  CAUTION
                </span>
              )}
              {item.safetyStatus === 'danger' && (
                <span className="bg-[#ba1a1a] text-white font-mono text-[11px] font-bold px-3 py-1 rounded-full uppercase tracking-wider animate-pulse">
                  HAZARD
                </span>
              )}
            </div>

            <p className="text-sm text-[#3d4a42] leading-relaxed">{item.forensicFlag}</p>

            <div className="flex flex-wrap items-center gap-2 font-mono text-xs text-[#6d7a72] mt-1">
              <span className="bg-[#e9efe9] text-[#171d19] px-2.5 py-1 rounded-md font-bold">
                {item.calories} kcal
              </span>
              <span className="bg-[#eff5ef] px-2 py-1 rounded-md">{item.protein}g P</span>
              <span className="bg-[#eff5ef] px-2 py-1 rounded-md">{item.fat}g F</span>
              <span className="bg-[#eff5ef] px-2 py-1 rounded-md">{item.carbs}g C</span>
              <span className="text-[11px] text-[#3d4a42] ml-auto font-sans font-medium">{item.recommendation}</span>
            </div>
          </div>
        ))}

        <div className="bg-[#006948] text-white p-4 flex justify-between items-center font-mono text-xs">
          <span>Ringkasan Nutrisi Piring</span>
          <div className="flex gap-3 font-bold text-sm">
            <span>{totals.totalProtein.toFixed(1)}g Protein</span>
            <span>{totals.totalCarbs.toFixed(1)}g Carbs</span>
          </div>
        </div>
      </div>
    </div>
  );
};
