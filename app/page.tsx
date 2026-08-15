'use client';

import React, { useState } from 'react';
import { Navbar } from '../components/Navbar';
import { CameraScanner } from '../components/CameraScanner';
import { FoodForensicTable } from '../components/FoodForensicTable';
import { SafetyTipsGrid } from '../components/SafetyTipsGrid';
import { FoodItemAnalysis, NutritionTotal } from '../lib/types';
import { Zap, ShieldCheck } from 'lucide-react';

export default function NutriSafeApp() {
  const [items, setItems] = useState<FoodItemAnalysis[]>([]);
  const [scannedImage, setScannedImage] = useState<string | null>(null);

  const handleAnalysisComplete = (newItems: FoodItemAnalysis[]) => {
    setItems(newItems);
  };

  const totals: NutritionTotal = items.reduce(
    (acc, curr) => {
      acc.totalCalories += curr.calories;
      acc.totalProtein += curr.protein;
      acc.totalFat += curr.fat;
      acc.totalCarbs += curr.carbs;
      acc.totalFiber += curr.fiber;
      if (curr.safetyStatus === 'danger') acc.overallSafety = 'danger';
      else if (curr.safetyStatus === 'warning' && acc.overallSafety !== 'danger')
        acc.overallSafety = 'warning';
      return acc;
    },
    {
      totalCalories: 0,
      totalProtein: 0,
      totalFat: 0,
      totalCarbs: 0,
      totalFiber: 0,
      overallSafety: 'safe',
    } as NutritionTotal
  );

  return (
    <div className="min-h-screen flex flex-col bg-[#f5fbf5] text-[#171d19]">
      <Navbar />

      <main className="flex-1 max-w-4xl w-full mx-auto px-4 py-6 flex flex-col items-center gap-6">
        {/* Intro Hero Badge & Header */}
        <div className="flex flex-col items-center gap-2 text-center max-w-lg">
          <span className="inline-flex items-center gap-1.5 bg-[#00855d] text-[#f5fff7] font-mono text-[11px] font-semibold px-4 py-1.5 rounded-full shadow-xs">
            <Zap className="w-3.5 h-3.5 fill-current" />
            100% Gratis • Tanpa Login • AI di Browser
          </span>
          <p className="text-sm md:text-base text-[#3d4a42] leading-relaxed mt-1">
            Cegah keracunan makanan dan pantau gizi harian Anda dengan pemindaian AI forensik.
          </p>
        </div>

        {/* Scanner HUD Container */}
        <section className="w-full max-w-xl">
          <CameraScanner
          onAnalysisComplete={handleAnalysisComplete}
          onImageCaptured={setScannedImage}
        />
        </section>

        {/* Diagnostic Results Table Container */}
        {items.length > 0 && (
          <section className="w-full max-w-4xl">
            <FoodForensicTable
              items={items}
              totals={totals}
              scannedImage={scannedImage}
            />
          </section>
        )}

        {/* Physical Forensic Indicators */}
        <section className="w-full max-w-xl">
          <SafetyTipsGrid />
        </section>
      </main>

      <footer className="w-full border-t border-[#e4eae4] bg-white py-4 mt-8 text-center text-xs text-[#6d7a72]">
        <div className="max-w-4xl mx-auto px-4 flex flex-col sm:flex-row justify-between items-center gap-2">
          <p>© 2026 NutriSafe AI — Platform Keamanan Pangan Klinis</p>
          <p className="flex items-center gap-1">
            <ShieldCheck className="w-3.5 h-3.5 text-[#006948]" /> Inferensi WASM On-Device
          </p>
        </div>
      </footer>
    </div>
  );
}
