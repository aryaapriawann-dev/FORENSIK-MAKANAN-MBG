'use client';

import React from 'react';
import Image from 'next/image';

export const Navbar: React.FC = () => {
  return (
    <header className="sticky top-0 w-full z-50 bg-[#f5fbf5]/80 backdrop-blur-xl shadow-xs border-b border-[#e4eae4]">
      <div className="max-w-5xl mx-auto h-20 px-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Image
            src="/logo.png"
            alt="NutriSafe AI — Pemindai Forensik Pangan"
            width={150}
            height={52}
            priority
            className="h-[44px] w-auto select-none drop-shadow-sm"
          />
          <div className="hidden sm:block">
            <span className="font-bold text-lg text-[#171d19] tracking-tight leading-none">
              NutriSafe <span className="text-[#006948]">AI</span>
            </span>
            <span className="block text-[10px] uppercase font-mono font-bold tracking-widest text-[#6d7a72] -mt-0.5">
              Pemindai Forensik
            </span>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <span className="hidden md:inline-flex items-center gap-1.5 bg-[#00855d] text-[#f5fff7] font-mono text-[11px] font-semibold px-3 py-1.5 rounded-full shadow-xs">
            100% Gratis • Tanpa Login
          </span>
        </div>
      </div>
    </header>
  );
};
