'use client';

import React from 'react';
import { Eye, AlertTriangle, ShieldCheck } from 'lucide-react';

export const SafetyTipsGrid: React.FC = () => {
  const tips = [
    {
      icon: Eye,
      title: 'Perubahan Warna Visual',
      description:
        'Warna pucat keabuan pada daging atau bercak hijau/pink pada nasi matang yang menandakan pembusukan bakteri.',
    },
    {
      icon: AlertTriangle,
      title: 'Jamur & Lendir',
      description:
        'Lapisan licin pada kacang-kacangan/tahu atau koloni spora jamur putih.',
    },
    {
      icon: ShieldCheck,
      title: 'Gas Fermentasi',
      description:
        'Busa abnormal pada kuah santan atau bau asam yang tajam.',
    },
  ];

  return (
    <div className="w-full mt-6 flex flex-col gap-3">
      <h4 className="font-bold text-sm text-[#171d19] px-1 uppercase tracking-wider font-mono">
        Indikator Fisik Forensik
      </h4>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        {tips.map((tip, index) => {
          const Icon = tip.icon;
          return (
            <div
              key={index}
              className="bg-white p-4 rounded-xl border border-[#dee4de] flex flex-col gap-2 shadow-2xs"
            >
              <div className="w-8 h-8 rounded-lg bg-[#eff5ef] flex items-center justify-center text-[#006948]">
                <Icon className="w-4 h-4" />
              </div>
              <h5 className="font-bold text-sm text-[#171d19]">{tip.title}</h5>
              <p className="text-xs text-[#3d4a42] leading-relaxed">{tip.description}</p>
            </div>
          );
        })}
      </div>
    </div>
  );
};
