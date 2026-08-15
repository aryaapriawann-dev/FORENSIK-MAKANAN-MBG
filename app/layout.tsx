import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'NutriSafe AI — Deteksi Forensik Pangan & Gizi',
  description:
    'Sistem Deteksi Forensik Kelayakan & Analisis Gizi Pangan Publik Berbasis AI On-Device.',
  icons: {
    icon: '/logo-icon.png',
    apple: '/logo-icon.png',
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="id">
      <body className="antialiased bg-slate-50 text-slate-900 min-h-screen">
        {children}
      </body>
    </html>
  );
}
