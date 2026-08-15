'use client';

import React, { useRef, useState, useEffect } from 'react';
import { Image as ImageIcon, Scan, Upload, RefreshCw } from 'lucide-react';
import { analyzeFoodImage } from '../lib/detector';
import { FoodItemAnalysis } from '../lib/types';

interface CameraScannerProps {
  onAnalysisComplete: (results: FoodItemAnalysis[]) => void;
  onImageCaptured?: (dataUrl: string) => void;
}

export const CameraScanner: React.FC<CameraScannerProps> = ({
  onAnalysisComplete,
  onImageCaptured,
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [streamActive, setStreamActive] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [statusMessage, setStatusMessage] = useState('Mempersiapkan inferensi AI...');

  useEffect(() => {
    let stream: MediaStream | null = null;
    const start = async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: 'environment', width: { ideal: 720 } },
          audio: false,
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          setStreamActive(true);
        }
      } catch {
        setStreamActive(false);
      }
    };
    start();
    return () => {
      stream?.getTracks().forEach((track) => track.stop());
    };
  }, []);

  const simulateProgressAndProcess = async (canvas: HTMLCanvasElement) => {
    setIsProcessing(true);
    setProgress(5);
    setStatusMessage('Memindai struktur matriks visual...');

    // Simulasi progress bar 1% - 100% yang mulus & interaktif
    const interval = setInterval(() => {
      setProgress((prev) => {
        if (prev < 30) {
          setStatusMessage('Mendeteksi spektrum warna & indikator pembusukan...');
          return prev + 5;
        } else if (prev < 65) {
          setStatusMessage('Menjalankan inferensi model AI On-Device (Transformers.js)...');
          return prev + 4;
        } else if (prev < 90) {
          setStatusMessage('Mencocokkan profil gizi dengan database TKPI Kemenkes...');
          return prev + 3;
        } else if (prev < 98) {
          return prev + 1;
        }
        return prev;
      });
    }, 80);

    try {
      const results = await analyzeFoodImage(canvas);
      clearInterval(interval);
      setProgress(100);
      setStatusMessage('Analisis forensik & gizi selesai 100%!');

      setTimeout(() => {
        onAnalysisComplete(results);
        setIsProcessing(false);
        setProgress(0);
      }, 500);
    } catch (err) {
      console.error(err);
      clearInterval(interval);
      setIsProcessing(false);
      setProgress(0);
    }
  };

  const captureAndScan = async () => {
    if (!videoRef.current || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const video = videoRef.current;

    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;

    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const photoDataUrl = canvas.toDataURL('image/jpeg', 0.85);
      onImageCaptured?.(photoDataUrl);
      await simulateProgressAndProcess(canvas);
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !canvasRef.current) return;

    const img = new Image();
    img.onload = async () => {
      const canvas = canvasRef.current!;
      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.drawImage(img, 0, 0);
        const photoDataUrl = canvas.toDataURL('image/jpeg', 0.85);
        onImageCaptured?.(photoDataUrl);
        await simulateProgressAndProcess(canvas);
      }
    };
    img.src = URL.createObjectURL(file);
  };

  return (
    <div className="relative w-full rounded-2xl bg-white shadow-lg overflow-hidden flex flex-col border border-[#dee4de]">
      {/* Viewfinder Container */}
      <div className="relative w-full aspect-[4/3] bg-[#dee4de] overflow-hidden flex items-center justify-center">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className={`w-full h-full object-cover ${!streamActive ? 'hidden' : ''}`}
        />

        {!streamActive && (
          <div className="flex flex-col items-center justify-center p-6 text-center text-[#3d4a42]">
            <ImageIcon className="w-12 h-12 mb-2 text-[#6d7a72]" />
            <p className="text-sm font-semibold">Pratinjau kamera tidak tersedia</p>
            <p className="text-xs text-[#6d7a72] mt-1">Unggah gambar makanan melalui tombol di bawah</p>
          </div>
        )}

        <canvas ref={canvasRef} className="hidden" />

        {/* HUD Reticle Overlays (Corner Brackets) */}
        <div className="absolute inset-4 pointer-events-none z-10">
          <div className="absolute top-0 left-0 w-8 h-8 flex">
            <div className="w-1 h-full bg-[#006948] rounded-l-full opacity-90"></div>
            <div className="h-1 w-full bg-[#006948] rounded-t-full opacity-90"></div>
          </div>
          <div className="absolute top-0 right-0 w-8 h-8 flex justify-end">
            <div className="h-1 w-full bg-[#006948] rounded-t-full opacity-90"></div>
            <div className="w-1 h-full bg-[#006948] rounded-r-full opacity-90"></div>
          </div>
          <div className="absolute bottom-0 left-0 w-8 h-8 flex items-end">
            <div className="w-1 h-full bg-[#006948] rounded-l-full opacity-90"></div>
            <div className="h-1 w-full bg-[#006948] rounded-b-full opacity-90"></div>
          </div>
          <div className="absolute bottom-0 right-0 w-8 h-8 flex items-end justify-end">
            <div className="h-1 w-full bg-[#006948] rounded-b-full opacity-90"></div>
            <div className="w-1 h-full bg-[#006948] rounded-r-full opacity-90"></div>
          </div>
        </div>

        {/* Animated Scan Line */}
        <div className="absolute top-0 left-0 w-full h-[2px] bg-[#006948] shadow-[0_0_20px_rgba(0,105,72,1)] z-10 animate-scan-line pointer-events-none" />

        {/* Fullscreen HUD Forensic Progress Overlay */}
        {isProcessing && (
          <div className="absolute inset-0 bg-[#0f172a]/85 backdrop-blur-md z-30 flex flex-col items-center justify-center p-6 text-white text-center">
            <RefreshCw className="w-10 h-10 animate-spin text-[#006948] mb-4" />
            <span className="font-mono text-3xl font-black text-[#006948] tracking-tight mb-2">
              {progress}%
            </span>

            {/* Progress Bar Container */}
            <div className="w-full max-w-xs bg-[#1e293b] rounded-full h-3.5 p-0.5 border border-[#334155] shadow-inner mb-3 overflow-hidden">
              <div
                className="bg-[#006948] h-full rounded-full transition-all duration-150 ease-out shadow-[0_0_12px_#006948]"
                style={{ width: `${progress}%` }}
              />
            </div>

            <p className="text-xs font-mono text-[#006948] font-bold uppercase tracking-wider animate-pulse">
              {statusMessage}
            </p>
            <p className="text-[11px] text-[#94a3b8] mt-1 font-sans">
              Analisis Forensik Fisik & Makronutrisi di Perangkat
            </p>
          </div>
        )}

        {/* HUD Status Pill */}
        {!isProcessing && (
          <div className="absolute bottom-4 left-1/2 -translate-x-1/2 bg-white/90 text-[#171d19] font-mono text-[12px] font-semibold px-4 py-2 rounded-full backdrop-blur-md shadow-md flex items-center gap-2 z-20">
            <span className="w-2.5 h-2.5 rounded-full bg-[#006948] animate-pulse"></span>
            <span>Siap memindai piring</span>
          </div>
        )}
      </div>

      {/* Action Buttons */}
      <div className="p-4 flex flex-col gap-3 bg-white">
        <button
          onClick={captureAndScan}
          disabled={isProcessing || !streamActive}
          className="w-full bg-[#006948] hover:bg-[#005137] text-white font-bold text-base py-4 rounded-xl flex items-center justify-center gap-2 shadow-md active:scale-[0.98] transition-transform cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <Scan className="w-5 h-5" />
          {isProcessing ? `Menganalisis (${progress}%)...` : 'Pindai & Analisis Makanan'}
        </button>

        <button
          onClick={() => fileInputRef.current?.click()}
          disabled={isProcessing}
          className="w-full bg-[#adedd3] text-[#306d58] font-semibold text-sm py-3 rounded-xl flex items-center justify-center gap-2 hover:bg-[#95d3ba] active:scale-[0.98] transition-transform cursor-pointer disabled:opacity-50"
        >
          <Upload className="w-4 h-4" />
          <span>Unggah Gambar dari Perangkat</span>
        </button>

        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={handleFileUpload}
        />
      </div>
    </div>
  );
};
