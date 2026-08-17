'use client';

import React, { useEffect, useState } from 'react';
import { analyzeFoodImage } from '../../lib/detector';
import { FoodItemAnalysis } from '../../lib/types';

const IMAGES = [
  { key: 'mbg', src: '/test_mbg_tray.png', label: 'Nampan MBG (nasi+ayam+sayur+sambal)' },
  { key: 'thali', src: '/test_plate.jpg', label: 'Piring Thali (test lama)' },
];

export default function VerifyForensic() {
  const [status, setStatus] = useState('Memuat gambar...');
  const [items, setItems] = useState<FoodItemAnalysis[] | null>(null);
  const [log, setLog] = useState<string[]>([]);
  const [imgKey, setImgKey] = useState('mbg');

  useEffect(() => {
    let cancelled = false;
    const src = IMAGES.find((i) => i.key === imgKey)!.src;
    async function run() {
      try {
        setItems(null);
        setStatus('Memuat gambar ' + src + ' ...');
        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.onload = async () => {
          if (cancelled) return;
          const canvas = document.createElement('canvas');
          canvas.width = img.naturalWidth;
          canvas.height = img.naturalHeight;
          const ctx = canvas.getContext('2d');
          if (!ctx) {
            setStatus('Gagal membuat canvas 2D');
            return;
          }
          ctx.drawImage(img, 0, 0);
          setStatus('Menjalankan analisis forensik multi-lauk...');
          const t0 = performance.now();
          const results = await analyzeFoodImage(canvas);
          const dt = (performance.now() - t0).toFixed(0);
          if (cancelled) return;
          setLog((l) => [...l, `[${src}] analyzed in ${dt}ms, ${results.length} items`]);
          setItems(results);
          setStatus(`SELESAI: ${results.length} komponen lauk terdeteksi`);
        };
        img.onerror = () => setStatus('Gagal load ' + src);
        img.src = src + '?t=' + Date.now();
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        setStatus('ERROR: ' + msg);
      }
    }
    run();
    return () => {
      cancelled = true;
    };
  }, [imgKey]);

  return (
    <div style={{ fontFamily: 'monospace', padding: 24, color: '#111' }}>
      <h1>Verify Forensic Pipeline</h1>
      <div style={{ marginBottom: 12 }}>
        {IMAGES.map((i) => (
          <button
            key={i.key}
            onClick={() => setImgKey(i.key)}
            style={{
              marginRight: 8,
              padding: '6px 12px',
              cursor: 'pointer',
              background: i.key === imgKey ? '#006948' : '#ddd',
              color: i.key === imgKey ? 'white' : '#111',
              border: '1px solid #999',
              borderRadius: 6,
            }}
          >
            {i.label}
          </button>
        ))}
      </div>
      <p>Status: <b>{status}</b></p>
      {log.map((l, i) => (
        <div key={i} style={{ color: '#666' }}>{l}</div>
      ))}
      <hr />
      {items && (
        <table border={1} cellPadding={6} style={{ borderCollapse: 'collapse', width: '100%' }}>
          <thead>
            <tr style={{ background: '#006948', color: 'white' }}>
              <th>#</th><th>Nama</th><th>Kategori</th><th>Conf</th><th>Status</th>
              <th>kkal</th><th>Prot</th><th>Lemak</th><th>Karbo</th><th>Box (x,y,w,h)</th>
            </tr>
          </thead>
          <tbody>
            {items.map((it, i) => (
              <tr key={it.id} style={{ background: it.safetyStatus === 'danger' ? '#ffe0e0' : 'white' }}>
                <td>{i + 1}</td>
                <td>{it.name}</td>
                <td>{it.category}</td>
                <td>{it.confidence}%</td>
                <td>{it.safetyStatus}</td>
                <td>{it.calories}</td>
                <td>{it.protein}</td>
                <td>{it.fat}</td>
                <td>{it.carbs}</td>
                <td>{it.box ? `${it.box.x.toFixed(2)},${it.box.y.toFixed(2)},${it.box.width.toFixed(2)},${it.box.height.toFixed(2)}` : '-'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {items && items.length === 0 && (
        <p style={{ color: 'red' }}>TIDAK ADA ITEM TERDETEKSI.</p>
      )}
    </div>
  );
}

