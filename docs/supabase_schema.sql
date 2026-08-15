-- ====================================================================
-- NUTRISAFE AI — SUPABASE DATABASE SCHEMA (FULL PRODUCTION)
-- ====================================================================

-- 1. TABEL MASTER GIZI & STANDAR PANGAN INDONESIA (TKPI Kemenkes RI)
CREATE TABLE IF NOT EXISTS public.nutrition_master (
    id BIGSERIAL PRIMARY KEY,
    food_code TEXT UNIQUE NOT NULL,
    food_name TEXT NOT NULL,
    category TEXT NOT NULL, -- Pokok, Lauk Hewani, Lauk Nabati, Sayur, Pelengkap
    serving_size_gram REAL DEFAULT 100,
    calories REAL NOT NULL,
    protein REAL NOT NULL,
    fat REAL NOT NULL,
    carbs REAL NOT NULL,
    fiber REAL NOT NULL,
    shelf_life_hours INT DEFAULT 12,
    spoilage_signs TEXT[] DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 2. TABEL LOG FORENSIK PEMINDAIAN (ANONYMOUS SCAN HISTORY)
CREATE TABLE IF NOT EXISTS public.forensic_scan_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    detected_items JSONB NOT NULL, -- Menyimpan array item hasil inferensi AI
    total_calories REAL NOT NULL,
    total_protein REAL NOT NULL,
    total_fat REAL NOT NULL,
    total_carbs REAL NOT NULL,
    overall_safety_status TEXT NOT NULL CHECK (overall_safety_status IN ('safe', 'warning', 'danger')),
    scanned_at TIMESTAMPTZ DEFAULT NOW()
);

-- ====================================================================
-- ROW LEVEL SECURITY (RLS) CONFIGURATION
-- ====================================================================

-- Aktifkan RLS pada kedua tabel
ALTER TABLE public.nutrition_master ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.forensic_scan_logs ENABLE ROW LEVEL SECURITY;

-- Policy 1: Akses Read-Only Publik untuk Data Master Gizi (Guest First / Tanpa Login)
CREATE POLICY "Allow Public Read Nutrition Master"
ON public.nutrition_master
FOR SELECT
TO anon, authenticated
USING (true);

-- Policy 2: Publik dapat membuat log hasil scan anonim tanpa login
CREATE POLICY "Allow Public Insert Forensic Scan Logs"
ON public.forensic_scan_logs
FOR INSERT
TO anon, authenticated
WITH CHECK (true);

-- Policy 3: Publik dapat membaca log riwayat scan anonim
CREATE POLICY "Allow Public Read Forensic Scan Logs"
ON public.forensic_scan_logs
FOR SELECT
TO anon, authenticated
USING (true);

-- ====================================================================
-- DATA SEEDER MASTER GIZI & INDIKATOR FORENSIK PANGAN LOKAL
-- ====================================================================

INSERT INTO public.nutrition_master 
(food_code, food_name, category, serving_size_gram, calories, protein, fat, carbs, fiber, shelf_life_hours, spoilage_signs)
VALUES
('NASI_PUTIH', 'Nasi Putih', 'Pokok', 150, 195.0, 3.6, 0.4, 42.9, 0.3, 18, ARRAY['Menguning abnormal', 'Bercak spora merah/pink', 'Berbau asam/basi', 'Berlendir']),
('AYAM_GORENG', 'Ayam Goreng', 'Lauk Hewani', 100, 260.0, 27.0, 16.0, 0.0, 0.0, 24, ARRAY['Warna puyeh keabuan di dekat tulang', 'Bau tengik minyak teroksidasi', 'Permukaan lengket berlendir']),
('TEMPE_GORENG', 'Tempe Goreng', 'Lauk Nabati', 50, 118.0, 9.0, 7.5, 4.0, 1.4, 24, ARRAY['Bau amonia/sangit tajam', 'Bercak hitam basah berlendir']),
('TAHU_GORENG', 'Tahu Goreng', 'Lauk Nabati', 50, 58.0, 4.8, 3.5, 2.1, 0.8, 16, ARRAY['Rasa masam tajam', 'Lendir licin permukaan', 'Tekstur lembek hancur berair']),
('SAYUR_SOP', 'Sayur Sop Bening', 'Sayur', 150, 45.0, 1.5, 0.5, 9.0, 1.8, 12, ARRAY['Kuah berbusa fermentasi', 'Kuah keruh asam', 'Sayuran layu hancur berlendir']),
('SAYUR_LODEH', 'Sayur Lodeh (Santan)', 'Sayur', 150, 120.0, 2.8, 8.5, 9.2, 2.1, 8, ARRAY['Santan pecah menggumpal', 'Busa gas abnormal', 'Rasa masam']),
('TELUR_DADAR', 'Telur Dadar', 'Lauk Hewani', 60, 154.0, 9.3, 12.0, 1.2, 0.0, 16, ARRAY['Bau tengik', 'Perubahan warna kehijauan gelap', 'Berlendir']),
('SAMBAL_TERASI', 'Sambal Terasi', 'Pelengkap', 20, 35.0, 0.8, 2.1, 3.2, 0.5, 36, ARRAY['Bintik spora jamur putih/abu-abu', 'Gelembung gas fermentasi liar'])
ON CONFLICT (food_code) DO NOTHING;
