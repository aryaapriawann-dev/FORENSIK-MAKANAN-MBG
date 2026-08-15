"""
Dataset Generator & Compiler for NutriSafe AI
Compiles Indonesian food composition dataset (TKPI - Kemenkes RI)
and visual/microbiological food forensic spoilage database.
Maps Vision AI classes (Food-101, ImageNet labels) to real Indonesian dishes and accurate nutritional & forensic metrics.
"""

import json
import os

# Comprehensive Indonesian Food & Forensic Nutrition Dataset
DATASET = [
    # --- POKOK (STAPLES) ---
    {
        "food_code": "NASI_PUTIH",
        "food_name": "Nasi Putih",
        "category": "Pokok",
        "serving_size_gram": 150,
        "calories": 195.0,
        "protein": 3.6,
        "fat": 0.4,
        "carbs": 42.9,
        "fiber": 0.3,
        "shelf_life_hours": 18,
        "aliases": ["rice", "white rice", "steamed rice", "nasi"],
        "spoilage_signs": ["Menguning abnormal", "Bercak spora merah/pink (Bacillus cereus)", "Berbau asam/basi", "Berlendir dan berair"]
    },
    {
        "food_code": "NASI_GORENG",
        "food_name": "Nasi Goreng",
        "category": "Pokok",
        "serving_size_gram": 200,
        "calories": 330.0,
        "protein": 7.8,
        "fat": 12.5,
        "carbs": 46.2,
        "fiber": 1.2,
        "shelf_life_hours": 14,
        "aliases": ["fried rice", "nasi goreng"],
        "spoilage_signs": ["Bau tengik menyengat", "Tekstur lembek basah berlendir", "Bercak jamur putih"]
    },
    {
        "food_code": "NASI_UDUK",
        "food_name": "Nasi Uduk",
        "category": "Pokok",
        "serving_size_gram": 150,
        "calories": 230.0,
        "protein": 4.2,
        "fat": 6.8,
        "carbs": 38.5,
        "fiber": 0.5,
        "shelf_life_hours": 10,
        "aliases": ["coconut rice", "nasi uduk", "nasi gurih"],
        "spoilage_signs": ["Santan terfermentasi asam", "Lendir pekat", "Aroma basi tajam"]
    },
    {
        "food_code": "NASI_KUNING",
        "food_name": "Nasi Kuning",
        "category": "Pokok",
        "serving_size_gram": 150,
        "calories": 220.0,
        "protein": 4.0,
        "fat": 5.5,
        "carbs": 39.0,
        "fiber": 0.6,
        "shelf_life_hours": 12,
        "aliases": ["yellow rice", "turmeric rice", "nasi kuning"],
        "spoilage_signs": ["Warna kuning pudar keabuan", "Bau masam", "Berlendir"]
    },
    {
        "food_code": "MIE_GORENG",
        "food_name": "Mie Goreng",
        "category": "Pokok",
        "serving_size_gram": 150,
        "calories": 280.0,
        "protein": 6.5,
        "fat": 11.0,
        "carbs": 39.0,
        "fiber": 1.5,
        "shelf_life_hours": 16,
        "aliases": ["fried noodles", "noodles", "chow mein", "mie goreng", "bakmi"],
        "spoilage_signs": ["Mie hancur berlendir licin", "Bau asam/basi", "Bercak kapang"]
    },
    {
        "food_code": "KENTANG_REBUS",
        "food_name": "Kentang Rebus",
        "category": "Pokok",
        "serving_size_gram": 150,
        "calories": 130.0,
        "protein": 3.0,
        "fat": 0.2,
        "carbs": 29.5,
        "fiber": 2.5,
        "shelf_life_hours": 20,
        "aliases": ["potato", "boiled potato", "mashed potato"],
        "spoilage_signs": ["Tekstur lembek berlendir", "Bintik hitam/kehijauan (solanin)", "Bau masam"]
    },
    {
        "food_code": "ROTI_TAWAR",
        "food_name": "Roti Tawar / Gandum",
        "category": "Pokok",
        "serving_size_gram": 70,
        "calories": 175.0,
        "protein": 5.8,
        "fat": 2.1,
        "carbs": 33.0,
        "fiber": 1.8,
        "shelf_life_hours": 72,
        "aliases": ["bread", "toast", "sandwich", "roti", "white bread"],
        "spoilage_signs": ["Koloni jamur (kapang) bercak hijau/hitam/putih fuzzy", "Aroma apek/kapur", "Tekstur rapuh mengering abnormal"]
    },

    # --- LAUK HEWANI (ANIMAL PROTEIN) ---
    {
        "food_code": "AYAM_GORENG",
        "food_name": "Ayam Goreng",
        "category": "Lauk Hewani",
        "serving_size_gram": 100,
        "calories": 260.0,
        "protein": 27.0,
        "fat": 16.0,
        "carbs": 0.0,
        "fiber": 0.0,
        "shelf_life_hours": 24,
        "aliases": ["fried chicken", "chicken wings", "ayam goreng", "ayam krispi", "chicken"],
        "spoilage_signs": ["Warna pudar keabuan dekat tulang", "Bau tengik/asam busuk", "Permukaan lengket/berlendir licin"]
    },
    {
        "food_code": "AYAM_BAKAR",
        "food_name": "Ayam Bakar",
        "category": "Lauk Hewani",
        "serving_size_gram": 100,
        "calories": 210.0,
        "protein": 26.5,
        "fat": 10.5,
        "carbs": 2.5,
        "fiber": 0.0,
        "shelf_life_hours": 20,
        "aliases": ["grilled chicken", "roasted chicken", "bbq chicken", "ayam bakar"],
        "spoilage_signs": ["Bumbu kecap berlendir asam", "Bau tengik busuk", "Bercak kapang abu-abu"]
    },
    {
        "food_code": "RENDANG_DAGING",
        "food_name": "Rendang Daging Sapi",
        "category": "Lauk Hewani",
        "serving_size_gram": 100,
        "calories": 290.0,
        "protein": 28.0,
        "fat": 19.0,
        "carbs": 3.0,
        "fiber": 1.0,
        "shelf_life_hours": 72,
        "aliases": ["rendang", "beef rendang", "beef", "steak", "pot roast"],
        "spoilage_signs": ["Minyak bumbu berbusa/tengik", "Daging berubah rasa masam", "Bintik jamur putih di lapisan minyak"]
    },
    {
        "food_code": "TELUR_DADAR",
        "food_name": "Telur Dadar",
        "category": "Lauk Hewani",
        "serving_size_gram": 60,
        "calories": 154.0,
        "protein": 9.3,
        "fat": 12.0,
        "carbs": 1.2,
        "fiber": 0.0,
        "shelf_life_hours": 16,
        "aliases": ["omelet", "omelette", "fried egg", "telur dadar", "egg"],
        "spoilage_signs": ["Bau tengik hidrogen sulfida", "Warna kehijauan gelap abnormal", "Permukaan berlendir"]
    },
    {
        "food_code": "TELUR_REBUS",
        "food_name": "Telur Rebus",
        "category": "Lauk Hewani",
        "serving_size_gram": 55,
        "calories": 78.0,
        "protein": 6.3,
        "fat": 5.3,
        "carbs": 0.6,
        "fiber": 0.0,
        "shelf_life_hours": 24,
        "aliases": ["boiled egg", "hard boiled egg", "telur rebus", "poached egg"],
        "spoilage_signs": ["Kuning telur menghitam basah", "Bau busuk belerang menyengat", "Putih telur berair lembek"]
    },
    {
        "food_code": "IKAN_GORENG",
        "food_name": "Ikan Goreng",
        "category": "Lauk Hewani",
        "serving_size_gram": 100,
        "calories": 190.0,
        "protein": 22.0,
        "fat": 11.0,
        "carbs": 0.0,
        "fiber": 0.0,
        "shelf_life_hours": 18,
        "aliases": ["fried fish", "fish and chips", "ikan goreng", "fish"],
        "spoilage_signs": ["Bau amonia/anyir busuk tajam", "Daging hancur lembek", "Permukaan lengket berlendir"]
    },
    {
        "food_code": "IKAN_BAKAR",
        "food_name": "Ikan Bakar",
        "category": "Lauk Hewani",
        "serving_size_gram": 100,
        "calories": 165.0,
        "protein": 24.0,
        "fat": 7.0,
        "carbs": 1.5,
        "fiber": 0.0,
        "shelf_life_hours": 16,
        "aliases": ["grilled fish", "ikan bakar", "grilled salmon"],
        "spoilage_signs": ["Aroma anyir asam", "Daging lembek hancur berlendir", "Bercak jamur"]
    },
    {
        "food_code": "SATE_AYAM",
        "food_name": "Sate Ayam (Bumbu Kacang)",
        "category": "Lauk Hewani",
        "serving_size_gram": 100,
        "calories": 245.0,
        "protein": 20.0,
        "fat": 15.0,
        "carbs": 8.0,
        "fiber": 1.5,
        "shelf_life_hours": 16,
        "aliases": ["satay", "chicken satay", "sate ayam", "skewers"],
        "spoilage_signs": ["Bumbu kacang asam berbusa", "Daging berlendir", "Bau tengik minyak kacang"]
    },

    # --- LAUK NABATI (PLANT PROTEIN) ---
    {
        "food_code": "TEMPE_GORENG",
        "food_name": "Tempe Goreng",
        "category": "Lauk Nabati",
        "serving_size_gram": 50,
        "calories": 118.0,
        "protein": 9.0,
        "fat": 7.5,
        "carbs": 4.0,
        "fiber": 1.4,
        "shelf_life_hours": 24,
        "aliases": ["tempeh", "fried tempeh", "tempe goreng", "tempe"],
        "spoilage_signs": ["Bau amonia/sangit tajam", "Bercak hitam basah berlendir", "Rasa pahit menyengat"]
    },
    {
        "food_code": "TEMPE_OREK",
        "food_name": "Orek Tempe Manis",
        "category": "Lauk Nabati",
        "serving_size_gram": 60,
        "calories": 145.0,
        "protein": 9.5,
        "fat": 6.8,
        "carbs": 12.0,
        "fiber": 1.6,
        "shelf_life_hours": 36,
        "aliases": ["orek tempe", "sweet tempeh"],
        "spoilage_signs": ["Gula berbusa/fermentasi", "Bau asam menyengat", "Tekstur lembek berair"]
    },
    {
        "food_code": "TAHU_GORENG",
        "food_name": "Tahu Goreng",
        "category": "Lauk Nabati",
        "serving_size_gram": 50,
        "calories": 58.0,
        "protein": 4.8,
        "fat": 3.5,
        "carbs": 2.1,
        "fiber": 0.8,
        "shelf_life_hours": 16,
        "aliases": ["tofu", "fried tofu", "tahu goreng", "tahu"],
        "spoilage_signs": ["Rasa masam tajam", "Lendir licin tebal di permukaan", "Tekstur hancur berair"]
    },
    {
        "food_code": "TAHU_ISI",
        "food_name": "Tahu Isi / Bakwan",
        "category": "Lauk Nabati",
        "serving_size_gram": 75,
        "calories": 135.0,
        "protein": 4.5,
        "fat": 8.5,
        "carbs": 11.0,
        "fiber": 1.2,
        "shelf_life_hours": 14,
        "aliases": ["fritter", "stuffed tofu", "bakwan", "gorengan", "spring roll"],
        "spoilage_signs": ["Sayuran dalam berbau asam", "Tepung berminyak tengik berlendir", "Rasa masam"]
    },

    # --- SAYUR (VEGETABLES) ---
    {
        "food_code": "SAYUR_SOP",
        "food_name": "Sayur Sop Bening",
        "category": "Sayur",
        "serving_size_gram": 150,
        "calories": 45.0,
        "protein": 1.5,
        "fat": 0.5,
        "carbs": 9.0,
        "fiber": 1.8,
        "shelf_life_hours": 12,
        "aliases": ["soup", "vegetable soup", "clear soup", "sayur sop", "broth"],
        "spoilage_signs": ["Kuah berbusa fermentasi bakteri", "Kuah keruh masam", "Sayuran layu hancur berlendir"]
    },
    {
        "food_code": "SAYUR_LODEH",
        "food_name": "Sayur Lodeh (Santan)",
        "category": "Sayur",
        "serving_size_gram": 150,
        "calories": 120.0,
        "protein": 2.8,
        "fat": 8.5,
        "carbs": 9.2,
        "fiber": 2.1,
        "shelf_life_hours": 8,
        "aliases": ["curry soup", "coconut soup", "lodeh", "sayur lodeh"],
        "spoilage_signs": ["Santan pecah menggumpal", "Busa gas abnormal", "Rasa masam/kecut"]
    },
    {
        "food_code": "SAYUR_ASEM",
        "food_name": "Sayur Asem",
        "category": "Sayur",
        "serving_size_gram": 150,
        "calories": 52.0,
        "protein": 1.8,
        "fat": 0.6,
        "carbs": 10.5,
        "fiber": 2.0,
        "shelf_life_hours": 18,
        "aliases": ["tamarind soup", "sayur asem", "sour soup"],
        "spoilage_signs": ["Kuah berbusa pekat", "Sayuran berlendir licin", "Aroma fermentasi alkoholik"]
    },
    {
        "food_code": "TUMIS_KANGKUNG",
        "food_name": "Tumis Kangkung / Sayur Hijau",
        "category": "Sayur",
        "serving_size_gram": 100,
        "calories": 65.0,
        "protein": 2.5,
        "fat": 3.8,
        "carbs": 5.5,
        "fiber": 2.2,
        "shelf_life_hours": 10,
        "aliases": ["stir-fry", "water spinach", "kangkung", "sauteed vegetables", "greens", "spinach"],
        "spoilage_signs": ["Daun menghitam hancur berlendir", "Air tumisan berbuih masam", "Bau apek pembusukan daun"]
    },
    {
        "food_code": "CAPCAY",
        "food_name": "Capcay Kuah / Goreng",
        "category": "Sayur",
        "serving_size_gram": 150,
        "calories": 95.0,
        "protein": 3.5,
        "fat": 4.5,
        "carbs": 11.0,
        "fiber": 2.8,
        "shelf_life_hours": 12,
        "aliases": ["capcay", "stir fried mixed vegetables", "mixed vegetables"],
        "spoilage_signs": ["Kuah mengental berlendir asam", "Wortel/brokoli lembek membusuk", "Gelembung gas"]
    },
    {
        "food_code": "GADO_GADO",
        "food_name": "Gado-Gado / Pecel",
        "category": "Sayur",
        "serving_size_gram": 200,
        "calories": 295.0,
        "protein": 11.0,
        "fat": 14.5,
        "carbs": 32.0,
        "fiber": 5.2,
        "shelf_life_hours": 8,
        "aliases": ["salad", "gado gado", "pecel", "peanut salad"],
        "spoilage_signs": ["Bumbu kacang masam berbuih", "Sayuran berlendir", "Aroma basi menyengat"]
    },

    # --- PELENGKAP & BUAH (CONDIMENTS & FRUITS) ---
    {
        "food_code": "SAMBAL_TERASI",
        "food_name": "Sambal Terasi / Sambal Ulek",
        "category": "Pelengkap",
        "serving_size_gram": 20,
        "calories": 35.0,
        "protein": 0.8,
        "fat": 2.1,
        "carbs": 3.2,
        "fiber": 0.5,
        "shelf_life_hours": 36,
        "aliases": ["chili", "hot sauce", "salsa", "sambal", "sambal terasi", "chili paste"],
        "spoilage_signs": ["Bintik spora jamur putih/abu-abu di permukaan", "Gelembung gas fermentasi liar", "Bau asam alkoholik"]
    },
    {
        "food_code": "KERUPUK",
        "food_name": "Kerupuk Putih / Kerupuk Udang",
        "category": "Pelengkap",
        "serving_size_gram": 20,
        "calories": 100.0,
        "protein": 0.6,
        "fat": 5.2,
        "carbs": 12.8,
        "fiber": 0.1,
        "shelf_life_hours": 168,
        "aliases": ["cracker", "chips", "crisps", "kerupuk", "krupuk"],
        "spoilage_signs": ["Melempem berbau tengik minyak berat", "Bercak jamur abu-abu jika lembap"]
    },
    {
        "food_code": "PISANG",
        "food_name": "Pisang Segar",
        "category": "Buah",
        "serving_size_gram": 100,
        "calories": 89.0,
        "protein": 1.1,
        "fat": 0.3,
        "carbs": 22.8,
        "fiber": 2.6,
        "shelf_life_hours": 72,
        "aliases": ["banana", "pisang"],
        "spoilage_signs": ["Kulit menghitam basah lembek", "Daging buah berair fermentasi alkoholik", "Bintik jamur putih"]
    },
    {
        "food_code": "JERUK",
        "food_name": "Jeruk Manis",
        "category": "Buah",
        "serving_size_gram": 100,
        "calories": 47.0,
        "protein": 0.9,
        "fat": 0.1,
        "carbs": 11.8,
        "fiber": 2.4,
        "shelf_life_hours": 96,
        "aliases": ["orange", "citrus", "jeruk", "tangerine"],
        "spoilage_signs": ["Kapang hijau/putih kebiruan (Penicillium digitatum)", "Kulit lembek berair busuk", "Aroma masam fermentasi"]
    },
    {
        "food_code": "APEL",
        "food_name": "Apel Segar",
        "category": "Buah",
        "serving_size_gram": 100,
        "calories": 52.0,
        "protein": 0.3,
        "fat": 0.2,
        "carbs": 13.8,
        "fiber": 2.4,
        "shelf_life_hours": 120,
        "aliases": ["apple", "apel"],
        "spoilage_signs": ["Bercak cokelat busuk melekuk", "Tekstur lembek berongga", "Kapang putih/hitam"]
    },
    {
        "food_code": "SEMANGKA",
        "food_name": "Semangka Segar",
        "category": "Buah",
        "serving_size_gram": 150,
        "calories": 45.0,
        "protein": 0.9,
        "fat": 0.2,
        "carbs": 11.2,
        "fiber": 0.6,
        "shelf_life_hours": 24,
        "aliases": ["watermelon", "semangka", "melon"],
        "spoilage_signs": ["Daging buah berlendir licin", "Aroma masam fermentasi gas", "Lapisan busa di permukaan"]
    }
]

def export_typescript(dataset, filepath):
    ts_code = """// AUTO-GENERATED BY scripts/generate_dataset.py
// Standar Gizi Pangan Indonesia (TKPI Kemenkes RI) & Database Forensik Makanan

import { NutritionMasterItem } from './types';

export interface ExtendedNutritionItem extends NutritionMasterItem {
  aliases: string[];
}

export const COMPREHENSIVE_FOOD_DATASET: ExtendedNutritionItem[] = """
    ts_code += json.dumps(dataset, indent=2, ensure_ascii=False)
    ts_code += """;

export const FALLBACK_NUTRITION_DATA: NutritionMasterItem[] = COMPREHENSIVE_FOOD_DATASET.map(
  ({ aliases, ...item }) => item
);

export function findNutritionByText(query: string): ExtendedNutritionItem | null {
  if (!query || !query.trim()) return null;
  const q = query.toLowerCase().trim();
  
  // 1. Exact match on food_code or aliases
  for (const item of COMPREHENSIVE_FOOD_DATASET) {
    if (item.food_code.toLowerCase() === q) return item;
    if (item.aliases.some((alias) => alias.toLowerCase() === q)) return item;
    if (item.food_name.toLowerCase() === q) return item;
  }

  // 2. Partial match on aliases
  for (const item of COMPREHENSIVE_FOOD_DATASET) {
    if (item.aliases.some((alias) => q.includes(alias) || alias.includes(q))) return item;
    if (item.food_name.toLowerCase().includes(q) || q.includes(item.food_name.toLowerCase())) return item;
  }

  return null;
}

export function findFallbackNutrition(query: string): NutritionMasterItem | null {
  return findNutritionByText(query);
}
"""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(ts_code)
    print(f"[OK] TypeScript dataset written to: {filepath}")

def export_sql_seeder(dataset, filepath):
    sql = """-- AUTO-GENERATED SEEDER: TKPI & Forensic Safety Database
INSERT INTO public.nutrition_master 
(food_code, food_name, category, serving_size_gram, calories, protein, fat, carbs, fiber, shelf_life_hours, spoilage_signs)
VALUES
"""
    rows = []
    for item in dataset:
        signs = []
        for s in item["spoilage_signs"]:
            escaped = s.replace("'", "''")
            signs.append(f"'{escaped}'")
        signs_escaped = ", ".join(signs)
        row = f"('{item['food_code']}', '{item['food_name']}', '{item['category']}', {item['serving_size_gram']}, {item['calories']}, {item['protein']}, {item['fat']}, {item['carbs']}, {item['fiber']}, {item['shelf_life_hours']}, ARRAY[{signs_escaped}])"
        rows.append(row)
    
    sql += ",\n".join(rows)
    sql += "\nON CONFLICT (food_code) DO UPDATE SET\n"
    sql += "  calories = EXCLUDED.calories,\n"
    sql += "  protein = EXCLUDED.protein,\n"
    sql += "  fat = EXCLUDED.fat,\n"
    sql += "  carbs = EXCLUDED.carbs,\n"
    sql += "  fiber = EXCLUDED.fiber,\n"
    sql += "  shelf_life_hours = EXCLUDED.shelf_life_hours,\n"
    sql += "  spoilage_signs = EXCLUDED.spoilage_signs;\n"

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(sql)
    print(f"[OK] SQL Seeder written to: {filepath}")

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ts_out = os.path.join(base_dir, "lib", "nutritionFallback.ts")
    sql_out = os.path.join(base_dir, "docs", "seed_comprehensive_dataset.sql")
    
    export_typescript(DATASET, ts_out)
    export_sql_seeder(DATASET, sql_out)
    print(f"Total compiled food & forensic items: {len(DATASET)}")

if __name__ == "__main__":
    main()
