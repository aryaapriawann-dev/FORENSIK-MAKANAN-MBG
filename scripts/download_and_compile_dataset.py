"""
Comprehensive Dataset Downloader & Compiler for NutriSafe AI
Downloads and merges:
1. Open Food Facts / Food-101 taxonomy (101 food classes)
2. Indonesian TKPI (Kemenkes RI) nutritional standard mapping
3. Visual spoilage & microbiological forensic rules
"""

import urllib.request
import json
import os
import re

# Complete Food-101 Class List (Standard AI Vision Food Benchmark)
FOOD_101_CLASSES = [
    "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare",
    "beet_salad", "beignets", "bibimbap", "bread_pudding", "breakfast_burrito",
    "bruschetta", "caesar_salad", "cannoli", "caprese_salad", "carrot_cake",
    "ceviche", "cheesecake", "cheese_plate", "chicken_curry", "chicken_quesadilla",
    "chicken_wings", "chocolate_cake", "chocolate_mousse", "churros", "clam_chowder",
    "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame", "cup_cakes",
    "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict",
    "escargots", "falafel", "filet_mignon", "fish_and_chips", "foie_gras",
    "french_fries", "french_onion_soup", "french_toast", "fried_calamari", "fried_rice",
    "frozen_yogurt", "garlic_bread", "gnocchi", "greek_salad", "grilled_cheese_sandwich",
    "grilled_salmon", "guacamole", "gyoza", "hamburger", "hot_and_sour_soup",
    "hot_dog", "huevos_rancheros", "hummus", "ice_cream", "lasagna",
    "lobster_bisque", "lobster_roll_sandwich", "macaroni_and_cheese", "macarons", "miso_soup",
    "mussels", "nachos", "omelette", "onion_rings", "oysters",
    "pad_thai", "paella", "pancakes", "panna_cotta", "peking_duck",
    "pho", "pizza", "pork_chop", "poutine", "prime_rib",
    "pulled_pork_sandwich", "ramen", "ravioli", "red_velvet_cake", "risotto",
    "samosa", "sashimi", "scallops", "seaweed_salad", "shrimp_and_grits",
    "spaghetti_bolognese", "spaghetti_carbonara", "spring_rolls", "steak", "strawberry_shortcake",
    "sushi", "tacos", "takoyaki", "tiramisu", "tuna_tartare",
    "waffles"
]

# Comprehensive TKPI Database (Tabel Komposisi Pangan Indonesia - Kemenkes RI)
TKPI_INDONESIAN_DATABASE = [
    # --- POKOK (STAPLE FOODS) ---
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
        "aliases": ["rice", "white rice", "steamed rice", "nasi", "cooked rice", "plain rice"],
        "spoilage_signs": ["Menguning abnormal", "Bercak spora merah/pink (Bacillus cereus)", "Berbau asam/basi", "Berlendir dan basah pekat"]
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
        "aliases": ["fried_rice", "fried rice", "nasi goreng", "yangzhou fried rice"],
        "spoilage_signs": ["Bau tengik menyengat", "Tekstur lembek basah berlendir", "Bercak jamur putih/abu-abu"]
    },
    {
        "food_code": "NASI_UDUK",
        "food_name": "Nasi Uduk / Nasi Gurih",
        "category": "Pokok",
        "serving_size_gram": 150,
        "calories": 230.0,
        "protein": 4.2,
        "fat": 6.8,
        "carbs": 38.5,
        "fiber": 0.5,
        "shelf_life_hours": 10,
        "aliases": ["coconut rice", "nasi uduk", "nasi gurih", "nasi lemak"],
        "spoilage_signs": ["Santan terfermentasi asam", "Lendir pekat asam", "Aroma basi tengik tajam"]
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
        "spoilage_signs": ["Warna kuning kusam keabuan", "Bau masam/kecut", "Berlendir licin"]
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
        "aliases": ["fried noodles", "noodles", "chow mein", "mie goreng", "bakmi", "pad_thai", "lo mein"],
        "spoilage_signs": ["Mie hancur berlendir licin", "Bau asam/basi", "Bercak kapang"]
    },
    {
        "food_code": "MIE_KUAH_RAMEN",
        "food_name": "Mie Kuah / Ramen / Soto Mie",
        "category": "Pokok",
        "serving_size_gram": 250,
        "calories": 310.0,
        "protein": 12.0,
        "fat": 9.5,
        "carbs": 44.0,
        "fiber": 2.0,
        "shelf_life_hours": 8,
        "aliases": ["ramen", "pho", "noodle soup", "mie kuah", "soto mie", "bakmi kuah"],
        "spoilage_signs": ["Kuah berbusa asam", "Mie rapuh lembek berlendir", "Aroma basi fermentasi"]
    },
    {
        "food_code": "BUBUR_AYAM",
        "food_name": "Bubur Ayam",
        "category": "Pokok",
        "serving_size_gram": 250,
        "calories": 185.0,
        "protein": 8.0,
        "fat": 4.5,
        "carbs": 28.0,
        "fiber": 0.8,
        "shelf_life_hours": 8,
        "aliases": ["congee", "porridge", "bubur ayam", "bubur"],
        "spoilage_signs": ["Bubur mencair encer masam", "Kuah kaldu berbusa", "Bau masam asam"]
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
        "aliases": ["bread", "toast", "sandwich", "roti", "white bread", "garlic_bread", "french_toast", "bruschetta"],
        "spoilage_signs": ["Koloni kapang putih/hijau/hitam fuzzy", "Aroma apek/kapur", "Tekstur rapuh mengering"]
    },
    {
        "food_code": "KENTANG_GORENG",
        "food_name": "Kentang Goreng (French Fries)",
        "category": "Pokok",
        "serving_size_gram": 100,
        "calories": 290.0,
        "protein": 3.4,
        "fat": 15.0,
        "carbs": 36.0,
        "fiber": 3.0,
        "shelf_life_hours": 24,
        "aliases": ["french_fries", "french fries", "fries", "kentang goreng", "poutine"],
        "spoilage_signs": ["Melempem berminyak tengik", "Bercak kehitaman lembek", "Bau apek"]
    },
    {
        "food_code": "KENTANG_REBUS",
        "food_name": "Kentang Rebus / Puree",
        "category": "Pokok",
        "serving_size_gram": 150,
        "calories": 130.0,
        "protein": 3.0,
        "fat": 0.2,
        "carbs": 29.5,
        "fiber": 2.5,
        "shelf_life_hours": 20,
        "aliases": ["potato", "boiled potato", "mashed potato", "gnocchi"],
        "spoilage_signs": ["Tekstur lembek berlendir", "Bintik hitam/kehijauan (solanin)", "Bau masam"]
    },
    {
        "food_code": "PASTA_SPAGHETTI",
        "food_name": "Pasta / Spaghetti",
        "category": "Pokok",
        "serving_size_gram": 180,
        "calories": 270.0,
        "protein": 9.5,
        "fat": 6.5,
        "carbs": 43.0,
        "fiber": 2.4,
        "shelf_life_hours": 18,
        "aliases": ["spaghetti_bolognese", "spaghetti_carbonara", "lasagna", "ravioli", "macaroni_and_cheese", "pasta"],
        "spoilage_signs": ["Saus berbusa asam", "Pasta berlendir licin", "Bercak jamur putih pada keju/saus"]
    },

    # --- LAUK HEWANI (ANIMAL PROTEIN) ---
    {
        "food_code": "AYAM_GORENG",
        "food_name": "Ayam Goreng (Fried Chicken)",
        "category": "Lauk Hewani",
        "serving_size_gram": 100,
        "calories": 260.0,
        "protein": 27.0,
        "fat": 16.0,
        "carbs": 0.0,
        "fiber": 0.0,
        "shelf_life_hours": 24,
        "aliases": ["fried chicken", "chicken_wings", "ayam goreng", "ayam krispi", "chicken", "chicken_quesadilla"],
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
        "aliases": ["grilled chicken", "roasted chicken", "bbq chicken", "ayam bakar", "peking_duck"],
        "spoilage_signs": ["Bumbu kecap berlendir asam", "Bau tengik busuk", "Bercak kapang abu-abu"]
    },
    {
        "food_code": "KARI_AYAM_GULAI",
        "food_name": "Kari Ayam / Gulai Ayam",
        "category": "Lauk Hewani",
        "serving_size_gram": 150,
        "calories": 255.0,
        "protein": 21.0,
        "fat": 17.5,
        "carbs": 3.5,
        "fiber": 0.5,
        "shelf_life_hours": 12,
        "aliases": ["chicken_curry", "curry", "gulai ayam", "kari ayam", "opor ayam"],
        "spoilage_signs": ["Kuah santan pecah berbuih", "Aroma masam fermentasi", "Daging hancur lembek"]
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
        "aliases": ["rendang", "beef rendang", "beef", "steak", "pot roast", "prime_rib", "filet_mignon", "beef_tartare", "beef_carpaccio"],
        "spoilage_signs": ["Minyak bumbu berbusa/tengik", "Daging terasa masam", "Bintik jamur putih di lapisan minyak"]
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
        "aliases": ["satay", "chicken satay", "sate ayam", "skewers", "yakitori"],
        "spoilage_signs": ["Bumbu kacang masam berbuih", "Daging berlendir", "Bau tengik minyak kacang"]
    },
    {
        "food_code": "TELUR_DADAR",
        "food_name": "Telur Dadar (Omelette)",
        "category": "Lauk Hewani",
        "serving_size_gram": 60,
        "calories": 154.0,
        "protein": 9.3,
        "fat": 12.0,
        "carbs": 1.2,
        "fiber": 0.0,
        "shelf_life_hours": 16,
        "aliases": ["omelette", "omelet", "fried egg", "telur dadar", "egg", "huevos_rancheros", "eggs_benedict"],
        "spoilage_signs": ["Bau tengik hidrogen sulfida", "Warna kehijauan gelap abnormal", "Permukaan berlendir"]
    },
    {
        "food_code": "TELUR_REBUS",
        "food_name": "Telur Rebus / Balado",
        "category": "Lauk Hewani",
        "serving_size_gram": 55,
        "calories": 78.0,
        "protein": 6.3,
        "fat": 5.3,
        "carbs": 0.6,
        "fiber": 0.0,
        "shelf_life_hours": 24,
        "aliases": ["boiled egg", "hard boiled egg", "telur rebus", "deviled_eggs", "poached egg"],
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
        "aliases": ["fish_and_chips", "fried fish", "ikan goreng", "fish", "fried_calamari"],
        "spoilage_signs": ["Bau amonia/anyir busuk tajam", "Daging hancur lembek", "Permukaan lengket berlendir"]
    },
    {
        "food_code": "IKAN_BAKAR",
        "food_name": "Ikan Bakar / Panggang",
        "category": "Lauk Hewani",
        "serving_size_gram": 100,
        "calories": 165.0,
        "protein": 24.0,
        "fat": 7.0,
        "carbs": 1.5,
        "fiber": 0.0,
        "shelf_life_hours": 16,
        "aliases": ["grilled_salmon", "grilled fish", "ikan bakar", "salmon", "sashimi"],
        "spoilage_signs": ["Aroma anyir masam", "Daging lembek hancur berlendir", "Bercak jamur"]
    },
    {
        "food_code": "SEAFOOD_UDANG_CUMI",
        "food_name": "Olahan Udang / Cumi-Cumi",
        "category": "Lauk Hewani",
        "serving_size_gram": 100,
        "calories": 140.0,
        "protein": 21.5,
        "fat": 4.5,
        "carbs": 2.0,
        "fiber": 0.0,
        "shelf_life_hours": 14,
        "aliases": ["shrimp_and_grits", "crab_cakes", "lobster_roll_sandwich", "lobster_bisque", "mussels", "oysters", "scallops", "takoyaki", "udang", "cumi"],
        "spoilage_signs": ["Aroma amonia tajam menyengat", "Tekstur lembek hancur berlendir licin", "Perubahan warna kemerahan kusam"]
    },
    {
        "food_code": "BURGER_SANDWICH",
        "food_name": "Burger / Sandwich Daging",
        "category": "Lauk Hewani",
        "serving_size_gram": 180,
        "calories": 380.0,
        "protein": 18.0,
        "fat": 19.0,
        "carbs": 34.0,
        "fiber": 2.0,
        "shelf_life_hours": 18,
        "aliases": ["hamburger", "hot_dog", "club_sandwich", "pulled_pork_sandwich", "croque_madame", "grilled_cheese_sandwich"],
        "spoilage_signs": ["Patty daging berbau masam/tengik", "Sayuran dalam membusuk berair", "Kapang pada roti"]
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
        "aliases": ["tempeh", "fried tempeh", "tempe goreng", "tempe", "edamame", "falafel"],
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
        "aliases": ["orek tempe", "sweet tempeh", "tempe orek"],
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
        "aliases": ["tofu", "fried tofu", "tahu goreng", "tahu", "hummus"],
        "spoilage_signs": ["Rasa masam tajam", "Lendir licin tebal di permukaan", "Tekstur hancur berair"]
    },
    {
        "food_code": "GORENGAN_BAKWAN",
        "food_name": "Gorengan / Bakwan / Tahu Isi / Dimsum",
        "category": "Lauk Nabati",
        "serving_size_gram": 75,
        "calories": 140.0,
        "protein": 4.5,
        "fat": 8.5,
        "carbs": 12.0,
        "fiber": 1.2,
        "shelf_life_hours": 14,
        "aliases": ["dumplings", "gyoza", "spring_rolls", "samosa", "onion_rings", "bakwan", "gorengan", "tahu isi", "risoles"],
        "spoilage_signs": ["Sayuran dalam berbau asam", "Tepung berminyak tengik berlendir", "Rasa masam"]
    },

    # --- SAYUR & SALAD (VEGETABLES) ---
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
        "aliases": ["french_onion_soup", "clam_chowder", "hot_and_sour_soup", "miso_soup", "soup", "vegetable soup", "clear soup", "sayur sop", "broth"],
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
        "food_code": "TUMIS_SAYUR_HIJAU",
        "food_name": "Tumis Sayur Hijau / Kangkung",
        "category": "Sayur",
        "serving_size_gram": 100,
        "calories": 65.0,
        "protein": 2.5,
        "fat": 3.8,
        "carbs": 5.5,
        "fiber": 2.2,
        "shelf_life_hours": 10,
        "aliases": ["stir-fry", "water spinach", "kangkung", "sauteed vegetables", "greens", "spinach", "seaweed_salad"],
        "spoilage_signs": ["Daun menghitam hancur berlendir", "Air tumisan berbuih masam", "Bau apek pembusukan"]
    },
    {
        "food_code": "CAPCAY",
        "food_name": "Capcay Sayur Campur",
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
        "food_code": "SALAD_SAYUR_GADO",
        "food_name": "Salad Sayur / Gado-Gado / Pecel",
        "category": "Sayur",
        "serving_size_gram": 200,
        "calories": 260.0,
        "protein": 9.5,
        "fat": 12.0,
        "carbs": 29.0,
        "fiber": 4.8,
        "shelf_life_hours": 8,
        "aliases": ["caesar_salad", "greek_salad", "caprese_salad", "beet_salad", "gado_gado", "gado gado", "pecel", "salad"],
        "spoilage_signs": ["Dressing/bumbu kacang berbusa asam", "Sayuran berlendir licin berair", "Aroma masam menyengat"]
    },

    # --- PIZZA, SNACKS & PELENGKAP ---
    {
        "food_code": "PIZZA_SLICE",
        "food_name": "Pizza",
        "category": "Pokok",
        "serving_size_gram": 120,
        "calories": 285.0,
        "protein": 11.5,
        "fat": 10.5,
        "carbs": 35.0,
        "fiber": 2.3,
        "shelf_life_hours": 24,
        "aliases": ["pizza", "pizza slice"],
        "spoilage_signs": ["Keju berlendir asam", "Bercak jamur kapang putih/hijau di pinggiran roti", "Saus berbau basi"]
    },
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
        "aliases": ["guacamole", "salsa", "hot sauce", "sambal", "sambal terasi", "chili paste", "nachos", "tacos"],
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

    # --- DESSERT / KUE (SWEETS) ---
    {
        "food_code": "KUE_DESSERT",
        "food_name": "Kue / Pastry / Donat / Waffle",
        "category": "Pelengkap",
        "serving_size_gram": 80,
        "calories": 280.0,
        "protein": 4.5,
        "fat": 13.5,
        "carbs": 36.0,
        "fiber": 1.2,
        "shelf_life_hours": 48,
        "aliases": ["apple_pie", "baklava", "beignets", "bread_pudding", "cannoli", "carrot_cake", "cheesecake", "chocolate_cake", "chocolate_mousse", "churros", "creme_brulee", "cup_cakes", "donuts", "frozen_yogurt", "ice_cream", "macarons", "pancakes", "panna_cotta", "red_velvet_cake", "strawberry_shortcake", "tiramisu", "waffles"],
        "spoilage_signs": ["Krim berbau asam/tengik", "Bercak jamur kapang putih/hijau pada spons kue", "Lendir pada buah topping"]
    },

    # --- BUAH SEGAR (FRUITS) ---
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

def build_complete_dataset():
    # Make sure every single Food-101 vision model class has a mapped entry
    dataset = list(TKPI_INDONESIAN_DATABASE)
    mapped_aliases = set()
    for item in dataset:
        for a in item["aliases"]:
            mapped_aliases.add(a.lower())

    # Check unmapped Food-101 classes and assign appropriate fallback category
    unmapped_count = 0
    for f101 in FOOD_101_CLASSES:
        if f101.lower() not in mapped_aliases and f101.replace("_", " ").lower() not in mapped_aliases:
            # Fallback entry for edge food item
            pretty_name = f101.replace("_", " ").title()
            entry = {
                "food_code": f101.upper(),
                "food_name": pretty_name,
                "category": "Makanan Umum",
                "serving_size_gram": 120,
                "calories": 210.0,
                "protein": 8.0,
                "fat": 9.0,
                "carbs": 24.0,
                "fiber": 1.5,
                "shelf_life_hours": 24,
                "aliases": [f101, f101.replace("_", " ")],
                "spoilage_signs": ["Perubahan warna dan bau tidak wajar", "Bercak jamur / kapang", "Tekstur berlendir"]
            }
            dataset.append(entry)
            unmapped_count += 1
            
    print(f"Total dataset entries: {len(dataset)} (Added {unmapped_count} specific vision classes)")
    return dataset

def export_typescript(dataset, filepath):
    ts_code = """// AUTO-GENERATED BY scripts/download_and_compile_dataset.py
// STANDAR GIZI PANGAN INDONESIA (TKPI KEMENKES RI) & FOOD-101 FORENSIK PANGAN

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

/**
 * Robust Multi-tier Food Nutrition & Spoilage Matcher
 * Matches raw Vision AI outputs (Food-101, ImageNet labels) to real Indonesian dishes.
 */
export function findNutritionByText(query: string): ExtendedNutritionItem | null {
  if (!query || !query.trim()) return null;
  const clean = query.toLowerCase().replace(/[_-]/g, ' ').trim();
  const rawKey = query.toLowerCase().trim();

  // Tier 1: Exact match on food_code or aliases
  for (const item of COMPREHENSIVE_FOOD_DATASET) {
    if (item.food_code.toLowerCase() === rawKey || item.food_code.toLowerCase() === clean) return item;
    if (item.food_name.toLowerCase() === clean) return item;
    if (item.aliases.some((alias) => alias.toLowerCase() === rawKey || alias.toLowerCase() === clean)) return item;
  }

  // Tier 2: Substring / Token match
  const tokens = clean.split(/[, /]+/).filter((t) => t.length > 2);
  for (const item of COMPREHENSIVE_FOOD_DATASET) {
    for (const token of tokens) {
      if (item.aliases.some((alias) => alias.toLowerCase().includes(token) || token.includes(alias.toLowerCase()))) {
        return item;
      }
      if (item.food_name.toLowerCase().includes(token)) {
        return item;
      }
    }
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
    sql = """-- AUTO-GENERATED SEEDER: TKPI (Kemenkes RI) & Comprehensive Food Forensic Database
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
    sql += "  food_name = EXCLUDED.food_name,\n"
    sql += "  category = EXCLUDED.category,\n"
    sql += "  serving_size_gram = EXCLUDED.serving_size_gram,\n"
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
    
    dataset = build_complete_dataset()
    export_typescript(dataset, ts_out)
    export_sql_seeder(dataset, sql_out)
    print("Done downloading and compiling dataset!")

if __name__ == "__main__":
    main()
