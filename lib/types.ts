export interface BoundingBox {
  x: number; // 0 to 1 normalized
  y: number; // 0 to 1 normalized
  width: number; // 0 to 1 normalized
  height: number; // 0 to 1 normalized
}

export interface FoodItemAnalysis {
  id: string;
  name: string;
  category: string;
  confidence: number;
  safetyStatus: 'safe' | 'warning' | 'danger';
  forensicFlag: string;
  calories: number;
  protein: number;
  fat: number;
  carbs: number;
  fiber: number;
  vitaminA_mcg?: number;
  vitaminB_mg?: number;
  vitaminC_mg?: number;
  vitaminD_mcg?: number;
  calcium_mg?: number;
  iron_mg?: number;
  recommendation: string;
  box?: BoundingBox;
}

export interface NutritionTotal {
  totalCalories: number;
  totalProtein: number;
  totalFat: number;
  totalCarbs: number;
  totalFiber: number;
  totalVitaminA: number;
  totalVitaminB: number;
  totalVitaminC: number;
  totalVitaminD: number;
  totalCalcium: number;
  totalIron: number;
  overallSafety: 'safe' | 'warning' | 'danger';
}

export interface NutritionMasterItem {
  id?: number;
  food_code: string;
  food_name: string;
  category: string;
  serving_size_gram: number;
  calories: number;
  protein: number;
  fat: number;
  carbs: number;
  fiber: number;
  vitaminA_mcg?: number;
  vitaminB_mg?: number;
  vitaminC_mg?: number;
  vitaminD_mcg?: number;
  calcium_mg?: number;
  iron_mg?: number;
  shelf_life_hours: number;
  spoilage_signs: string[];
}

