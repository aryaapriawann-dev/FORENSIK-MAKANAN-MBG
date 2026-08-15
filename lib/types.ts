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
  recommendation: string;
}

export interface NutritionTotal {
  totalCalories: number;
  totalProtein: number;
  totalFat: number;
  totalCarbs: number;
  totalFiber: number;
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
  shelf_life_hours: number;
  spoilage_signs: string[];
}
