# AI Agent Execution Prompt & Master Instruction
## Project: NutriSafe AI (Food Forensic & Nutrition Scanner)

Salin seluruh prompt di bawah ini dan berikan langsung ke AI Agent / Coding Assistant (seperti Cursor, GitHub Copilot Workspace, Windsurf, Claude Code, Aider, atau Dev Agent lainnya) untuk mengeksekusi implementasi full-stack Next.js dari dokumen PRD, Design System, dan desain Stitch.

---

```markdown
# MISSION & SYSTEM ROLE
You are an expert Senior Full-Stack Next.js Engineer and Applied Machine Learning Specialist.
Your mission is to build, execute, and deliver a production-ready, zero-cost, guest-first web application named **"NutriSafe AI"** based strictly on the provided Product Requirements Document (`NutriSafe_PRD_and_MVP_Spec.md`), Design System Guide (`NutriSafe_Design_System.md`), and the Stitch UI layout specifications.

---

## 1. CORE ARCHITECTURAL CONSTRAINTS (NON-NEGOTIABLE)
1. **Zero-Cost & Free-Tier Only:** DO NOT integrate or call any paid cloud vision APIs (No OpenAI, Anthropic, or paid Google Vision API).
2. **Client-Side AI Inference:** Run image classification and forensic analysis directly inside the user's browser using `@huggingface/transformers` (WebAssembly / WebGPU fallback) or `onnxruntime-web`.
3. **No Authentication / Guest-First:** The app must be 100% accessible to the public without login, registration, or paywalls.
4. **Backend & Database:** Supabase Free Tier for the master nutrition database (`nutrition_master`), configured with Row Level Security (RLS) public read-only (`anon` role).
5. **Design & Color Fidelity:** Strictly adhere to the Clean Clinical White & Emerald Green palette defined in `NutriSafe_Design_System.md`.

---

## 2. REQUIRED TECH STACK
* **Framework:** Next.js 14+ (App Router, React 18/19, TypeScript)
* **Styling:** Tailwind CSS (configured with design tokens from `NutriSafe_Design_System.md`)
* **Icons:** `lucide-react`
* **In-Browser ML Runtime:** `@huggingface/transformers`
* **Database Client:** `@supabase/supabase-js`
* **Deployment Target:** Vercel (Hobby Tier - Rp 0)

---

## 3. STEP-BY-STEP EXECUTION PLAN

### STEP 1: Dependencies & Tailwind Setup
1. Verify and install dependencies:
   `npm install @huggingface/transformers @supabase/supabase-js lucide-react clsx tailwind-merge`
2. Update `tailwind.config.ts` with the custom colors (`brand`, `surface`, `forensic`), border radiuses, and shadow tokens from `NutriSafe_Design_System.md`.
3. Configure `next.config.js` to support WebAssembly / top-level await if required by Transformers.js:
   ```javascript
   /** @type {import('next').NextConfig} */
   const nextConfig = {
     webpack: (config) => {
       config.experiments = { ...config.experiments, topLevelAwait: true, asyncWebAssembly: true };
       return config;
     },
   };
   module.exports = nextConfig;
   ```

### STEP 2: Supabase Schema & Client Initialization
1. Create `lib/supabase.ts` with public anonymous client creation using `NEXT_PUBLIC_SUPABASE_URL` and `NEXT_PUBLIC_SUPABASE_ANON_KEY`.
2. Ensure database types and mock fallback data match the TKPI standards outlined in the PRD.
3. Provide an offline fallback in `lib/nutritionFallback.ts` containing Indonesian food data (Nasi Putih, Ayam Goreng, Tempe, Tahu, Sayur Sop, Sayur Lodeh, Telur Dadar, Sambal) so the application functions seamlessly even without a Supabase connection.

### STEP 3: Client-Side Machine Learning Pipeline (`lib/detector.ts`)
1. Implement browser caching and runtime setup for `@huggingface/transformers`.
2. Load an open-source vision classification model (e.g., `Xenova/food-classification-resnet-50` or ViT).
3. Implement the `analyzeFoodImage(canvas: HTMLCanvasElement)` function:
   * Process the captured canvas frame.
   * Classify detected food items.
   * Apply visual forensic rules: detect discoloration, suspicious texture/mold, and assign safety statuses:
     - `safe` (🟢 Fresh / Normal)
     - `warning` (🟡 Caution / High Oil / Near Spoilage)
     - `danger` (🔴 Hazard / Spoiled / Mold / Fermenting)
   * Map items to nutritional values (Calories, Protein, Fat, Carbs, Fiber).

### STEP 4: Component Implementation (Stitch UI Layout)
1. **`components/Navbar.tsx`**: Header with logo, health shield icon, and "100% Free • No Login • In-Browser AI" pill badge.
2. **`components/CameraScanner.tsx`**:
   * HTML5 WebCam stream with back-camera priority (`facingMode: 'environment'`).
   * Fallback button for file upload (`<input type="file" accept="image/*" />`).
   * Live viewfinder with modern HUD scanning target overlay lines.
   * Prominent Emerald Green CTA button: "Foto & Analisis Forensik".
   * Loading overlay with spinner while processing inference.
3. **`components/FoodForensicTable.tsx`**:
   * High-contrast, responsive table.
   * Columns: Komponen Lauk, Status Forensik (Pill Badge), Observasi Visual, Kalori (kkal), Protein, Lemak, Karbo, Rekomendasi Aksi.
   * Distinct color-coded badges matching the semantic design system.
   * Table summary footer calculating total meal calories and macronutrients.
4. **`components/SafetyTipsGrid.tsx`**: 3-card informational grid at the bottom highlighting quick visual spoilage signs (Discoloration, Slime/Mold, Acidic Odor/Froth).

### STEP 5: Page Assembly (`app/page.tsx`)
* Assemble all components into `app/page.tsx` with responsive layout (`max-w-5xl mx-auto`).
* Connect state management: Camera Capture $ightarrow$ ML Inference $ightarrow$ State Update $ightarrow$ Table Render.
* Add clean transitions and zero-layout-shift placeholders.

---

## 4. QUALITY ASSURANCE & VERIFICATION
* Run `npm run build` and ensure TypeScript compilation succeeds without errors.
* Verify responsive UI across mobile screen widths (360px - 430px) and desktop (>1024px).
* Ensure all elements adhere strictly to the Emerald Green & Crisp White color scheme.
```
