/**
 * Mock API service for demo-only deployment
 * Serves pre-computed predictions from static JSON files
 */

const MOCK_PAGES = [74, 79, 96, 102, 131, 144, 159, 181, 199, 254, 311, 320, 420, 439, 474];

// Map page numbers to example folder names
const PAGE_TO_EXAMPLE: Record<number, string> = {
  74: 'example_01',
  79: 'example_02',
  96: 'example_03',
  102: 'example_04',
  131: 'example_05',
  144: 'example_06',
  159: 'example_07',
  181: 'example_08',
  199: 'example_09',
  254: 'example_10',
  311: 'example_11',
  320: 'example_12',
  420: 'example_13',
  439: 'example_14',
  474: 'example_15',
};

interface MockPrediction {
  page_num: number;
  image_path: string;
  image_url: string;
  tokens: Array<{
    text: string;
    bbox: number[];
    pred_label: string;
    pred_id: number;
    confidence: number;
  }>;
  groups?: any;
}

interface MockMeta {
  title: string;
  difficulty: string;
  description: string;
  tags: string[];
  page_num: number;
  expectedTitle: string;
  expectedKeywords: string[];
}

async function loadMockData(exampleId: string, file: string) {
  try {
    const response = await fetch(`/demo_examples/${exampleId}/${file}`);
    if (!response.ok) return null;
    return await response.json();
  } catch (error) {
    console.error(`Failed to load ${file} for ${exampleId}:`, error);
    return null;
  }
}

export async function getMockPrediction(pageNum: number): Promise<MockPrediction | null> {
  const exampleId = PAGE_TO_EXAMPLE[pageNum];
  if (!exampleId) return null;

  return await loadMockData(exampleId, 'prediction.json');
}

export async function getMockMeta(pageNum: number): Promise<MockMeta | null> {
  const exampleId = PAGE_TO_EXAMPLE[pageNum];
  if (!exampleId) return null;

  return await loadMockData(exampleId, 'meta.json');
}

export async function getMockDemoBundle() {
  // Load meta for each page to get titles
  const featured_pages = await Promise.all(
    MOCK_PAGES.map(async (page) => {
      const meta = await getMockMeta(page);
      return {
        page_id: page,
        png_id: String(page).padStart(4, '0'),
        page_num: page,
        title: meta?.title || `Page ${page}`,
        recipe_confidence: 0.85 + Math.random() * 0.1,
        is_recipe_page: true
      };
    })
  );

  return {
    health: { status: 'ok', mode: 'mock' },
    default_page: 79,
    default_page_id: 79,
    featured_pages,
    status: 'ok',
    message: 'Mock demo data',
    pages_total: MOCK_PAGES.length,
    pages_with_images: MOCK_PAGES.length,
    featured_mode: 'curated_recipe',
  };
}

export function isMockMode(): boolean {
  return !process.env.NEXT_PUBLIC_API_BASE_URL || process.env.NEXT_PUBLIC_USE_MOCK === 'true';
}

export function getAvailableMockPages(): number[] {
  return MOCK_PAGES;
}
