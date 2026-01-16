/**
 * Mock API service for demo-only deployment
 * Serves pre-computed predictions from static JSON files
 */

const MOCK_PAGES = [79, 96];

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
  const exampleId = pageNum === 79 ? 'example_01' : pageNum === 96 ? 'example_02' : null;
  if (!exampleId) return null;

  return await loadMockData(exampleId, 'prediction.json');
}

export async function getMockMeta(pageNum: number): Promise<MockMeta | null> {
  const exampleId = pageNum === 79 ? 'example_01' : pageNum === 96 ? 'example_02' : null;
  if (!exampleId) return null;

  return await loadMockData(exampleId, 'meta.json');
}

export async function getMockDemoBundle() {
  return {
    default_page: 79,
    default_page_id: 79,
    available_pages: MOCK_PAGES,
    featured_pages: [
      { page_num: 79, title: 'Unfermented Grape Juice', difficulty: 'easy' },
      { page_num: 96, title: 'Coffee Cakes (Brioche)', difficulty: 'medium' },
    ],
  };
}

export function isMockMode(): boolean {
  return !process.env.NEXT_PUBLIC_API_BASE_URL || process.env.NEXT_PUBLIC_USE_MOCK === 'true';
}

export function getAvailableMockPages(): number[] {
  return MOCK_PAGES;
}
