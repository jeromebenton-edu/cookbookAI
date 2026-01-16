# CookbookAI Frontend

A Next.js (App Router) frontend for the CookbookAI portfolio project. It ships with pre-parsed recipe JSON files and placeholder page images, plus an AI Parse View that visualizes LayoutLMv3 tokens and confidence.

## Local Development

```bash
npm install
npm run dev
```

Open `http://localhost:3000`.

## Environment Variables

- `NEXT_PUBLIC_API_BASE_URL` - FastAPI base URL (example: `http://localhost:8000`). When not set, the upload page uses mock data.

## Deploy to Vercel

- Import the `frontend/` directory as a new Vercel project.
- Set `NEXT_PUBLIC_API_BASE_URL` if the backend is deployed.
- Run `npm run build` to verify the build.

## Data Sources

Recipe JSON lives in `public/recipes/boston/*.json` with placeholder page scans in `public/recipes/boston/pages/`.
