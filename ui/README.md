# Intentmind Demo UI

Small React/Vite demo for the FastAPI chat endpoint in the repository root.

## Development

```bash
npm install
npm run dev
```

The app expects the backend to expose `POST /api/chat`. In local development,
run the FastAPI app from the repository root and proxy or serve the UI behind
the same origin.

## Checks

```bash
npm run lint
npm run build
```
