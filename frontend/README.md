# S2S Frontend — React + Vite + Tailwind

## Prerequisites

- [Node.js](https://nodejs.org/) >= 18 (includes npm)

## Quick Start

```bash
cd frontend
npm install
npm run dev
```

Opens at `http://localhost:5173`. The Vite dev server proxies `/ws` to the backend at `localhost:8000`.

## Build for Production

```bash
npm run build
npm run preview
```

## Stack

- **React 18** — UI framework
- **Vite 5** — bundler & dev server
- **Tailwind CSS 3** — utility-first styling
- **Framer Motion 11** — animations (Reactive Orb)
- **Lucide React** — icons

## UI Components

- **ReactiveOrb** — central animated orb with 4 states (idle, listening, thinking, speaking)
- **TranscriptWindow** — translucent chat showing last 3 exchanges
- **Controls** — mic toggle, connect button, clear conversation
