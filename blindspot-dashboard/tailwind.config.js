/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        background: {
          DEFAULT: "#0b0c13",
          secondary: "#11131d",
          elevated: "#181a27",
          subtle: "#1f2233",
        },
        border: {
          DEFAULT: "#26293d",
          muted: "#1d2030",
          accent: "#3d4263",
        },
        accent: {
          DEFAULT: "#6366f1",
          hover: "#4f46e5",
          light: "#818cf8",
          subtle: "rgba(99, 102, 241, 0.12)",
        },
        brand: {
          blue: "#3b82f6",
          violet: "#8b5cf6",
          indigo: "#6366f1",
        },
        status: {
          active: "#10b981",
          idle: "#f59e0b",
          stopped: "#ef4444",
          tracking: "#06b6d4",
        }
      },
      fontFamily: {
        sans: [
          "-apple-system",
          "BlinkMacSystemFont",
          "Segoe UI",
          "Roboto",
          "Helvetica Neue",
          "Arial",
          "sans-serif",
        ],
        mono: [
          "JetBrains Mono",
          "SFMono-Regular",
          "Consolas",
          "Courier New",
          "monospace",
        ],
      },
    },
  },
  plugins: [],
}
