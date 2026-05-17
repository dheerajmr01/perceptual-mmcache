/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Cream theme — warm off-white background, deep stone ink.
        cream: {
          50: "#fbf7ef",
          100: "#f6efe0",
          200: "#ebe1cb",
          300: "#dccfb0",
          400: "#c8b78f",
          500: "#a89770",
        },
        ink: {
          50: "#f5f4f2",
          100: "#e7e5e1",
          200: "#cfcbc4",
          300: "#a8a39a",
          400: "#78736b",
          500: "#5a554d",
          600: "#3f3a33",
          700: "#2d2925",
          800: "#1f1d19",
          900: "#14110e",
        },
        // Accent palette — desaturated, matte, readable on cream.
        baseline: "#78736b", // warm gray
        perceptual: "#3f7d57", // muted sage / forest
        highlight: "#a06425", // rust / amber-earth
        ocean: "#385c7e", // muted navy for tertiary accent
      },
      fontFamily: {
        sans: [
          "Inter",
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "Segoe UI",
          "Roboto",
          "sans-serif",
        ],
        mono: [
          "JetBrains Mono",
          "Fira Code",
          "ui-monospace",
          "SFMono-Regular",
          "Menlo",
          "monospace",
        ],
      },
      boxShadow: {
        // Matte = no glow, just paper-weight lift.
        matte: "0 1px 0 0 rgba(31, 29, 25, 0.04), 0 1px 2px 0 rgba(31, 29, 25, 0.04)",
        "matte-md":
          "0 1px 0 0 rgba(31, 29, 25, 0.05), 0 4px 12px -2px rgba(31, 29, 25, 0.06)",
      },
    },
  },
  plugins: [],
};
