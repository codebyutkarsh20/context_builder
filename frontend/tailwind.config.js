/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: [
    './index.html',
    './src/**/*.{ts,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        zinc: {
          850: '#1f1f23',
        },
      },
    },
  },
  plugins: [],
}
