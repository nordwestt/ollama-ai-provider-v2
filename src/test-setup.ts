// Polyfill for MSW in Node.js environment

// Setup localStorage polyfill
Object.defineProperty(globalThis, 'localStorage', {
  value: {
    getItem: () => null,
    setItem: () => {},
    removeItem: () => {},
    clear: () => {},
    length: 0,
    key: () => null,
  },
  writable: true,
});

// Setup sessionStorage polyfill
Object.defineProperty(globalThis, 'sessionStorage', {
  value: {
    getItem: () => null,
    setItem: () => {},
    removeItem: () => {},
    clear: () => {},
    length: 0,
    key: () => null,
  },
  writable: true,
});