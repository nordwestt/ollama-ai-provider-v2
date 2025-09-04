# Ollama Provider V2 for Vercel AI SDK

Use Ollama with the Vercel AI SDK, implementing the official Ollama API. This provider has minimal dependencies and is web-compatible out of the box.

[![npm version](https://badge.fury.io/js/ollama-ai-provider-v2.svg)](https://badge.fury.io/js/ollama-ai-provider-v2)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.8+-blue.svg)](https://www.typescriptlang.org/)
[![Node.js](https://img.shields.io/badge/Node.js-18+-green.svg)](https://nodejs.org/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-yellow.svg)](https://opensource.org/licenses/Apache-2.0)

## Why Choose Ollama Provider V2?

- ✅ **Minimal Dependencies** - Lean codebase with just 2 core dependencies
- ✅ **Universal Compatibility** - Works seamlessly in Node.js, browsers, and edge environments
- ✅ **Direct Ollama Integration** - Uses official Ollama API endpoints for maximum compatibility
- ✅ **Advanced Features** - Tool calling, streaming, thinking mode, embeddings, and completion models
- ✅ **Type Safety** - Full TypeScript support with comprehensive type definitions
- ✅ **Zero Configuration** - Works out-of-the-box with sensible defaults
- ✅ **Actively Maintained** - Regular updates and AI SDK v5+ compatibility

## Quick Start

```bash
npm install ollama-ai-provider-v2 ai
```

```typescript
import { ollama } from 'ollama-ai-provider-v2';
import { generateText } from 'ai';

// Works everywhere - Node.js, browsers, edge environments
const { text } = await generateText({
  model: ollama('llama3.2'),
  prompt: 'Write a haiku about coding',
});

console.log(text);
```

## Core Features

### Streaming Text Generation

```typescript
import { streamText } from 'ai';

const { textStream } = await streamText({
  model: ollama('llama3.2'),
  prompt: 'Tell me a story about artificial intelligence',
});

for await (const chunk of textStream) {
  process.stdout.write(chunk);
}
```

### Tool Calling Support

```typescript
import { generateText, tool } from 'ai';
import { z } from 'zod';

const { text, toolCalls } = await generateText({
  model: ollama('llama3.2'),
  prompt: 'What is the weather like in San Francisco?',
  tools: {
    getWeather: tool({
      description: 'Get current weather for a location',
      inputSchema: z.object({
        location: z.string().describe('City name'),
        unit: z.enum(['celsius', 'fahrenheit']).optional(),
      }),
      execute: async ({ location, unit = 'celsius' }) => {
        return { temp: 18, unit, condition: 'sunny' };
      },
    }),
  },
});
```

### Reasoning Mode (Thinking)

Unique feature for models that support chain-of-thought reasoning:

```typescript
const { text } = await generateText({
  model: ollama('deepseek-r1:7b'),
  providerOptions: { ollama: { think: true } },
  prompt: 'Solve this complex math problem step by step: 2x + 5 = 17',
});
```

### Advanced Ollama Options

Access Ollama's native parameters while maintaining AI SDK compatibility:

```typescript
const { text } = await generateText({
  model: ollama('llama3.2'),
  providerOptions: {
    ollama: {
      options: {
        seed: 123,           // Deterministic outputs
        num_ctx: 8192,       // Context window size
        repeat_penalty: 1.1, // Control repetition
        top_k: 40,          // Advanced sampling
        min_p: 0.1,         // Minimum probability
      },
    },
  },
  prompt: 'Write a detailed analysis of Alice in Wonderland',
  temperature: 0.8, // Standard AI SDK parameters work too
});
```

### Embeddings

```typescript
import { embed } from 'ai';

// Single embedding
const { embedding } = await embed({
  model: ollama.embedding('nomic-embed-text'),
  value: 'Hello world',
});

console.log('Embedding dimensions:', embedding.length);

// Batch embeddings
const texts = ['Hello world', 'How are you?', 'AI is amazing'];
const results = await Promise.all(
  texts.map((text) =>
    embed({
      model: ollama.embedding('all-minilm'),
      value: text,
    })
  )
);
```

### Completion Models

For use cases requiring completion-style generation:

```typescript
const { text } = await generateText({
  model: ollama.completion('codellama:code'),
  prompt: 'def fibonacci(n):\n    if n <= 1:\n        return n\n    else:\n        return fibonacci(n-1) + fibonacci(n-2)\n\n# Optimize this function:\n',
});
```

## Custom Ollama Instance

Connect to remote Ollama servers or custom configurations:

```typescript
import { createOllama } from 'ollama-ai-provider-v2';

const customOllama = createOllama({
  baseURL: 'https://my-ollama-server.com/api',
  headers: {
    'Authorization': 'Bearer your-token',
  },
});

const { text } = await generateText({
  model: customOllama('llama3.2'),
  prompt: 'Hello from remote server!',
});
```

## Supported Models

Works with any model in your Ollama installation, including:

- **Chat Models**: `llama3.2`, `mistral`, `phi4-mini`, `qwen2.5`, `codellama`, `gemma3`
- **Vision Models**: `llava`, `llama3.2-vision`, `minicpm-v`
- **Reasoning Models**: `deepseek-r1:7b`, `deepseek-r1:1.5b`, `deepseek-r1:8b`
- **Code Models**: `codellama:code`, `codellama:python`, `deepseek-coder-v2`
- **Embedding Models**: `nomic-embed-text`, `all-minilm`, `mxbai-embed-large`

## Prerequisites

1. [Ollama](https://ollama.com) installed and running
2. AI SDK v5+ (`ai` package)
3. Node.js 18+ for development

```bash
# Start Ollama
ollama serve

# Pull a model
ollama pull llama3.2
```

## Contributing

Contributions are welcome! Here's how to get started:

1. **Fork the repository**
2. **Install dependencies**: `pnpm install`
3. **Build the project**: `pnpm build`
4. **Run tests**: `pnpm test`
5. **Make your changes**
6. **Test locally**: Copy `dist/*` to your project's `node_modules/ollama-ai-provider-v2/dist`
7. **Submit a pull request**

## License

Apache-2.0 © [nordwestt](https://github.com/nordwestt/ollama-ai-provider-v2)

See [LICENSE.md](./LICENSE.md) for details.