# AI SDK - Ollama Provider

The **[Ollama provider](https://sdk.vercel.ai/providers/ai-sdk-providers/ollama)** for the [AI SDK](https://sdk.vercel.ai/docs)
contains language model support for the Ollama chat and completion APIs and embedding model support for the Ollama embeddings API.

## Setup

The Ollama provider is available in the `@ai-sdk/ollama` module. You can install it with

```bash
npm i @ai-sdk/ollama
```

## Provider Instance

You can import the default provider instance `ollama` from `@ai-sdk/ollama`:

```ts
import { ollama } from '@ai-sdk/ollama';
```

## Example

```ts
import { ollama } from '@ai-sdk/ollama';
import { generateText } from 'ai';

const { text } = await generateText({
  model: ollama('gpt-4-turbo'),
  prompt: 'Write a vegetarian lasagna recipe for 4 people.',
});
```

## Documentation

Please check out the **[Ollama provider documentation](https://sdk.vercel.ai/providers/ai-sdk-providers/ollama)** for more information.
