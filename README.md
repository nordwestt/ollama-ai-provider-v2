# Ollama Provider V2 for the Vercel AI SDK

The **[Ollama Provider V2](https://github.com/nordwestt/ollama-ai-provider-v2)** for the [AI SDK](https://sdk.vercel.ai/docs) has been created as the original [ollama-ai-provider](https://www.npmjs.com/package/ollama-ai-provider) was not being actively maintained. 

This provider now supports: 
- tool streaming and calling for models
- enable/disable thinking

## Setup

The Ollama provider is available in the `ollama-ai-provider-v2` module. You can install it with

```bash
npm i ollama-ai-provider-v2
```

## Provider Instance

You can import the default provider instance `ollama` from `ollama-ai-provider-v2`:

```ts
import { ollama } from 'ollama-ai-provider-v2';
```

## Example

```ts
import { ollama } from 'ollama-ai-provider-v2';
import { generateText } from 'ai';

const { text } = await generateText({
  model: ollama('llama3.2:latest'),
  prompt: 'Write a meaty lasagna recipe for 4 people.',
});
```

## Thinking mode toggle example

```ts
import { ollama } from 'ollama-ai-provider-v2';
import { generateText } from 'ai';

const { text } = await generateText({
  model: ollama('qwen3:4b'),
  providerOptions: { ollama: { think: true } },
  prompt: 'Write a meaty lasagna recipe for 4 people, but really think about it',
});
```


## Documentation

Please check out the **[Ollama provider documentation](https://github.com/nordwestt/ollama-ai-provider-v2)** for more information.
