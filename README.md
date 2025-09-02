# Ollama Provider V2 for the Vercel AI SDK (now for SDK 5)

The **[Ollama Provider V2](https://github.com/nordwestt/ollama-ai-provider-v2)** for the [AI SDK](https://sdk.vercel.ai/docs) has been created as the original [ollama-ai-provider](https://www.npmjs.com/package/ollama-ai-provider) was not being actively maintained. 

This provider now supports: 
- tool streaming and calling for models
- enable/disable thinking

## Setup

The Ollama provider is available in the `ollama-ai-provider-v2` module. You can install it with

```bash
npm i ollama-ai-provider-v2
```

To update an existing installation to the new major version that supports AI SDK 5, simply do
```bash
npm update ollama-ai-provider-v2
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

## Example using a fixed seed - where you always get the same response for the same question

```ts
import { ollama } from 'ollama-ai-provider-v2';
import { generateText } from 'ai';

const { text } = await generateText({
  model: ollama('llama3.2'),
  providerOptions: { ollama: { options: {seed: 123} } },
  prompt: 'Write a meaty lasagna recipe for 4 people',
});
```


## Documentation

Please check out the **[Ollama provider documentation](https://github.com/nordwestt/ollama-ai-provider-v2)** for more information.

## How to contribute

### Step 1. Fork the repository
### Step 2. Make the changes to the code
### Step 3. Build and test for yourself

Run following commands to install dependencies and build the project:
```
npm run install
npm run build
```

This produces the built files in the `dist` folder. 
Simply copy these files over to your project at `yourproject/node_modules/ollama-ai-provider-v2/dist`.
Now you can test the changes in your project and verify that they work as expected. 

### Step 4. Open a PR to this project with your forked branch
