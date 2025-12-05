import { describe, it, expect } from 'vitest';
import { validateThinkParameter } from './ollama-chat-settings';

describe('validateThinkParameter', () => {
  describe('GPT-OSS models', () => {
    it('should accept valid string levels', () => {
      expect(validateThinkParameter('gpt-oss:7b', 'low')).toBe('low');
      expect(validateThinkParameter('gpt-oss:7b', 'medium')).toBe('medium');
      expect(validateThinkParameter('gpt-oss:7b', 'high')).toBe('high');
    });

    it('should convert boolean true to medium level', () => {
      expect(validateThinkParameter('gpt-oss:7b', true)).toBe('medium');
    });

    it('should convert boolean false to undefined', () => {
      expect(validateThinkParameter('gpt-oss:7b', false)).toBeUndefined();
    });

    it('should handle undefined input', () => {
      expect(validateThinkParameter('gpt-oss:7b', undefined)).toBeUndefined();
    });

    it('should be case insensitive for model detection', () => {
      expect(validateThinkParameter('GPT-OSS:7b', 'high')).toBe('high');
      expect(validateThinkParameter('Gpt-Oss:7b', 'medium')).toBe('medium');
    });
  });

  describe('Regular models', () => {
    const regularModels = ['llama3.2', 'deepseek-r1:7b', 'qwen3:8b', 'mistral:7b'];

    regularModels.forEach((modelId) => {
      describe(`${modelId}`, () => {
        it('should accept boolean values', () => {
          expect(validateThinkParameter(modelId, true)).toBe(true);
          expect(validateThinkParameter(modelId, false)).toBe(false);
        });

        it('should convert string levels to true', () => {
          expect(validateThinkParameter(modelId, 'low')).toBe(true);
          expect(validateThinkParameter(modelId, 'medium')).toBe(true);
          expect(validateThinkParameter(modelId, 'high')).toBe(true);
        });

        it('should handle undefined input', () => {
          expect(validateThinkParameter(modelId, undefined)).toBeUndefined();
        });
      });
    });
  });
});