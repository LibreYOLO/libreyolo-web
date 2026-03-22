import * as ort from 'onnxruntime-web';
import { configureRuntime } from '../utils/runtime';
import type { ExecutionProvider, RuntimeConfig, OnnxRunResult } from '../types';

export interface CoreEngineOptions {
  modelPath: string;
  providers?: ExecutionProvider[];
  runtime?: RuntimeConfig;
  /** Progress callback during model download (0-1) */
  onProgress?: (progress: number) => void;
}

export class CoreEngine {
  private session: ort.InferenceSession | null = null;
  private currentProvider: ExecutionProvider | null = null;

  constructor(private options: CoreEngineOptions) {
    configureRuntime(options.runtime);
  }

  async init(): Promise<void> {
    if (this.session) {
      await this.session.release();
      this.session = null;
      this.currentProvider = null;
    }

    // If onProgress is set and model is a URL, fetch with progress tracking
    let modelSource: string | Uint8Array = this.options.modelPath;
    if (this.options.onProgress && this.isUrl(this.options.modelPath)) {
      modelSource = await this.fetchWithProgress(
        this.options.modelPath,
        this.options.onProgress
      );
    }

    const providersToTry = this.options.providers ?? ['webgpu', 'wasm'];

    for (const provider of providersToTry) {
      try {
        console.log(`[LibreYOLO] Attempting ${provider} backend...`);
        await this.loadSession(provider, modelSource);
        this.currentProvider = provider;
        console.log(`[LibreYOLO] Successfully loaded with ${provider}`);
        return;
      } catch (error) {
        console.warn(`[LibreYOLO] ${provider} failed:`, error);
      }
    }

    throw new Error('[LibreYOLO] Failed to initialize with any backend');
  }

  private isUrl(path: string): boolean {
    return path.startsWith('http://') || path.startsWith('https://');
  }

  private async fetchWithProgress(
    url: string,
    onProgress: (progress: number) => void
  ): Promise<Uint8Array> {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`[LibreYOLO] Failed to download model: ${response.status} ${response.statusText}`);
    }

    const contentLength = Number(response.headers.get('content-length') || 0);
    if (!contentLength || !response.body) {
      // No content-length or no streaming — fall back to simple fetch
      onProgress(0);
      const buffer = await response.arrayBuffer();
      onProgress(1);
      return new Uint8Array(buffer);
    }

    const reader = response.body.getReader();
    const chunks: Uint8Array[] = [];
    let received = 0;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.length;
      onProgress(received / contentLength);
    }

    // Combine chunks into a single Uint8Array
    const buffer = new Uint8Array(received);
    let offset = 0;
    for (const chunk of chunks) {
      buffer.set(chunk, offset);
      offset += chunk.length;
    }

    return buffer;
  }

  private async loadSession(
    provider: ExecutionProvider,
    modelSource: string | Uint8Array
  ): Promise<void> {
    const sessionOptions: ort.InferenceSession.SessionOptions = {
      executionProviders: [provider],
      graphOptimizationLevel: 'all',
    };

    if (typeof modelSource === 'string') {
      this.session = await ort.InferenceSession.create(modelSource, sessionOptions);
    } else {
      this.session = await ort.InferenceSession.create(modelSource, sessionOptions);
    }
  }

  async run(inputs: Record<string, ort.Tensor>): Promise<OnnxRunResult> {
    if (!this.session) {
      throw new Error('[LibreYOLO] Session not initialized. Call init() first.');
    }
    return await this.session.run(inputs) as OnnxRunResult;
  }

  get inputNames(): readonly string[] {
    return this.session?.inputNames ?? [];
  }

  get outputNames(): readonly string[] {
    return this.session?.outputNames ?? [];
  }

  get provider(): ExecutionProvider | null {
    return this.currentProvider;
  }

  async release(): Promise<void> {
    await this.session?.release();
    this.session = null;
    this.currentProvider = null;
  }
}
