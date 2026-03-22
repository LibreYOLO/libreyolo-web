import * as ort from 'onnxruntime-web';
import type { RuntimeConfig } from '../types';

/**
 * Default CDN URL for ONNX Runtime WASM files.
 * Override via RuntimeConfig.wasmPaths if using a different onnxruntime-web version.
 */
export const ORT_WASM_CDN = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';

let isConfigured = false;

/**
 * Configure ONNX Runtime Web environment.
 * Called automatically before creating inference sessions.
 * Safe to call multiple times (only configures once).
 */
export function configureRuntime(config: RuntimeConfig = {}): void {
  if (isConfigured) return;

  ort.env.wasm.wasmPaths = config.wasmPaths ?? ORT_WASM_CDN;

  const defaultThreads =
    typeof navigator !== 'undefined'
      ? Math.min(4, navigator.hardwareConcurrency || 2)
      : 2;
  ort.env.wasm.numThreads = config.numThreads ?? defaultThreads;

  ort.env.wasm.simd = config.simd ?? true;
  ort.env.wasm.proxy = config.proxy ?? true;

  isConfigured = true;
}

/**
 * Check if WebGPU is available in the current browser.
 */
export async function isWebGPUAvailable(): Promise<boolean> {
  if (typeof navigator === 'undefined') return false;
  if (!('gpu' in navigator)) return false;

  try {
    const gpu = (navigator as Navigator & { gpu?: { requestAdapter(): Promise<unknown> } }).gpu;
    if (!gpu) return false;
    const adapter = await gpu.requestAdapter();
    return adapter !== null;
  } catch {
    return false;
  }
}

/**
 * Reset runtime configuration (useful for testing).
 */
export function resetRuntimeConfig(): void {
  isConfigured = false;
}
