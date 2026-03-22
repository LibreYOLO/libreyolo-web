// Main class and factory functions
export { LIBREYOLO } from './LibreYOLO';
export { createModel, loadModel } from './factory';

// Types
export type {
  LibreYOLOOptions,
  Detection,
  DetectionResult,
  ImageInput,
  ExecutionProvider,
  RuntimeConfig,
  ScaleInfo,
  ModelFamily,
} from './types';

export { COCO_CLASSES } from './types';

/** Path to bundled sample image (parkour scene, 1280x852). Same as Python libreyolo's SAMPLE_IMAGE. */
export const SAMPLE_IMAGE = new URL('../assets/parkour.jpg', import.meta.url).href;

// Model zoo
export { MODEL_ZOO, isZooModel, listModels } from './zoo';
export type { ZooModel, ZooModelName } from './zoo';

// UI utilities
export { BoxOverlay } from './ui/BoxOverlay';
export type { BoxOverlayOptions, DrawOptions } from './ui/BoxOverlay';

// Advanced: Direct access to internals
export { CoreEngine } from './core/CoreEngine';
export type { CoreEngineOptions } from './core/CoreEngine';

export { PreProcessor } from './processors/PreProcessor';

export { PostProcessor } from './processors/PostProcessor';
export type { PostProcessorConfig, PostProcessOverrides } from './processors/PostProcessor';

// Utilities
export { configureRuntime, isWebGPUAvailable, ORT_WASM_CDN } from './utils/runtime';
