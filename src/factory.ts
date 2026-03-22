import { LIBREYOLO } from './LibreYOLO';
import type { LibreYOLOOptions } from './types';
import type { ZooModelName } from './zoo';
import { MODEL_ZOO, isZooModel } from './zoo';

/**
 * Resolve a model path: if it's a zoo model name, expand to URL + config.
 * Otherwise return as-is.
 */
function resolveModelPath(
  modelPath: ZooModelName | (string & {}),
  options: LibreYOLOOptions = {}
): { resolvedPath: string; resolvedOptions: LibreYOLOOptions } {
  if (isZooModel(modelPath)) {
    const zoo = MODEL_ZOO[modelPath];
    console.log(`[LibreYOLO] Zoo model "${modelPath}" → downloading ${zoo.description} from HuggingFace`);
    return {
      resolvedPath: zoo.url,
      resolvedOptions: {
        modelFamily: zoo.family,
        inputSize: zoo.inputSize,
        ...options,
      },
    };
  }
  return { resolvedPath: modelPath, resolvedOptions: options };
}

/**
 * Create a LIBREYOLO instance without initializing.
 * Call init() or let the first predict() auto-initialize.
 *
 * Accepts a standard libreyolo model name (e.g. `'LibreYOLOXn'`) or a URL/path.
 *
 * @example
 * ```typescript
 * const model = createModel('LibreYOLOXn');
 * await model.init();
 * const result = await model.predict(imageElement);
 * ```
 */
export function createModel(
  modelPath: ZooModelName | (string & {}),
  options?: LibreYOLOOptions
): LIBREYOLO {
  const { resolvedPath, resolvedOptions } = resolveModelPath(modelPath, options);
  return new LIBREYOLO(resolvedPath, resolvedOptions);
}

/**
 * Create and initialize a model in one step.
 *
 * Accepts a standard libreyolo model name (e.g. `'LibreYOLOXn'`) or a URL/path.
 *
 * @example
 * ```typescript
 * // Zoo model — auto-downloads from HuggingFace, zero config
 * const model = await loadModel('LibreYOLOXn');
 *
 * // Custom model — specify family
 * const model = await loadModel('./my_model.onnx', { modelFamily: 'yolo9' });
 * ```
 */
export async function loadModel(
  modelPath: ZooModelName | (string & {}),
  options?: LibreYOLOOptions
): Promise<LIBREYOLO> {
  const { resolvedPath, resolvedOptions } = resolveModelPath(modelPath, options);
  const model = new LIBREYOLO(resolvedPath, resolvedOptions);
  await model.init();
  return model;
}
