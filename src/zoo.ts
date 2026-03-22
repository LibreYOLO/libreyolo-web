import type { ModelFamily } from './types';

const HF_BASE = 'https://huggingface.co/LibreYOLO/libreyolo-web/resolve/main';

export interface ZooModel {
  url: string;
  family: ModelFamily;
  inputSize: number;
  description: string;
}

/** Standard libreyolo model names — provides IDE autocompletion */
export type ZooModelName =
  // YOLOX
  | 'LibreYOLOXn' | 'LibreYOLOXt' | 'LibreYOLOXs'
  | 'LibreYOLOXm' | 'LibreYOLOXl' | 'LibreYOLOXx'
  // YOLO9
  | 'LibreYOLO9t' | 'LibreYOLO9s' | 'LibreYOLO9m' | 'LibreYOLO9c'
  // RF-DETR
  | 'LibreRFDETRn' | 'LibreRFDETRs' | 'LibreRFDETRm' | 'LibreRFDETRl';

/**
 * Pre-configured models hosted on HuggingFace.
 * Uses standard libreyolo naming: LibreYOLOXn, LibreYOLO9t, LibreRFDETRs, etc.
 *
 * @example
 * ```typescript
 * const model = await loadModel('LibreYOLOXn');
 * ```
 */
export const MODEL_ZOO: Record<ZooModelName, ZooModel> = {
  // YOLOX family
  'LibreYOLOXn':  { url: `${HF_BASE}/yolox_n.onnx`, family: 'yolox', inputSize: 416, description: 'YOLOX Nano (416, 3.6MB)' },
  'LibreYOLOXt':  { url: `${HF_BASE}/yolox_t.onnx`, family: 'yolox', inputSize: 416, description: 'YOLOX Tiny (416, 19MB)' },
  'LibreYOLOXs':  { url: `${HF_BASE}/yolox_s.onnx`, family: 'yolox', inputSize: 640, description: 'YOLOX Small (640, 34MB)' },
  'LibreYOLOXm':  { url: `${HF_BASE}/yolox_m.onnx`, family: 'yolox', inputSize: 640, description: 'YOLOX Medium (640, 97MB)' },
  'LibreYOLOXl':  { url: `${HF_BASE}/yolox_l.onnx`, family: 'yolox', inputSize: 640, description: 'YOLOX Large (640, 207MB)' },
  'LibreYOLOXx':  { url: `${HF_BASE}/yolox_x.onnx`, family: 'yolox', inputSize: 640, description: 'YOLOX XLarge (640, 378MB)' },

  // YOLO9 family
  'LibreYOLO9t':  { url: `${HF_BASE}/yolo9_t.onnx`, family: 'yolo9', inputSize: 640, description: 'YOLO9 Tiny (640, 8MB)' },
  'LibreYOLO9s':  { url: `${HF_BASE}/yolo9_s.onnx`, family: 'yolo9', inputSize: 640, description: 'YOLO9 Small (640, 28MB)' },
  'LibreYOLO9m':  { url: `${HF_BASE}/yolo9_m.onnx`, family: 'yolo9', inputSize: 640, description: 'YOLO9 Medium (640, 77MB)' },
  'LibreYOLO9c':  { url: `${HF_BASE}/yolo9_c.onnx`, family: 'yolo9', inputSize: 640, description: 'YOLO9 Compact (640, 97MB)' },

  // RF-DETR family
  'LibreRFDETRn': { url: `${HF_BASE}/rfdetr_n.onnx`, family: 'rfdetr', inputSize: 384, description: 'RF-DETR Nano (384, 103MB)' },
  'LibreRFDETRs': { url: `${HF_BASE}/rfdetr_s.onnx`, family: 'rfdetr', inputSize: 512, description: 'RF-DETR Small (512, 109MB)' },
  'LibreRFDETRm': { url: `${HF_BASE}/rfdetr_m.onnx`, family: 'rfdetr', inputSize: 576, description: 'RF-DETR Medium (576, 115MB)' },
  'LibreRFDETRl': { url: `${HF_BASE}/rfdetr_l.onnx`, family: 'rfdetr', inputSize: 704, description: 'RF-DETR Large (704, 116MB)' },
};

/**
 * Check if a string is a known zoo model name.
 */
export function isZooModel(name: string): name is ZooModelName {
  return name in MODEL_ZOO;
}

/**
 * List all available zoo models.
 */
export function listModels(): { name: ZooModelName; model: ZooModel }[] {
  return (Object.entries(MODEL_ZOO) as [ZooModelName, ZooModel][]).map(
    ([name, model]) => ({ name, model })
  );
}
