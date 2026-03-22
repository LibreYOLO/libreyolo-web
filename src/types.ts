import type { Tensor } from 'onnxruntime-web';

// ============ Model Families ============

/**
 * Supported model families. Each family has different
 * preprocessing, output tensor layout, and postprocessing.
 *
 * - `yolo`  — YOLOv8 / v11 / v26 style (xywh, no objectness)
 * - `yolo9` — LibreYOLO YOLO9 (xyxy boxes, direct resize)
 * - `yolox` — YOLOX (objectness score, BGR input, 0-255 range)
 * - `rfdetr` — RF-DETR transformer (normalized boxes, no NMS)
 */
export type ModelFamily = 'yolo' | 'yolox' | 'yolo9' | 'rfdetr';

// ============ Backend Configuration ============

export type ExecutionProvider = 'webgpu' | 'wasm';

export interface RuntimeConfig {
  /** CDN URL for WASM files. Defaults to jsDelivr CDN */
  wasmPaths?: string;
  /** Number of WASM threads. Defaults to min(4, navigator.hardwareConcurrency) */
  numThreads?: number;
  /** Enable SIMD optimization. Defaults to true */
  simd?: boolean;
  /** Use WASM proxy for better performance. Defaults to true */
  proxy?: boolean;
}

// ============ Model Configuration ============

export interface LibreYOLOOptions {
  /** Confidence threshold for detections. Default: 0.25 */
  confThres?: number;
  /** IoU threshold for NMS. Default: 0.45 */
  iouThres?: number;
  /** Maximum number of detections. Default: 300 */
  maxDet?: number;
  /**
   * Backend selection.
   * - "auto": Try WebGPU > WASM (default)
   * - Array: Custom fallback order
   * - Single string: Force specific backend
   */
  device?: 'auto' | ExecutionProvider | ExecutionProvider[];
  /** Input image size. Default: 640 */
  inputSize?: number;
  /** Class names for labeling. Default: COCO 80 classes */
  classNames?: string[];
  /** Runtime configuration for WASM/threading */
  runtime?: RuntimeConfig;
  /**
   * Model family for correct pre/post-processing.
   * - "auto": Auto-detect from ONNX output structure (default)
   * - Explicit: "yolo", "yolox", "yolo9", "rfdetr"
   */
  modelFamily?: ModelFamily | 'auto';
  /** Progress callback during model download (0-1). Useful for loading bars. */
  onProgress?: (progress: number) => void;
}

// ============ Detection Results ============

/** Single detection result */
export interface Detection {
  /** Class ID (0-indexed) */
  classId: number;
  /** Confidence score (0-1) */
  confidence: number;
  /** Bounding box [x1, y1, x2, y2] in pixel coordinates */
  bbox: [number, number, number, number];
  /** Class label (if classNames provided) */
  label?: string;
}

/**
 * Detection result format matching Python libreyolo.
 * Primary return type for predict().
 */
export interface DetectionResult {
  /** Array of bounding boxes [[x1,y1,x2,y2], ...] */
  boxes: number[][];
  /** Array of confidence scores */
  scores: number[];
  /** Array of class IDs */
  classes: number[];
  /** Total number of detections */
  numDetections: number;
  /** Array of Detection objects */
  detections: Detection[];
}

// ============ Input Types ============

/** Supported input types for inference (types that can be drawn to canvas) */
export type ImageInput =
  | HTMLImageElement
  | HTMLVideoElement
  | HTMLCanvasElement
  | ImageBitmap;

// ============ Internal Types ============

export interface ScaleInfo {
  scale: number;
  offsetX: number;
  offsetY: number;
  originalWidth: number;
  originalHeight: number;
  scaledWidth: number;
  scaledHeight: number;
  /** The target input size used for preprocessing */
  inputSize: number;
}

/** ONNX inference result type */
export type OnnxRunResult = {
  readonly [name: string]: Tensor;
};

// ============ COCO Classes ============

/** COCO 80 class names */
export const COCO_CLASSES = [
  'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
  'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
  'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
  'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
  'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
  'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
  'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
  'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
  'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
  'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
  'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
  'toothbrush'
] as const;
