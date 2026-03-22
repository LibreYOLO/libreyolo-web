import * as ort from 'onnxruntime-web';
import { CoreEngine } from './core/CoreEngine';
import { PreProcessor } from './processors/PreProcessor';
import { PostProcessor } from './processors/PostProcessor';
import type {
  LibreYOLOOptions,
  ExecutionProvider,
  ImageInput,
  Detection,
  DetectionResult,
  ModelFamily,
} from './types';
import { COCO_CLASSES } from './types';

const DEFAULT_OPTIONS = {
  confThres: 0.25,
  iouThres: 0.45,
  maxDet: 300,
  device: 'auto' as const,
  inputSize: 640,
  modelFamily: 'auto' as const,
};

/**
 * LibreYOLO Web - Multi-Family Object Detection in the Browser
 *
 * Supports YOLOX, YOLO9, RF-DETR, and YOLOv8/v11/v26 with correct
 * per-family preprocessing and postprocessing. Powered by ONNX Runtime
 * with WebGPU and WASM backends.
 *
 * @example
 * ```typescript
 * import { loadModel } from 'libreyolo-web';
 *
 * const model = await loadModel('./yolox_s.onnx', { modelFamily: 'yolox' });
 * const result = await model.predict(imageElement);
 * ```
 */
export class LIBREYOLO {
  private engine: CoreEngine;
  private preProcessor: PreProcessor;
  private postProcessor!: PostProcessor;
  private isInitialized = false;
  private classNames: string[];
  private resolvedFamily: ModelFamily | null = null;

  readonly confThres: number;
  readonly iouThres: number;
  readonly maxDet: number;
  readonly inputSize: number;
  private readonly requestedFamily: ModelFamily | 'auto';

  constructor(
    modelPath: string,
    options: LibreYOLOOptions = {}
  ) {
    this.confThres = options.confThres ?? DEFAULT_OPTIONS.confThres;
    this.iouThres = options.iouThres ?? DEFAULT_OPTIONS.iouThres;
    this.maxDet = options.maxDet ?? DEFAULT_OPTIONS.maxDet;
    this.inputSize = options.inputSize ?? DEFAULT_OPTIONS.inputSize;
    this.classNames = options.classNames ?? [...COCO_CLASSES];
    this.requestedFamily = options.modelFamily ?? DEFAULT_OPTIONS.modelFamily;

    const providers = this.resolveProviders(options.device);

    this.engine = new CoreEngine({
      modelPath,
      providers,
      runtime: options.runtime,
      onProgress: options.onProgress,
    });

    this.preProcessor = new PreProcessor();
  }

  private resolveProviders(
    device: LibreYOLOOptions['device']
  ): ExecutionProvider[] {
    if (device === 'auto' || device === undefined) {
      return ['webgpu', 'wasm'];
    }
    if (Array.isArray(device)) {
      return device;
    }
    return [device];
  }

  /**
   * Initialize the model (load ONNX session).
   * Auto-detects model family from ONNX output structure if not specified.
   */
  async init(): Promise<void> {
    if (this.isInitialized) return;
    await this.engine.init();

    if (this.requestedFamily === 'auto') {
      this.resolvedFamily = this.detectFamily();
    } else {
      this.resolvedFamily = this.requestedFamily;
    }

    this.postProcessor = new PostProcessor({
      confidenceThreshold: this.confThres,
      iouThreshold: this.iouThres,
      maxDetections: this.maxDet,
      classNames: this.classNames,
    });

    this.isInitialized = true;
    console.log(`[LibreYOLO] Model family: ${this.resolvedFamily}`);
  }

  /**
   * Auto-detect model family from ONNX output structure.
   * Multiple outputs -> RF-DETR, otherwise -> yolo (safe default).
   * For yolox/yolo9, specify modelFamily explicitly.
   */
  private detectFamily(): ModelFamily {
    const outputNames = this.engine.outputNames;
    if (outputNames.length >= 2) {
      return 'rfdetr';
    }
    return 'yolo';
  }

  /** The resolved model family (available after init) */
  get modelFamily(): ModelFamily | null {
    return this.resolvedFamily;
  }

  /**
   * Run inference on an image.
   * Returns DetectionResult with uniform output regardless of model family.
   */
  async predict(
    image: ImageInput,
    options?: { confThres?: number; iouThres?: number; maxDet?: number }
  ): Promise<DetectionResult> {
    if (!this.isInitialized) {
      await this.init();
    }

    const family = this.resolvedFamily!;
    const confThres = options?.confThres ?? this.confThres;
    const iouThres = options?.iouThres ?? this.iouThres;
    const maxDet = options?.maxDet ?? this.maxDet;

    // 1. Preprocess (family-aware)
    const { tensorData, scaleInfo } = this.preProcessor.process(
      image,
      this.inputSize,
      family
    );

    // 2. Create ONNX tensor and run inference
    const inputTensor = new ort.Tensor(
      'float32',
      tensorData,
      [1, 3, this.inputSize, this.inputSize]
    );
    const inputName = this.engine.inputNames[0] || 'images';

    let results: Awaited<ReturnType<typeof this.engine.run>>;
    try {
      results = await this.engine.run({ [inputName]: inputTensor });
    } finally {
      inputTensor.dispose();
    }

    // 3. Postprocess (family-aware)
    const outputNames = [...this.engine.outputNames];

    let detections: Detection[];
    try {
      detections = this.postProcessor.process(
        results,
        outputNames,
        scaleInfo,
        family,
        { confThres, iouThres, maxDet }
      );
    } finally {
      for (const name of outputNames) {
        const tensor = results[name] as ort.Tensor;
        tensor?.dispose?.();
      }
    }

    // 4. Uniform output across all families
    return this.formatResult(detections);
  }

  /**
   * JS-idiomatic detection method.
   * Returns an array of Detection objects.
   */
  async detect(
    image: ImageInput,
    options?: { confThres?: number; iouThres?: number; maxDet?: number }
  ): Promise<Detection[]> {
    const result = await this.predict(image, options);
    return result.detections;
  }

  private formatResult(detections: Detection[]): DetectionResult {
    return {
      boxes: detections.map((d) => [...d.bbox]),
      scores: detections.map((d) => d.confidence),
      classes: detections.map((d) => d.classId),
      numDetections: detections.length,
      detections,
    };
  }

  get provider(): ExecutionProvider | null {
    return this.engine.provider;
  }

  async release(): Promise<void> {
    this.preProcessor.cleanup();
    await this.engine.release();
    this.isInitialized = false;
    this.resolvedFamily = null;
  }
}
