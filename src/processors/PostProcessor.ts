import type { ScaleInfo, Detection, ModelFamily, OnnxRunResult } from '../types';
import { iou, xywh2xyxy, clamp } from '../utils/math';

/**
 * COCO 91-class → 80-class mapping.
 * RF-DETR models trained on COCO output 81 logits (background + 80 classes in COCO91 IDs).
 * This maps COCO91 indices (1-90) → COCO80 indices (0-79). Index 0 is background.
 */
const COCO91_TO_COCO80: Record<number, number> = {
  1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 9:8, 10:9,
  11:10, 13:11, 14:12, 15:13, 16:14, 17:15, 18:16, 19:17, 20:18, 21:19,
  22:20, 23:21, 24:22, 25:23, 27:24, 28:25, 31:26, 32:27, 33:28, 34:29,
  35:30, 36:31, 37:32, 38:33, 39:34, 40:35, 41:36, 42:37, 43:38, 44:39,
  46:40, 47:41, 48:42, 49:43, 50:44, 51:45, 52:46, 53:47, 54:48, 55:49,
  56:50, 57:51, 58:52, 59:53, 60:54, 61:55, 62:56, 63:57, 64:58, 65:59,
  67:60, 70:61, 72:62, 73:63, 74:64, 75:65, 76:66, 77:67, 78:68, 79:69,
  80:70, 81:71, 82:72, 84:73, 85:74, 86:75, 87:76, 88:77, 89:78, 90:79,
};

export interface PostProcessorConfig {
  confidenceThreshold: number;
  iouThreshold: number;
  maxDetections: number;
  classNames?: string[];
}

export interface PostProcessOverrides {
  confThres?: number;
  iouThres?: number;
  maxDet?: number;
}

export class PostProcessor {
  constructor(
    private config: PostProcessorConfig = {
      confidenceThreshold: 0.25,
      iouThreshold: 0.45,
      maxDetections: 300,
    }
  ) {}

  /**
   * Process ONNX inference results into detections.
   * Dispatches to family-specific parsing.
   */
  process(
    results: OnnxRunResult,
    outputNames: string[],
    scaleInfo: ScaleInfo,
    modelFamily: ModelFamily,
    overrides?: PostProcessOverrides
  ): Detection[] {
    if (outputNames.length === 0) {
      throw new Error('[LibreYOLO] No output tensors from model.');
    }

    // Validate: detect likely family mismatch and give helpful error
    this.validateFamily(results, outputNames, modelFamily);

    const confThres = overrides?.confThres ?? this.config.confidenceThreshold;
    const iouThres = overrides?.iouThres ?? this.config.iouThreshold;
    const maxDet = overrides?.maxDet ?? this.config.maxDetections;

    switch (modelFamily) {
      case 'yolo':
        return this.processYolo(results, outputNames, scaleInfo, confThres, iouThres, maxDet);
      case 'yolo9':
        return this.processYolo9(results, outputNames, scaleInfo, confThres, iouThres, maxDet);
      case 'yolox':
        return this.processYolox(results, outputNames, scaleInfo, confThres, iouThres, maxDet);
      case 'rfdetr':
        return this.processRfdetr(results, outputNames, scaleInfo, confThres, maxDet);
    }
  }

  // ============================================================
  // Smart error detection — suggest correct modelFamily
  // ============================================================

  private validateFamily(
    results: OnnxRunResult,
    outputNames: string[],
    modelFamily: ModelFamily
  ): void {
    const numOutputs = outputNames.length;
    const firstOutput = results[outputNames[0]];
    const dims = firstOutput?.dims;
    if (!dims || dims.length < 2) return;

    let suggested: string | null = null;

    if (numOutputs >= 2 && modelFamily !== 'rfdetr') {
      // Multiple outputs = RF-DETR
      suggested = `Output has ${numOutputs} tensors, which looks like RF-DETR. Did you mean modelFamily: 'rfdetr'?`;
    } else if (numOutputs === 1 && modelFamily === 'rfdetr') {
      suggested = `Output has only 1 tensor, but RF-DETR expects 2+. Did you mean modelFamily: 'yolo' or 'yolox'?`;
    } else if (dims.length === 3 && numOutputs === 1) {
      const [, d1, d2] = dims;
      // YOLOX: (B, N, 5+nc) — d1 >> d2 and (d2 - 5) makes sense as class count
      const looksLikeYolox = d1 > d2 && (d2 - 5) > 0 && (d2 - 5) <= 1000;
      // YOLO/YOLO9: (B, 4+nc, N) — d1 < d2 and (d1 - 4) makes sense as class count
      const looksLikeYolo = d1 < d2 && (d1 - 4) > 0 && (d1 - 4) <= 1000;

      if (looksLikeYolox && (modelFamily === 'yolo' || modelFamily === 'yolo9')) {
        suggested = `Output shape [${dims}] looks like YOLOX (B, ${d1}, ${d2}) — has objectness column. Did you mean modelFamily: 'yolox'?`;
      } else if (looksLikeYolo && modelFamily === 'yolox') {
        suggested = `Output shape [${dims}] looks like YOLO/YOLO9 (B, ${d1}, ${d2}) — no objectness. Did you mean modelFamily: 'yolo' or 'yolo9'?`;
      }
    }

    if (suggested) {
      console.warn(`[LibreYOLO] Possible model family mismatch: ${suggested}`);
    }
  }

  // ============================================================
  // YOLO (v8 / v11 / v26) — xywh center, no objectness
  // Output: (B, 4+nc, N) or transposed (B, N, 4+nc)
  // ============================================================

  private processYolo(
    results: OnnxRunResult,
    outputNames: string[],
    scaleInfo: ScaleInfo,
    confThres: number,
    iouThres: number,
    maxDet: number
  ): Detection[] {
    const output = results[outputNames[0]];
    const dims = output.dims;
    const data = output.data as Float32Array;

    // Auto-detect layout: (B, 4+nc, N) vs (B, N, 4+nc)
    const isTransposed = dims[1] > dims[2];
    const numClasses = isTransposed ? dims[2] - 4 : dims[1] - 4;
    const numCandidates = isTransposed ? dims[1] : dims[2];

    const boxes: Detection[] = [];

    for (let i = 0; i < numCandidates; i++) {
      let maxScore = -Infinity;
      let classId = -1;

      for (let c = 4; c < numClasses + 4; c++) {
        const score = isTransposed
          ? data[i * (numClasses + 4) + c]
          : data[c * numCandidates + i];
        if (score > maxScore) {
          maxScore = score;
          classId = c - 4;
        }
      }

      if (maxScore >= confThres) {
        let x: number, y: number, w: number, h: number;
        if (isTransposed) {
          x = data[i * (numClasses + 4) + 0];
          y = data[i * (numClasses + 4) + 1];
          w = data[i * (numClasses + 4) + 2];
          h = data[i * (numClasses + 4) + 3];
        } else {
          x = data[0 * numCandidates + i];
          y = data[1 * numCandidates + i];
          w = data[2 * numCandidates + i];
          h = data[3 * numCandidates + i];
        }

        const [x1, y1, x2, y2] = xywh2xyxy(x, y, w, h);
        boxes.push({
          classId,
          confidence: maxScore,
          bbox: [x1, y1, x2, y2],
          label: this.config.classNames?.[classId],
        });
      }
    }

    return this.applyNmsAndRescale(boxes, iouThres, maxDet, (det) =>
      this.rescaleLetterbox(det, scaleInfo)
    );
  }

  // ============================================================
  // YOLO9 — xyxy boxes (already decoded), no objectness
  // Output: (B, 4+nc, N) — first 4 are x1,y1,x2,y2
  // ============================================================

  private processYolo9(
    results: OnnxRunResult,
    outputNames: string[],
    scaleInfo: ScaleInfo,
    confThres: number,
    iouThres: number,
    maxDet: number
  ): Detection[] {
    const output = results[outputNames[0]];
    const dims = output.dims;
    const data = output.data as Float32Array;

    // YOLO9 outputs (B, 4+nc, N) — channel-first
    const isTransposed = dims[1] > dims[2];
    const numClasses = isTransposed ? dims[2] - 4 : dims[1] - 4;
    const numCandidates = isTransposed ? dims[1] : dims[2];

    const boxes: Detection[] = [];

    for (let i = 0; i < numCandidates; i++) {
      let maxScore = -Infinity;
      let classId = -1;

      for (let c = 4; c < numClasses + 4; c++) {
        const score = isTransposed
          ? data[i * (numClasses + 4) + c]
          : data[c * numCandidates + i];
        if (score > maxScore) {
          maxScore = score;
          classId = c - 4;
        }
      }

      if (maxScore >= confThres) {
        // YOLO9: first 4 values are already xyxy (not xywh)
        let x1: number, y1: number, x2: number, y2: number;
        if (isTransposed) {
          x1 = data[i * (numClasses + 4) + 0];
          y1 = data[i * (numClasses + 4) + 1];
          x2 = data[i * (numClasses + 4) + 2];
          y2 = data[i * (numClasses + 4) + 3];
        } else {
          x1 = data[0 * numCandidates + i];
          y1 = data[1 * numCandidates + i];
          x2 = data[2 * numCandidates + i];
          y2 = data[3 * numCandidates + i];
        }

        boxes.push({
          classId,
          confidence: maxScore,
          bbox: [x1, y1, x2, y2],
          label: this.config.classNames?.[classId],
        });
      }
    }

    return this.applyNmsAndRescale(boxes, iouThres, maxDet, (det) =>
      this.rescaleResize(det, scaleInfo)
    );
  }

  // ============================================================
  // YOLOX — xywh center + objectness, conf = obj × cls
  // Output: (B, N, 5+nc) — always row-major
  // ============================================================

  private processYolox(
    results: OnnxRunResult,
    outputNames: string[],
    scaleInfo: ScaleInfo,
    confThres: number,
    iouThres: number,
    maxDet: number
  ): Detection[] {
    const output = results[outputNames[0]];
    const dims = output.dims;
    const data = output.data as Float32Array;

    // YOLOX: (B, N, 5+nc) — N is dim[1], stride is dim[2]
    const numCandidates = dims[1];
    const stride = dims[2]; // 5 + numClasses
    const numClasses = stride - 5;

    const boxes: Detection[] = [];

    for (let i = 0; i < numCandidates; i++) {
      const base = i * stride;
      const objectness = data[base + 4];

      // Early exit: if objectness is below threshold, final conf can't pass either
      // (since conf = objectness * cls_score and cls_score <= 1)
      if (objectness < confThres) continue;

      let maxClsScore = -Infinity;
      let classId = -1;

      for (let c = 0; c < numClasses; c++) {
        const score = data[base + 5 + c];
        if (score > maxClsScore) {
          maxClsScore = score;
          classId = c;
        }
      }

      const confidence = objectness * maxClsScore;

      if (confidence >= confThres) {
        const cx = data[base + 0];
        const cy = data[base + 1];
        const w = data[base + 2];
        const h = data[base + 3];
        const [x1, y1, x2, y2] = xywh2xyxy(cx, cy, w, h);

        boxes.push({
          classId,
          confidence,
          bbox: [x1, y1, x2, y2],
          label: this.config.classNames?.[classId],
        });
      }
    }

    return this.applyNmsAndRescale(boxes, iouThres, maxDet, (det) =>
      this.rescaleLetterbox(det, scaleInfo)
    );
  }

  // ============================================================
  // RF-DETR — normalized cxcywh, sigmoid logits, NO NMS (top-K)
  // Output: 2+ tensors (boxes + logits) or single (B, Q, 4+nc)
  // ============================================================

  private processRfdetr(
    results: OnnxRunResult,
    outputNames: string[],
    scaleInfo: ScaleInfo,
    confThres: number,
    maxDet: number
  ): Detection[] {
    let boxData: Float32Array;
    let logitData: Float32Array;
    let numQueries: number;
    let numClasses: number;

    if (outputNames.length >= 2) {
      // Separate tensors: boxes (B, Q, 4) + logits (B, Q, nc)
      const boxesTensor = results[outputNames[0]];
      const logitsTensor = results[outputNames[1]];
      numQueries = boxesTensor.dims[1];
      numClasses = logitsTensor.dims[2];
      boxData = boxesTensor.data as Float32Array;
      logitData = logitsTensor.data as Float32Array;
    } else {
      // Single concatenated tensor: (B, Q, 4+nc)
      const output = results[outputNames[0]];
      numQueries = output.dims[1];
      const totalCols = output.dims[2];
      numClasses = totalCols - 4;
      const rawData = output.data as Float32Array;

      // Split into box and logit data
      boxData = new Float32Array(numQueries * 4);
      logitData = new Float32Array(numQueries * numClasses);

      for (let q = 0; q < numQueries; q++) {
        const base = q * totalCols;
        boxData[q * 4 + 0] = rawData[base + 0];
        boxData[q * 4 + 1] = rawData[base + 1];
        boxData[q * 4 + 2] = rawData[base + 2];
        boxData[q * 4 + 3] = rawData[base + 3];
        for (let c = 0; c < numClasses; c++) {
          logitData[q * numClasses + c] = rawData[base + 4 + c];
        }
      }
    }

    // Detect COCO91 format (81 classes = background + 80 COCO classes)
    const isCoco91 = numClasses === 91 || numClasses === 81;
    // If 81 classes: index 0 is background, 1-80 are COCO91 IDs
    // If 91 classes: indices 0-90, same mapping applies
    const classOffset = numClasses === 81 ? 1 : 0;

    const { originalWidth, originalHeight } = scaleInfo;
    const boxes: Detection[] = [];

    for (let q = 0; q < numQueries; q++) {
      // Apply sigmoid to logits to get class probabilities
      let maxScore = -Infinity;
      let rawClassId = -1;

      // Skip index 0 (background) for COCO91 format (both 81 and 91 class variants)
      const startIdx = isCoco91 ? 1 : 0;

      for (let c = startIdx; c < numClasses; c++) {
        const logit = logitData[q * numClasses + c];
        const score = 1 / (1 + Math.exp(-logit)); // sigmoid
        if (score > maxScore) {
          maxScore = score;
          rawClassId = c;
        }
      }

      if (maxScore >= confThres) {
        // Map COCO91 class ID → COCO80 class ID
        let classId = rawClassId;
        if (isCoco91) {
          const coco91Id = rawClassId + classOffset; // adjust for 81-class format
          const mapped = COCO91_TO_COCO80[numClasses === 81 ? rawClassId : coco91Id];
          if (mapped === undefined) continue; // skip unmapped classes (gaps in COCO91)
          classId = mapped;
        }

        // Boxes are normalized cxcywh [0,1]
        const cx = boxData[q * 4 + 0];
        const cy = boxData[q * 4 + 1];
        const w = boxData[q * 4 + 2];
        const h = boxData[q * 4 + 3];

        // Convert to xyxy in pixel coordinates
        const [nx1, ny1, nx2, ny2] = xywh2xyxy(cx, cy, w, h);
        const x1 = clamp(nx1 * originalWidth, 0, originalWidth);
        const y1 = clamp(ny1 * originalHeight, 0, originalHeight);
        const x2 = clamp(nx2 * originalWidth, 0, originalWidth);
        const y2 = clamp(ny2 * originalHeight, 0, originalHeight);

        boxes.push({
          classId,
          confidence: maxScore,
          bbox: [x1, y1, x2, y2],
          label: this.config.classNames?.[classId],
        });
      }
    }

    // RF-DETR: No NMS — just top-K by confidence
    boxes.sort((a, b) => b.confidence - a.confidence);
    return boxes.slice(0, maxDet);
  }

  // ============================================================
  // Shared helpers
  // ============================================================

  private applyNmsAndRescale(
    boxes: Detection[],
    iouThres: number,
    maxDet: number,
    rescaleFn: (det: Detection) => Detection
  ): Detection[] {
    boxes.sort((a, b) => b.confidence - a.confidence);

    // Keep more candidates for NMS, then limit
    const topBoxes = boxes.slice(0, maxDet * 3);
    const nmsBoxes = this.nms(topBoxes, iouThres);
    const finalBoxes = nmsBoxes.slice(0, maxDet);

    return finalBoxes.map(rescaleFn);
  }

  /**
   * Per-class NMS: apply NMS independently within each class,
   * so overlapping boxes of different classes are preserved.
   */
  private nms(boxes: Detection[], threshold: number): Detection[] {
    if (boxes.length === 0) return [];

    // Group by class
    const byClass = new Map<number, Detection[]>();
    for (const box of boxes) {
      const list = byClass.get(box.classId);
      if (list) {
        list.push(box);
      } else {
        byClass.set(box.classId, [box]);
      }
    }

    const selected: Detection[] = [];

    for (const classBoxes of byClass.values()) {
      // Greedy NMS within this class (already sorted by confidence)
      const active = [...classBoxes];
      while (active.length > 0) {
        const best = active.shift()!;
        selected.push(best);

        for (let i = active.length - 1; i >= 0; i--) {
          if (iou(best.bbox, active[i].bbox) > threshold) {
            active.splice(i, 1);
          }
        }
      }
    }

    // Re-sort by confidence (classes were processed separately)
    selected.sort((a, b) => b.confidence - a.confidence);
    return selected;
  }

  /** Rescale from letterbox input space to original image coordinates */
  private rescaleLetterbox(detection: Detection, scaleInfo: ScaleInfo): Detection {
    const [x1, y1, x2, y2] = detection.bbox;
    const { scale, offsetX, offsetY, originalWidth, originalHeight } = scaleInfo;

    return {
      ...detection,
      bbox: [
        clamp((x1 - offsetX) / scale, 0, originalWidth),
        clamp((y1 - offsetY) / scale, 0, originalHeight),
        clamp((x2 - offsetX) / scale, 0, originalWidth),
        clamp((y2 - offsetY) / scale, 0, originalHeight),
      ],
    };
  }

  /** Rescale from direct-resize input space to original image coordinates */
  private rescaleResize(detection: Detection, scaleInfo: ScaleInfo): Detection {
    const [x1, y1, x2, y2] = detection.bbox;
    const { originalWidth, originalHeight, inputSize } = scaleInfo;

    const scaleX = originalWidth / inputSize;
    const scaleY = originalHeight / inputSize;

    return {
      ...detection,
      bbox: [
        clamp(x1 * scaleX, 0, originalWidth),
        clamp(y1 * scaleY, 0, originalHeight),
        clamp(x2 * scaleX, 0, originalWidth),
        clamp(y2 * scaleY, 0, originalHeight),
      ],
    };
  }
}
