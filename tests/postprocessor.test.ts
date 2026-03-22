/**
 * Integration test: verify PostProcessor produces correct detections
 * for each model family using raw ONNX output tensors saved from Python.
 *
 * We load the binary tensor data, wrap it in onnxruntime-web-compatible
 * Tensor objects, and feed them through our PostProcessor. Then compare
 * against Python reference results.
 */

import { describe, it, expect } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';
import { PostProcessor } from '../src/processors/PostProcessor';
import { COCO_CLASSES } from '../src/types';
import type { ScaleInfo, ModelFamily, OnnxRunResult } from '../src/types';

const MODELS_DIR = path.join(__dirname, 'models');

// Load reference results from Python
const referenceResults = JSON.parse(
  fs.readFileSync(path.join(MODELS_DIR, 'reference_results.json'), 'utf-8')
);
const testMetadata = JSON.parse(
  fs.readFileSync(path.join(MODELS_DIR, 'test_metadata.json'), 'utf-8')
);

/** Load a binary float32 tensor file and create a mock Tensor-like object */
function loadTensor(filename: string, shape: number[]) {
  const buf = fs.readFileSync(path.join(MODELS_DIR, filename));
  const data = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
  return { data, dims: shape } as any;
}

function makeScaleInfo(meta: any): ScaleInfo {
  return {
    scale: meta.scale,
    offsetX: meta.offset_x,
    offsetY: meta.offset_y,
    originalWidth: meta.original_width,
    originalHeight: meta.original_height,
    scaledWidth: meta.input_size,
    scaledHeight: meta.input_size,
    inputSize: meta.input_size,
  };
}

describe('PostProcessor integration tests', () => {
  const postProcessor = new PostProcessor({
    confidenceThreshold: 0.25,
    iouThreshold: 0.45,
    maxDetections: 300,
    classNames: [...COCO_CLASSES],
  });

  it('YOLOX: detects persons in parkour image', () => {
    const meta = testMetadata.yolox;
    const ref = referenceResults.yolox;

    const outputInfo = meta.outputs[0];
    const tensor = loadTensor(outputInfo.file, outputInfo.shape);

    const results: OnnxRunResult = { [outputInfo.name]: tensor };
    const outputNames = [outputInfo.name];
    const scaleInfo = makeScaleInfo(meta);

    const detections = postProcessor.process(
      results, outputNames, scaleInfo, 'yolox' as ModelFamily
    );

    console.log(`YOLOX: ${detections.length} detections (ref: ${ref.num_detections /* Python ref key */})`);
    detections.forEach((d, i) => {
      console.log(`  [${i}] cls=${d.classId} (${d.label}) conf=${d.confidence.toFixed(3)} bbox=[${d.bbox.map(b => b.toFixed(1))}]`);
    });

    // Should detect same number of objects (allowing ±1 for threshold edge cases)
    expect(detections.length).toBeGreaterThanOrEqual(ref.num_detections /* Python ref key */ - 1);
    expect(detections.length).toBeLessThanOrEqual(ref.num_detections /* Python ref key */ + 2);

    // All detections should be persons (class 0) for this image
    for (const det of detections) {
      expect(det.classId).toBe(0);
      expect(det.label).toBe('person');
      expect(det.confidence).toBeGreaterThan(0.25);
    }

    // Check boxes are in reasonable pixel coordinates
    for (const det of detections) {
      const [x1, y1, x2, y2] = det.bbox;
      expect(x1).toBeGreaterThanOrEqual(0);
      expect(y1).toBeGreaterThanOrEqual(0);
      expect(x2).toBeLessThanOrEqual(meta.original_width + 1);
      expect(y2).toBeLessThanOrEqual(meta.original_height + 1);
      expect(x2).toBeGreaterThan(x1);
      expect(y2).toBeGreaterThan(y1);
    }

    // Compare top detection box with reference (allow ~20px tolerance for rounding)
    if (detections.length > 0 && ref.num_detections /* Python ref key */ > 0) {
      // Find matching reference detection by IoU
      const topDet = detections[0];
      const refBoxes = ref.boxes as number[][];
      let bestIou = 0;
      for (const refBox of refBoxes) {
        const ix1 = Math.max(topDet.bbox[0], refBox[0]);
        const iy1 = Math.max(topDet.bbox[1], refBox[1]);
        const ix2 = Math.min(topDet.bbox[2], refBox[2]);
        const iy2 = Math.min(topDet.bbox[3], refBox[3]);
        const inter = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1);
        const area1 = (topDet.bbox[2] - topDet.bbox[0]) * (topDet.bbox[3] - topDet.bbox[1]);
        const area2 = (refBox[2] - refBox[0]) * (refBox[3] - refBox[1]);
        const iouVal = inter / (area1 + area2 - inter);
        bestIou = Math.max(bestIou, iouVal);
      }
      console.log(`  Top detection IoU with best ref match: ${bestIou.toFixed(3)}`);
      expect(bestIou).toBeGreaterThan(0.5);
    }
  });

  it('YOLO9: detects persons in parkour image', () => {
    const meta = testMetadata.yolo9;
    const ref = referenceResults.yolo9;

    const outputInfo = meta.outputs[0];
    const tensor = loadTensor(outputInfo.file, outputInfo.shape);

    const results: OnnxRunResult = { [outputInfo.name]: tensor };
    const outputNames = [outputInfo.name];
    const scaleInfo = makeScaleInfo(meta);

    const detections = postProcessor.process(
      results, outputNames, scaleInfo, 'yolo9' as ModelFamily
    );

    console.log(`YOLO9: ${detections.length} detections (ref: ${ref.num_detections /* Python ref key */})`);
    detections.forEach((d, i) => {
      console.log(`  [${i}] cls=${d.classId} (${d.label}) conf=${d.confidence.toFixed(3)} bbox=[${d.bbox.map(b => b.toFixed(1))}]`);
    });

    expect(detections.length).toBeGreaterThanOrEqual(ref.num_detections /* Python ref key */ - 1);
    expect(detections.length).toBeLessThanOrEqual(ref.num_detections /* Python ref key */ + 2);

    for (const det of detections) {
      expect(det.classId).toBe(0);
      expect(det.label).toBe('person');
      expect(det.confidence).toBeGreaterThan(0.25);
    }

    for (const det of detections) {
      const [x1, y1, x2, y2] = det.bbox;
      expect(x1).toBeGreaterThanOrEqual(0);
      expect(y1).toBeGreaterThanOrEqual(0);
      expect(x2).toBeLessThanOrEqual(meta.original_width + 1);
      expect(y2).toBeLessThanOrEqual(meta.original_height + 1);
      expect(x2).toBeGreaterThan(x1);
      expect(y2).toBeGreaterThan(y1);
    }

    if (detections.length > 0 && ref.num_detections /* Python ref key */ > 0) {
      const topDet = detections[0];
      const refBoxes = ref.boxes as number[][];
      let bestIou = 0;
      for (const refBox of refBoxes) {
        const ix1 = Math.max(topDet.bbox[0], refBox[0]);
        const iy1 = Math.max(topDet.bbox[1], refBox[1]);
        const ix2 = Math.min(topDet.bbox[2], refBox[2]);
        const iy2 = Math.min(topDet.bbox[3], refBox[3]);
        const inter = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1);
        const area1 = (topDet.bbox[2] - topDet.bbox[0]) * (topDet.bbox[3] - topDet.bbox[1]);
        const area2 = (refBox[2] - refBox[0]) * (refBox[3] - refBox[1]);
        const iouVal = inter / (area1 + area2 - inter);
        bestIou = Math.max(bestIou, iouVal);
      }
      console.log(`  Top detection IoU with best ref match: ${bestIou.toFixed(3)}`);
      expect(bestIou).toBeGreaterThan(0.5);
    }
  });

  it('RF-DETR: detects persons in parkour image', () => {
    const meta = testMetadata.rfdetr;
    const ref = referenceResults.rfdetr;

    // RF-DETR has 2 outputs
    const boxOutput = meta.outputs[0];
    const logitOutput = meta.outputs[1];

    const boxTensor = loadTensor(boxOutput.file, boxOutput.shape);
    const logitTensor = loadTensor(logitOutput.file, logitOutput.shape);

    const results: OnnxRunResult = {
      [boxOutput.name]: boxTensor,
      [logitOutput.name]: logitTensor,
    };
    const outputNames = [boxOutput.name, logitOutput.name];
    const scaleInfo = makeScaleInfo(meta);

    const detections = postProcessor.process(
      results, outputNames, scaleInfo, 'rfdetr' as ModelFamily
    );

    console.log(`RF-DETR: ${detections.length} detections (ref: ${ref.num_detections /* Python ref key */})`);
    detections.forEach((d, i) => {
      console.log(`  [${i}] cls=${d.classId} (${d.label}) conf=${d.confidence.toFixed(3)} bbox=[${d.bbox.map(b => b.toFixed(1))}]`);
    });

    // RF-DETR should find persons (class 0 in COCO80)
    expect(detections.length).toBeGreaterThanOrEqual(3);

    const personDets = detections.filter(d => d.classId === 0);
    expect(personDets.length).toBeGreaterThanOrEqual(3);
    console.log(`  Person detections: ${personDets.length}`);

    for (const det of detections) {
      expect(det.confidence).toBeGreaterThan(0.25);
      const [x1, y1, x2, y2] = det.bbox;
      expect(x1).toBeGreaterThanOrEqual(0);
      expect(y1).toBeGreaterThanOrEqual(0);
      expect(x2).toBeLessThanOrEqual(meta.original_width + 1);
      expect(y2).toBeLessThanOrEqual(meta.original_height + 1);
      expect(x2).toBeGreaterThan(x1);
      expect(y2).toBeGreaterThan(y1);
    }

    // Check IoU with reference
    if (detections.length > 0 && ref.num_detections /* Python ref key */ > 0) {
      const topDet = detections[0];
      const refBoxes = ref.boxes as number[][];
      let bestIou = 0;
      for (const refBox of refBoxes) {
        const ix1 = Math.max(topDet.bbox[0], refBox[0]);
        const iy1 = Math.max(topDet.bbox[1], refBox[1]);
        const ix2 = Math.min(topDet.bbox[2], refBox[2]);
        const iy2 = Math.min(topDet.bbox[3], refBox[3]);
        const inter = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1);
        const area1 = (topDet.bbox[2] - topDet.bbox[0]) * (topDet.bbox[3] - topDet.bbox[1]);
        const area2 = (refBox[2] - refBox[0]) * (refBox[3] - refBox[1]);
        const iouVal = inter / (area1 + area2 - inter);
        bestIou = Math.max(bestIou, iouVal);
      }
      console.log(`  Top detection IoU with best ref match: ${bestIou.toFixed(3)}`);
      expect(bestIou).toBeGreaterThan(0.5);
    }
  });
});
