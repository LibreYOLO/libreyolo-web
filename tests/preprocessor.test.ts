/**
 * Regression tests for PreProcessor dimension resolution.
 *
 * Context: predict(HTMLVideoElement) silently returned zero detections because
 * the preprocessor fell through to image.width (which is 0 for video elements
 * without an explicit width attribute). The correct intrinsic dimensions for
 * HTMLVideoElement live on videoWidth/videoHeight.
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { PreProcessor } from '../src/processors/PreProcessor';
import type { ImageInput, ModelFamily } from '../src/types';

class FakeCtx {
  fillStyle = '';
  fillRect() {}
  drawImage() {}
  getImageData(_x: number, _y: number, w: number, h: number) {
    return { data: new Uint8ClampedArray(w * h * 4) };
  }
}

class FakeOffscreenCanvas {
  constructor(public width: number, public height: number) {}
  getContext() {
    return new FakeCtx();
  }
}

beforeAll(() => {
  (globalThis as any).OffscreenCanvas = FakeOffscreenCanvas;
});

function fakeImage(naturalWidth: number, naturalHeight: number): ImageInput {
  return { naturalWidth, naturalHeight, width: 0, height: 0 } as unknown as ImageInput;
}

function fakeVideo(videoWidth: number, videoHeight: number, attrWidth = 0, attrHeight = 0): ImageInput {
  return { videoWidth, videoHeight, width: attrWidth, height: attrHeight } as unknown as ImageInput;
}

function fakeCanvas(width: number, height: number): ImageInput {
  return { width, height } as unknown as ImageInput;
}

describe('PreProcessor dimension resolution', () => {
  const families: ModelFamily[] = ['yolo', 'yolox', 'yolo9', 'rfdetr'];

  it('HTMLImageElement: reads naturalWidth/naturalHeight', () => {
    const pp = new PreProcessor();
    const { scaleInfo } = pp.process(fakeImage(1280, 852), 640, 'yolox');
    expect(scaleInfo.originalWidth).toBe(1280);
    expect(scaleInfo.originalHeight).toBe(852);
    expect(Number.isFinite(scaleInfo.scale)).toBe(true);
  });

  it('HTMLVideoElement: reads videoWidth/videoHeight (the bug fix)', () => {
    for (const family of families) {
      const pp = new PreProcessor();
      const { scaleInfo } = pp.process(fakeVideo(1920, 1080), 640, family);
      expect(scaleInfo.originalWidth).toBe(1920);
      expect(scaleInfo.originalHeight).toBe(1080);
    }
  });

  it('HTMLVideoElement: prefers videoWidth over the width attribute', () => {
    const pp = new PreProcessor();
    // width attribute set to something misleading; intrinsic size should win.
    const { scaleInfo } = pp.process(fakeVideo(1280, 720, 300, 150), 640, 'yolox');
    expect(scaleInfo.originalWidth).toBe(1280);
    expect(scaleInfo.originalHeight).toBe(720);
  });

  it('HTMLCanvasElement / ImageBitmap: reads width/height', () => {
    const pp = new PreProcessor();
    const { scaleInfo } = pp.process(fakeCanvas(800, 600), 640, 'yolo9');
    expect(scaleInfo.originalWidth).toBe(800);
    expect(scaleInfo.originalHeight).toBe(600);
  });

  it('throws a clear error on zero-dimension input (no more silent zero detections)', () => {
    const pp = new PreProcessor();
    expect(() => pp.process(fakeVideo(0, 0), 640, 'yolox')).toThrow(/zero or unresolved dimensions/);
    expect(() => pp.process(fakeImage(0, 0), 640, 'yolo')).toThrow(/zero or unresolved dimensions/);
  });

  it('letterbox scale is finite for video input across all families', () => {
    for (const family of families) {
      const pp = new PreProcessor();
      const { scaleInfo } = pp.process(fakeVideo(1920, 1080), 640, family);
      expect(Number.isFinite(scaleInfo.scale)).toBe(true);
      expect(Number.isFinite(scaleInfo.scaledWidth)).toBe(true);
      expect(Number.isFinite(scaleInfo.scaledHeight)).toBe(true);
    }
  });
});
