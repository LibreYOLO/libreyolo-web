import type { ImageInput, ScaleInfo, ModelFamily } from '../types';

// ============ Preprocessing Configs Per Family ============

interface PreprocessConfig {
  letterbox: boolean;
  /** Letterbox alignment: 'center' (standard YOLO) or 'top-left' (YOLOX) */
  letterboxAlign: 'center' | 'top-left';
  normalize: 'div255' | 'imagenet' | 'none';
  channelOrder: 'rgb' | 'bgr';
  padColor: number;
}

const FAMILY_PREPROCESS: Record<ModelFamily, PreprocessConfig> = {
  yolo:  { letterbox: true,  letterboxAlign: 'center',   normalize: 'div255',   channelOrder: 'rgb', padColor: 114 },
  yolo9: { letterbox: false, letterboxAlign: 'center',   normalize: 'div255',   channelOrder: 'rgb', padColor: 114 },
  yolox: { letterbox: true,  letterboxAlign: 'top-left', normalize: 'none',     channelOrder: 'bgr', padColor: 114 },
  rfdetr:{ letterbox: false, letterboxAlign: 'center',   normalize: 'imagenet', channelOrder: 'rgb', padColor: 0   },
};

const IMAGENET_MEAN = [0.485, 0.456, 0.406];
const IMAGENET_STD  = [0.229, 0.224, 0.225];

export class PreProcessor {
  private canvas: OffscreenCanvas | HTMLCanvasElement;
  private ctx: OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D | null;

  constructor() {
    if (typeof OffscreenCanvas !== 'undefined') {
      this.canvas = new OffscreenCanvas(640, 640);
    } else {
      this.canvas = document.createElement('canvas');
    }
    this.ctx = this.canvas.getContext('2d', { willReadFrequently: true }) as
      | OffscreenCanvasRenderingContext2D
      | CanvasRenderingContext2D
      | null;
  }

  /** Release canvas resources */
  cleanup(): void {
    this.ctx = null;
  }

  process(
    image: ImageInput,
    targetSize: number = 640,
    modelFamily: ModelFamily = 'yolo'
  ): { tensorData: Float32Array; scaleInfo: ScaleInfo } {
    if (!this.ctx) throw new Error('PreProcessor: Failed to get 2D context');

    const config = FAMILY_PREPROCESS[modelFamily];

    this.canvas.width = targetSize;
    this.canvas.height = targetSize;

    const imgWidth = 'naturalWidth' in image ? image.naturalWidth : image.width;
    const imgHeight = 'naturalHeight' in image ? image.naturalHeight : image.height;

    let scale: number;
    let scaledWidth: number;
    let scaledHeight: number;
    let offsetX: number;
    let offsetY: number;

    if (config.letterbox) {
      // Letterbox: preserve aspect ratio with padding
      scale = Math.min(targetSize / imgWidth, targetSize / imgHeight);
      scaledWidth = imgWidth * scale;
      scaledHeight = imgHeight * scale;

      if (config.letterboxAlign === 'top-left') {
        // YOLOX: image at top-left, padding on right/bottom
        offsetX = 0;
        offsetY = 0;
      } else {
        // Standard YOLO: image centered, padding on all sides
        offsetX = (targetSize - scaledWidth) / 2;
        offsetY = (targetSize - scaledHeight) / 2;
      }

      const c = config.padColor;
      this.ctx.fillStyle = `rgb(${c}, ${c}, ${c})`;
      this.ctx.fillRect(0, 0, targetSize, targetSize);
      this.ctx.drawImage(image, offsetX, offsetY, scaledWidth, scaledHeight);
    } else {
      // Direct resize: stretch to fill (no aspect ratio preservation)
      scale = 1; // not meaningful for non-letterbox
      scaledWidth = targetSize;
      scaledHeight = targetSize;
      offsetX = 0;
      offsetY = 0;

      this.ctx.drawImage(image, 0, 0, targetSize, targetSize);
    }

    const imageData = this.ctx.getImageData(0, 0, targetSize, targetSize);
    const { data } = imageData;
    const float32Data = new Float32Array(3 * targetSize * targetSize);

    // Channel mapping: RGB [0,1,2] or BGR [2,1,0]
    const channelMap = config.channelOrder === 'bgr' ? [2, 1, 0] : [0, 1, 2];

    let i = 0;
    for (let c = 0; c < 3; c++) {
      const srcChannel = channelMap[c];
      for (let h = 0; h < targetSize; h++) {
        for (let w = 0; w < targetSize; w++) {
          const pixelIndex = (h * targetSize + w) * 4;
          const raw = data[pixelIndex + srcChannel];

          switch (config.normalize) {
            case 'div255':
              float32Data[i++] = raw / 255.0;
              break;
            case 'imagenet':
              float32Data[i++] = (raw / 255.0 - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
              break;
            case 'none':
              float32Data[i++] = raw;
              break;
          }
        }
      }
    }

    return {
      tensorData: float32Data,
      scaleInfo: {
        scale,
        offsetX,
        offsetY,
        originalWidth: imgWidth,
        originalHeight: imgHeight,
        scaledWidth,
        scaledHeight,
        inputSize: targetSize,
      },
    };
  }
}
