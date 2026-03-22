import type { Detection } from '../types';

export interface BoxOverlayOptions {
  canvas: HTMLCanvasElement;
  colors?: string[];
  lineWidth?: number;
  fontSize?: number;
  fontFamily?: string;
  showLabels?: boolean;
  showConfidence?: boolean;
  /** Semi-transparent fill inside boxes. Default: true */
  fillBoxes?: boolean;
  /** Fill opacity (0-1). Default: 0.1 */
  fillOpacity?: number;
}

export interface DrawOptions {
  originalWidth?: number;
  originalHeight?: number;
}

/** High-contrast color palette — readable on both light and dark images */
const DEFAULT_COLORS = [
  '#FF3838', '#FF9D97', '#FF701F', '#FFB21D', '#CFD231',
  '#48F90A', '#92CC17', '#3DDB86', '#1A9334', '#00D4BB',
  '#2C99A8', '#00C2FF', '#344593', '#6473FF', '#0018EC',
  '#8438FF', '#520085', '#CB38FF', '#FF95C8', '#FF37C7',
];

export class BoxOverlay {
  private ctx: CanvasRenderingContext2D;
  private colors: string[];

  constructor(private options: BoxOverlayOptions) {
    const ctx = options.canvas.getContext('2d');
    if (!ctx) throw new Error('BoxOverlay: Could not get 2D context');
    this.ctx = ctx;
    this.colors = options.colors || DEFAULT_COLORS;
  }

  /**
   * Draw bounding boxes on the canvas.
   */
  draw(detections: Detection[], drawOptions?: DrawOptions): void {
    const canvas = this.options.canvas;
    this.ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (detections.length === 0) return;

    const scaleX = drawOptions?.originalWidth
      ? canvas.width / drawOptions.originalWidth
      : 1;
    const scaleY = drawOptions?.originalHeight
      ? canvas.height / drawOptions.originalHeight
      : 1;

    const lineWidth = this.options.lineWidth || 2;
    const fontSize = this.options.fontSize || 16;
    const fontFamily = this.options.fontFamily || 'Arial';
    const fillBoxes = this.options.fillBoxes !== false;
    const fillOpacity = this.options.fillOpacity ?? 0.1;

    this.ctx.lineWidth = lineWidth;
    this.ctx.font = `bold ${fontSize}px ${fontFamily}`;

    for (const det of detections) {
      const [x1, y1, x2, y2] = det.bbox;

      const sx1 = x1 * scaleX;
      const sy1 = y1 * scaleY;
      const sx2 = x2 * scaleX;
      const sy2 = y2 * scaleY;
      const w = sx2 - sx1;
      const h = sy2 - sy1;

      // Skip invalid boxes
      if (w <= 0 || h <= 0) continue;

      const color = this.colors[det.classId % this.colors.length];

      // Semi-transparent fill inside box
      if (fillBoxes) {
        this.ctx.fillStyle = color;
        this.ctx.globalAlpha = fillOpacity;
        this.ctx.fillRect(sx1, sy1, w, h);
        this.ctx.globalAlpha = 1;
      }

      // Box outline
      this.ctx.strokeStyle = color;
      this.ctx.strokeRect(sx1, sy1, w, h);

      // Label
      if (this.options.showLabels !== false) {
        const labelText = det.label || `Class ${det.classId}`;
        const scoreText =
          this.options.showConfidence !== false
            ? ` ${(det.confidence * 100).toFixed(1)}%`
            : '';
        const fullText = labelText + scoreText;

        const textMetrics = this.ctx.measureText(fullText);
        const textHeight = fontSize * 1.2;
        const textWidth = textMetrics.width + 8;

        // Flip label below box if it would go above the canvas
        const labelAbove = sy1 - textHeight >= 0;
        const labelY = labelAbove ? sy1 - textHeight : sy1;
        const textY = labelAbove ? sy1 - 4 : sy1 + textHeight - 4;

        // Clamp label X to canvas bounds
        const labelX = Math.min(Math.max(sx1, 0), canvas.width - textWidth);

        // Dark background for readability
        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        this.ctx.fillRect(labelX, labelY, textWidth, textHeight);

        // Colored left accent bar
        this.ctx.fillStyle = color;
        this.ctx.fillRect(labelX, labelY, 3, textHeight);

        // White text
        this.ctx.fillStyle = '#fff';
        this.ctx.fillText(fullText, labelX + 6, textY);
      }
    }
  }

  clear(): void {
    this.ctx.clearRect(
      0,
      0,
      this.options.canvas.width,
      this.options.canvas.height
    );
  }
}
