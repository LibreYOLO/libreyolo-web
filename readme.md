# LibreYOLO Web

Multi-family YOLO object detection in the browser. The web companion to [libreyolo](https://github.com/LibreYOLO/libreyolo) for Python.

```typescript
import { loadModel } from 'libreyolo-web';

const model = await loadModel('LibreYOLOXn');
const result = await model.predict(imageElement);
// { boxes, scores, classes, numDetections, detections }
```

Supports **YOLOX**, **YOLO9**, **RF-DETR**, and **YOLOv8/v11/v26** with correct per-family preprocessing and postprocessing. 14 pre-trained models available from the [model zoo](#model-zoo), auto-downloaded from HuggingFace.

Powered by ONNX Runtime with **WebGPU** and **WASM** backends.

## Why This Library

Each YOLO family differs in input preprocessing, output tensor format, and postprocessing. Swapping YOLOv8 for YOLOX means rewriting your pre/post-processing code. This library handles it all — same `predict()` call, same output format, regardless of model family.

| What varies per family | This library handles it |
|---|---|
| Letterbox vs resize, RGB vs BGR, /255 vs ImageNet norm | Per-family preprocessing |
| xywh vs xyxy, objectness column, sigmoid logits | Per-family tensor parsing |
| Per-class NMS vs top-K selection | Per-family postprocessing |
| COCO91 vs COCO80 class IDs | Automatic mapping |

## Installation

```bash
npm install libreyolo-web onnxruntime-web
```

## Quick Start

### Option 1: Zoo model (zero config)

```typescript
import { loadModel } from 'libreyolo-web';

// Auto-downloads from HuggingFace, sets correct family + input size
const model = await loadModel('LibreYOLOXn');
const result = await model.predict(imageElement);

console.log(`Found ${result.numDetections} objects`);
for (const det of result.detections) {
  console.log(`${det.label} ${(det.confidence * 100).toFixed(1)}% at [${det.bbox}]`);
}

await model.release();
```

### Option 2: Your own ONNX model

```typescript
import { loadModel } from 'libreyolo-web';

const model = await loadModel('./my_model.onnx', {
  modelFamily: 'yolo9',
  inputSize: 640,
});
const result = await model.predict(imageElement);
```

### Drawing bounding boxes

```typescript
import { loadModel, BoxOverlay } from 'libreyolo-web';

const model = await loadModel('LibreYOLO9t');
const result = await model.predict(imageElement);

const overlay = new BoxOverlay({
  canvas: document.getElementById('overlay'),
  lineWidth: 3,
  fontSize: 14,
});

overlay.draw(result.detections, {
  originalWidth: imageElement.naturalWidth,
  originalHeight: imageElement.naturalHeight,
});
```

### With a loading bar

```typescript
const model = await loadModel('LibreRFDETRs', {
  onProgress: (p) => {
    progressBar.style.width = `${(p * 100).toFixed(0)}%`;
  },
});
```

## Model Zoo

14 pre-trained models on [HuggingFace](https://huggingface.co/LibreYOLO/libreyolo-web). Uses standard libreyolo naming — IDE autocompletion included.

### YOLOX — Fast, anchor-free

| Name | Input | Size | Speed* |
|------|-------|------|--------|
| `LibreYOLOXn` | 416 | 3.6MB | ~12ms |
| `LibreYOLOXt` | 416 | 19MB | ~15ms |
| `LibreYOLOXs` | 640 | 34MB | ~18ms |
| `LibreYOLOXm` | 640 | 97MB | ~30ms |
| `LibreYOLOXl` | 640 | 207MB | - |
| `LibreYOLOXx` | 640 | 378MB | - |

### YOLO9 — Anchor-free with DFL

| Name | Input | Size | Speed* |
|------|-------|------|--------|
| `LibreYOLO9t` | 640 | 8MB | ~28ms |
| `LibreYOLO9s` | 640 | 28MB | ~35ms |
| `LibreYOLO9m` | 640 | 77MB | - |
| `LibreYOLO9c` | 640 | 97MB | - |

### RF-DETR — Transformer, no NMS needed

| Name | Input | Size | Speed* |
|------|-------|------|--------|
| `LibreRFDETRn` | 384 | 103MB | ~75ms |
| `LibreRFDETRs` | 512 | 109MB | ~85ms |
| `LibreRFDETRm` | 576 | 115MB | - |
| `LibreRFDETRl` | 704 | 116MB | - |

*Inference time on WebGPU (M-series Mac). First run is slower due to shader compilation.

```typescript
import { listModels } from 'libreyolo-web';

// See all available models
for (const { name, model } of listModels()) {
  console.log(`${name} — ${model.description}`);
}
```

## Supported Model Families

| Family | `modelFamily` | Preprocessing | Postprocessing |
|--------|--------------|---------------|----------------|
| **YOLOv8/v11/v26** | `'yolo'` | Centered letterbox, /255, RGB | xywh→xyxy, per-class NMS |
| **YOLOX** | `'yolox'` | Top-left letterbox, 0-255, BGR | Objectness × class score, per-class NMS |
| **YOLO9** | `'yolo9'` | Direct resize, /255, RGB | xyxy direct, per-class NMS |
| **RF-DETR** | `'rfdetr'` | Direct resize, ImageNet norm, RGB | Sigmoid logits, top-K (no NMS), COCO91→80 mapping |

All families produce the same `DetectionResult`:

```typescript
interface DetectionResult {
  boxes: number[][];        // [[x1, y1, x2, y2], ...]
  scores: number[];         // [0.95, 0.87, ...]
  classes: number[];        // [0, 17, ...]
  numDetections: number;
  detections: Detection[];  // [{ classId, confidence, bbox, label }, ...]
}
```

## API Reference

### `loadModel(name, options?)`

Create and initialize a model. Accepts a zoo model name or URL/path.

```typescript
const model = await loadModel('LibreYOLOXs');
// or
const model = await loadModel('./model.onnx', { modelFamily: 'yolox', inputSize: 640 });
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `modelFamily` | `'auto' \| 'yolo' \| 'yolox' \| 'yolo9' \| 'rfdetr'` | `'auto'` | Model family (auto-set for zoo models) |
| `inputSize` | `number` | `640` | Model input resolution (auto-set for zoo models) |
| `confThres` | `number` | `0.25` | Confidence threshold |
| `iouThres` | `number` | `0.45` | NMS IoU threshold |
| `maxDet` | `number` | `300` | Maximum detections |
| `device` | `'auto' \| 'webgpu' \| 'wasm'` | `'auto'` | Backend selection |
| `classNames` | `string[]` | COCO 80 | Custom class names |
| `onProgress` | `(p: number) => void` | - | Download progress (0-1) |

### `model.predict(image, options?)`

Run detection. Returns `DetectionResult`.

### `model.detect(image, options?)`

Run detection. Returns `Detection[]`.

### `model.release()`

Free model resources.

### `BoxOverlay`

```typescript
const overlay = new BoxOverlay({
  canvas: HTMLCanvasElement,
  lineWidth?: number,        // Default: 2
  fontSize?: number,         // Default: 16
  showLabels?: boolean,      // Default: true
  showConfidence?: boolean,  // Default: true
  fillBoxes?: boolean,       // Semi-transparent fill (default: true)
  fillOpacity?: number,      // Fill opacity 0-1 (default: 0.1)
});

overlay.draw(detections, { originalWidth, originalHeight });
overlay.clear();
```

### Sample Image

A bundled test image is included for quick testing:

```typescript
import { SAMPLE_IMAGE, loadModel, BoxOverlay } from 'libreyolo-web';

const img = new Image();
img.src = SAMPLE_IMAGE;
await new Promise(r => img.onload = r);

const model = await loadModel('LibreYOLOXn');
const result = await model.predict(img);
```

## Backends

| Backend | Browser Coverage | Notes |
|---------|-----------------|-------|
| **WebGPU** | ~70%+ (Chrome, Edge, Firefox 147+, Safari 26) | GPU acceleration, fastest |
| **WASM** | ~98% | CPU fallback, works everywhere |

Default: WebGPU > WASM. Force a backend with `device: 'wasm'`.

## Using with Custom Models

### From Python libreyolo

```python
from libreyolo import LibreYOLO

model = LibreYOLO('LibreYOLOXs.pt')  # or LibreYOLO9t.pt, LibreRFDETRs.pt
model.export(format='onnx', simplify=True)
# RF-DETR needs: model.export(format='onnx', opset=17, simplify=True)
```

Then in the browser:

```typescript
const model = await loadModel('./LibreYOLOXs.onnx', { modelFamily: 'yolox', inputSize: 640 });
```

### From other frameworks

Any ONNX model with standard YOLO output format works. Set `modelFamily` to match your model's architecture:

- **Ultralytics YOLOv8/v11**: `modelFamily: 'yolo'`
- **Custom YOLOX**: `modelFamily: 'yolox'`
- **Custom YOLOv9**: `modelFamily: 'yolo9'`

## Vite / Bundler Configuration

```typescript
// vite.config.ts
export default defineConfig({
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
});
```

For WASM threading (optional, improves WASM performance):

```typescript
// vite.config.ts
server: {
  headers: {
    'Cross-Origin-Opener-Policy': 'same-origin',
    'Cross-Origin-Embedder-Policy': 'require-corp',
  },
},
```

## Development

```bash
git clone https://github.com/xuban-ceccon/libreyolo-web
cd libreyolo-web
npm install
npm run build       # Build
npm run typecheck   # Type checks
npm run test        # Tests
npm run example     # Demo at localhost:5173
```

## Publishing to npm

1. Create a **Classic Automation token** at [npmjs.com/settings/tokens](https://www.npmjs.com/settings/ehxuban11/tokens/) — this bypasses 2FA for CLI publishing
2. Under **Packages and scopes**, set permissions to **Read and Write**
3. Add the token to your `.env` file:
   ```
   NPM_TOKEN=npm_xxxxxxxxxxxx
   ```
4. Build and publish:
   ```bash
   source .env
   npm config set //registry.npmjs.org/:_authToken=$NPM_TOKEN
   npm run build
   npm publish --access public
   ```
5. Bump version for subsequent releases:
   ```bash
   npm version patch   # 0.0.1 → 0.0.2
   npm run build
   npm publish --access public
   ```

## License

MIT
