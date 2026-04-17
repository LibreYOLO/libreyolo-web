# LibreYOLO Web

[![npm](https://img.shields.io/npm/v/libreyolo-web)](https://www.npmjs.com/package/libreyolo-web)
[![CI](https://github.com/LibreYOLO/libreyolo-web/actions/workflows/ci.yml/badge.svg)](https://github.com/LibreYOLO/libreyolo-web/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-libreyolo.com-blue)](https://www.libreyolo.com/docs)

Object detection in the browser. 100% MIT Licensed.

The web companion to [libreyolo](https://github.com/LibreYOLO/libreyolo). Same models, same license, no AGPL. YOLOX, YOLO9, RF-DETR, and YOLOv8/v11/v26 running in the browser on WebGPU or WASM.

![LibreYOLO Web Detection](assets/parkour.jpg)

## Install

```bash
npm install libreyolo-web onnxruntime-web
```

## Quick Start

```typescript
import { loadModel } from 'libreyolo-web';

const model = await loadModel('LibreYOLOXn');
const result = await model.predict(imageElement);

console.log(`Found ${result.numDetections} objects`);
```

That's it. The model auto-downloads from HuggingFace and handles its own preprocessing.

## Drawing Boxes

```typescript
import { loadModel, BoxOverlay } from 'libreyolo-web';

const model = await loadModel('LibreYOLO9t');
const result = await model.predict(imageElement);

new BoxOverlay({ canvas: myCanvas }).draw(result.detections);
```

## Model Zoo

14 pre-trained models, ready to go: `LibreYOLOXn`, `LibreYOLO9s`, `LibreRFDETRm`, and friends. Full list and benchmarks at [libreyolo.com/docs](https://www.libreyolo.com/docs).

```typescript
import { listModels } from 'libreyolo-web';
listModels().forEach(({ name }) => console.log(name));
```

## Your Own Model

```typescript
const model = await loadModel('./my_model.onnx', {
  modelFamily: 'yolox',  // 'yolo' | 'yolox' | 'yolo9' | 'rfdetr'
  inputSize: 640,
});
```

Export from the Python sister project:

```python
from libreyolo import LibreYOLO
LibreYOLO('LibreYOLOXs.pt').export(format='onnx', simplify=True)
```

## Docs

Everything else (full API reference, bundler config, backend tuning) lives at [libreyolo.com/docs](https://www.libreyolo.com/docs).

## License

MIT. Truly MIT. No AGPL.
