export function iou(boxA: number[], boxB: number[]): number {
  const x1 = Math.max(boxA[0], boxB[0]);
  const y1 = Math.max(boxA[1], boxB[1]);
  const x2 = Math.min(boxA[2], boxB[2]);
  const y2 = Math.min(boxA[3], boxB[3]);

  const intersectionArea = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]);
  const boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]);

  const unionArea = boxAArea + boxBArea - intersectionArea;

  return unionArea === 0 ? 0 : intersectionArea / unionArea;
}

export function xywh2xyxy(x: number, y: number, w: number, h: number): [number, number, number, number] {
  return [x - w / 2, y - h / 2, x + w / 2, y + h / 2];
}

export function clamp(val: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, val));
}

