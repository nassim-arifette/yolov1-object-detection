/**
 * TypeScript port of protocol.py (target encoding), metrics.py (IoU, NMS) of this repository.
 * Parity with Python on every VOC2007 record is tested in test/yolo.test.ts.
 */
export const CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'] as const;

export type Box = [number, number, number, number]; // VOC 1-based inclusive pixels: x1, y1, x2, y2
export interface VocObject { class_id: number; bbox: Box; difficult: boolean }
export interface VocRecord { image_id: string; width: number; height: number; objects: VocObject[] }

/** VOC 1-based inclusive pixels -> continuous normalised (cx, cy, w, h). */
export function normalizedBox(o: VocObject, r: VocRecord): [number, number, number, number] {
  const [x1, y1, x2, y2] = o.bbox;
  return [(x1 - 1 + x2) / (2 * r.width), (y1 - 1 + y2) / (2 * r.height), (x2 - x1 + 1) / r.width, (y2 - y1 + 1) / r.height];
}

export interface CellTarget { index: number; row: number; col: number; class_id: number; x: number; y: number; w: number; h: number }

/** First non-difficult object in XML order wins its cell; every later one in the same cell is dropped. */
export function cellTargets(r: VocRecord, S = 7): { selected: Map<string, CellTarget>; dropped: number[]; cellOf: (number[] | null)[] } {
  const selected = new Map<string, CellTarget>();
  const dropped: number[] = [];
  const cellOf: (number[] | null)[] = [];
  r.objects.forEach((o, index) => {
    if (o.difficult) { cellOf.push(null); return; }
    const [x, y, w, h] = normalizedBox(o, r);
    const row = Math.min(S - 1, Math.floor(y * S)), col = Math.min(S - 1, Math.floor(x * S));
    cellOf.push([row, col]);
    const key = `${row},${col}`;
    if (selected.has(key)) dropped.push(index);
    else selected.set(key, { index, row, col, class_id: o.class_id, x: x * S - col, y: y * S - row, w, h });
  });
  return { selected, dropped, cellOf };
}

export function iou(a: Box, b: Box): number {
  const inter = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]) + 1) * Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]) + 1);
  const areaA = Math.max(0, a[2] - a[0] + 1) * Math.max(0, a[3] - a[1] + 1);
  const areaB = Math.max(0, b[2] - b[0] + 1) * Math.max(0, b[3] - b[1] + 1);
  const union = areaA + areaB - inter;
  return union > 0 ? inter / union : 0;
}

export interface Detection { class_id: number; score: number; bbox: Box; id?: number }

/** Class-specific greedy NMS, as in metrics.nms (one image). */
export function nms(dets: Detection[], threshold = 0.5): Detection[] {
  const kept: Detection[] = [];
  const groups = new Map<number, Detection[]>();
  for (const d of dets) { if (!groups.has(d.class_id)) groups.set(d.class_id, []); groups.get(d.class_id)!.push(d); }
  for (const g of groups.values()) {
    let pending = [...g].sort((a, b) => b.score - a.score);
    while (pending.length) {
      const chosen = pending.shift()!;
      kept.push(chosen);
      pending = pending.filter((d) => iou(chosen.bbox, d.bbox) <= threshold);
    }
  }
  return kept;
}
