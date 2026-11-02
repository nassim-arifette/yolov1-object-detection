import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { cellTargets, iou, nms, type Detection, type VocRecord } from '../src/yolo';

const records: (VocRecord & { dropped: number[] })[] = JSON.parse(readFileSync(new URL('./fixtures/test_targets.json', import.meta.url), 'utf8'));

describe('TypeScript grid encoder matches protocol.cell_targets', () => {
  it('drops exactly the same objects on all 4,952 VOC2007 test images', () => {
    expect(records.length).toBe(4952);
    for (const r of records) expect(cellTargets(r).dropped, r.image_id).toEqual(r.dropped);
  });
});

describe('IoU and NMS follow metrics.py', () => {
  it('uses 1-based inclusive pixels', () => {
    expect(iou([1, 1, 10, 10], [1, 1, 10, 10])).toBe(1);
    expect(iou([1, 1, 10, 10], [11, 1, 20, 10])).toBe(0);
    expect(iou([1, 1, 10, 10], [6, 1, 15, 10])).toBeCloseTo(50 / 150, 12);
  });
  it('suppresses only within a class', () => {
    const d: Detection[] = [
      { class_id: 0, score: 0.9, bbox: [1, 1, 10, 10] },
      { class_id: 0, score: 0.8, bbox: [2, 2, 11, 11] },
      { class_id: 1, score: 0.7, bbox: [2, 2, 11, 11] },
    ];
    expect(nms(d, 0.5).map((x) => x.score).sort()).toEqual([0.7, 0.9]);
  });
});
