/* ========================================================================= */
/* SECTION: Types & Interfaces                                               */
/* ========================================================================= */

export interface RNG {
  next(): number;
}

/* ========================================================================= */
/* SECTION: Core Functions                                                   */
/* ========================================================================= */

export function createRng(seed: number): RNG {
  // Mulberry32 PRNG
  return {
    next: function () {
      let t = (seed += 0x6d2b79f5);
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    },
  };
}

export function randInt(rng: RNG, lo: number, hi: number): number {
  return Math.floor(rng.next() * (hi - lo + 1)) + lo;
}

export function clamp(value: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, value));
}
