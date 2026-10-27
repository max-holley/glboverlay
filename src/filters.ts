/** One Euro filter (Casiez et al. 2012): low jitter when still, low lag when moving. */
export class OneEuro {
  private xPrev = 0;
  private dxPrev = 0;
  private tPrev = -1;

  constructor(
    public minCutoff = 1,
    public beta = 0,
    public dCutoff = 1,
  ) {}

  reset(): void {
    this.tPrev = -1;
  }

  filter(x: number, t: number): number {
    if (this.tPrev < 0) {
      this.xPrev = x;
      this.dxPrev = 0;
      this.tPrev = t;
      return x;
    }
    const dt = Math.max(t - this.tPrev, 1e-3);
    this.tPrev = t;
    const dx = (x - this.xPrev) / dt;
    const aD = alpha(dt, this.dCutoff);
    this.dxPrev = aD * dx + (1 - aD) * this.dxPrev;
    const cutoff = this.minCutoff + this.beta * Math.abs(this.dxPrev);
    const a = alpha(dt, cutoff);
    this.xPrev = a * x + (1 - a) * this.xPrev;
    return this.xPrev;
  }
}

function alpha(dt: number, cutoff: number): number {
  const tau = 1 / (2 * Math.PI * cutoff);
  return 1 / (1 + tau / dt);
}
