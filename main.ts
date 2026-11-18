// GTA VI launch-day Twitch viewers forecaster.
// Reads one case per *.in file (schema makefun/twitch-ccu-case/v1) and writes
// one forecast per case: for hours 2..24 of the release day, five quantiles
// (5%, 25%, 50%, 75%, 95%) of that hour's Twitch viewer count.
//
// Model: for every past launch in the case's comparables, take the log ratio
// of each release-day hour's viewers to that launch's hour-1 print. For each
// hour, regress that log ratio on two features of the launch -- the log of its
// hour-1 print and the log of its mean viewers in the 7 days before release
// (a launch that already had a big hour 1, or a big pre-release audience,
// grows less through the day). Console-only launches count triple, because
// GTA VI is a console launch with a rolling-midnight unlock, unlike PC
// launches whose hour 1 is the global peak. Coefficients and residual spreads
// are smoothed across neighbouring hours, and the features are clipped to the
// console launches' range so an unprecedented launch size saturates instead
// of extrapolating to a falling curve. The spread is the residual spread
// widened by the regression leverage of the target case, then scaled by 0.7
// (calibrated on the four practice launches). Everything is deterministic:
// no clock, no randomness, no network, standard library only.

import { readdirSync, readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";

const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
const Z = [-1.6448536269514722, -0.6744897501960817, 0, 0.6744897501960817, 1.6448536269514722];
const CONSOLE_WEIGHT = 3;
const SPREAD_SCALE = 0.7;
const MIN_SPREAD = 0.08;
const SMOOTH_HALF = 2;

type Row = {
  logP1: number;
  logPre: number;
  console: boolean;
  // log ratio of hour h's viewers to hour-1 viewers, h = 2..24 (index 0 = hour 2)
  logRatio: (number | null)[];
};

function log(x: number): number {
  return Math.log(Math.max(x, 1));
}

// Solve a small dense linear system by Gaussian elimination with partial
// pivoting. Returns null when the matrix is (near-)singular.
function solve(A: number[][], b: number[]): number[] | null {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
    if (Math.abs(M[piv][col]) < 1e-12) return null;
    [M[col], M[piv]] = [M[piv], M[col]];
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = M[r][col] / M[col][col];
      for (let c = col; c <= n; c++) M[r][c] -= f * M[col][c];
    }
  }
  return M.map((row, i) => row[n] / M[i][i]);
}

// Invert a small dense matrix via Gauss-Jordan. Returns null if singular.
function invert(A: number[][]): number[][] | null {
  const n = A.length;
  const M = A.map((row, i) => [...row, ...Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))]);
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
    if (Math.abs(M[piv][col]) < 1e-12) return null;
    [M[col], M[piv]] = [M[piv], M[col]];
    const pv = M[col][col];
    for (let c = 0; c < 2 * n; c++) M[col][c] /= pv;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = M[r][col];
      for (let c = 0; c < 2 * n; c++) M[r][c] -= f * M[col][c];
    }
  }
  return M.map((row) => row.slice(n));
}

function mean(xs: number[]): number {
  return xs.reduce((a, b) => a + b, 0) / xs.length;
}

function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  const n = s.length;
  return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
}

// Moving average over neighbouring hours; edges use the smaller window.
function smooth(vals: number[]): number[] {
  return vals.map((_, i) => {
    const lo = Math.max(0, i - SMOOTH_HALF);
    const hi = Math.min(vals.length, i + SMOOTH_HALF + 1);
    return mean(vals.slice(lo, hi));
  });
}

function parseCase(text: string): any {
  return JSON.parse(text);
}

// Mean viewers of the points in the 7 days before dayStart (ISO strings).
function preReleaseMean(points: any[], dayStart: string): number {
  const start = Date.parse(dayStart);
  const weekAgo = start - 7 * 24 * 3600 * 1000;
  const vals: number[] = [];
  for (const p of points || []) {
    const t = Date.parse(p.from ?? p.at ?? "");
    if (Number.isNaN(t)) continue;
    if (t >= weekAgo && t < start && typeof p.viewers === "number" && p.viewers > 0) {
      vals.push(p.viewers);
    }
  }
  return vals.length ? mean(vals) : 1;
}

function buildRows(kase: any): Row[] {
  const rows: Row[] = [];
  for (const comp of kase.comparables ?? []) {
    if (typeof comp.unlock === "string" && comp.unlock.includes("Not a launch")) continue;
    const hourly: any[] = comp.hourly ?? [];
    const day = String(comp.day_start ?? "").slice(0, 10);
    let idx = -1;
    for (let i = 0; i < hourly.length; i++) {
      if (String(hourly[i].from ?? "").startsWith(day)) {
        idx = i;
        break;
      }
    }
    if (idx < 0) continue;
    const v1 = hourly[idx].viewers;
    if (typeof v1 !== "number" || v1 <= 0) continue;
    const pre: number[] = [];
    for (let i = Math.max(0, idx - 168); i < idx; i++) {
      const v = hourly[i].viewers;
      if (typeof v === "number" && v > 0) pre.push(v);
    }
    const logRatio: (number | null)[] = [];
    for (let h = 2; h <= 24; h++) {
      const j = idx + h - 1;
      const v = j < hourly.length ? hourly[j].viewers : null;
      logRatio.push(typeof v === "number" && v > 0 ? Math.log(v / v1) : null);
    }
    rows.push({
      logP1: log(v1),
      logPre: log(pre.length ? mean(pre) : 1),
      console: typeof comp.platforms === "string" ? !comp.platforms.includes("PC") : false,
      logRatio,
    });
  }
  return rows;
}

// Weighted least squares of y on [1, x1, x2]; returns beta and the weighted
// normal-equations matrix (for the leverage), or null when singular.
function wols(
  xs: number[][],
  ys: number[],
  ws: number[],
): { beta: number[]; ata: number[][] } | null {
  const k = 3;
  const ata: number[][] = Array.from({ length: k }, () => new Array(k).fill(0));
  const aty: number[] = new Array(k).fill(0);
  for (let t = 0; t < xs.length; t++) {
    const a = [1, xs[t][0], xs[t][1]];
    for (let i = 0; i < k; i++) {
      aty[i] += ws[t] * a[i] * ys[t];
      for (let j = 0; j < k; j++) ata[i][j] += ws[t] * a[i] * a[j];
    }
  }
  const beta = solve(ata, aty);
  if (!beta) return null;
  return { beta, ata };
}

function forecast(kase: any): { hours: { hour: number; q: number[] }[] } {
  const p1 = Math.max(1, kase.known?.[0]?.viewers ?? 1);
  const rows = buildRows(kase);
  const dayStart = String(kase.day_start ?? "");
  const targetPre = preReleaseMean(kase.history_hourly ?? [], dayStart);
  let targetLogP1 = log(p1);
  let targetLogPre = log(targetPre);

  // Clip the target's features to the console launches' range, so a launch
  // bigger than any comparable saturates instead of extrapolating.
  const consoles = rows.filter((r) => r.console);
  if (consoles.length) {
    const loP = Math.min(...consoles.map((r) => r.logP1));
    const hiP = Math.max(...consoles.map((r) => r.logP1));
    const loR = Math.min(...consoles.map((r) => r.logPre));
    const hiR = Math.max(...consoles.map((r) => r.logPre));
    targetLogP1 = Math.min(Math.max(targetLogP1, loP), hiP);
    targetLogPre = Math.min(Math.max(targetLogPre, loR), hiR);
  }

  // Fallback shape: the flat house baseline (median = hour-1 print, spread
  // widening over the day), used when a hour has too little training data.
  const fallbackMu = 0;
  const fallbackS = (h: number) => 0.15 * Math.sqrt(h - 1);

  const betas: (number[] | null)[] = [];
  const sds: number[] = [];
  const levs: number[] = [];
  for (let i = 0; i < 23; i++) {
    const pts = rows.filter((r) => r.logRatio[i] !== null);
    if (pts.length < 4) {
      betas.push(null);
      sds.push(fallbackS(i + 2));
      levs.push(0);
      continue;
    }
    const xs = pts.map((r) => [r.logP1, r.logPre]);
    const ys = pts.map((r) => r.logRatio[i] as number);
    const ws = pts.map((r) => (r.console ? CONSOLE_WEIGHT : 1));
    const fit = wols(xs, ys, ws);
    if (!fit) {
      betas.push(null);
      sds.push(fallbackS(i + 2));
      levs.push(0);
      continue;
    }
    const { beta, ata } = fit;
    let s2 = 0;
    let wsum = 0;
    for (let t = 0; t < pts.length; t++) {
      const pred = beta[0] + beta[1] * xs[t][0] + beta[2] * xs[t][1];
      const resid = ys[t] - pred;
      s2 += ws[t] * resid * resid;
      wsum += ws[t];
    }
    const sd = Math.sqrt(s2 / Math.max(1, wsum - 3));
    let lev = 0;
    const inv = invert(ata);
    if (inv) {
      const x0 = [1, targetLogP1, targetLogPre];
      let h0 = 0;
      for (let a = 0; a < 3; a++)
        for (let b = 0; b < 3; b++) h0 += x0[a] * inv[a][b] * x0[b];
      lev = Math.max(0, h0);
    }
    betas.push(beta);
    sds.push(sd);
    levs.push(lev);
  }

  // Smooth coefficients and spreads across neighbouring hours so the
  // forecast curve has no jagged jumps from independent per-hour fits.
  const b0 = smooth(betas.map((b) => (b ? b[0] : fallbackMu)));
  const b1 = smooth(betas.map((b) => (b ? b[1] : 0)));
  const b2 = smooth(betas.map((b) => (b ? b[2] : 0)));
  const sSmooth = smooth(sds);
  const lSmooth = smooth(levs);

  const hours: { hour: number; q: number[] }[] = [];
  for (let i = 0; i < 23; i++) {
    const h = i + 2;
    const hasFit = betas[i] !== null;
    const mu = hasFit ? b0[i] + b1[i] * targetLogP1 + b2[i] * targetLogPre : fallbackMu;
    const raw = hasFit ? sSmooth[i] : fallbackS(h);
    const s = Math.max(
      hasFit ? Math.sqrt(raw * raw * (1 + lSmooth[i])) * SPREAD_SCALE : raw,
      MIN_SPREAD,
    );
    const q = Z.map((z) => {
      const v = p1 * Math.exp(mu + z * s);
      const rounded = Math.round(Math.min(Math.max(v, 1), 1e9) * 100) / 100;
      return rounded;
    });
    // Quantiles must be non-decreasing; rounding can only make neighbours equal.
    for (let k = 1; k < q.length; k++) if (q[k] < q[k - 1]) q[k] = q[k - 1];
    hours.push({ hour: h, q });
  }
  return { hours };
}

function main(): void {
  const inputDir = process.env.MAKEFUN_INPUT_DIR || "/input";
  const outputDir = process.env.MAKEFUN_OUTPUT_DIR || "/output";
  mkdirSync(outputDir, { recursive: true });
  const files = readdirSync(inputDir)
    .filter((f) => f.endsWith(".in"))
    .sort();
  for (const file of files) {
    const kase = parseCase(readFileSync(join(inputDir, file), "utf8"));
    const out = forecast(kase);
    const id = file.replace(/\.in$/, "");
    writeFileSync(join(outputDir, `${id}.out`), JSON.stringify(out));
  }
}

main();
