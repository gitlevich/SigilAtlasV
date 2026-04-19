/**
 * RelevanceFilter — SigilML expression mirrored from Python.
 *
 * A membrane separating intent (the expression) from the resulting slice.
 * The same AST feeds both `/slice` (image_ids) and `/spacelike` (attractors
 * and proximity subspace derived from the same atoms).
 */

import type { Attractor, ContrastControl, RangeFilter } from "./types";

// --- Atoms ---

export interface ThingAtom {
  type: "thing";
  name: string;
}

export interface TargetImageAtom {
  type: "target_image";
  image_id: string;
}

export interface ContrastAtom {
  type: "contrast";
  pole_a: string;
  pole_b: string;
  band_min: number;
  band_max: number;
}

export interface RangeAtom {
  type: "range";
  dimension: string;
  min: number;
  max: number;
}

// --- Composites ---

export interface AndNode {
  type: "and";
  children: Expression[];
}

export interface OrNode {
  type: "or";
  children: Expression[];
}

export interface NotNode {
  type: "not";
  child: Expression;
}

export type Atom = ThingAtom | TargetImageAtom | ContrastAtom | RangeAtom;
export type Expression = Atom | AndNode | OrNode | NotNode;

// --- Builders ---

export function thing(name: string): ThingAtom {
  return { type: "thing", name };
}

export function targetImage(image_id: string): TargetImageAtom {
  return { type: "target_image", image_id };
}

export function contrast(c: ContrastControl): ContrastAtom {
  return {
    type: "contrast",
    pole_a: c.pole_a,
    pole_b: c.pole_b,
    band_min: c.band_min,
    band_max: c.band_max,
  };
}

export function range(r: RangeFilter): RangeAtom {
  return { type: "range", dimension: r.dimension, min: r.min, max: r.max };
}

export function and(children: Expression[]): AndNode {
  return { type: "and", children };
}

/**
 * Compose the RelevanceFilter from the UI's current state.
 *
 * Two possible sources for the attractor sub-tree, mutually exclusive:
 *   - `attractorExpression` (a SigilML boolean expression parsed from text), or
 *   - flat `attractors` pills, which combine as a single peer-group under AND.
 *
 * ContrastControl widgets and range sliders are always ANDed on top.
 * Returns null when nothing constrains the slice — the whole corpus survives.
 */
export function buildFilter(args: {
  attractors: Attractor[];
  attractorExpression: Expression | null;
  contrastControls: ContrastControl[];
  rangeFilters: RangeFilter[];
}): Expression | null {
  const children: Expression[] = [];

  if (args.attractorExpression) {
    children.push(args.attractorExpression);
  } else {
    for (const a of args.attractors) {
      children.push(
        a.kind === "thing" ? thing(a.ref) : targetImage(a.ref),
      );
    }
  }
  for (const cc of args.contrastControls) {
    children.push(contrast(cc));
  }
  for (const rf of args.rangeFilters) {
    children.push(range(rf));
  }

  if (children.length === 0) return null;
  if (children.length === 1) return children[0];
  return and(children);
}
