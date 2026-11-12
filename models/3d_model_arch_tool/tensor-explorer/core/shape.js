/** Small, safe symbolic dimension AST. No eval or string execution. */
export const dim = {
  sym: name => ({ op: "symbol", name }),
  c: value => ({ op: "const", value }),
  add: (...args) => ({ op: "add", args }),
  mul: (...args) => ({ op: "mul", args }),
  div: (a, b) => ({ op: "div", args: [a, b] }),
};

export function normalizeExpr(expr) {
  if (typeof expr === "number") return dim.c(expr);
  if (typeof expr === "string") return dim.sym(expr);
  if (!expr || typeof expr !== "object") throw new TypeError("Invalid dimension expression");
  return expr;
}

export function evaluateDim(expr, env = {}) {
  expr = normalizeExpr(expr);
  if (expr.op === "const") return expr.value;
  if (expr.op === "symbol") {
    const value = env[expr.name];
    return Number.isFinite(value) ? value : undefined;
  }
  const values = expr.args.map(x => evaluateDim(x, env));
  if (values.some(x => x === undefined)) return undefined;
  if (expr.op === "add") return values.reduce((a, b) => a + b, 0);
  if (expr.op === "mul") return values.reduce((a, b) => a * b, 1);
  if (expr.op === "div") {
    if (values[1] === 0) throw new RangeError("Dimension division by zero");
    return values[0] / values[1];
  }
  throw new TypeError(`Unsupported dimension operation: ${expr.op}`);
}

export function formatDim(expr) {
  expr = normalizeExpr(expr);
  if (expr.op === "const") return String(expr.value);
  if (expr.op === "symbol") return expr.name;
  const [a, b] = expr.args;
  if (expr.op === "mul") return expr.args.map(formatDim).join(" × ");
  if (expr.op === "add") return expr.args.map(formatDim).join(" + ");
  if (expr.op === "div") return `(${formatDim(a)} / ${formatDim(b)})`;
  return "?";
}

export function evaluateShape(shape, env = {}) {
  return shape.map(x => evaluateDim(x, env));
}

export function formatShape(shape, env = null) {
  const values = env ? evaluateShape(shape, env) : shape.map(() => undefined);
  return `[${shape.map((x, i) => values[i] === undefined ? formatDim(x) : values[i].toLocaleString("en-US")).join(", ")}]`;
}

export function numel(shape, env = {}) {
  const values = evaluateShape(shape, env);
  if (values.some(x => x === undefined)) return undefined;
  return values.reduce((a, b) => a * b, 1);
}

export function assertPositiveInteger(name, value) {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new RangeError(`${name} must be a positive safe integer; got ${value}`);
  }
  return value;
}

export function validateDivisibility(numerator, denominator, label = "dimension") {
  if (!Number.isInteger(numerator) || !Number.isInteger(denominator) || denominator <= 0 || numerator % denominator !== 0) {
    throw new RangeError(`${label}: ${numerator} must be divisible by ${denominator}`);
  }
  return true;
}
