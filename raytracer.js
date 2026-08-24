/**
 * Reference Raytracer - JavaScript (Node.js) Port
 * =================================================
 * Line-for-line port of raytracer.py. Same algorithm, same function
 * names, same loop order -- only the syntax changes.
 *
 * Fairness rules (see raytracer.py for full explanation):
 *   1. Scene is hardcoded from scene_data.js (generated from the frozen
 *      scene_spec.json) -- no parsing happens inside the timed render loop.
 *   2. Timing is split into scene setup / render / file write. Only
 *      "render" counts toward the cross-language benchmark.
 *   3. No worker threads, no external raytracing libraries. Plain
 *      synchronous scalar loops, same as the Python version.
 */

const fs = require("fs");
const { CAMERA, SPHERES } = require("./scene_data.js");

// ---------------------------------------------------------------------------
// Vec3
// ---------------------------------------------------------------------------

class Vec3 {
  constructor(x = 0, y = 0, z = 0) {
    this.x = x;
    this.y = y;
    this.z = z;
  }
  add(o) { return new Vec3(this.x + o.x, this.y + o.y, this.z + o.z); }
  sub(o) { return new Vec3(this.x - o.x, this.y - o.y, this.z - o.z); }
  neg() { return new Vec3(-this.x, -this.y, -this.z); }
  mulVec(o) { return new Vec3(this.x * o.x, this.y * o.y, this.z * o.z); }
  mul(t) { return new Vec3(this.x * t, this.y * t, this.z * t); }
  div(t) { return this.mul(1.0 / t); }
  lengthSquared() { return this.x * this.x + this.y * this.y + this.z * this.z; }
  length() { return Math.sqrt(this.lengthSquared()); }
  unit() { return this.div(this.length()); }
  nearZero() {
    const eps = 1e-8;
    return Math.abs(this.x) < eps && Math.abs(this.y) < eps && Math.abs(this.z) < eps;
  }
}

function dot(a, b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

function cross(a, b) {
  return new Vec3(
    a.y * b.z - a.z * b.y,
    a.z * b.x - a.x * b.z,
    a.x * b.y - a.y * b.x
  );
}

function reflect(v, n) { return v.sub(n.mul(2.0 * dot(v, n))); }

function refract(uv, n, etaiOverEtat) {
  const cosTheta = Math.min(dot(uv.neg(), n), 1.0);
  const rOutPerp = uv.add(n.mul(cosTheta)).mul(etaiOverEtat);
  const rOutParallel = n.mul(-Math.sqrt(Math.abs(1.0 - rOutPerp.lengthSquared())));
  return rOutPerp.add(rOutParallel);
}

function randRange(a, b) { return a + Math.random() * (b - a); }

function randomVec(a = 0.0, b = 1.0) {
  return new Vec3(randRange(a, b), randRange(a, b), randRange(a, b));
}

function randomInUnitSphere() {
  while (true) {
    const p = randomVec(-1.0, 1.0);
    if (p.lengthSquared() < 1.0) return p;
  }
}

function randomUnitVector() { return randomInUnitSphere().unit(); }

function randomInUnitDisk() {
  while (true) {
    const p = new Vec3(randRange(-1, 1), randRange(-1, 1), 0.0);
    if (p.lengthSquared() < 1.0) return p;
  }
}

// ---------------------------------------------------------------------------
// Ray
// ---------------------------------------------------------------------------

class Ray {
  constructor(origin, direction) {
    this.origin = origin;
    this.direction = direction;
  }
  at(t) { return this.origin.add(this.direction.mul(t)); }
}

// ---------------------------------------------------------------------------
// Materials
// ---------------------------------------------------------------------------

class Lambertian {
  constructor(albedo) { this.albedo = albedo; }
  scatter(rIn, rec) {
    let scatterDir = rec.normal.add(randomUnitVector());
    if (scatterDir.nearZero()) scatterDir = rec.normal;
    return { didScatter: true, attenuation: this.albedo, scattered: new Ray(rec.p, scatterDir) };
  }
}

class Metal {
  constructor(albedo, fuzz) { this.albedo = albedo; this.fuzz = Math.min(fuzz, 1.0); }
  scatter(rIn, rec) {
    let reflected = reflect(rIn.direction.unit(), rec.normal);
    reflected = reflected.add(randomInUnitSphere().mul(this.fuzz));
    const scattered = new Ray(rec.p, reflected);
    const didScatter = dot(scattered.direction, rec.normal) > 0;
    return { didScatter, attenuation: this.albedo, scattered };
  }
}

class Dielectric {
  constructor(refIdx) { this.refIdx = refIdx; }
  static reflectance(cosine, refIdx) {
    let r0 = (1 - refIdx) / (1 + refIdx);
    r0 = r0 * r0;
    return r0 + (1 - r0) * Math.pow(1 - cosine, 5);
  }
  scatter(rIn, rec) {
    const attenuation = new Vec3(1.0, 1.0, 1.0);
    const refractionRatio = rec.frontFace ? 1.0 / this.refIdx : this.refIdx;

    const unitDir = rIn.direction.unit();
    const cosTheta = Math.min(dot(unitDir.neg(), rec.normal), 1.0);
    const sinTheta = Math.sqrt(1.0 - cosTheta * cosTheta);

    const cannotRefract = refractionRatio * sinTheta > 1.0;
    let direction;
    if (cannotRefract || Dielectric.reflectance(cosTheta, refractionRatio) > Math.random()) {
      direction = reflect(unitDir, rec.normal);
    } else {
      direction = refract(unitDir, rec.normal, refractionRatio);
    }
    return { didScatter: true, attenuation, scattered: new Ray(rec.p, direction) };
  }
}

class CheckerLambertian {
  constructor(odd, even, scale = 1.0) { this.odd = odd; this.even = even; this.scale = scale; }
  scatter(rIn, rec) {
    const s = this.scale;
    const sines = Math.sin(s * rec.p.x) * Math.sin(s * rec.p.y) * Math.sin(s * rec.p.z);
    const albedo = sines < 0 ? this.odd : this.even;
    let scatterDir = rec.normal.add(randomUnitVector());
    if (scatterDir.nearZero()) scatterDir = rec.normal;
    return { didScatter: true, attenuation: albedo, scattered: new Ray(rec.p, scatterDir) };
  }
}

function buildMaterial(m) {
  switch (m.type) {
    case "lambertian":
      return new Lambertian(new Vec3(...m.albedo));
    case "metal":
      return new Metal(new Vec3(...m.albedo), m.fuzz);
    case "dielectric":
      return new Dielectric(m.refIdx);
    case "checker":
      return new CheckerLambertian(new Vec3(...m.odd), new Vec3(...m.even), m.scale);
    default:
      throw new Error("unknown material type: " + m.type);
  }
}

// ---------------------------------------------------------------------------
// Hittables
// ---------------------------------------------------------------------------

class Sphere {
  constructor(center, radius, material) {
    this.center = center;
    this.radius = radius;
    this.material = material;
  }
  hit(r, tMin, tMax) {
    const oc = r.origin.sub(this.center);
    const a = r.direction.lengthSquared();
    const halfB = dot(oc, r.direction);
    const c = oc.lengthSquared() - this.radius * this.radius;
    const discriminant = halfB * halfB - a * c;
    if (discriminant < 0) return null;
    const sqrtd = Math.sqrt(discriminant);

    let root = (-halfB - sqrtd) / a;
    if (root < tMin || root > tMax) {
      root = (-halfB + sqrtd) / a;
      if (root < tMin || root > tMax) return null;
    }

    const p = r.at(root);
    const outwardNormal = p.sub(this.center).div(this.radius);
    const frontFace = dot(r.direction, outwardNormal) < 0;
    return {
      t: root,
      p,
      frontFace,
      normal: frontFace ? outwardNormal : outwardNormal.neg(),
      material: this.material,
    };
  }
}

function hitWorld(spheres, r, tMin, tMax) {
  let closest = tMax;
  let hitRec = null;
  for (const s of spheres) {
    const rec = s.hit(r, tMin, closest);
    if (rec !== null) {
      closest = rec.t;
      hitRec = rec;
    }
  }
  return hitRec;
}

// ---------------------------------------------------------------------------
// Camera (thin-lens model)
// ---------------------------------------------------------------------------

class Camera {
  constructor(lookfrom, lookat, vup, vfov, aspectRatio, aperture, focusDist) {
    const theta = (vfov * Math.PI) / 180;
    const h = Math.tan(theta / 2);
    const viewportHeight = 2.0 * h;
    const viewportWidth = aspectRatio * viewportHeight;

    this.w = lookfrom.sub(lookat).unit();
    this.u = cross(vup, this.w).unit();
    this.v = cross(this.w, this.u);

    this.origin = lookfrom;
    this.horizontal = this.u.mul(viewportWidth * focusDist);
    this.vertical = this.v.mul(viewportHeight * focusDist);
    this.lowerLeft = this.origin
      .sub(this.horizontal.div(2))
      .sub(this.vertical.div(2))
      .sub(this.w.mul(focusDist));
    this.lensRadius = aperture / 2;
  }
  getRay(s, t) {
    const rd = randomInUnitDisk().mul(this.lensRadius);
    const offset = this.u.mul(rd.x).add(this.v.mul(rd.y));
    const origin = this.origin.add(offset);
    const direction = this.lowerLeft
      .add(this.horizontal.mul(s))
      .add(this.vertical.mul(t))
      .sub(this.origin)
      .sub(offset);
    return new Ray(origin, direction);
  }
}

// ---------------------------------------------------------------------------
// Scene -- hardcoded from scene_data.js (frozen scene_spec.json)
// ---------------------------------------------------------------------------

function buildScene() {
  return SPHERES.map(
    (s) => new Sphere(new Vec3(...s.center), s.radius, buildMaterial(s.material))
  );
}

// ---------------------------------------------------------------------------
// Shading
// ---------------------------------------------------------------------------

function skyColor(r) {
  const unitDir = r.direction.unit();
  const t = 0.5 * (unitDir.y + 1.0);
  return new Vec3(1, 1, 1).mul(1.0 - t).add(new Vec3(0.5, 0.7, 1.0).mul(t));
}

function rayColor(r, spheres, depth) {
  if (depth <= 0) return new Vec3(0, 0, 0);

  const rec = hitWorld(spheres, r, 0.001, Infinity);
  if (rec !== null) {
    const { didScatter, attenuation, scattered } = rec.material.scatter(r, rec);
    if (didScatter) {
      return attenuation.mulVec(rayColor(scattered, spheres, depth - 1));
    }
    return new Vec3(0, 0, 0);
  }
  return skyColor(r);
}

// ---------------------------------------------------------------------------
// Main render
// ---------------------------------------------------------------------------

function writePPM(path, width, height, rows) {
  const stream = fs.createWriteStream(path);
  stream.write(`P3\n${width} ${height}\n255\n`);
  stream.write(rows.join("\n"));
  stream.write("\n");
  stream.end();
}

function render(width, height, samplesPerPixel, maxDepth, outPath) {
  const t0 = process.hrtime.bigint();

  const aspectRatio = width / height;
  const cam = new Camera(
    new Vec3(...CAMERA.lookfrom),
    new Vec3(...CAMERA.lookat),
    new Vec3(...CAMERA.vup),
    CAMERA.vfov,
    aspectRatio,
    CAMERA.aperture,
    CAMERA.focus_dist
  );
  const spheres = buildScene();

  const t1 = process.hrtime.bigint();

  const rows = [];
  for (let j = height - 1; j >= 0; j--) {
    const row = [];
    for (let i = 0; i < width; i++) {
      let color = new Vec3(0, 0, 0);
      for (let s = 0; s < samplesPerPixel; s++) {
        const su = (i + Math.random()) / (width - 1);
        const sv = (j + Math.random()) / (height - 1);
        color = color.add(rayColor(cam.getRay(su, sv), spheres, maxDepth));
      }

      const scale = 1.0 / samplesPerPixel;
      const r_ = Math.sqrt(color.x * scale);
      const g_ = Math.sqrt(color.y * scale);
      const b_ = Math.sqrt(color.z * scale);

      const ir = Math.floor(256 * Math.min(Math.max(r_, 0.0), 0.999));
      const ig = Math.floor(256 * Math.min(Math.max(g_, 0.0), 0.999));
      const ib = Math.floor(256 * Math.min(Math.max(b_, 0.0), 0.999));
      row.push(`${ir} ${ig} ${ib}`);
    }
    rows.push(row.join(" "));

    process.stderr.write(`\rScanlines remaining: ${String(j).padStart(4)} `);
  }

  const t2 = process.hrtime.bigint();

  writePPM(outPath, width, height, rows);

  const t3 = process.hrtime.bigint();

  process.stderr.write("\nDone.\n");
  const ns = (a, b) => Number(b - a) / 1e9;
  console.log(`scene_setup_seconds: ${ns(t0, t1).toFixed(4)}`);
  console.log(`render_seconds:      ${ns(t1, t2).toFixed(4)}`);
  console.log(`file_write_seconds:  ${ns(t2, t3).toFixed(4)}`);
  console.log(`total_seconds:       ${ns(t0, t3).toFixed(4)}`);
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

function parseArgs() {
  const args = { width: 400, height: 225, samples: 20, depth: 20, out: "render.ppm" };
  const argv = process.argv.slice(2);
  for (let i = 0; i < argv.length; i += 2) {
    const key = argv[i].replace(/^--/, "");
    const val = argv[i + 1];
    if (key === "out") args.out = val;
    else args[key] = Number(val);
  }
  return args;
}

if (require.main === module) {
  const args = parseArgs();
  render(args.width, args.height, args.samples, args.depth, args.out);
}
