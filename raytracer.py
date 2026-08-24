"""
Reference Raytracer - Python Implementation
=============================================
Baseline for the "Same App, Four Languages" comparison series
(C, Rust, Python, JavaScript).

Fairness rules that every port must follow:
  1. The scene (sphere positions, radii, materials, colors, camera) is
     fixed and frozen in scene_spec.json. Every language HARDCODES these
     exact values as native constants -- nobody parses JSON inside the
     timed render loop, so we only ever measure raytracing performance,
     never file I/O or parsing speed.
  2. Timing is split into three phases: scene setup, render, file write.
     Only "render" counts toward the cross-language benchmark number.
  3. No multithreading, no SIMD, no external raytracing libraries. Plain
     scalar loops in every language -- the comparison reflects the
     language itself, not a library's optimization work.

Algorithm: a standard Whitted-style raytracer --
  - Lambertian (diffuse), Metal (reflective + fuzz), Dielectric (glass)
  - Sphere-only scene (the "ground" is just a very large sphere)
  - Antialiasing via multisampling
  - Depth of field via a thin-lens camera model
  - Recursive ray bounces up to MAX_DEPTH
"""

import json
import math
import random
import sys
import time


# ---------------------------------------------------------------------------
# Vec3
# ---------------------------------------------------------------------------

class Vec3:
    __slots__ = ("x", "y", "z")

    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = x, y, z

    def __add__(self, o): return Vec3(self.x + o.x, self.y + o.y, self.z + o.z)
    def __sub__(self, o): return Vec3(self.x - o.x, self.y - o.y, self.z - o.z)
    def __neg__(self): return Vec3(-self.x, -self.y, -self.z)

    def __mul__(self, o):
        if isinstance(o, Vec3):
            return Vec3(self.x * o.x, self.y * o.y, self.z * o.z)
        return Vec3(self.x * o, self.y * o, self.z * o)

    __rmul__ = __mul__

    def __truediv__(self, t):
        return self * (1.0 / t)

    def length_squared(self):
        return self.x * self.x + self.y * self.y + self.z * self.z

    def length(self):
        return math.sqrt(self.length_squared())

    def unit(self):
        return self / self.length()

    def near_zero(self):
        eps = 1e-8
        return abs(self.x) < eps and abs(self.y) < eps and abs(self.z) < eps


def dot(a, b):
    return a.x * b.x + a.y * b.y + a.z * b.z


def cross(a, b):
    return Vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)


def reflect(v, n):
    return v - n * (2.0 * dot(v, n))


def refract(uv, n, etai_over_etat):
    cos_theta = min(dot(-uv, n), 1.0)
    r_out_perp = (uv + n * cos_theta) * etai_over_etat
    r_out_parallel = n * -math.sqrt(abs(1.0 - r_out_perp.length_squared()))
    return r_out_perp + r_out_parallel


def random_vec(a=0.0, b=1.0):
    return Vec3(random.uniform(a, b), random.uniform(a, b), random.uniform(a, b))


def random_in_unit_sphere():
    while True:
        p = random_vec(-1.0, 1.0)
        if p.length_squared() < 1.0:
            return p


def random_unit_vector():
    return random_in_unit_sphere().unit()


def random_in_unit_disk():
    while True:
        p = Vec3(random.uniform(-1, 1), random.uniform(-1, 1), 0.0)
        if p.length_squared() < 1.0:
            return p


# ---------------------------------------------------------------------------
# Ray
# ---------------------------------------------------------------------------

class Ray:
    __slots__ = ("origin", "direction")

    def __init__(self, origin, direction):
        self.origin, self.direction = origin, direction

    def at(self, t):
        return self.origin + self.direction * t


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------

class Lambertian:
    def __init__(self, albedo):
        self.albedo = albedo

    def scatter(self, r_in, rec):
        scatter_dir = rec.normal + random_unit_vector()
        if scatter_dir.near_zero():
            scatter_dir = rec.normal
        return True, self.albedo, Ray(rec.p, scatter_dir)


class Metal:
    def __init__(self, albedo, fuzz):
        self.albedo = albedo
        self.fuzz = min(fuzz, 1.0)

    def scatter(self, r_in, rec):
        reflected = reflect(r_in.direction.unit(), rec.normal)
        reflected = reflected + random_in_unit_sphere() * self.fuzz
        scattered = Ray(rec.p, reflected)
        ok = dot(scattered.direction, rec.normal) > 0
        return ok, self.albedo, scattered


class Dielectric:
    def __init__(self, ref_idx):
        self.ref_idx = ref_idx

    @staticmethod
    def _reflectance(cosine, ref_idx):
        r0 = (1 - ref_idx) / (1 + ref_idx)
        r0 = r0 * r0
        return r0 + (1 - r0) * ((1 - cosine) ** 5)

    def scatter(self, r_in, rec):
        attenuation = Vec3(1.0, 1.0, 1.0)
        refraction_ratio = (1.0 / self.ref_idx) if rec.front_face else self.ref_idx

        unit_dir = r_in.direction.unit()
        cos_theta = min(dot(-unit_dir, rec.normal), 1.0)
        sin_theta = math.sqrt(1.0 - cos_theta * cos_theta)

        cannot_refract = refraction_ratio * sin_theta > 1.0
        if cannot_refract or self._reflectance(cos_theta, refraction_ratio) > random.random():
            direction = reflect(unit_dir, rec.normal)
        else:
            direction = refract(unit_dir, rec.normal, refraction_ratio)

        return True, attenuation, Ray(rec.p, direction)


class CheckerLambertian:
    """Ground material: alternates between two albedos based on world position."""

    def __init__(self, odd, even, scale=1.0):
        self.odd, self.even, self.scale = odd, even, scale

    def scatter(self, r_in, rec):
        s = self.scale
        sines = math.sin(s * rec.p.x) * math.sin(s * rec.p.y) * math.sin(s * rec.p.z)
        albedo = self.odd if sines < 0 else self.even
        scatter_dir = rec.normal + random_unit_vector()
        if scatter_dir.near_zero():
            scatter_dir = rec.normal
        return True, albedo, Ray(rec.p, scatter_dir)


# ---------------------------------------------------------------------------
# Hittables
# ---------------------------------------------------------------------------

class HitRecord:
    __slots__ = ("p", "normal", "t", "front_face", "material")


class Sphere:
    def __init__(self, center, radius, material):
        self.center, self.radius, self.material = center, radius, material

    def hit(self, r, t_min, t_max):
        oc = r.origin - self.center
        a = r.direction.length_squared()
        half_b = dot(oc, r.direction)
        c = oc.length_squared() - self.radius * self.radius
        discriminant = half_b * half_b - a * c
        if discriminant < 0:
            return None
        sqrtd = math.sqrt(discriminant)

        root = (-half_b - sqrtd) / a
        if root < t_min or root > t_max:
            root = (-half_b + sqrtd) / a
            if root < t_min or root > t_max:
                return None

        rec = HitRecord()
        rec.t = root
        rec.p = r.at(root)
        outward_normal = (rec.p - self.center) / self.radius
        rec.front_face = dot(r.direction, outward_normal) < 0
        rec.normal = outward_normal if rec.front_face else -outward_normal
        rec.material = self.material
        return rec


def hit_world(spheres, r, t_min, t_max):
    closest = t_max
    hit_rec = None
    for s in spheres:
        rec = s.hit(r, t_min, closest)
        if rec is not None:
            closest = rec.t
            hit_rec = rec
    return hit_rec


# ---------------------------------------------------------------------------
# Camera (thin-lens model, gives depth-of-field blur)
# ---------------------------------------------------------------------------

class Camera:
    def __init__(self, lookfrom, lookat, vup, vfov, aspect_ratio, aperture, focus_dist):
        theta = math.radians(vfov)
        h = math.tan(theta / 2)
        viewport_height = 2.0 * h
        viewport_width = aspect_ratio * viewport_height

        self.w = (lookfrom - lookat).unit()
        self.u = cross(vup, self.w).unit()
        self.v = cross(self.w, self.u)

        self.origin = lookfrom
        self.horizontal = self.u * viewport_width * focus_dist
        self.vertical = self.v * viewport_height * focus_dist
        self.lower_left = (self.origin - self.horizontal / 2 - self.vertical / 2
                            - self.w * focus_dist)
        self.lens_radius = aperture / 2

    def get_ray(self, s, t):
        rd = random_in_unit_disk() * self.lens_radius
        offset = self.u * rd.x + self.v * rd.y
        origin = self.origin + offset
        direction = (self.lower_left + self.horizontal * s + self.vertical * t
                     - self.origin - offset)
        return Ray(origin, direction)


# ---------------------------------------------------------------------------
# Scene -- fixed & seeded so it's identical every run
# ---------------------------------------------------------------------------

NAVY = Vec3(0.06, 0.13, 0.32)
ORANGE = Vec3(0.92, 0.46, 0.09)


def build_scene():
    """
    Checkered ground + three 'hero' spheres (glass / diffuse / metal) +
    a deterministic field of small spheres. Seeded RNG guarantees the
    exact same scene every run -- these values get frozen into
    scene_spec.json so C/Rust/JS can hardcode the identical scene.
    """
    random.seed(1337)
    spheres = []

    ground_mat = CheckerLambertian(NAVY, Vec3(0.92, 0.92, 0.90), scale=1.0)
    spheres.append(Sphere(Vec3(0, -1000, 0), 1000, ground_mat))

    positions = []
    for a in range(-6, 6):
        for b in range(-6, 6):
            center = Vec3(a + 0.9 * random.random(), 0.2, b + 0.9 * random.random())
            if (center - Vec3(4, 0.2, 0)).length() <= 0.9:
                continue
            if (center - Vec3(-4, 0.2, 0)).length() <= 0.9:
                continue
            if (center - Vec3(0, 0.2, 0)).length() <= 1.2:
                continue
            positions.append(center)

    for center in positions:
        choose = random.random()
        if choose < 0.75:
            if random.random() < 0.18:
                albedo = NAVY if random.random() < 0.5 else ORANGE
            else:
                albedo = random_vec() * random_vec()
            spheres.append(Sphere(center, 0.2, Lambertian(albedo)))
        elif choose < 0.92:
            albedo = random_vec(0.5, 1.0)
            fuzz = random.uniform(0, 0.5)
            spheres.append(Sphere(center, 0.2, Metal(albedo, fuzz)))
        else:
            spheres.append(Sphere(center, 0.2, Dielectric(1.5)))

    spheres.append(Sphere(Vec3(0, 1, 0), 1.0, Dielectric(1.5)))
    spheres.append(Sphere(Vec3(-4, 1, 0), 1.0, Lambertian(ORANGE)))
    spheres.append(Sphere(Vec3(4, 1, 0), 1.0, Metal(NAVY, 0.02)))

    return spheres


def export_scene_spec(spheres, path):
    """Dumps the exact frozen scene to JSON -- the porting reference doc
    for C/Rust/JS. Not read back in at render time by any language."""

    def mat_to_dict(m):
        if isinstance(m, Lambertian):
            return {"type": "lambertian", "albedo": [m.albedo.x, m.albedo.y, m.albedo.z]}
        if isinstance(m, Metal):
            return {"type": "metal", "albedo": [m.albedo.x, m.albedo.y, m.albedo.z], "fuzz": m.fuzz}
        if isinstance(m, Dielectric):
            return {"type": "dielectric", "ref_idx": m.ref_idx}
        if isinstance(m, CheckerLambertian):
            return {
                "type": "checker",
                "odd": [m.odd.x, m.odd.y, m.odd.z],
                "even": [m.even.x, m.even.y, m.even.z],
                "scale": m.scale,
            }
        raise ValueError("unknown material")

    data = {
        "camera": {
            "lookfrom": [13, 2, 3],
            "lookat": [0, 0, 0],
            "vup": [0, 1, 0],
            "vfov": 20.0,
            "aperture": 0.1,
            "focus_dist": 10.0,
        },
        "spheres": [
            {
                "center": [s.center.x, s.center.y, s.center.z],
                "radius": s.radius,
                "material": mat_to_dict(s.material),
            }
            for s in spheres
        ],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Shading
# ---------------------------------------------------------------------------

def sky_color(r):
    unit_dir = r.direction.unit()
    t = 0.5 * (unit_dir.y + 1.0)
    return Vec3(1.0, 1.0, 1.0) * (1.0 - t) + Vec3(0.5, 0.7, 1.0) * t


def ray_color(r, spheres, depth):
    if depth <= 0:
        return Vec3(0, 0, 0)

    rec = hit_world(spheres, r, 0.001, math.inf)
    if rec is not None:
        did_scatter, attenuation, scattered = rec.material.scatter(r, rec)
        if did_scatter:
            return attenuation * ray_color(scattered, spheres, depth - 1)
        return Vec3(0, 0, 0)

    return sky_color(r)


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def write_ppm(path, width, height, rows):
    with open(path, "w") as f:
        f.write(f"P3\n{width} {height}\n255\n")
        f.write("\n".join(rows))
        f.write("\n")


def render(width, height, samples_per_pixel, max_depth, out_path, spec_path=None):
    t0 = time.perf_counter()

    aspect_ratio = width / height
    cam = Camera(Vec3(13, 2, 3), Vec3(0, 0, 0), Vec3(0, 1, 0), 20.0, aspect_ratio, 0.1, 10.0)
    spheres = build_scene()

    if spec_path:
        export_scene_spec(spheres, spec_path)

    t1 = time.perf_counter()

    rows = []
    for j in range(height - 1, -1, -1):
        row = []
        for i in range(width):
            color = Vec3(0, 0, 0)
            for _ in range(samples_per_pixel):
                s = (i + random.random()) / (width - 1)
                t = (j + random.random()) / (height - 1)
                color = color + ray_color(cam.get_ray(s, t), spheres, max_depth)

            scale = 1.0 / samples_per_pixel
            r_ = math.sqrt(color.x * scale)
            g_ = math.sqrt(color.y * scale)
            b_ = math.sqrt(color.z * scale)

            ir = int(256 * min(max(r_, 0.0), 0.999))
            ig = int(256 * min(max(g_, 0.0), 0.999))
            ib = int(256 * min(max(b_, 0.0), 0.999))
            row.append(f"{ir} {ig} {ib}")
        rows.append(" ".join(row))

        sys.stderr.write(f"\rScanlines remaining: {j:4d} ")
        sys.stderr.flush()

    t2 = time.perf_counter()

    write_ppm(out_path, width, height, rows)

    t3 = time.perf_counter()

    sys.stderr.write("\nDone.\n")
    print(f"scene_setup_seconds: {t1 - t0:.4f}")
    print(f"render_seconds:      {t2 - t1:.4f}")
    print(f"file_write_seconds:  {t3 - t2:.4f}")
    print(f"total_seconds:       {t3 - t0:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=400)
    parser.add_argument("--height", type=int, default=225)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--depth", type=int, default=20)
    parser.add_argument("--out", type=str, default="render.ppm")
    parser.add_argument("--spec", type=str, default="scene_spec.json")
    args = parser.parse_args()

    render(args.width, args.height, args.samples, args.depth, args.out, args.spec)
