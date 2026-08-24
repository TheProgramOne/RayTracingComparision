//! Reference Raytracer - Rust Port
//! ==================================
//! Same algorithm as raytracer.py / raytracer.js / raytracer.c. Materials
//! are an `enum` with per-variant data (Rust's answer to C's tagged struct
//! and Python/JS's class hierarchy -- arguably the cleanest of the four).
//!
//! Fairness rules (see raytracer.py for full explanation):
//!   1. Scene is hardcoded (scene_data.rs, generated from the frozen
//!      scene_spec.json) -- no parsing happens inside the timed render loop.
//!   2. Timing is split into scene setup / render / file write. Only
//!      "render" counts toward the cross-language benchmark.
//!   3. Single-threaded, zero external crates -- not even for random
//!      numbers (see `Rng` below). Same scalar loops as the other ports.
//!
//! Build:  cargo build --release

use std::env;
use std::fs::File;
use std::io::Write;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::time::Instant;

mod scene_data;
use scene_data::{
    CAM_APERTURE, CAM_FOCUS_DIST, CAM_LOOKAT, CAM_LOOKFROM, CAM_VFOV, CAM_VUP, SCENE_SPHERES,
};

// ---------------------------------------------------------------------------
// Vec3
// ---------------------------------------------------------------------------

#[derive(Copy, Clone, Debug)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    const fn new(x: f64, y: f64, z: f64) -> Self {
        Vec3 { x, y, z }
    }

    fn dot(self, o: Vec3) -> f64 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }
    fn cross(self, o: Vec3) -> Vec3 {
        Vec3::new(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }
    fn length_squared(self) -> f64 {
        self.dot(self)
    }
    fn length(self) -> f64 {
        self.length_squared().sqrt()
    }
    fn unit(self) -> Vec3 {
        self / self.length()
    }
    fn near_zero(self) -> bool {
        let eps = 1e-8;
        self.x.abs() < eps && self.y.abs() < eps && self.z.abs() < eps
    }
    fn reflect(self, n: Vec3) -> Vec3 {
        self - n * (2.0 * self.dot(n))
    }
    fn refract(self, n: Vec3, etai_over_etat: f64) -> Vec3 {
        let cos_theta = (-self).dot(n).min(1.0);
        let r_out_perp = (self + n * cos_theta) * etai_over_etat;
        let r_out_parallel = n * -((1.0 - r_out_perp.length_squared()).abs().sqrt());
        r_out_perp + r_out_parallel
    }
}

impl Add for Vec3 {
    type Output = Vec3;
    fn add(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }
}
impl Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x - o.x, self.y - o.y, self.z - o.z)
    }
}
impl Neg for Vec3 {
    type Output = Vec3;
    fn neg(self) -> Vec3 {
        Vec3::new(-self.x, -self.y, -self.z)
    }
}
impl Mul<f64> for Vec3 {
    type Output = Vec3;
    fn mul(self, t: f64) -> Vec3 {
        Vec3::new(self.x * t, self.y * t, self.z * t)
    }
}
impl Mul<Vec3> for Vec3 {
    type Output = Vec3;
    fn mul(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x * o.x, self.y * o.y, self.z * o.z)
    }
}
impl Div<f64> for Vec3 {
    type Output = Vec3;
    fn div(self, t: f64) -> Vec3 {
        self * (1.0 / t)
    }
}

// ---------------------------------------------------------------------------
// RNG -- hand-rolled xorshift64, threaded explicitly through every call.
//
// Python and JS both reach for an invisible global `random()`. Rust has no
// such global without reaching for `unsafe` or a thread_local -- the
// idiomatic move is to make the RNG state an explicit value that gets
// passed as `&mut Rng` everywhere it's needed. More typing, but it means
// there is no hidden mutable state anywhere in this program.
// ---------------------------------------------------------------------------

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Rng { state: seed }
    }
    fn next_f64(&mut self) -> f64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        (x >> 11) as f64 * (1.0 / 9007199254740992.0) // 2^53
    }
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }
    fn vec3(&mut self, lo: f64, hi: f64) -> Vec3 {
        Vec3::new(self.range(lo, hi), self.range(lo, hi), self.range(lo, hi))
    }
    fn in_unit_sphere(&mut self) -> Vec3 {
        loop {
            let p = self.vec3(-1.0, 1.0);
            if p.length_squared() < 1.0 {
                return p;
            }
        }
    }
    fn unit_vector(&mut self) -> Vec3 {
        self.in_unit_sphere().unit()
    }
    fn in_unit_disk(&mut self) -> Vec3 {
        loop {
            let p = Vec3::new(self.range(-1.0, 1.0), self.range(-1.0, 1.0), 0.0);
            if p.length_squared() < 1.0 {
                return p;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Ray
// ---------------------------------------------------------------------------

#[derive(Copy, Clone)]
struct Ray {
    origin: Vec3,
    direction: Vec3,
}

impl Ray {
    fn at(self, t: f64) -> Vec3 {
        self.origin + self.direction * t
    }
}

// ---------------------------------------------------------------------------
// Materials -- an enum with per-variant data. This is Rust's replacement
// for C's flat tagged struct and Python/JS's class hierarchy; `match`
// forces every variant to be handled, so a new material can't be added
// without the compiler pointing at every place that needs updating.
// ---------------------------------------------------------------------------

#[derive(Copy, Clone)]
pub enum Material {
    Lambertian { albedo: Vec3 },
    Metal { albedo: Vec3, fuzz: f64 },
    Dielectric { ref_idx: f64 },
    Checker { odd: Vec3, even: Vec3, scale: f64 },
}

fn reflectance(cosine: f64, ref_idx: f64) -> f64 {
    let mut r0 = (1.0 - ref_idx) / (1.0 + ref_idx);
    r0 *= r0;
    r0 + (1.0 - r0) * (1.0 - cosine).powi(5)
}

impl Material {
    /// Returns None if the ray is absorbed, Some((attenuation, scattered)) otherwise.
    fn scatter(&self, rng: &mut Rng, r_in: &Ray, rec: &HitRecord<'_>) -> Option<(Vec3, Ray)> {
        match *self {
            Material::Lambertian { albedo } => {
                let mut dir = rec.normal + rng.unit_vector();
                if dir.near_zero() {
                    dir = rec.normal;
                }
                Some((albedo, Ray { origin: rec.p, direction: dir }))
            }
            Material::Metal { albedo, fuzz } => {
                let reflected = r_in.direction.unit().reflect(rec.normal) + rng.in_unit_sphere() * fuzz;
                let scattered = Ray { origin: rec.p, direction: reflected };
                if scattered.direction.dot(rec.normal) > 0.0 {
                    Some((albedo, scattered))
                } else {
                    None
                }
            }
            Material::Dielectric { ref_idx } => {
                let attenuation = Vec3::new(1.0, 1.0, 1.0);
                let refraction_ratio = if rec.front_face { 1.0 / ref_idx } else { ref_idx };

                let unit_dir = r_in.direction.unit();
                let cos_theta = (-unit_dir).dot(rec.normal).min(1.0);
                let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();

                let cannot_refract = refraction_ratio * sin_theta > 1.0;
                let direction = if cannot_refract || reflectance(cos_theta, refraction_ratio) > rng.next_f64() {
                    unit_dir.reflect(rec.normal)
                } else {
                    unit_dir.refract(rec.normal, refraction_ratio)
                };
                Some((attenuation, Ray { origin: rec.p, direction }))
            }
            Material::Checker { odd, even, scale } => {
                let sines = (scale * rec.p.x).sin() * (scale * rec.p.y).sin() * (scale * rec.p.z).sin();
                let albedo = if sines < 0.0 { odd } else { even };
                let mut dir = rec.normal + rng.unit_vector();
                if dir.near_zero() {
                    dir = rec.normal;
                }
                Some((albedo, Ray { origin: rec.p, direction: dir }))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Hittables
//
// HitRecord borrows its material from the sphere it hit rather than
// copying it -- so it carries a lifetime, `'a`, which the compiler uses to
// guarantee that record can never outlive the scene it points into. This
// is the borrow checker made visible: C has the same pointer, silently,
// with no such guarantee.
// ---------------------------------------------------------------------------

struct HitRecord<'a> {
    p: Vec3,
    normal: Vec3,
    t: f64,
    front_face: bool,
    material: &'a Material,
}

#[derive(Copy, Clone)]
pub struct Sphere {
    pub center: Vec3,
    pub radius: f64,
    pub material: Material,
}

impl Sphere {
    fn hit(&self, r: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord<'_>> {
        let oc = r.origin - self.center;
        let a = r.direction.length_squared();
        let half_b = oc.dot(r.direction);
        let c = oc.length_squared() - self.radius * self.radius;
        let discriminant = half_b * half_b - a * c;
        if discriminant < 0.0 {
            return None;
        }
        let sqrtd = discriminant.sqrt();

        let mut root = (-half_b - sqrtd) / a;
        if root < t_min || root > t_max {
            root = (-half_b + sqrtd) / a;
            if root < t_min || root > t_max {
                return None;
            }
        }

        let p = r.at(root);
        let outward_normal = (p - self.center) / self.radius;
        let front_face = r.direction.dot(outward_normal) < 0.0;
        let normal = if front_face { outward_normal } else { -outward_normal };

        Some(HitRecord {
            p,
            normal,
            t: root,
            front_face,
            material: &self.material,
        })
    }
}

fn hit_world<'a>(spheres: &'a [Sphere], r: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord<'a>> {
    let mut closest = t_max;
    let mut result = None;
    for s in spheres.iter() {
        if let Some(rec) = s.hit(r, t_min, closest) {
            closest = rec.t;
            result = Some(rec);
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Camera (thin-lens model)
// ---------------------------------------------------------------------------

struct Camera {
    origin: Vec3,
    lower_left: Vec3,
    horizontal: Vec3,
    vertical: Vec3,
    u: Vec3,
    v: Vec3,
    lens_radius: f64,
}

impl Camera {
    fn new(
        lookfrom: Vec3,
        lookat: Vec3,
        vup: Vec3,
        vfov: f64,
        aspect_ratio: f64,
        aperture: f64,
        focus_dist: f64,
    ) -> Self {
        let theta = vfov.to_radians();
        let h = (theta / 2.0).tan();
        let viewport_height = 2.0 * h;
        let viewport_width = aspect_ratio * viewport_height;

        let w = (lookfrom - lookat).unit();
        let u = vup.cross(w).unit();
        let v = w.cross(u);

        let origin = lookfrom;
        let horizontal = u * (viewport_width * focus_dist);
        let vertical = v * (viewport_height * focus_dist);
        let lower_left = origin - horizontal / 2.0 - vertical / 2.0 - w * focus_dist;

        Camera {
            origin,
            lower_left,
            horizontal,
            vertical,
            u,
            v,
            lens_radius: aperture / 2.0,
        }
    }

    fn get_ray(&self, rng: &mut Rng, s: f64, t: f64) -> Ray {
        let rd = rng.in_unit_disk() * self.lens_radius;
        let offset = self.u * rd.x + self.v * rd.y;
        let origin = self.origin + offset;
        let direction = self.lower_left + self.horizontal * s + self.vertical * t - self.origin - offset;
        Ray { origin, direction }
    }
}

// ---------------------------------------------------------------------------
// Shading
// ---------------------------------------------------------------------------

fn sky_color(r: &Ray) -> Vec3 {
    let unit_dir = r.direction.unit();
    let t = 0.5 * (unit_dir.y + 1.0);
    Vec3::new(1.0, 1.0, 1.0) * (1.0 - t) + Vec3::new(0.5, 0.7, 1.0) * t
}

fn ray_color(rng: &mut Rng, r: &Ray, spheres: &[Sphere], depth: i32) -> Vec3 {
    if depth <= 0 {
        return Vec3::new(0.0, 0.0, 0.0);
    }

    if let Some(rec) = hit_world(spheres, r, 0.001, f64::INFINITY) {
        if let Some((attenuation, scattered)) = rec.material.scatter(rng, r, &rec) {
            return attenuation * ray_color(rng, &scattered, spheres, depth - 1);
        }
        return Vec3::new(0.0, 0.0, 0.0);
    }
    sky_color(r)
}

// ---------------------------------------------------------------------------
// Main render
// ---------------------------------------------------------------------------

fn render(width: usize, height: usize, samples_per_pixel: u32, max_depth: i32, out_path: &str) {
    let t0 = Instant::now();

    let aspect_ratio = width as f64 / height as f64;
    let cam = Camera::new(
        CAM_LOOKFROM,
        CAM_LOOKAT,
        CAM_VUP,
        CAM_VFOV,
        aspect_ratio,
        CAM_APERTURE,
        CAM_FOCUS_DIST,
    );
    let spheres: &[Sphere] = &SCENE_SPHERES;

    let t1 = Instant::now();

    let mut rng = Rng::new(88172645463325252);
    let mut rows: Vec<String> = Vec::with_capacity(height);

    for j in (0..height).rev() {
        let mut row = String::with_capacity(width * 12);
        for i in 0..width {
            let mut color = Vec3::new(0.0, 0.0, 0.0);
            for _ in 0..samples_per_pixel {
                let su = (i as f64 + rng.next_f64()) / (width as f64 - 1.0);
                let sv = (j as f64 + rng.next_f64()) / (height as f64 - 1.0);
                let r = cam.get_ray(&mut rng, su, sv);
                color = color + ray_color(&mut rng, &r, spheres, max_depth);
            }

            let scale = 1.0 / samples_per_pixel as f64;
            let r_ = (color.x * scale).sqrt();
            let g_ = (color.y * scale).sqrt();
            let b_ = (color.z * scale).sqrt();

            let ir = (256.0 * r_.clamp(0.0, 0.999)) as i32;
            let ig = (256.0 * g_.clamp(0.0, 0.999)) as i32;
            let ib = (256.0 * b_.clamp(0.0, 0.999)) as i32;

            if i > 0 {
                row.push(' ');
            }
            row.push_str(&format!("{} {} {}", ir, ig, ib));
        }
        rows.push(row);
        eprint!("\rScanlines remaining: {:4} ", j);
    }

    let t2 = Instant::now();

    let mut f = File::create(out_path).expect("failed to create output file");
    write!(f, "P3\n{} {}\n255\n", width, height).unwrap();
    for row in &rows {
        writeln!(f, "{}", row).unwrap();
    }

    let t3 = Instant::now();

    eprintln!("\nDone.");
    println!("scene_setup_seconds: {:.4}", (t1 - t0).as_secs_f64());
    println!("render_seconds:      {:.4}", (t2 - t1).as_secs_f64());
    println!("file_write_seconds:  {:.4}", (t3 - t2).as_secs_f64());
    println!("total_seconds:       {:.4}", (t3 - t0).as_secs_f64());
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut width: usize = 400;
    let mut height: usize = 225;
    let mut samples: u32 = 20;
    let mut depth: i32 = 20;
    let mut out = String::from("render.ppm");

    let mut i = 1;
    while i + 1 < args.len() {
        match args[i].as_str() {
            "--width" => width = args[i + 1].parse().expect("--width must be an integer"),
            "--height" => height = args[i + 1].parse().expect("--height must be an integer"),
            "--samples" => samples = args[i + 1].parse().expect("--samples must be an integer"),
            "--depth" => depth = args[i + 1].parse().expect("--depth must be an integer"),
            "--out" => out = args[i + 1].clone(),
            _ => {}
        }
        i += 2;
    }

    render(width, height, samples, depth, &out);
}
