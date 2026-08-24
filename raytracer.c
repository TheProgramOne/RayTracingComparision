static inline double random_double_range(double lo, double hi) { return lo + (hi - lo) * random_double(); }
static inline Vec3 random_vec(double lo, double hi) {
    return (Vec3){random_double_range(lo, hi), random_double_range(lo, hi), random_double_range(lo, hi)};
}
static inline Vec3 random_in_unit_sphere(void) {
    while (1) {
        Vec3 p = random_vec(-1.0, 1.0);
        if (vec3_length_squared(p) < 1.0) return p;
    }
}
static inline Vec3 random_unit_vector(void) { return vec3_unit(random_in_unit_sphere()); }
static inline Vec3 random_in_unit_disk(void) {
    while (1) {
        Vec3 p = {random_double_range(-1, 1), random_double_range(-1, 1), 0.0};
        if (vec3_length_squared(p) < 1.0) return p;
    }
}

/* --------------------------------------------------------------------- */
/* Ray                                                                    */
/* --------------------------------------------------------------------- */

typedef struct { Vec3 origin, direction; } Ray;
static inline Vec3 ray_at(Ray r, double t) { return vec3_add(r.origin, vec3_mul(r.direction, t)); }

/* --------------------------------------------------------------------- */
/* Materials -- flat tagged struct (C has no classes/polymorphism)       */
/* --------------------------------------------------------------------- */

typedef enum { MAT_LAMBERTIAN, MAT_METAL, MAT_DIELECTRIC, MAT_CHECKER } MatType;

typedef struct {
    MatType type;
    Vec3 albedo;     /* lambertian, metal */
    double fuzz;     /* metal */
    double ref_idx;  /* dielectric */
    Vec3 odd, even;  /* checker */
    double scale;    /* checker */
} Material;

typedef struct {
    Vec3 p, normal;
    double t;
    int front_face;
    const Material *material;
} HitRecord;

static double reflectance(double cosine, double ref_idx) {
    double r0 = (1 - ref_idx) / (1 + ref_idx);
    r0 = r0 * r0;
    return r0 + (1 - r0) * pow(1 - cosine, 5);
}

/* Returns 1 if the ray scatters, 0 if absorbed. Fills attenuation/scattered. */
static int material_scatter(const Material *mat, Ray r_in, const HitRecord *rec,
                             Vec3 *attenuation, Ray *scattered) {
    switch (mat->type) {
        case MAT_LAMBERTIAN: {
            Vec3 dir = vec3_add(rec->normal, random_unit_vector());
            if (vec3_near_zero(dir)) dir = rec->normal;
            *attenuation = mat->albedo;
            *scattered = (Ray){rec->p, dir};
            return 1;
        }
        case MAT_METAL: {
            Vec3 reflected = vec3_reflect(vec3_unit(r_in.direction), rec->normal);
            reflected = vec3_add(reflected, vec3_mul(random_in_unit_sphere(), mat->fuzz));
            *attenuation = mat->albedo;
            *scattered = (Ray){rec->p, reflected};
            return vec3_dot(scattered->direction, rec->normal) > 0;
        }
        case MAT_DIELECTRIC: {
            *attenuation = (Vec3){1.0, 1.0, 1.0};
            double refraction_ratio = rec->front_face ? (1.0 / mat->ref_idx) : mat->ref_idx;

            Vec3 unit_dir = vec3_unit(r_in.direction);
            double cos_theta = fmin(vec3_dot(vec3_neg(unit_dir), rec->normal), 1.0);
            double sin_theta = sqrt(1.0 - cos_theta * cos_theta);

            int cannot_refract = refraction_ratio * sin_theta > 1.0;
            Vec3 direction;
            if (cannot_refract || reflectance(cos_theta, refraction_ratio) > random_double()) {
                direction = vec3_reflect(unit_dir, rec->normal);
            } else {
                direction = vec3_refract(unit_dir, rec->normal, refraction_ratio);
            }
            *scattered = (Ray){rec->p, direction};
            return 1;
        }
        case MAT_CHECKER: {
            double s = mat->scale;
            double sines = sin(s * rec->p.x) * sin(s * rec->p.y) * sin(s * rec->p.z);
            Vec3 dir = vec3_add(rec->normal, random_unit_vector());
            if (vec3_near_zero(dir)) dir = rec->normal;
            *attenuation = (sines < 0) ? mat->odd : mat->even;
            *scattered = (Ray){rec->p, dir};
            return 1;
        }
    }
    return 0;
}

/* --------------------------------------------------------------------- */
/* Sphere                                                                 */
/* --------------------------------------------------------------------- */

typedef struct {
    Vec3 center;
    double radius;
    Material material;
} Sphere;

static int sphere_hit(const Sphere *s, Ray r, double t_min, double t_max, HitRecord *rec) {
    Vec3 oc = vec3_sub(r.origin, s->center);
    double a = vec3_length_squared(r.direction);
    double half_b = vec3_dot(oc, r.direction);
    double c = vec3_length_squared(oc) - s->radius * s->radius;
    double discriminant = half_b * half_b - a * c;
    if (discriminant < 0) return 0;
    double sqrtd = sqrt(discriminant);

    double root = (-half_b - sqrtd) / a;
    if (root < t_min || root > t_max) {
        root = (-half_b + sqrtd) / a;
        if (root < t_min || root > t_max) return 0;
    }

    rec->t = root;
    rec->p = ray_at(r, root);
    Vec3 outward_normal = vec3_div(vec3_sub(rec->p, s->center), s->radius);
    rec->front_face = vec3_dot(r.direction, outward_normal) < 0;
    rec->normal = rec->front_face ? outward_normal : vec3_neg(outward_normal);
    rec->material = &s->material;
    return 1;
}

static int hit_world(const Sphere *spheres, int count, Ray r, double t_min, double t_max, HitRecord *rec) {
    HitRecord temp;
    int hit_anything = 0;
    double closest = t_max;
    for (int i = 0; i < count; i++) {
        if (sphere_hit(&spheres[i], r, t_min, closest, &temp)) {
            hit_anything = 1;
            closest = temp.t;
            *rec = temp;
        }
    }
    return hit_anything;
}

/* --------------------------------------------------------------------- */
/* Camera (thin-lens model)                                               */
/* --------------------------------------------------------------------- */

typedef struct {
    Vec3 origin, lower_left, horizontal, vertical;
    Vec3 u, v, w;
    double lens_radius;
} Camera;

static Camera camera_init(Vec3 lookfrom, Vec3 lookat, Vec3 vup, double vfov,
                           double aspect_ratio, double aperture, double focus_dist) {
    double theta = vfov * M_PI / 180.0;
    double h = tan(theta / 2);
    double viewport_height = 2.0 * h;
    double viewport_width = aspect_ratio * viewport_height;

    Camera cam;
    cam.w = vec3_unit(vec3_sub(lookfrom, lookat));
    cam.u = vec3_unit(vec3_cross(vup, cam.w));
    cam.v = vec3_cross(cam.w, cam.u);

    cam.origin = lookfrom;
    cam.horizontal = vec3_mul(cam.u, viewport_width * focus_dist);
    cam.vertical = vec3_mul(cam.v, viewport_height * focus_dist);
    cam.lower_left = vec3_sub(vec3_sub(vec3_sub(cam.origin, vec3_div(cam.horizontal, 2)),
                                        vec3_div(cam.vertical, 2)),
                               vec3_mul(cam.w, focus_dist));
    cam.lens_radius = aperture / 2;
    return cam;
}

static Ray camera_get_ray(const Camera *cam, double s, double t) {
    Vec3 rd = vec3_mul(random_in_unit_disk(), cam->lens_radius);
    Vec3 offset = vec3_add(vec3_mul(cam->u, rd.x), vec3_mul(cam->v, rd.y));
    Vec3 origin = vec3_add(cam->origin, offset);
    Vec3 direction = vec3_sub(vec3_sub(vec3_add(vec3_add(cam->lower_left, vec3_mul(cam->horizontal, s)),
                                                 vec3_mul(cam->vertical, t)),
                                        cam->origin),
                               offset);
    return (Ray){origin, direction};
}

/* --------------------------------------------------------------------- */
/* Scene -- hardcoded, compiled directly into the binary                 */
/* --------------------------------------------------------------------- */

#include "scene_data.h"

/* --------------------------------------------------------------------- */
/* Shading                                                                */
/* --------------------------------------------------------------------- */

static Vec3 sky_color(Ray r) {
    Vec3 unit_dir = vec3_unit(r.direction);
    double t = 0.5 * (unit_dir.y + 1.0);
    return vec3_add(vec3_mul((Vec3){1, 1, 1}, 1.0 - t), vec3_mul((Vec3){0.5, 0.7, 1.0}, t));
}

static Vec3 ray_color(Ray r, const Sphere *spheres, int count, int depth) {
    if (depth <= 0) return (Vec3){0, 0, 0};

    HitRecord rec;
    if (hit_world(spheres, count, r, 0.001, INFINITY, &rec)) {
        Vec3 attenuation;
        Ray scattered;
        if (material_scatter(rec.material, r, &rec, &attenuation, &scattered)) {
            return vec3_mul_vec(attenuation, ray_color(scattered, spheres, count, depth - 1));
        }
        return (Vec3){0, 0, 0};
    }
    return sky_color(r);
}

/* --------------------------------------------------------------------- */
/* Main render                                                            */
/* --------------------------------------------------------------------- */

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static void render(int width, int height, int samples_per_pixel, int max_depth, const char *out_path) {
    double t0 = now_seconds();

    double aspect_ratio = (double)width / (double)height;
    Camera cam = camera_init(CAM_LOOKFROM, CAM_LOOKAT, CAM_VUP, CAM_VFOV,
                              aspect_ratio, CAM_APERTURE, CAM_FOCUS_DIST);
    /* Scene is a static compiled-in array (scene_data.h) -- there is no
       runtime construction step here, unlike Python/JS which build sphere
       objects at startup. That's a real structural difference worth
       calling out on camera. */
    const Sphere *spheres = SCENE_SPHERES;
    const int sphere_count = SPHERE_COUNT;

    double t1 = now_seconds();

    /* Buffer the whole image in memory, same shape as the Python/JS row lists. */
    char **rows = malloc(sizeof(char *) * height);
    const int MAX_ROW_LEN = width * 12 + 16; /* "255 255 255 " per pixel, generous */

    int row_idx = 0;
    for (int j = height - 1; j >= 0; j--) {
        char *row = malloc(MAX_ROW_LEN);
        int pos = 0;
        for (int i = 0; i < width; i++) {
            Vec3 color = {0, 0, 0};
            for (int s = 0; s < samples_per_pixel; s++) {
                double su = (i + random_double()) / (width - 1);
                double sv = (j + random_double()) / (height - 1);
                Ray r = camera_get_ray(&cam, su, sv);
                color = vec3_add(color, ray_color(r, spheres, sphere_count, max_depth));
            }

            double scale = 1.0 / samples_per_pixel;
            double r_ = sqrt(color.x * scale);
            double g_ = sqrt(color.y * scale);
            double b_ = sqrt(color.z * scale);

            int ir = (int)(256 * fmin(fmax(r_, 0.0), 0.999));
            int ig = (int)(256 * fmin(fmax(g_, 0.0), 0.999));
            int ib = (int)(256 * fmin(fmax(b_, 0.0), 0.999));

            pos += sprintf(row + pos, i == 0 ? "%d %d %d" : " %d %d %d", ir, ig, ib);
        }
        rows[row_idx++] = row;
        fprintf(stderr, "\rScanlines remaining: %4d ", j);
    }

    double t2 = now_seconds();

    FILE *f = fopen(out_path, "w");
    fprintf(f, "P3\n%d %d\n255\n", width, height);
    for (int k = 0; k < height; k++) {
        fputs(rows[k], f);
        fputc('\n', f);
        free(rows[k]);
    }
    fclose(f);
    free(rows);

    double t3 = now_seconds();

    fprintf(stderr, "\nDone.\n");
    printf("scene_setup_seconds: %.4f\n", t1 - t0);
    printf("render_seconds:      %.4f\n", t2 - t1);
    printf("file_write_seconds:  %.4f\n", t3 - t2);
    printf("total_seconds:       %.4f\n", t3 - t0);
}

/* --------------------------------------------------------------------- */
/* CLI                                                                    */
/* --------------------------------------------------------------------- */

int main(int argc, char **argv) {
    int width = 400, height = 225, samples = 20, depth = 20;
    const char *out = "render.ppm";

    for (int i = 1; i < argc - 1; i += 2) {
        if (strcmp(argv[i], "--width") == 0) width = atoi(argv[i + 1]);
        else if (strcmp(argv[i], "--height") == 0) height = atoi(argv[i + 1]);
        else if (strcmp(argv[i], "--samples") == 0) samples = atoi(argv[i + 1]);
        else if (strcmp(argv[i], "--depth") == 0) depth = atoi(argv[i + 1]);
        else if (strcmp(argv[i], "--out") == 0) out = argv[i + 1];
    }

    render(width, height, samples, depth, out);
    return 0;
}
