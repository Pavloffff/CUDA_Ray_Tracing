#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <memory>
#include <float.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define CSC(call)                                                   \
do {                                                                \
    cudaError_t res = call;                                         \
    if (res != cudaSuccess) {                                       \
        fprintf(stderr, "ERROR in %s:%d. Message: %s\n",            \
                __FILE__, __LINE__, cudaGetErrorString(res));       \
        exit(0);                                                    \
    }                                                               \
} while(0)

typedef unsigned char uchar;

__host__ __device__ uchar4 operator*(double b, uchar4 a) {
    return {
        (uchar)fmin(255.0, a.x * b),
        (uchar)fmin(255.0, a.y * b),
        (uchar)fmin(255.0, a.z * b),
        (uchar)fmin(255.0, a.w * b)
    };
}

__host__ __device__ uchar4 operator+(uchar4 b, uchar4 a) {
    return {
        (uchar)fmin(255.0, (double)(a.x + b.x)),
        (uchar)fmin(255.0, (double)(a.y + b.y)),
        (uchar)fmin(255.0, (double)(a.z + b.z)),
        (uchar)fmin(255.0, (double)(a.w + b.w)),
    };
}

__host__ __device__ bool is_near(double x, double eps = 1e-5) {
    return std::fabs(x) < eps;
}

__host__ __device__ double dot(double3 a, double3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ double3 prod(double3 a, double3 b) {
    return {
        a.y * b.z - a.z * b.y, 
        a.z * b.x - a.x * b.z,  
        a.x * b.y - a.y * b.x
    };
}

__host__ __device__ double3 norm(double3 v) {
    double l = std::sqrt(dot(v, v));
    if (l < 1e-12) return {0,0,0};
    return {v.x / l, v.y / l, v.z / l};
}

__host__ __device__ double3 diff(double3 a, double3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__host__ __device__ double3 add(double3 a, double3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__host__ __device__ double3 prod_num(double3 a, double b) {
    return {a.x * b, a.y * b, a.z * b};
}

__host__ __device__ double3 lerp(double3 a, double3 b, double t) {
    double3 ab = diff(b, a);
    double3 mod_ab = prod_num(ab, t);
    double3 res = add(a, mod_ab);
    return res;
}

__host__ __device__ double3 mult(double3 a, double3 b, double3 c, double3 v) {
    return {
        a.x * v.x + b.x * v.y + c.x * v.z,
        a.y * v.x + b.y * v.y + c.y * v.z,
        a.z * v.x + b.z * v.y + c.z * v.z
    };
}

__host__ __device__ enum class LightType {
    AMBIENT,
    POINT
};

__host__ __device__ struct Light {
    LightType type;
    double intensity;
    double3 position;
    double3 direction;
};

__host__ __device__ struct IntersectDto {
    bool res = false;
    double tOut = 0.0;
};

__host__ __device__ struct Point {
    double3 a;
    double radius = 0.05;
};

__host__ __device__ void point_init(Point *point, double3 A) {
    point->a = A;
}

__host__ __device__ uchar4 point_at(Point point, double3 P = {0, 0, 0}) {
    return {255, 255, 255, 255};
}

__host__ __device__ IntersectDto point_intersect(Point point, double3 pos, double3 dir) {
    IntersectDto ans;

    double3 L = diff(point.a, pos);
    double t = dot(L, dir) / dot(dir, dir);

    if (t < 0.0) {
        ans.res = false;
        return ans;
    }

    double3 closest_point = add(pos, prod_num(dir, t));
    double distance_squared = dot(diff(point.a, closest_point), diff(point.a, closest_point));

    if (distance_squared <= point.radius * point.radius) {
        ans.res = true;
        ans.tOut = t;
        return ans;
    }

    ans.res = false;
    return ans;
}

__host__ __device__ double3 point_normal() {       
    return {0.0, 0.0, 0.0};
}

__host__ __device__ struct Trig {
    double3 a, b, c;
    uchar4 color;
    double r = 0.0;
    double tr = 0.0;
};

__host__ __device__ void trig_init(Trig *trig, double3 A, double3 B, double3 C, uchar4 col) {
    trig->a = A; trig->b = B; trig->c = C;
    trig->color = col;
    trig->r = 0.4;
    trig->tr = 0.6;
    // r = 0.2;
    // tr = 0.1;
}

__host__ __device__ uchar4 trig_at(Trig trig, double3 P = {0, 0, 0}) {
    double3 AB = diff(trig.b, trig.a);
    double3 AC = diff(trig.c, trig.a);
    double3 AP = diff(P, trig.a);

    double3 crossABAC = prod(AB, AC);
    double denom = dot(crossABAC, crossABAC);
    if (is_near(denom)) {
        return {0, 0, 0, 255};
    }

    double3 crossAPAC = prod(AP, AC); 
    double u = dot(crossAPAC, crossABAC) / denom;

    double3 crossABAP = prod(AB, AP);
    double v = dot(crossABAP, crossABAC) / denom;

    double w = 1.0 - u - v;
    double edge_width = 5e-2;

    if ((u >= 0.0 && v >= 0.0 && w >= 0.0) && (
        is_near(u, edge_width) || is_near(v, edge_width) || is_near(w, edge_width)) ) 
    {
        return {0, 0, 0, 255};
    }

    return trig.color;
}

__host__ __device__ IntersectDto trig_intersect(Trig trig, double3 pos, const double3 dir) {
    IntersectDto ans;
    
    double3 e1 = diff(trig.b, trig.a);
    double3 e2 = diff(trig.c, trig.a);
    double3 p  = prod(dir, e2);
    double div = dot(p, e1);

    if (std::fabs(div) < 1e-12) {
        ans.res = false;
        return ans;
    }

    double3 t = diff(pos, trig.a);
    double u = dot(p, t) / div;
    if (u < 0.0 || u > 1.0) {
        ans.res = false;
        return ans;
    }

    double3 q = prod(t, e1);
    double v = dot(q, dir) / div;
    if (v < 0.0 || (u + v) > 1.0) {
        ans.res = false;
        return ans;
    }

    double ts = dot(q, e2) / div;
    if (ts < 0.0) {
        ans.res = false;
        return ans;
    }

    ans.res = true;
    ans.tOut = ts;
    return ans;
}

__host__ __device__ double3 trig_normal(Trig trig) {
    double3 e1 = diff(trig.b, trig.a);
    double3 e2 = diff(trig.c, trig.a);
    double3 n = prod(e2, e1);
    return norm(n);
}

__host__ __device__ struct Rect {
    double3 a, b, c, d;
	int texW, texH;
    uchar4 color;
    double r = 0.0;
    double tr = 0.0;
};

__host__ __device__ void rect_init(Rect *rect, double3 A, double3 B, double3 C, double3 D, uchar4 col, int texW, int texH) {
    rect->a = A;  rect->b = B;  rect->c = C; rect->d = D;
    rect->color = col;
    rect->texH = texH;
    rect->texW = texW;
    rect->r = 0.5;
    rect->tr = 0.0;
}

__host__ __device__ IntersectDto rect_intersect(Rect rect, double3 pos, double3 dir) {
    double tMin = DBL_MAX;
    IntersectDto hit;

    {
        Trig t1;
        trig_init(&t1, rect.a, rect.b, rect.d, rect.color);
        IntersectDto t = trig_intersect(t1, pos, dir);
        if (t.res) {
            if (t.tOut < tMin) {
                tMin = t.tOut;
                hit.res = true;
            }
        }
    }
    {
        Trig t2;
        trig_init(&t2, rect.d, rect.b, rect.c, rect.color);
        IntersectDto t = trig_intersect(t2, pos, dir);
        if (t.res) {
            if (t.tOut < tMin) {
                tMin = t.tOut;
                hit.res = true;
            }
        }
    }

    if (hit.res) {
        hit.tOut = tMin;
    }
    return hit;
}

__host__ __device__ uchar4 rect_at(Rect rect, uchar4 *tex, double3 a = {0, 0, 0}) {
    double3 a0 = diff(a, rect.a);
    double3 c0 = diff(rect.c, rect.a);
    double eps = 1e-12;
    if (fabs(c0.x) < eps) c0.x = (c0.x < 0 ? -eps : eps);
    if (fabs(c0.y) < eps) c0.y = (c0.y < 0 ? -eps : eps);
    int x = round((a0.x / c0.x) * rect.texW); 
    int y = round((a0.y / c0.y) * rect.texH); 
    if (x < 0) x = 0;
    if (x >= rect.texW) x = rect.texW - 1;
    if (y < 0) y = 0;
    if (y >= rect.texH) y = rect.texH - 1;
    return tex[x + rect.texW * y];
};

__host__ __device__ double3 rect_normal(Rect rect) {
    double3 e1 = diff(rect.b, rect.a);
    double3 e2 = diff(rect.c, rect.a);
    double3 n  = prod(e1, e2);
    return norm(n);
}

__host__ __device__ double3 transform_figure(double3 v, double scale, double3 shift) {
    return double3 {
        v.x * scale + shift.x,
        v.y * scale + shift.y,
        v.z * scale + shift.z
    };
};

double3 *build_space(Rect *rects, uchar4 *floor_tex, int w, int h, int cnt_lights_on_edge) {
    // rects[0] = Rect(
    //     double3{-5, -5, 0}, 
    //     double3{ 5, -5, 0},
    //     double3{ 5,  5, 0},
    //     double3{-5,  5, 0},
    //     uchar4{0, 0, 255, 255},
    //     floor_tex,
    //     w,
    //     h
    // );
    rect_init(
        &rects[0],
        double3{-5, -5, 0}, 
        double3{ 5, -5, 0},
        double3{ 5,  5, 0},
        double3{-5,  5, 0},
        uchar4{0, 0, 255, 255},
        w,
        h
    );

    // double oct_scale = 1.5;
    // double3 oct_shift = {-2.5, 2.5, 0};
    // double3 oct_v0 = {0, 0, 3};  // top
    // double3 oct_v1 = {0, 0, 1};  // bottom
    // double3 oct_v2 = {0, 1, 2};  // front
    // double3 oct_v3 = {0,-1, 2};  // back
    // double3 oct_v4 = {1, 0, 2};  // right
    // double3 oct_v5 = {-1,0, 2};  // left
    // double3 oct_V0 = transform_figure(oct_v0, oct_scale, oct_shift);
    // double3 oct_V1 = transform_figure(oct_v1, oct_scale, oct_shift);
    // double3 oct_V2 = transform_figure(oct_v2, oct_scale, oct_shift);
    // double3 oct_V3 = transform_figure(oct_v3, oct_scale, oct_shift);
    // double3 oct_V4 = transform_figure(oct_v4, oct_scale, oct_shift);
    // double3 oct_V5 = transform_figure(oct_v5, oct_scale, oct_shift);
    // uchar4 oct_color = {100, 100, 100, 255};
    // polygons[1] = new Trig(oct_V0, oct_V2, oct_V4, oct_color);
    // polygons[2] = new Trig(oct_V0, oct_V4, oct_V3, oct_color);
    // polygons[3] = new Trig(oct_V0, oct_V3, oct_V5, oct_color);
    // polygons[4] = new Trig(oct_V0, oct_V5, oct_V2, oct_color);
    // polygons[5] = new Trig(oct_V1, oct_V4, oct_V2, oct_color);
    // polygons[6] = new Trig(oct_V1, oct_V3, oct_V4, oct_color);
    // polygons[7] = new Trig(oct_V1, oct_V5, oct_V3, oct_color);
    // polygons[8] = new Trig(oct_V1, oct_V2, oct_V5, oct_color);
    // int total_points = 1;
    // double3 *center_points = new double3[total_points];
    // double3 oct_point = oct_V0;
    // oct_point = add(oct_point, oct_V1);
    // oct_point = add(oct_point, oct_V2);
    // oct_point = add(oct_point, oct_V3);
    // oct_point = add(oct_point, oct_V4);
    // oct_point = add(oct_point, oct_V5);
    // oct_point = prod_num(oct_point, 1.0/6);
    // center_points[0] = oct_point;

    // int num_triangles = 8;              // Количество треугольников
    // int num_edges_per_triangle = 3;     // Количество рёбер на треугольник
    // int points_per_edge = 2; // Точек на каждом ребре

    // int point_index = 9; // Начинаем добавлять точки после треугольников

    // // Цикл по треугольникам
    // for (int i = 1; i <= num_triangles; i++) {
    //     Trig *triangle = static_cast<Trig*>(polygons[i]);

    //     double3 vA = triangle->a;
    //     double3 vB = triangle->b;
    //     double3 vC = triangle->c;

    //     // Три ребра: (A->B), (B->C), (C->A)
    //     double3 edges[3][2] = {
    //         {vA, vB},
    //         {vB, vC},
    //         {vC, vA}
    //     };

    //     for (int e = 0; e < num_edges_per_triangle; e++) {
    //         double3 p1 = edges[e][0];
    //         double3 p2 = edges[e][1];

    //         // Создаём точки на каждом ребре
    //         double step = 1.0 / (points_per_edge + 1);

    //         for (int j = 1; j <= points_per_edge; j++) {
    //             double t = j * step;
    //             double3 point_position = lerp(p1, p2, t);
    //             polygons[point_index++] = new Point(point_position);
    //         }
    //     }
    // }

    // return center_points;
    return nullptr;
}

void init_lights(Light *lights, double3 *trig_center_points, int n_trig_center_points) {
    Light amb;
    amb.type = LightType::AMBIENT;
    amb.intensity = 0.5; 
    lights[0] = amb;
    // for (int i = 0; i < n_trig_center_points; i++) {
    //     Light pnt;
    //     pnt.type = LightType::POINT;
    //     pnt.intensity = 0.8;
    //     pnt.position = trig_center_points[i];
    //     lights[i + 1] = pnt;
    // }
    // {
    //     Light pnt;
    //     pnt.type = LightType::POINT;
    //     pnt.intensity = 10.8;
    //     pnt.position  = {-4.0, 4.0, 3};
    //     lights[1] = pnt;
    // }
    // {
    //     Light pnt;
    //     pnt.type = LightType::POINT;
    //     pnt.intensity = 10.0;
    //     pnt.position  = {-1., -1., 1.};
    //     lights[1] = pnt;
    // }
}

__host__ __device__ double compute_shadow(Rect *rects, int n_rects, double3 O, double3 D, double t_min, double t_max) {
    double shadow_index = 1.0;
    for (int k = 0; k < n_rects; k++) {
        IntersectDto t = rect_intersect(rects[k], O, D);
        if (t.res) {
            if (t.tOut >= t_min && t.tOut <= t_max) {
                shadow_index *= rects[k].tr;
            }
        }
    }
    return shadow_index;
}

__host__ __device__ double3 reflect(double3 R, double3 N) {
    double dotNR = dot(N, R);
    double3 tmp = prod_num(N, 2.0*dotNR);
    return diff(tmp, R);
}

__host__ __device__ double compute_lighting(Rect *rects, int n_rects, Light *lights, int n_lights, double3 P, double3 N, double3 V, int specular) {
    double i = 0.0;
    for (int j = 0; j < n_lights; j++) {
        Light light = lights[j];

        if (light.type == LightType::AMBIENT) {
            i += light.intensity;
        }
        else {
            double3 L;
            double t_max; 
            if (light.type == LightType::POINT) {
                L = diff(light.position, P);
                t_max = 1.0;
            } else {
                L = light.direction; 
                t_max = 1e17;
            }

            // int shadowIndex;
            double shadowT = compute_shadow(rects, n_rects, P, L, 0.001, t_max);
            // double shadowT = 1.0;
            // if (closest_intersection(polygons, n_polygons, P, L, 0.001, t_max, shadowIndex, shadowT)) {
            //     continue;
            // }

            double n_dot_l = dot(N, L);
            if (n_dot_l > 0.0) {
                i += shadowT * light.intensity * n_dot_l / (std::sqrt(dot(N,N)) * std::sqrt(dot(L,L)));
            }

            if (specular >= 0) {
                double3 R = reflect(L, N); 
                double r_dot_v = dot(R, V);
                if (r_dot_v > 0.0) {
                    i += shadowT * light.intensity 
                         * std::pow(r_dot_v/(std::sqrt(dot(R,R))*std::sqrt(dot(V,V))), specular);
                }
            }
        }
    }

    return i;
}

__host__ __device__ double3 refract(const double3& ray_dir, double3& a_normal) 
{
    double3 res = {0, 0, 0};
    double eta = 1.5;
    double cos_theta = -dot(a_normal, ray_dir);

    if (cos_theta < 0) {
        cos_theta *= -1.0;
        a_normal = prod_num(a_normal, -1.0);
        eta = 1.0 / eta;
    }
    float k = 1.0 - eta*eta*(1.0-cos_theta*cos_theta);

    if (k >= 0.0) {
        double3 coef1 = prod_num(ray_dir, eta);
        double3 coef2 = prod_num(a_normal, (eta * cos_theta - sqrt(k)));
        double3 coef_sum = add(coef1, coef2);
        res = norm(coef_sum);
    }

    return res;
}

template<int depth>
__host__ __device__ uchar4 ray(Rect *rects, int n_rects, Light *lights, int n_lights, double3 pos, double3 dir, int max_depth, uchar4 *floor_tex) {
    const double light_reps = 0.125;
    for (int k = 0; k < n_lights; k++) {
        if (lights[k].type == LightType::POINT) {
            double3 light_dir = diff(lights[k].position, pos);
            double distance_to_light = std::sqrt(dot(light_dir, light_dir));
            light_dir = norm(light_dir);
            if (fabs(dot(light_dir, dir) - 1.0) < light_reps && distance_to_light < light_reps) {
                return {255, 255, 255, 255};
            }
        }
    }
    int k_min = -1;
    double ts_min = DBL_MAX;
    for (int k = 0; k < n_rects; k++) {
        IntersectDto t = rect_intersect(rects[k], pos, dir);
        if (t.res) {
            if (t.tOut < ts_min && t.tOut > 1e-6) {
                ts_min = t.tOut;
                k_min = k;
            }
        }
    }
    if (k_min == -1) {
        return {0, 0, 0, 255};
    }

    double t = ts_min;
    double3 P = add(pos, prod_num(dir, t));
    uchar4 base_color = rect_at(rects[k_min], floor_tex, P);
    double3 N = rect_normal(rects[k_min]);
    double3 V = {-dir.x, -dir.y, -dir.z};
    double intensity = compute_lighting(rects, n_rects, lights, n_lights, P, N, V, 110);
    uchar4 local_color = intensity * base_color;

        // bool a = rects[k_min].r;

    if (rects[k_min].r <= 0.0) {
        return local_color;
    }
        // return {a, 0, 0, 255};

    double3 reflected_dir = reflect(prod_num(dir, -1.0), N);
    reflected_dir = norm(reflected_dir);
    uchar4 reflected_color = ray<depth + 1>(rects, n_rects, lights, 
                                            n_lights, P, reflected_dir, max_depth, floor_tex);

    uchar4 refracted_color = reflected_color;
    double3 refracted_dir = refract(dir, N);
    if (std::sqrt(dot(refracted_dir, refracted_dir)) > 1e-6) {
        refracted_color = ray<depth + 1>(rects, n_rects, lights, 
                                         n_lights, P, refracted_dir, max_depth, floor_tex);
    }

    uchar4 out_color = rects[k_min].r * reflected_color + 
        (1 - rects[k_min].r - rects[k_min].tr) * local_color + 
        rects[k_min].tr * refracted_color;
    return out_color;
}

template<>
__host__ __device__ uchar4 ray<4>(Rect *rects, int n_rects, Light *lights, int n_lights, double3 pos, double3 dir, int max_depth, uchar4 *floor_tex) {
    return {0, 0, 0, 255};
}

__host__ __device__ uchar4 ssaa_pixel(const uchar4* big_data, int bigW, int bigH, int x, int y, int k) {
    long sumR = 0, sumG = 0, sumB = 0, sumA = 0;
    for (int j = 0; j < k; j++) {
        for (int i = 0; i < k; i++) {
            int xx = x*k + i;
            int yy = y*k + j;
            int idx = yy * bigW + xx;
            uchar4 c = big_data[idx];
            sumR += c.x;
            sumG += c.y;
            sumB += c.z;
            sumA += c.w;
        }
    }
    int count = k * k;
    uchar4 out;
    out.x = fmin(255.0, (double)(sumR / count));
    out.y = fmin(255.0, (double)(sumG / count));
    out.z = fmin(255.0, (double)(sumB / count));
    out.w = fmin(255.0, (double)(sumA / count));
    return out;
}

__global__ void ssaa_kernel(uchar4 *big_data, uchar4 *small_data, int bigW, int bigH, int w, int h, int k) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
   	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;
    int i, j;

    for (i = idy; i < w; i += offsety) {
        for (j = idx; j < h; j += offsetx) {
            small_data[i * w + j] = ssaa_pixel(big_data, bigW, bigH, j, i, k);
        }
    }
}

__global__ void render_kernel(Rect *rects, int n_rects, Light *lights, int n_lights, double3 pc, double3 pv, int w, int h, double angle, uchar4 *data, int max_depth, uchar4 *floor_tex) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
   	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;
    int i, j;
    
    double dw = 2.0 / (w - 1.0);
    double dh = 2.0 / (h - 1.0);
    double z  = 1.0 / std::tan(angle * M_PI / 360.0);

    double3 bz = norm(diff(pv, pc));
    double3 bx = norm(prod(bz, {0.0, 0.0, 1.0}));
    double3 by = norm(prod(bx, bz));
    for (i = idy; i < w; i += offsety) {
        for (j = idx; j < h; j += offsetx) {
            double3 v = {
                -1.0 + dw * i,
                (-1.0 + dh * j) * (double)h / (double)w,
                z
            };
            double3 dir = mult(bx, by, bz, v);
            dir = norm(dir);
            data[(h - 1 - j)*w + i] = ray<0>(
                rects, n_rects, lights, n_lights, pc, dir, max_depth, floor_tex
            );
        }
    }
}

__host__ __device__ void render(Rect *rects, int n_rects, Light *lights, int n_lights, double3 pc, double3 pv, int w, int h, double angle, uchar4 *data, int max_depth, uchar4 *floor_tex) {
    double dw = 2.0 / (w - 1.0);
    double dh = 2.0 / (h - 1.0);
    double z  = 1.0 / std::tan(angle * M_PI / 360.0);

    double3 bz = norm(diff(pv, pc));
    double3 bx = norm(prod(bz, {0.0, 0.0, 1.0}));
    double3 by = norm(prod(bx, bz));

    for (int i = 0; i < w; i++) {
        for (int j = 0; j < h; j++) {
            double3 v = {
                -1.0 + dw * i,
                (-1.0 + dh * j) * (double)h / (double)w,
                z
            };
            double3 dir = mult(bx, by, bz, v);
            dir = norm(dir);
            data[(h - 1 - j)*w + i] = ray<0>(
                rects, n_rects, lights, n_lights, pc, dir, max_depth, floor_tex
            );
        }
    }
}

int main(int argc, char const *argv[])
{
    int cuda = 1;

    int frame_cnt = atoi(argv[1]);
    int max_depth = 10;
    int n_rects = 1;
    Rect *rects = (Rect *) malloc(sizeof(Rect) * n_rects);
    int texW, texH;
   	FILE *fp = fopen("floor.data", "rb");
    fread(&texW, sizeof(int), 1, fp);
	fread(&texH, sizeof(int), 1, fp);
    uchar4 *floor_tex = (uchar4 *)malloc(sizeof(uchar4) * texW * texH);
    fread(floor_tex, sizeof(uchar4), texW * texH, fp);
    fclose(fp);

    int cnt_lights_on_center = 1;
    double3 *trig_center_points = build_space(rects, floor_tex, texW, texH, cnt_lights_on_center);
    // int n_lights = 3 * cnt_lights_on_edge * (n_rects - 1) + 1;
    int n_lights = 1;
    Light *lights = (Light *) malloc(sizeof(Light) *n_lights);
    init_lights(lights, trig_center_points, n_lights - 1);
    // printf("%lf %lf %lf\n", trig_center_points[0].x, trig_center_points[0].y, trig_center_points[0].z);
    int w = 500, h = 500;
    int k = 1;
    int bigW = w * k;
    int bigH = h * k;
    uchar4 *data_big = (uchar4*)malloc(sizeof(uchar4) * bigW * bigH);
    uchar4 *data_small = (uchar4*)malloc(sizeof(uchar4) * w * h);
    double3 pc, pv;
    char buff[256];
    
    uchar4 *data_big_dev;
    uchar4 *data_small_dev;
    Rect *rects_dev;
    uchar4 *floor_tex_dev;
    Light *lights_dev;
    if (cuda) {
        CSC(cudaMalloc(&data_big_dev, sizeof(uchar4) * bigW * bigH));
        CSC(cudaMemcpy(data_big_dev, data_big, sizeof(uchar4) * bigW * bigH, cudaMemcpyHostToDevice));
        CSC(cudaMalloc(&data_small_dev, sizeof(uchar4) * w * h));
        CSC(cudaMemcpy(data_small_dev, data_small, sizeof(uchar4) * w * h, cudaMemcpyHostToDevice));
        CSC(cudaMalloc(&rects_dev, sizeof(Rect) * n_rects));
        CSC(cudaMemcpy(rects_dev, rects, sizeof(Rect) * n_rects, cudaMemcpyHostToDevice));
        CSC(cudaMalloc(&lights_dev, sizeof(Light) * n_lights));
        CSC(cudaMemcpy(lights_dev, lights, sizeof(Light) * n_lights, cudaMemcpyHostToDevice));
        CSC(cudaMalloc(&floor_tex_dev, sizeof(uchar4) * texW * texH));
        CSC(cudaMemcpy(floor_tex_dev, floor_tex, sizeof(uchar4) * texW * texH, cudaMemcpyHostToDevice));
    }

    for(int idx = 0; idx < frame_cnt; idx++) {
        pc = {
            6.0 * std::sin(0.05 * idx), 
            6.0 * std::cos(0.05 * idx), 
            5.0 + 2.0 * std::sin(0.1 * idx)
        };
        pv = {
            3.0 * std::sin(0.05 * idx + M_PI), 
            3.0 * std::cos(0.05 * idx + M_PI), 
            0.0
        };
        
        if (cuda) {
            render_kernel<<<dim3(2, 2), dim3(2, 2)>>>(rects_dev, n_rects,
                lights_dev, n_lights, pc, pv, bigW, bigH, 120.0, data_big_dev, max_depth, floor_tex_dev);
            // printf("%d\n", cudaGetLastError());
        } else {
            render(rects, n_rects, lights, n_lights,
                pc, pv, bigW, bigH, 120.0, data_big, max_depth, floor_tex);
        }
        // CSC(cudaMemcpy(data_big_dev, data_big, sizeof(uchar4) * bigW * bigH, cudaMemcpyHostToDevice));
        if (cuda) {
            ssaa_kernel<<<dim3(2, 2), dim3(2, 2)>>>(data_big_dev, data_small_dev,
                bigW, bigH, w, h, k);
        } else {
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    data_small[y * w + x] = ssaa_pixel(data_big, bigW, bigH, x, y, k);
                }
            }
        }

        if (cuda) {
            CSC(cudaMemcpy(data_small, data_small_dev, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));
        }
        sprintf(buff, "frames/frame%d.out", idx);
        printf("Generating [%d/%d]: %s\n", idx+1, frame_cnt, buff);

        FILE *out = fopen(buff, "wb");
        fwrite(&w, sizeof(int), 1, out);
        fwrite(&h, sizeof(int), 1, out);
        fwrite(data_small, sizeof(uchar4), w*h, out);
        fclose(out);
        // printf("%d\n", cudaGetLastError());
    }

    if (cuda) {
        CSC(cudaFree(lights_dev));
        CSC(cudaFree(rects_dev));
        CSC(cudaFree(data_small_dev));
        CSC(cudaFree(data_big_dev));
    }
    free(trig_center_points);
    free(data_big);
    free(data_small);
    free(floor_tex);
    free(rects);
    free(lights);
    return 0;
}
