#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <memory>
#include <float.h>
#include <time.h>
#include <sys/time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__host__ __device__ double pow_bin(double base, double exp) {
    if (exp == 0.0) return 1.0;
    if (exp < 0.0) {
        base = 1.0 / base;
        exp = -exp;
    }

    double result = 1.0;
    double current_base = base;
    int integer_part = static_cast<int>(exp);

    while (integer_part > 0) {
        if (integer_part % 2 == 1) {
            result *= current_base;
        }
        current_base *= current_base;
        integer_part /= 2;
    }

    double fractional_part = exp - static_cast<int>(exp);
    if (fractional_part > 0.0) {
        double fractional_result = 1.0;
        double precision = 1e-7;
        double x = base;
        for (int i = 0; i < 100; ++i) {
            fractional_result *= 1.0 + fractional_part * (x - 1.0) / (i + 1);
            if (x - fractional_result < precision) break;
        }
        result *= fractional_result;
    }

    return result;
}

__host__ __device__ double sqrt_custom(double x) {
    if (x < 0.0) return -1.0;
    if (x == 0.0) return 0.0;

    double result = x;
    double epsilon = 1e-7;
    double previous_result = 0.0;

    while (fabs(result - previous_result) > epsilon) {
        previous_result = result;
        result = 0.5 * (result + x / result);
    }

    return result;
}

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
    return fabs(x) < eps;
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
    double l = sqrt_custom(dot(v, v));
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

__host__ __device__ double3 reflect(double3 R, double3 N) {
    double dotNR = dot(N, R);
    double3 tmp = prod_num(N, 2.0*dotNR);
    return diff(tmp, R);
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
        double3 coef2 = prod_num(a_normal, (eta * cos_theta - sqrt_custom(k)));
        double3 coef_sum = add(coef1, coef2);
        res = norm(coef_sum);
    }

    return res;
}

enum class LightType {
    AMBIENT,
    POINT
};

struct Light {
    LightType type;
    double intensity;
    double3 position;
    double3 direction;
};

struct IntersectDto {
    bool res = false;
    double tOut = 0.0;
};

struct Trig {
    double3 a, b, c;
    uchar4 color;
    double r = 0.0;
    double tr = 0.0;
};

__host__ __device__ void trig_init(Trig *trig, double3 A, double3 B, double3 C, uchar4 col, double r, double tr) {
    trig->a = A; trig->b = B; trig->c = C;
    trig->color = col;
    trig->r = r;
    trig->tr = tr;
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

    if (fabs(div) < 1e-12) {
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

struct Rect {
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
        trig_init(&t1, rect.a, rect.b, rect.d, rect.color, 0.1, 0.6);
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
        trig_init(&t2, rect.d, rect.b, rect.c, rect.color, 0.1, 0.6);
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

struct FigureDto {
    double3 center;
    double3 ncolor;
    uchar4 color;
    double radius;
    double r;
    double tr;
    int num_points;
};

struct FloorDto {
    double3 a;
    double3 b;
    double3 c;
    double3 d;
    double3 color;
};

void build_space(Rect *rects, Trig *trigs, uchar4 *floor_tex, int w, int h, FigureDto *figure_data, FloorDto floor_data) {
    rect_init(
        &rects[0],
        floor_data.a,
        floor_data.b,
        floor_data.c,
        floor_data.d,
        uchar4{0, 0, 255, 255},
        w,
        h
    );

    double oct_radius = 1.5;
    double oct_scale = figure_data[0].radius / oct_radius;
    double3 oct_shift = figure_data[0].center;
    double3 oct_v0 = {0, 0, 1};
    double3 oct_v1 = {0, 0,-1};
    double3 oct_v2 = {0, 1, 0};
    double3 oct_v3 = {0,-1, 0};
    double3 oct_v4 = {1, 0, 0};
    double3 oct_v5 = {-1,0, 0};
    double3 oct_V0 = transform_figure(oct_v0, oct_scale, oct_shift);
    double3 oct_V1 = transform_figure(oct_v1, oct_scale, oct_shift);
    double3 oct_V2 = transform_figure(oct_v2, oct_scale, oct_shift);
    double3 oct_V3 = transform_figure(oct_v3, oct_scale, oct_shift);
    double3 oct_V4 = transform_figure(oct_v4, oct_scale, oct_shift);
    double3 oct_V5 = transform_figure(oct_v5, oct_scale, oct_shift);
    uchar4 oct_color = figure_data[0].color;
    trig_init(&trigs[0], oct_V0, oct_V2, oct_V4, oct_color, figure_data[0].r, figure_data[0].tr);
    trig_init(&trigs[1], oct_V0, oct_V4, oct_V3, oct_color, figure_data[0].r, figure_data[0].tr);
    trig_init(&trigs[2], oct_V0, oct_V3, oct_V5, oct_color, figure_data[0].r, figure_data[0].tr);
    trig_init(&trigs[3], oct_V0, oct_V5, oct_V2, oct_color, figure_data[0].r, figure_data[0].tr);
    trig_init(&trigs[4], oct_V1, oct_V4, oct_V2, oct_color, figure_data[0].r, figure_data[0].tr);
    trig_init(&trigs[5], oct_V1, oct_V3, oct_V4, oct_color, figure_data[0].r, figure_data[0].tr);
    trig_init(&trigs[6], oct_V1, oct_V5, oct_V3, oct_color, figure_data[0].r, figure_data[0].tr);
    trig_init(&trigs[7], oct_V1, oct_V2, oct_V5, oct_color, figure_data[0].r, figure_data[0].tr);

    double dodec_radius = 1.732;
    double dodec_scale = figure_data[1].radius / dodec_radius;
    double3 dodec_shift = figure_data[1].center;
    double3 dodec_v0 = {1.0, 1.0, 1.0};
    double3 dodec_v1 = {1.0, 1.0, -1.0};
    double3 dodec_v2 = {1.0, -1.0, 1.0};
    double3 dodec_v3 = {1.0, -1.0, -1.0};
    double3 dodec_v4 = {-1.0, 1.0, 1.0};
    double3 dodec_v5 = {-1.0, 1.0, -1.0};
    double3 dodec_v6 = {-1.0, -1.0, 1.0};
    double3 dodec_v7 = {-1.0, -1.0, -1.0};
    double3 dodec_v8 = {0.0, 0.618034, 1.61803};
    double3 dodec_v9 = {0.0, 0.618034, -1.61803};
    double3 dodec_v10 = {0.0, -0.618034, 1.61803};
    double3 dodec_v11 = {0.0, -0.618034, -1.61803};
    double3 dodec_v12 = {0.618034, 1.61803, 0.0};
    double3 dodec_v13 = {0.618034, -1.61803, 0.0};
    double3 dodec_v14 = {-0.618034, 1.61803, 0.0};
    double3 dodec_v15 = {-0.618034, -1.61803, 0.0};
    double3 dodec_v16 = {1.61803, 0.0, 0.618034};
    double3 dodec_v17 = {1.61803, 0.0, -0.618034};
    double3 dodec_v18 = {-1.61803, 0.0, 0.618034};
    double3 dodec_v19 = {-1.61803, 0.0, -0.618034};
    double3 dodec_V0 = transform_figure(dodec_v0, dodec_scale, dodec_shift);
    double3 dodec_V1 = transform_figure(dodec_v1, dodec_scale, dodec_shift);
    double3 dodec_V2 = transform_figure(dodec_v2, dodec_scale, dodec_shift);
    double3 dodec_V3 = transform_figure(dodec_v3, dodec_scale, dodec_shift);
    double3 dodec_V4 = transform_figure(dodec_v4, dodec_scale, dodec_shift);
    double3 dodec_V5 = transform_figure(dodec_v5, dodec_scale, dodec_shift);
    double3 dodec_V6 = transform_figure(dodec_v6, dodec_scale, dodec_shift);
    double3 dodec_V7 = transform_figure(dodec_v7, dodec_scale, dodec_shift);
    double3 dodec_V8 = transform_figure(dodec_v8, dodec_scale, dodec_shift);
    double3 dodec_V9 = transform_figure(dodec_v9, dodec_scale, dodec_shift);
    double3 dodec_V10 = transform_figure(dodec_v10, dodec_scale, dodec_shift);
    double3 dodec_V11 = transform_figure(dodec_v11, dodec_scale, dodec_shift);
    double3 dodec_V12 = transform_figure(dodec_v12, dodec_scale, dodec_shift);
    double3 dodec_V13 = transform_figure(dodec_v13, dodec_scale, dodec_shift);
    double3 dodec_V14 = transform_figure(dodec_v14, dodec_scale, dodec_shift);
    double3 dodec_V15 = transform_figure(dodec_v15, dodec_scale, dodec_shift);
    double3 dodec_V16 = transform_figure(dodec_v16, dodec_scale, dodec_shift);
    double3 dodec_V17 = transform_figure(dodec_v17, dodec_scale, dodec_shift);
    double3 dodec_V18 = transform_figure(dodec_v18, dodec_scale, dodec_shift);
    double3 dodec_V19 = transform_figure(dodec_v19, dodec_scale, dodec_shift);
    uchar4 dodec_color = figure_data[1].color;
    trig_init(&trigs[8], dodec_V8, dodec_V4, dodec_V14, dodec_color, figure_data[1].r, figure_data[1].tr);
    trig_init(&trigs[9], dodec_V8, dodec_V14, dodec_V0, dodec_color, figure_data[1].r, figure_data[1].tr);
    trig_init(&trigs[10], dodec_V0, dodec_V14, dodec_V12, dodec_color, figure_data[1].r, figure_data[1].tr);
    trig_init(&trigs[11], dodec_V16, dodec_V0, dodec_V12, dodec_color, figure_data[1].r, figure_data[1].tr);
    trig_init(&trigs[12], dodec_V16, dodec_V12, dodec_V1, dodec_color, figure_data[1].r, figure_data[1].tr);
    trig_init(&trigs[13], dodec_V16, dodec_V1, dodec_V7, dodec_color, figure_data[1].r, figure_data[1].tr);
    trig_init(&trigs[14], dodec_V10, dodec_V8, dodec_V0, dodec_color, figure_data[1].r, figure_data[1].tr);
    trig_init(&trigs[15], dodec_V10, dodec_V0, dodec_V16, dodec_color, figure_data[1].r, figure_data[1].tr);
    trig_init(&trigs[16], dodec_V10, dodec_V16, dodec_V2, dodec_color, figure_data[1].r, figure_data[1].tr);
    trig_init(&trigs[17], dodec_V10, dodec_V6, dodec_V8, dodec_color, figure_data[1].r, figure_data[1].tr);
    trig_init(&trigs[18], dodec_V8, dodec_V6, dodec_V18, dodec_color, figure_data[1].r, figure_data[1].tr);
    trig_init(&trigs[19], dodec_V8, dodec_V18, dodec_V4, dodec_color, figure_data[1].r, figure_data[1].tr);
    trig_init(&trigs[21], dodec_V4, dodec_V18, dodec_V19, dodec_color, figure_data[1].r, figure_data[1].tr);
    trig_init(&trigs[22], dodec_V4, dodec_V19, dodec_V14, dodec_color, figure_data[1].r, figure_data[1].tr);
    trig_init(&trigs[23], dodec_V14, dodec_V19, dodec_V5, dodec_color, figure_data[1].r, figure_data[1].tr);
    trig_init(&trigs[24], dodec_V14, dodec_V5, dodec_V12, dodec_color, figure_data[1].r, figure_data[1].tr);
    trig_init(&trigs[25], dodec_V12, dodec_V5, dodec_V9, dodec_color, figure_data[1].r, figure_data[1].tr);
    trig_init(&trigs[26], dodec_V12, dodec_V9, dodec_V1, dodec_color, figure_data[1].r, figure_data[1].tr);
    trig_init(&trigs[27], dodec_V19, dodec_V7, dodec_V11, dodec_color, figure_data[1].r, figure_data[1].tr);
    trig_init(&trigs[28], dodec_V19, dodec_V11, dodec_V5, dodec_color, figure_data[1].r, figure_data[1].tr);
    trig_init(&trigs[29], dodec_V5, dodec_V11, dodec_V9, dodec_color, figure_data[1].r, figure_data[1].tr);
    trig_init(&trigs[30], dodec_V6, dodec_V15, dodec_V18, dodec_color, figure_data[1].r, figure_data[1].tr);
    trig_init(&trigs[31], dodec_V18, dodec_V15, dodec_V7, dodec_color, figure_data[1].r, figure_data[1].tr);
    trig_init(&trigs[32], dodec_V18, dodec_V7, dodec_V19, dodec_color, figure_data[1].r, figure_data[1].tr);
    trig_init(&trigs[33], dodec_V10, dodec_V2, dodec_V13, dodec_color, figure_data[1].r, figure_data[1].tr);
    trig_init(&trigs[34], dodec_V10, dodec_V13, dodec_V15, dodec_color, figure_data[1].r, figure_data[1].tr);
    trig_init(&trigs[36], dodec_V10, dodec_V15, dodec_V6, dodec_color, figure_data[1].r, figure_data[1].tr);
    trig_init(&trigs[37], dodec_V2, dodec_V16, dodec_V17, dodec_color, figure_data[1].r, figure_data[1].tr);
    trig_init(&trigs[38], dodec_V2, dodec_V17, dodec_V3, dodec_color, figure_data[1].r, figure_data[1].tr);
    trig_init(&trigs[39], dodec_V2, dodec_V3, dodec_V13, dodec_color, figure_data[1].r, figure_data[1].tr);
    trig_init(&trigs[40], dodec_V13, dodec_V3, dodec_V15, dodec_color, figure_data[1].r, figure_data[1].tr);
    trig_init(&trigs[41], dodec_V15, dodec_V13, dodec_V11, dodec_color, figure_data[1].r, figure_data[1].tr);
    trig_init(&trigs[42], dodec_V15, dodec_V1, dodec_V7, dodec_color, figure_data[1].r, figure_data[1].tr);
    trig_init(&trigs[43], dodec_V17, dodec_V1, dodec_V9, dodec_color, figure_data[1].r, figure_data[1].tr);
    trig_init(&trigs[44], dodec_V3, dodec_V17, dodec_V9, dodec_color, figure_data[1].r, figure_data[1].tr);
    trig_init(&trigs[45], dodec_V17, dodec_V11, dodec_V3, dodec_color, figure_data[1].r, figure_data[1].tr);

    double ekos_radius = 2.118;
    double ekos_scale = figure_data[2].radius / ekos_radius;
    double3 ekos_shift = figure_data[2].center;
    double3 ekos_v0 = {1.0, 1.61803, 0.0};
    double3 ekos_v1 = {-1.0, 1.61803, 0.0};
    double3 ekos_v2 = {1.0, -1.61803, 0.0};
    double3 ekos_v3 = {-1.0, -1.61803, 0.0};
    double3 ekos_v4 = {0.0, 1.0, 1.61803};
    double3 ekos_v5 = {0.0, -1.0, 1.61803};
    double3 ekos_v6 = {0.0, 1.0, -1.61803};
    double3 ekos_v7 = {0.0, -1.0, -1.61803};
    double3 ekos_v8 = {1.61803, 0.0, 1.0};
    double3 ekos_v9 = {-1.61803, 0.0, 1.0};
    double3 ekos_v10 = {1.61803, 0.0, -1.0};
    double3 ekos_v11 = {-1.61803, 0.0, -1.0};
    double3 ekos_V0 = transform_figure(ekos_v0, ekos_scale, ekos_shift);
    double3 ekos_V1 = transform_figure(ekos_v1, ekos_scale, ekos_shift);
    double3 ekos_V2 = transform_figure(ekos_v2, ekos_scale, ekos_shift);
    double3 ekos_V3 = transform_figure(ekos_v3, ekos_scale, ekos_shift);
    double3 ekos_V4 = transform_figure(ekos_v4, ekos_scale, ekos_shift);
    double3 ekos_V5 = transform_figure(ekos_v5, ekos_scale, ekos_shift);
    double3 ekos_V6 = transform_figure(ekos_v6, ekos_scale, ekos_shift);
    double3 ekos_V7 = transform_figure(ekos_v7, ekos_scale, ekos_shift);
    double3 ekos_V8 = transform_figure(ekos_v8, ekos_scale, ekos_shift);
    double3 ekos_V9 = transform_figure(ekos_v9, ekos_scale, ekos_shift);
    double3 ekos_V10 = transform_figure(ekos_v10, ekos_scale, ekos_shift);
    double3 ekos_V11 = transform_figure(ekos_v11, ekos_scale, ekos_shift);
    uchar4 ekos_color = figure_data[2].color;
    trig_init(&trigs[46], ekos_V5, ekos_V4, ekos_V8, ekos_color, figure_data[2].r, figure_data[2].tr);
    trig_init(&trigs[47], ekos_V4, ekos_V9, ekos_V1, ekos_color, figure_data[2].r, figure_data[2].tr);
    trig_init(&trigs[48], ekos_V4, ekos_V5, ekos_V9, ekos_color, figure_data[2].r, figure_data[2].tr);
    trig_init(&trigs[49], ekos_V5, ekos_V3, ekos_V9, ekos_color, figure_data[2].r, figure_data[2].tr);
    trig_init(&trigs[50], ekos_V4, ekos_V1, ekos_V0, ekos_color, figure_data[2].r, figure_data[2].tr);
    trig_init(&trigs[51], ekos_V8, ekos_V4, ekos_V0, ekos_color, figure_data[2].r, figure_data[2].tr);
    trig_init(&trigs[52], ekos_V2, ekos_V5, ekos_V8, ekos_color, figure_data[2].r, figure_data[2].tr);
    trig_init(&trigs[53], ekos_V8, ekos_V4, ekos_V1, ekos_color, figure_data[2].r, figure_data[2].tr);
    trig_init(&trigs[54], ekos_V5, ekos_V2, ekos_V3, ekos_color, figure_data[2].r, figure_data[2].tr);
    trig_init(&trigs[55], ekos_V10, ekos_V2, ekos_V8, ekos_color, figure_data[2].r, figure_data[2].tr);
    trig_init(&trigs[56], ekos_V10, ekos_V8, ekos_V0, ekos_color, figure_data[2].r, figure_data[2].tr);
    trig_init(&trigs[57], ekos_V7, ekos_V2, ekos_V10, ekos_color, figure_data[2].r, figure_data[2].tr);
    trig_init(&trigs[58], ekos_V7, ekos_V10, ekos_V6, ekos_color, figure_data[2].r, figure_data[2].tr);
    trig_init(&trigs[59], ekos_V0, ekos_V6, ekos_V10, ekos_color, figure_data[2].r, figure_data[2].tr);
    trig_init(&trigs[60], ekos_V0, ekos_V1, ekos_V1, ekos_color, figure_data[2].r, figure_data[2].tr);
    trig_init(&trigs[61], ekos_V1, ekos_V9, ekos_V11, ekos_color, figure_data[2].r, figure_data[2].tr);
    trig_init(&trigs[62], ekos_V1, ekos_V11, ekos_V6, ekos_color, figure_data[2].r, figure_data[2].tr);
    trig_init(&trigs[63], ekos_V11, ekos_V7, ekos_V6, ekos_color, figure_data[2].r, figure_data[2].tr);
    trig_init(&trigs[64], ekos_V9, ekos_V3, ekos_V11, ekos_color, figure_data[2].r, figure_data[2].tr);
    trig_init(&trigs[65], ekos_V3, ekos_V7, ekos_V11, ekos_color, figure_data[2].r, figure_data[2].tr);
    trig_init(&trigs[66], ekos_V3, ekos_V2, ekos_V7, ekos_color, figure_data[2].r, figure_data[2].tr);
}

struct LightDto {
    double3 position;
    double3 color;
};

void init_lights(Light *lights, LightDto *lights_data, int n_lights) {
    Light amb;
    amb.type = LightType::AMBIENT;
    amb.intensity = 0.5; 
    lights[0] = amb;
    for (int i = 1; i < n_lights; i++) {
        Light pnt;
        pnt.type = LightType::POINT;
        pnt.intensity = 1.0;
        pnt.position = lights_data[i - 1].position;
        lights[i] = pnt;
    }
}

__host__ __device__ double compute_shadow(Rect *rects, int n_rects, Trig *trigs, int n_trigs, double3 O, double3 D, double t_min, double t_max) {
    double shadow_index = 1.0;
    for (int k = 0; k < n_rects; k++) {
        IntersectDto t = rect_intersect(rects[k], O, D);
        if (t.res) {
            if (t.tOut >= t_min && t.tOut <= t_max) {
                shadow_index *= rects[k].tr;
            }
        }
    }
    for (int k = 0; k < n_trigs; k++) {
        IntersectDto t = trig_intersect(trigs[k], O, D);
        if (t.res) {
            if (t.tOut >= t_min && t.tOut <= t_max) {
                shadow_index *= trigs[k].tr;
            }
        }
    }
    return shadow_index;
}

__host__ __device__ double compute_lighting(Rect *rects, int n_rects, Trig *trigs, int n_trigs, Light *lights, int n_lights, double3 P, double3 N, double3 V, int specular) {
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

            double shadowT = compute_shadow(rects, n_rects, trigs, n_trigs, P, L, 0.001, t_max);

            double n_dot_l = dot(N, L);
            if (n_dot_l > 0.0) {
                i += shadowT * light.intensity * n_dot_l / (sqrt_custom(dot(N,N)) * sqrt_custom(dot(L,L)));
            }

            if (specular >= 0) {
                double3 R = reflect(L, N); 
                double r_dot_v = dot(R, V);
                if (r_dot_v > 0.0) {
                    i += shadowT * light.intensity 
                         * pow_bin(r_dot_v/(sqrt_custom(dot(R,R))*sqrt_custom(dot(V,V))), specular);
                }
            }
        }
    }

    return i;
}

template<int depth>
__host__ __device__ uchar4 ray(Rect *rects, int n_rects, Trig *trigs, int n_trigs, Light *lights, int n_lights, double3 pos, double3 dir, int max_depth, uchar4 *floor_tex) {
    const double light_reps = 0.125;
    for (int k = 0; k < n_lights; k++) {
        if (lights[k].type == LightType::POINT) {
            double3 light_dir = diff(lights[k].position, pos);
            double distance_to_light = sqrt_custom(dot(light_dir, light_dir));
            light_dir = norm(light_dir);
            if (fabs(dot(light_dir, dir) - 1.0) < light_reps && distance_to_light < light_reps) {
                return {255, 255, 255, 255};
            }
        }
    }
    int k_min = -1, k_type = -1;
    double ts_min = DBL_MAX;
    for (int k = 0; k < n_rects; k++) {
        IntersectDto t = rect_intersect(rects[k], pos, dir);
        if (t.res) {
            if (t.tOut < ts_min && t.tOut > 1e-6) {
                ts_min = t.tOut;
                k_min = k;
                k_type = 0;
            }
        }
    }
    for (int k = 0; k < n_trigs; k++) {
        IntersectDto t = trig_intersect(trigs[k], pos, dir);
        if (t.res) {
            if (t.tOut < ts_min && t.tOut > 1e-6) {
                ts_min = t.tOut;
                k_min = k;
                k_type = 1;
            }
        }
    }

    if (k_min == -1) {
        return {0, 0, 0, 255};
    }

    double t = ts_min;
    double3 P = add(pos, prod_num(dir, t));
    uchar4 base_color = {255, 255, 255, 255};
    double3 N = {0.0, 0.0, 0.0};
    if (k_type == 0) {
        base_color = rect_at(rects[k_min], floor_tex, P);
        N = rect_normal(rects[k_min]);
    } else if (k_type == 1) {
        base_color = trig_at(trigs[k_min], P);
        N = trig_normal(trigs[k_min]);
    }
    
    double3 V = {-dir.x, -dir.y, -dir.z};
    double intensity = compute_lighting(rects, n_rects, trigs, n_trigs, lights, n_lights, P, N, V, 110);
    uchar4 local_color = intensity * base_color;
    
    if (k_type == 0) {
        if (rects[k_min].r <= 0.0) {
            return local_color;
        }
    } else if (k_type == 1) {
        if (trigs[k_min].r <= 0.0) {
            return local_color;
        }
    }

    double3 reflected_dir = reflect(prod_num(dir, -1.0), N);
    reflected_dir = norm(reflected_dir);
    uchar4 reflected_color = ray<depth + 1>(rects, n_rects, trigs, n_trigs, lights, 
                                            n_lights, P, reflected_dir, max_depth, floor_tex);

    uchar4 refracted_color = reflected_color;
    double3 refracted_dir = refract(dir, N);
    if (sqrt_custom(dot(refracted_dir, refracted_dir)) > 1e-6) {
        refracted_color = ray<depth + 1>(rects, n_rects, trigs, n_trigs, lights,
                                         n_lights, P, refracted_dir, max_depth, floor_tex);
    }

    uchar4 out_color = local_color;
    if (k_type == 0) {
        out_color = rects[k_min].r * reflected_color + 
            (1 - rects[k_min].r - rects[k_min].tr) * local_color + 
            rects[k_min].tr * refracted_color;
    } else if (k_type == 1) {
        out_color = trigs[k_min].r * reflected_color + 
            (1 - trigs[k_min].r - trigs[k_min].tr) * local_color + 
            trigs[k_min].tr * refracted_color;
    }
    return out_color;
}

template<>
__host__ __device__ uchar4 ray<4>(Rect *rects, int n_rects, Trig *trigs, int n_trigs, Light *lights, int n_lights, double3 pos, double3 dir, int max_depth, uchar4 *floor_tex) {
    return {0, 0, 0, 255};
}

__global__ void render_kernel(Rect *rects, int n_rects, Trig *trigs, int n_trigs, Light *lights, int n_lights, double3 pc, double3 pv, int w, int h, double angle, uchar4 *data, int max_depth, uchar4 *floor_tex) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
   	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;
    int i, j;
    
    double dw = 2.0 / (w - 1.0);
    double dh = 2.0 / (h - 1.0);
    double z  = 1.0 / tan(angle * M_PI / 360.0);

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
                rects, n_rects, trigs, n_trigs, lights, n_lights, pc, dir, max_depth, floor_tex
            );
        }
    }
}

void render(Rect *rects, int n_rects, Trig *trigs, int n_trigs, Light *lights, int n_lights, double3 pc, double3 pv, int w, int h, double angle, uchar4 *data, int max_depth, uchar4 *floor_tex) {
    double dw = 2.0 / (w - 1.0);
    double dh = 2.0 / (h - 1.0);
    double z  = 1.0 / tan(angle * M_PI / 360.0);

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
                rects, n_rects, trigs, n_trigs, lights, n_lights, pc, dir, max_depth, floor_tex
            );
        }
    }
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

    for (i = idy; i < h; i += offsety) {
        for (j = idx; j < w; j += offsetx) {
            small_data[i * w + j] = ssaa_pixel(big_data, bigW, bigH, j, i, k);
        }
    }
}

int main(int argc, char const *argv[])
{
    int cuda = 1;
    if (argc == 2) {
        if (strcmp(argv[1], "--cpu") == 0) {
            cuda = 0;
        } else if (strcmp(argv[1], "--default") == 0) {
            printf("300\n");
            printf("floor.data\n");
            printf("frames/frame%s.out\n", "%d");
            printf("1280 720 120\n");
            printf("6.0 5.0 0.0 2.0 1.0 0.05 0.1 0.05 0.0 0.0\n");
            printf("3.0 0.0 3.1415926 1.5 0.5 0.03 0.07 0.02 0.0 0.0\n");
            printf("-2.5 2.5 2 0.9843 0.4745 0.0156 2.25 0.1 0.6\n");
            printf("2.5 2.5 2 0.49 0.3216 0.0078 1.732 0.1 0.6\n");
            printf("2.5 -2.5 2 0.949 0.773 0.05 2.118 0.1 0.6\n");
            printf("-5 -5 0\n");
            printf("5 -5 0\n");
            printf("5 5 0\n");
            printf("-5 5 0\n");
            printf("2\n");
            printf("-4 4 3 1 1 1\n");
            printf("5 0 4 1 1 1\n");
            printf("4 3\n");
            return 0;
        }
    }

    int frame_cnt = 126;
    char floor_file_path[1024];
    char frame_files_path[1024];
    int w = 640, h = 480;
    double angle = 120.0;

    double r_c0 = 6.0, z_c0 = 5.0, phi_c0 = 0.0;
    double A_c_r = 2.0, A_c_z = 1.0;
    double omega_c_r = 0.05, omega_c_z = 0.1, omega_c_phi = 0.05;
    double p_c_r = 0.0, p_c_z = 0.0;

    double r_n0 = 3.0, z_n0 = 0.0, phi_n0 = M_PI;
    double A_n_r = 1.5, A_n_z = 0.5;
    double omega_n_r = 0.03, omega_n_z = 0.07, omega_n_phi = 0.02;
    double p_n_r = 0.0, p_n_z = 0.0;

    FigureDto *figures_data = (FigureDto *) malloc(sizeof(FigureDto) * 3);
    FloorDto floor_data;
    
    int n_lights = 3;
    LightDto *lights_data;
    int max_depth = 10;
    int k = 3;

    scanf("%d", &frame_cnt);
    getchar();
    scanf("%s", floor_file_path);
    getchar();
    scanf("%s", frame_files_path);
    getchar();
    scanf("%d %d %lf", &w, &h, &angle);
    getchar();
    scanf("%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf", &r_c0, &z_c0, &phi_c0,
          &A_c_r, &A_c_z, &omega_c_r, &omega_c_z, &omega_c_phi, &p_c_r, &p_c_z);
    getchar();
    scanf("%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf", &r_n0, &z_n0, &phi_n0,
          &A_n_r, &A_n_z, &omega_n_r, &omega_n_z, &omega_n_phi, &p_n_r, &p_n_z);
    getchar();
    for (int idx = 0; idx < 3; idx++) {
        scanf("%lf %lf %lf", &figures_data[idx].center.x, &figures_data[idx].center.y, &figures_data[idx].center.z);
        getchar();
        scanf("%lf %lf %lf", &figures_data[idx].ncolor.x, &figures_data[idx].ncolor.y, &figures_data[idx].ncolor.z);
        getchar();
        figures_data[idx].color = {
            (uchar)(figures_data[idx].ncolor.x * 255),
            (uchar)(figures_data[idx].ncolor.y * 255),
            (uchar)(figures_data[idx].ncolor.z * 255),
            255,
        };
        scanf("%lf", &figures_data[idx].radius);
        getchar();
        scanf("%lf", &figures_data[idx].r);
        getchar();
        scanf("%lf", &figures_data[idx].tr);
        getchar();
        scanf("%d", &figures_data[idx].num_points);
        getchar();
    }
    scanf("%lf %lf %lf", &floor_data.a.x, &floor_data.a.y, &floor_data.a.z);
    getchar();
    scanf("%lf %lf %lf", &floor_data.b.x, &floor_data.b.y, &floor_data.b.z);
    getchar();
    scanf("%lf %lf %lf", &floor_data.c.x, &floor_data.c.y, &floor_data.c.z);
    getchar();
    scanf("%lf %lf %lf", &floor_data.d.x, &floor_data.d.y, &floor_data.d.z);
    getchar();
    scanf("%lf %lf %lf", &floor_data.color.x, &floor_data.color.y, &floor_data.color.z);
    getchar();
    scanf("%d", &n_lights);
    getchar();
    n_lights += 1;
    lights_data = (LightDto *) malloc(sizeof(LightDto) * n_lights);
    for (int idx = 1; idx < n_lights; idx++) {
        scanf("%lf %lf %lf", &lights_data[idx].position.x, &lights_data[idx].position.y, &lights_data[idx].position.z);
        scanf("%lf %lf %lf", &lights_data[idx].color.x, &lights_data[idx].color.y, &lights_data[idx].color.z);
    }
    scanf("%d %d", &max_depth, &k);

    int n_rects = 1;
    Rect *rects = (Rect *) malloc(sizeof(Rect) * n_rects);
    int n_trigs = 67;
    Trig *trigs = (Trig *) malloc(sizeof(Trig) * n_trigs);
    int texW, texH;
   	FILE *fp = fopen(floor_file_path, "rb");
    fread(&texW, sizeof(int), 1, fp);
	fread(&texH, sizeof(int), 1, fp);
    uchar4 *floor_tex = (uchar4 *)malloc(sizeof(uchar4) * texW * texH);
    fread(floor_tex, sizeof(uchar4), texW * texH, fp);
    fclose(fp);

    build_space(rects, trigs, floor_tex, texW, texH, figures_data, floor_data);
    Light *lights = (Light *) malloc(sizeof(Light) * n_lights);
    init_lights(lights, lights_data, n_lights);
    
    int bigW = w * k;
    int bigH = h * k;
    uchar4 *data_big = (uchar4*)malloc(sizeof(uchar4) * bigW * bigH);
    uchar4 *data_small = (uchar4*)malloc(sizeof(uchar4) * w * h);
    double3 pc, pv;
    char buff[256];
    
    uchar4 *data_big_dev;
    uchar4 *data_small_dev;
    Rect *rects_dev;
    Trig *trigs_dev;
    uchar4 *floor_tex_dev;
    Light *lights_dev;
    if (cuda) {
        CSC(cudaMalloc(&data_big_dev, sizeof(uchar4) * bigW * bigH));
        CSC(cudaMemcpy(data_big_dev, data_big, sizeof(uchar4) * bigW * bigH, cudaMemcpyHostToDevice));
        CSC(cudaMalloc(&data_small_dev, sizeof(uchar4) * w * h));
        CSC(cudaMemcpy(data_small_dev, data_small, sizeof(uchar4) * w * h, cudaMemcpyHostToDevice));
        CSC(cudaMalloc(&rects_dev, sizeof(Rect) * n_rects));
        CSC(cudaMemcpy(rects_dev, rects, sizeof(Rect) * n_rects, cudaMemcpyHostToDevice));
        CSC(cudaMalloc(&trigs_dev, sizeof(Trig) * n_trigs));
        CSC(cudaMemcpy(trigs_dev, trigs, sizeof(Trig) * n_trigs, cudaMemcpyHostToDevice));
        CSC(cudaMalloc(&lights_dev, sizeof(Light) * n_lights));
        CSC(cudaMemcpy(lights_dev, lights, sizeof(Light) * n_lights, cudaMemcpyHostToDevice));
        CSC(cudaMalloc(&floor_tex_dev, sizeof(uchar4) * texW * texH));
        CSC(cudaMemcpy(floor_tex_dev, floor_tex, sizeof(uchar4) * texW * texH, cudaMemcpyHostToDevice));
    }

    float start_cpu, end_cpu;
    cudaEvent_t start, stop;
    float gpu_time = 0.0;
    if (cuda) {
        CSC(cudaEventCreate(&start));
        CSC(cudaEventCreate(&stop));
    }
    for(int idx = 0; idx < frame_cnt; idx++) {
        if (cuda) {
            CSC(cudaEventRecord(start, 0));
        } else {
            start_cpu = clock();
        }
        double t = idx * 1.0;
        double r_c = r_c0 + A_c_r * sin(omega_c_r * t + p_c_r);
        double z_c = z_c0 + A_c_z * sin(omega_c_z * t + p_c_z);
        double phi_c = phi_c0 + omega_c_phi * t;

        pc = {
            r_c * cos(phi_c),
            r_c * sin(phi_c),
            z_c
        };

        double r_n = r_n0 + A_n_r * sin(omega_n_r * t + p_n_r);
        double z_n = z_n0 + A_n_z * sin(omega_n_z * t + p_n_z);
        double phi_n = phi_n0 + omega_n_phi * t;

        pv = {
            r_n * cos(phi_n),
            r_n * sin(phi_n),
            z_n
        };
        
        if (cuda) {
            render_kernel<<<dim3(16, 16), dim3(16, 16)>>>(rects_dev, n_rects, trigs_dev, n_trigs,
                lights_dev, n_lights, pc, pv, bigW, bigH, angle, data_big_dev, max_depth, floor_tex_dev);
        } else {
            render(rects, n_rects, trigs, n_trigs, lights, n_lights,
                pc, pv, bigW, bigH, angle, data_big, max_depth, floor_tex);
        }
        if (cuda) {
            ssaa_kernel<<<dim3(16, 16), dim3(16, 16)>>>(data_big_dev, data_small_dev,
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
        sprintf(buff, frame_files_path, idx);
        // printf("Generating [%d/%d]: %s\n", idx+1, frame_cnt, buff);

        FILE *out = fopen(buff, "wb");
        fwrite(&w, sizeof(int), 1, out);
        fwrite(&h, sizeof(int), 1, out);
        fwrite(data_small, sizeof(uchar4), w*h, out);
        fclose(out);

        if (cuda) {
            CSC(cudaEventRecord(stop, 0));
            CSC(cudaEventSynchronize(stop));
            CSC(cudaEventElapsedTime(&gpu_time, start, stop));
            printf("%f %d\n", gpu_time, w * h * k * k);
        } else {
            end_cpu = clock();
            printf("%f %d\n", (double)((end_cpu - start_cpu) / 1000), w * h * k * k);
        }
    }

    if (cuda) {
        CSC(cudaFree(lights_dev));
        CSC(cudaFree(rects_dev));
        CSC(cudaFree(trigs_dev));
        CSC(cudaFree(data_small_dev));
        CSC(cudaFree(data_big_dev));
    }
    free(data_big);
    free(data_small);
    free(lights);
    free(trigs);
    free(rects);
    free(lights_data);
    free(figures_data);
    free(floor_tex);
    return 0;
}
