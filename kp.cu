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

__host__ __device__ double dot(const double3 &a, const double3 &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ double3 prod(const double3 &a, const double3 &b) {
    return {
        a.y * b.z - a.z * b.y, 
        a.z * b.x - a.x * b.z,  
        a.x * b.y - a.y * b.x
    };
}

__host__ __device__ double3 norm(const double3 &v) {
    double l = std::sqrt(dot(v, v));
    if (l < 1e-12) return {0,0,0};
    return {v.x / l, v.y / l, v.z / l};
}

__host__ __device__ double3 diff(const double3 &a, const double3 &b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__host__ __device__ double3 add(const double3 &a, const double3 &b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__host__ __device__ double3 prod_num(const double3 &a, double b) {
    return {a.x * b, a.y * b, a.z * b};
}

__host__ __device__ double3 lerp(const double3 &a, const double3 &b, double t) {
    double3 ab = diff(b, a);
    double3 mod_ab = prod_num(ab, t);
    double3 res = add(a, mod_ab);
    return res;
}

__host__ __device__ double3 mult(const double3 &a, const double3 &b, 
                                 const double3 &c, const double3 &v) {
    return {
        a.x * v.x + b.x * v.y + c.x * v.z,
        a.y * v.x + b.y * v.y + c.y * v.z,
        a.z * v.x + b.z * v.y + c.z * v.z
    };
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

class Polygon {
public:
    uchar4 color;
    double r = 0.0;
    double tr = 0.0;

    __host__ __device__ virtual uchar4 at(double3 a = {0, 0, 0}) const = 0;
    __host__ __device__ virtual bool intersect(const double3 &pos, 
                                            const double3 &dir, double &tOut) const = 0;
    __host__ __device__ virtual double3 normal() const = 0;
    __host__ __device__ virtual ~Polygon() {}
};

class Trig : public Polygon {
public:
    double3 a, b, c;

    __host__ __device__ Trig(const double3 &A, const double3 &B, const double3 &C, uchar4 col) {
        a = A;  b = B;  c = C;
        color = col;
        r = 0.2;
        tr = 0.6;
    }

    __host__ __device__ uchar4 at(double3 P = {0, 0, 0}) const override {
        double3 AB = diff(b, a);
        double3 AC = diff(c, a);
        double3 AP = diff(P, a);

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

        return color;
    }

    __host__ __device__ bool intersect(const double3 &pos, const double3 &dir, double &tOut) const override {
        double3 e1 = diff(b, a);
        double3 e2 = diff(c, a);
        double3 p  = prod(dir, e2);
        double div = dot(p, e1);

        if (std::fabs(div) < 1e-12) {
            return false;
        }

        double3 t = diff(pos, a);
        double u = dot(p, t) / div;
        if (u < 0.0 || u > 1.0) {
            return false;
        }

        double3 q = prod(t, e1);
        double v = dot(q, dir) / div;
        if (v < 0.0 || (u + v) > 1.0) {
            return false;
        }

        double ts = dot(q, e2) / div;
        if (ts < 0.0) {
            return false;
        }
        tOut = ts;
        return true;
    }

    __host__ __device__ double3 normal() const override {
        double3 e1 = diff(b, a);
        double3 e2 = diff(c, a);
        double3 n = prod(e2, e1);
        return norm(n);
    }
};

class Rect : public Polygon {
public:
    double3 a, b, c, d;
	uchar4 *tex;
	int texW, texH;
    
    __host__ __device__ Rect(const double3 &A, const double3 &B, const double3 &C, 
        const double3 &D, uchar4 col, uchar4 *tex, int texW, int texH) {
        a = A;  b = B;  c = C;  d = D;
        color = col;
        this->tex = tex;
        this->texH = texH;
        this->texW = texW;
        r = 0.5;
        tr = 0.0;
    }

    __host__ __device__ uchar4 at(double3 a = {0, 0, 0}) const override {
        double3 a0 = diff(a, this->a);
        double3 c0 = diff(this->c, this->a);
        double eps = 1e-12;
        if (fabs(c0.x) < eps) c0.x = (c0.x < 0 ? -eps : eps);
        if (fabs(c0.y) < eps) c0.y = (c0.y < 0 ? -eps : eps);
        int x = round((a0.x / c0.x) * this->texW); 
        int y = round((a0.y / c0.y) * this->texH); 
        if (x < 0) x = 0;
        if (x >= texW) x = texW - 1;
        if (y < 0) y = 0;
        if (y >= texH) y = texH - 1;
        return this->tex[x + this->texW * y];
    };

    __host__ __device__ bool intersect(const double3 &pos, const double3 &dir, double &tOut) const override {
        double tMin = DBL_MAX;
        bool hit = false;

        {
            Trig t1(a, b, d, color);
            double t;
            if (t1.intersect(pos, dir, t)) {
                if (t < tMin) {
                    tMin = t;
                    hit = true;
                }
            }
        }
        {
            Trig t2(d, b, c, color);
            double t;
            if (t2.intersect(pos, dir, t)) {
                if (t < tMin) {
                    tMin = t;
                    hit = true;
                }
            }
        }

        if (hit) {
            tOut = tMin;
        }
        return hit;
    }

    __host__ __device__ double3 normal() const override {
        double3 e1 = diff(b, a);
        double3 e2 = diff(c, a);
        double3 n  = prod(e1, e2);
        return norm(n);
    }
};

__host__ __device__ double3 transform_figure(const double3 &v, double scale, double3 shift) {
    return double3 {
        v.x * scale + shift.x,
        v.y * scale + shift.y,
        v.z * scale + shift.z
    };
};

__host__ __device__ double3 *build_space(Polygon **polygons, uchar4 *floor_tex, 
                                                int w, int h, int cnt_lights_on_edge) {
    polygons[0] = new Rect(
        double3{-5, -5, 0}, 
        double3{ 5, -5, 0},
        double3{ 5,  5, 0},
        double3{-5,  5, 0},
        uchar4{0, 0, 255, 255},
        floor_tex,
        w,
        h
    );

    double oct_scale = 1.5;
    double3 oct_shift = {-2.5, 2.5, 0};
    double3 oct_v0 = {0, 0, 3};  // top
    double3 oct_v1 = {0, 0, 1};  // bottom
    double3 oct_v2 = {0, 1, 2};  // front
    double3 oct_v3 = {0,-1, 2};  // back
    double3 oct_v4 = {1, 0, 2};  // right
    double3 oct_v5 = {-1,0, 2};  // left
    double3 oct_V0 = transform_figure(oct_v0, oct_scale, oct_shift);
    double3 oct_V1 = transform_figure(oct_v1, oct_scale, oct_shift);
    double3 oct_V2 = transform_figure(oct_v2, oct_scale, oct_shift);
    double3 oct_V3 = transform_figure(oct_v3, oct_scale, oct_shift);
    double3 oct_V4 = transform_figure(oct_v4, oct_scale, oct_shift);
    double3 oct_V5 = transform_figure(oct_v5, oct_scale, oct_shift);
    uchar4 oct_color = {100, 100, 100, 255};
    polygons[1] = new Trig(oct_V0, oct_V2, oct_V4, oct_color);
    polygons[2] = new Trig(oct_V0, oct_V4, oct_V3, oct_color);
    polygons[3] = new Trig(oct_V0, oct_V3, oct_V5, oct_color);
    polygons[4] = new Trig(oct_V0, oct_V5, oct_V2, oct_color);
    polygons[5] = new Trig(oct_V1, oct_V4, oct_V2, oct_color);
    polygons[6] = new Trig(oct_V1, oct_V3, oct_V4, oct_color);
    polygons[7] = new Trig(oct_V1, oct_V5, oct_V3, oct_color);
    polygons[8] = new Trig(oct_V1, oct_V2, oct_V5, oct_color);

    int num_triangles = 8;
    int num_edges_per_triangle = 3;
    int total_points = num_triangles * num_edges_per_triangle * cnt_lights_on_edge;
    double3 *edge_points = nullptr;
    if (cnt_lights_on_edge > 0) {
        edge_points = new double3[total_points];
        int write_idx = 0;
        for (int i = 1; i <= 8; i++) {
            Trig *triangle = static_cast<Trig*>(polygons[i]);
            double3 vA = triangle->a;
            double3 vB = triangle->b;
            double3 vC = triangle->c;
            double3 edges[3][2] = {
                {vA, vB},
                {vB, vC},
                {vC, vA}
            };
            double step = 1.0 / (cnt_lights_on_edge + 1);
            for (int e = 0; e < 3; e++) {
                double3 p1 = edges[e][0];
                double3 p2 = edges[e][1];
                for (int j = 1; j <= cnt_lights_on_edge; j++) {
                    double t = j * step;
                    double3 point = lerp(p1, p2, t);
                    edge_points[write_idx++] = point;
                }
            }
        }
    }

    return edge_points;
}

__host__ __device__ void init_lights(Light *lights, double3 *trig_edge_points, int n_trig_edge_points) {
    Light amb;
    amb.type = LightType::AMBIENT;
    amb.intensity = 0.5; 
    lights[0] = amb;
    for (int i = 1; i <= n_trig_edge_points; i++) {
        Light pnt;
        pnt.type = LightType::POINT;
        pnt.intensity = 0.2;
        pnt.position = trig_edge_points[i];
        lights[i] = pnt;
    }
    // {
    //     Light pnt;
    //     pnt.type = LightType::POINT;
    //     pnt.intensity = 0.8;
    //     pnt.position  = {-4.0, 4.0, 3};
    //     lights[1] = pnt;
    // }
    // {
    //     Light pnt;
    //     pnt.type = LightType::POINT;
    //     pnt.intensity = 1.0;
    //     pnt.position  = {0.0, 0.0, 0.01};
    //     lights[2] = pnt;
    // }
}

__host__ __device__ bool closest_intersection(Polygon **polygons, int n_polygons, const double3 &O, 
               const double3 &D, double t_min, double t_max, int &closest_index, double &closest_t) {
    closest_t = DBL_MAX;
    closest_index = -1;
    for (int k = 0; k < n_polygons; k++) {
        double t;
        if (polygons[k]->intersect(O, D, t)) {
            if (t >= t_min && t <= t_max && t < closest_t) {
                closest_t = t;
                closest_index = k;
            }
        }
    }
    return (closest_index != -1);
}

__host__ __device__ double3 reflect(const double3 &R, const double3 &N) {
    double dotNR = dot(N, R);
    double3 tmp = prod_num(N, 2.0*dotNR);
    return diff(tmp, R);
}

__host__ __device__ double compute_lighting(Polygon **polygons, int n_polygons, 
                                            Light *lights, int n_lights, const double3 &P, 
                                            const double3 &N, const double3 &V, int specular) {
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

            int shadowIndex;
            double shadowT;
            if (closest_intersection(polygons, n_polygons, P, L, 0.001, t_max, shadowIndex, shadowT)) {
                continue;
            }

            double n_dot_l = dot(N, L);
            if (n_dot_l > 0.0) {
                i += light.intensity * n_dot_l / ( std::sqrt(dot(N,N)) * std::sqrt(dot(L,L)));
            }

            if (specular >= 0) {
                double3 R = reflect(L, N); 
                double r_dot_v = dot(R, V);
                if (r_dot_v > 0.0) {
                    i += light.intensity 
                         * std::pow( r_dot_v/(std::sqrt(dot(R,R))*std::sqrt(dot(V,V))), specular);
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
__host__ __device__ uchar4 ray(Polygon **polygons, int n_polygons, Light *lights, int n_lights,
                                         const double3 &pos, const double3 &dir, int max_depth) {
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
    for (int k = 0; k < n_polygons; k++) {
        double t;
        if (polygons[k]->intersect(pos, dir, t)) {
            if (t < ts_min && t > 1e-6) {
                ts_min = t;
                k_min  = k;
            }
        }
    }
    if (k_min == -1) {
        return {0, 0, 0, 255};
    }

    double t = ts_min;
    double3 P = add(pos, prod_num(dir, t));
    uchar4 base_color = polygons[k_min]->at(P);
    double3 N = polygons[k_min]->normal();
    double3 V = {-dir.x, -dir.y, -dir.z};
    double intensity = compute_lighting(polygons, n_polygons, lights, n_lights, P, N, V, 110);
    uchar4 local_color = intensity * base_color;

    if (depth >= max_depth || polygons[k_min]->r <= 0.0) {
        return local_color;
    }

    double3 reflected_dir = reflect(prod_num(dir, -1.0), N);
    reflected_dir = norm(reflected_dir);
    uchar4 reflected_color = ray<depth + 1>(polygons, n_polygons, lights, 
                                            n_lights, P, reflected_dir, max_depth);

    uchar4 refracted_color = reflected_color;
    double3 refracted_dir = refract(dir, N);
    if (std::sqrt(dot(refracted_dir, refracted_dir)) > 1e-6) {
        refracted_color = ray<depth + 1>(polygons, n_polygons, lights, 
                                         n_lights, P, refracted_dir, max_depth);
    }

    uchar4 out_color = polygons[k_min]->r * reflected_color + 
        (1 - polygons[k_min]->r - polygons[k_min]->tr) * local_color + 
        polygons[k_min]->tr * refracted_color;
    return out_color;
}

template<>
__host__ __device__ uchar4 ray<100>(Polygon **polygons, int n_polygons, Light *lights, int n_lights,
                                              const double3 &pos, const double3 &dir, int max_depth) {
    return {0, 0, 0, 255};
}

__host__ __device__ uchar4 ssaa_pixel(const uchar4* big_data, int bigW, int bigH, int x, int y, int k)
{
    long sumR = 0, sumG = 0, sumB = 0, sumA = 0;
    for (int j = 0; j < k; j++) {
        for (int i = 0; i < k; i++) {
            int xx = x*k + i;
            int yy = y*k + j;
            int idx = yy * bigW + xx;
            const uchar4 &c = big_data[idx];
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

__host__ __device__ void render(Polygon **polygons, int n_polygons, Light *lights, int n_lights, const double3 &pc,
                    const double3 &pv, int w, int h, double angle, uchar4 *data, int max_depth) {
    double dw = 2.0 / (w - 1.0);    // это вынести в main
    double dh = 2.0 / (h - 1.0);
    double z  = 1.0 / std::tan(angle * M_PI / 360.0);

    double3 bz = norm(diff(pv, pc));
    double3 bx = norm(prod(bz, {0.0, 0.0, 1.0}));
    double3 by = norm(prod(bx, bz));

    for (int i = 0; i < w; i++) {    // Параллелить эти 2 for-а
        for (int j = 0; j < h; j++) {
            double3 v = {
                -1.0 + dw * i,
                (-1.0 + dh * j) * (double)h / (double)w,
                z
            };
            double3 dir = mult(bx, by, bz, v);
            dir = norm(dir);
            data[(h - 1 - j)*w + i] = ray<0>(
                polygons, n_polygons, lights, n_lights, pc, dir, max_depth
            );
        }
    }
}

int main(int argc, char const *argv[])
{
    int cuda = 0;

    int frame_cnt = atoi(argv[1]);
    int max_depth = 10;

    int n_polygons = 9;
    Polygon **polygons = new Polygon*[n_polygons];
    int texW, texH;
   	FILE *fp = fopen("floor.data", "rb");
    fread(&texW, sizeof(int), 1, fp);
	fread(&texH, sizeof(int), 1, fp);
    uchar4 *floor_tex = (uchar4 *)malloc(sizeof(uchar4) * texW * texH);
    fread(floor_tex, sizeof(uchar4), texW * texH, fp);
    fclose(fp);

    int cnt_lights_on_edge = 1;
    double3 *trig_edge_points = build_space(polygons, floor_tex, texW, texH, cnt_lights_on_edge);
    int n_lights = 3 * cnt_lights_on_edge * (n_polygons - 1) + 1;
    printf("%d\n", n_lights);
    Light *lights = new Light[n_lights];
    init_lights(lights, trig_edge_points, n_lights - 1);

    int w = 640, h = 480;
    int k = 3;
    int bigW = w * k;
    int bigH = h * k;
    uchar4 *data_big = (uchar4*)malloc(sizeof(uchar4) * bigW * bigH);
    uchar4 *data_small = (uchar4*)malloc(sizeof(uchar4) * w * h);
    double3 pc, pv;
    char buff[256];
    
    if (cuda) {

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

        render(polygons, n_polygons, lights, n_lights,
               pc, pv, bigW, bigH, 120.0, data_big, max_depth);     // вот тут параллелить

        for (int y = 0; y < h; y++) {   // параллелить данные 2 for-а
            for (int x = 0; x < w; x++) {
                data_small[y * w + x] = ssaa_pixel(data_big, bigW, bigH, x, y, k);
            }
        }
        sprintf(buff, "frames/frame%d.out", idx);
        printf("Generating [%d/%d]: %s\n", idx+1, frame_cnt, buff);

        FILE *out = fopen(buff, "wb");
        fwrite(&w, sizeof(int), 1, out);
        fwrite(&h, sizeof(int), 1, out);
        fwrite(data_small, sizeof(uchar4), w*h, out);
        fclose(out);
    }
    
    free(trig_edge_points);
    free(data_big);
    free(data_small);
    free(floor_tex);
    for (int i = 0; i < n_polygons; i++) {
        delete polygons[i];
    }
    delete[] polygons;
    delete[] lights;
    return 0;
}
