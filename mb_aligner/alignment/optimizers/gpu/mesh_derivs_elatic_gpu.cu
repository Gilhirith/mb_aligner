/*
 * GPU based implementation of the elastic mesh deriviatives computations.
 */


//#define FLOAT_INFINITY __int_as_float(0x7f800000)
#define FLOAT_INFINITY __int_as_float(-1)
#define SMALL_VALUE 0.0001

/* Huber Loss Functions */

inline __device__ float huber(
            const float value,
            const float target,
            const float sigma,
            const float d_value_dx,
            const float d_value_dy,
            float* d_huber_dx,
            float* d_huber_dy
            )
{
    float diff, a, b;

    diff = value - target;
    if (abs(diff) <= sigma) {
        a = (diff * diff) / 2;
        *d_huber_dx = diff * d_value_dx;
        *d_huber_dy = diff * d_value_dy;
        return a;
    } else {
        b = sigma * (abs(diff) - sigma / 2);
        *d_huber_dx = sigma * d_value_dx;
        *d_huber_dy = sigma * d_value_dy;
        return b;
    }
}


/* Regularized length function */
inline __device__ float reglen(
            const float vx,
            const float vy,
            const float d_vx_dx,
            const float v_vy_dy,
            float* d_reglen_dx,
            float* d_reglen_dy
            )
{
    float sq_len, sqrt_len;

    sq_len = vx * vx + vy * vy + SMALL_VALUE;
    sqrt_len = sqrt(sq_len);
    *d_reglen_dx = vx / sqrt_len;
    *d_reglen_dy = vy / sqrt_len;
    return sqrt_len;
}


/* Mesh cross-link derivs */
__global__ void crosslink_mesh_derivs(
            const float2* mesh1,
            const float2* mesh2,
            float2* d_cost_d_mesh1,
            float2* d_cost_d_mesh2,
            const uint3* indices1,
            const uint3* indices2,
            const int indices_num,
            const float3* barys1,
            const float3* barys2,
            const float all_weight,
            const float sigma,
            float* crosslink_costs
            )
{
    float px, py, qx, qy;
    int pidx0, pidx1, pidx2;
    int qidx0, qidx1, qidx2;
    float pb0, pb1, pb2;
    float qb0, qb1, qb2;
    float r, h;
    float dr_dx, dr_dy, dh_dx, dh_dy;
    float cost = 0;

    const int idx = blockDim.x * blockIdx.x + threadIdx.x;

    // Check that the given index is valid
    if (idx >= indices_num)
        return;

    pidx0 = indices1[idx].x;
    pidx1 = indices1[idx].y;
    pidx2 = indices1[idx].z;
    pb0 = barys1[idx].x;
    pb1 = barys1[idx].y;
    pb2 = barys1[idx].z;

    qidx0 = indices2[idx].x;
    qidx1 = indices2[idx].y;
    qidx2 = indices2[idx].z;
    qb0 = barys2[idx].x;
    qb1 = barys2[idx].y;
    qb2 = barys2[idx].z;

    px = (mesh1[pidx0].x * pb0 +
          mesh1[pidx1].x * pb1 +
          mesh1[pidx2].x * pb2);
    py = (mesh1[pidx0].y * pb0 +
          mesh1[pidx1].y * pb1 +
          mesh1[pidx2].y * pb2);

    qx = (mesh2[qidx0].x * qb0 +
          mesh2[qidx1].x * qb1 +
          mesh2[qidx2].x * qb2);
    qy = (mesh2[qidx0].y * qb0 +
          mesh2[qidx1].y * qb1 +
          mesh2[qidx2].y * qb2);

    r = reglen(px - qx, py - qy,
               1, 1,
               &(dr_dx), &(dr_dy));
    h = huber(r, 0, sigma,
              dr_dx, dr_dy,
              &(dh_dx), &(dh_dy));

    cost = h * all_weight;
    dh_dx *= all_weight;
    dh_dy *= all_weight;

    // update derivs
    atomicAdd(&d_cost_d_mesh1[pidx0].x, pb0 * dh_dx);
    atomicAdd(&d_cost_d_mesh1[pidx1].x, pb1 * dh_dx);
    atomicAdd(&d_cost_d_mesh1[pidx2].x, pb2 * dh_dx);
    atomicAdd(&d_cost_d_mesh1[pidx0].y, pb0 * dh_dy);
    atomicAdd(&d_cost_d_mesh1[pidx1].y, pb1 * dh_dy);
    atomicAdd(&d_cost_d_mesh1[pidx2].y, pb2 * dh_dy);
    // opposite direction for other end of spring, and distributed according to weight
    atomicAdd(&d_cost_d_mesh2[qidx0].x, -qb0 * dh_dx);
    atomicAdd(&d_cost_d_mesh2[qidx1].x, -qb1 * dh_dx);
    atomicAdd(&d_cost_d_mesh2[qidx2].x, -qb2 * dh_dx);
    atomicAdd(&d_cost_d_mesh2[qidx0].y, -qb0 * dh_dy);
    atomicAdd(&d_cost_d_mesh2[qidx1].y, -qb1 * dh_dy);
    atomicAdd(&d_cost_d_mesh2[qidx2].y, -qb2 * dh_dy);

    crosslink_costs[idx] = cost;
}

/* Mesh internal-links derivs */
__global__ void internal_mesh_derivs(
            const float2* mesh,
            float2* d_cost_d_mesh,
            const uint2* edge_indices,
            const int edge_indices_num,
            const float* rest_lengths,
            const float all_weight,
            const float sigma,
            float* edge_costs
            )
{
    int idx1, idx2;
    float px, py, qx, qy;
    float r, h;
    float dr_dx, dr_dy, dh_dx, dh_dy;
    float cost = 0;
    
    const int edge_idx = blockDim.x * blockIdx.x + threadIdx.x;
    // Check that the given edge index is valid
    if (edge_idx >= edge_indices_num)
        return;

    idx1 = edge_indices[edge_idx].x;
    idx2 = edge_indices[edge_idx].y;

    px = mesh[idx1].x;
    py = mesh[idx1].y;
    qx = mesh[idx2].x;
    qy = mesh[idx2].y;

    r = reglen(px - qx, py - qy,
               1.0, 1.0,
               &(dr_dx), &(dr_dy));
    h = huber(r, rest_lengths[edge_idx], sigma,
              dr_dx, dr_dy,
              &(dh_dx), &(dh_dy));

    cost = h * all_weight;
    dh_dx *= all_weight;
    dh_dy *= all_weight;

    // update derivs
    atomicAdd(&d_cost_d_mesh[idx1].x, dh_dx);
    atomicAdd(&d_cost_d_mesh[idx1].y, dh_dy);
    atomicAdd(&d_cost_d_mesh[idx2].x, -dh_dx);
    atomicAdd(&d_cost_d_mesh[idx2].y, -dh_dy);

    edge_costs[edge_idx] = cost;
}

/* Mesh area derivs (triangle area) */
__global__ void area_mesh_derivs(
            const float2* mesh,
            float2* d_cost_d_mesh,
            const uint3* triangle_indices,
            const int triangle_indices_num,
            const float* rest_areas,
            const float all_weight,
            float* triangle_costs
            )
{
    int idx0, idx1, idx2;
    float v01x, v01y, v02x, v02y, area, r_area;
    float cost, dc_da;
    float a1;

    const int triangle_idx = blockDim.x * blockIdx.x + threadIdx.x;
    // Check that the given edge index is valid
    if (triangle_idx >= triangle_indices_num)
        return;

    idx0 = triangle_indices[triangle_idx].x;
    idx1 = triangle_indices[triangle_idx].y;
    idx2 = triangle_indices[triangle_idx].z;

    v01x = mesh[idx1].x - mesh[idx0].x;
    v01y = mesh[idx1].y - mesh[idx0].y;
    v02x = mesh[idx2].x - mesh[idx0].x;
    v02y = mesh[idx2].y - mesh[idx0].y;

    area = 0.5 * (v02x * v01y - v01x * v02y);
    r_area = rest_areas[triangle_idx];
    if (area * r_area <= 0) {
        cost = FLOAT_INFINITY;
        dc_da = 0;
    } else {
        /*
        # cost is ((A - A_rest) / A) ^ 2 * A_rest  (last term is for area normalization)
        #
        #      / A  -  A     \ 2
        #      |        rest |     |       |
        #      | ----------- |   * | A     |
        #      \      A      /     |  rest |
        */
        a1 = ((area - r_area) / area);
        cost = all_weight * (a1 * a1);
        dc_da = 2 * all_weight * r_area * (area - r_area) / (area * area * area);
    }


    // update derivs
    atomicAdd(&d_cost_d_mesh[idx1].x, dc_da * 0.5 * (-v02y));
    atomicAdd(&d_cost_d_mesh[idx1].y, dc_da * 0.5 * (v02x));
    atomicAdd(&d_cost_d_mesh[idx2].x, dc_da * 0.5 * (v01y));
    atomicAdd(&d_cost_d_mesh[idx2].y, dc_da * 0.5 * (-v01x));

    // sum of negative of above
    atomicAdd(&d_cost_d_mesh[idx0].x, dc_da * 0.5 * (v02y - v01y));
    atomicAdd(&d_cost_d_mesh[idx0].y, dc_da * 0.5 * (v01x - v02x));
 
    triangle_costs[triangle_idx] = cost;
}


/* Mesh area derivs (triangle area) */
__global__ void update_mesh_and_momentum_grads(
            float2* mesh,
            const int pts_num,
            const float2* grads,
            float2* momentum_grads,
            const float momentum,
            const float stepsize
            )
{

    const int pt_idx = blockDim.x * blockIdx.x + threadIdx.x;
    // Check that the given point index is valid
    if (pt_idx >= pts_num)
        return;

    // Update the gradients with momentum
    // gradients_with_momentum[sec_idx] = gradients[sec_idx] + momentum * gradients_with_momentum[sec_idx]
    momentum_grads[pt_idx].x = grads[pt_idx].x + momentum * momentum_grads[pt_idx].x;
    momentum_grads[pt_idx].y = grads[pt_idx].y + momentum * momentum_grads[pt_idx].y;

    // Update the mesh points
    // meshes[sec_idx].pts -= stepsize * gradients_with_momentum[sec_idx]
    mesh[pt_idx].x -= stepsize * momentum_grads[pt_idx].x;
    mesh[pt_idx].y -= stepsize * momentum_grads[pt_idx].y;
}

