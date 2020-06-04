
#define EPS 0.00001

__device__ float distance(const float2 f1, const float2 f2)
{
    float2 v;
    v.x = f2.x - f1.x;
    v.y = f2.y - f1.y;
    return sqrt(v.x * v.x + v.y * v.y);
}

__device__ float distance_f2_f(const float2 f1, const float f2_x, const float f2_y)
{
    float2 v;
    v.x = f2_x - f1.x;
    v.y = f2_y - f1.y;
    return sqrt(v.x * v.x + v.y * v.y);
}

__device__ float distance_f_f(const float f1_x, const float f1_y, const float f2_x, const float f2_y)
{
    float2 v;
    v.x = f2_x - f1_x;
    v.y = f2_y - f1_y;
    return sqrt(v.x * v.x + v.y * v.y);
}


__device__ float2 rigid_transform(const float2 pt, const float3 params)
{
    float2 out;
    float cos_theta = cos(params.x);
    float sin_theta = sin(params.x);
    out.x = cos_theta * pt.x - sin_theta * pt.y + params.y;
    out.y = sin_theta * pt.x + cos_theta * pt.y + params.z;
    return out;
}

__device__ float compute_cost_L2(const int cur_match_idx, const float2 *matches_src, const float2 *matches_dst, const float3* tiles_params, const int* match_src_idx_to_tile_idx, const int* match_dst_idx_to_tile_idx)
{
    float2 src_pt_transformed = rigid_transform(matches_src[cur_match_idx], tiles_params[match_src_idx_to_tile_idx[cur_match_idx]]);
    float2 dst_pt_transformed = rigid_transform(matches_dst[cur_match_idx], tiles_params[match_dst_idx_to_tile_idx[cur_match_idx]]);
    float dist = distance(src_pt_transformed, dst_pt_transformed);
    return dist;
}


__global__ void compute_cost_huber(const float2 *matches_src, const float2 *matches_dst, const int matches_num, const float3* tiles_params, const int* match_src_idx_to_tile_idx, const int* match_dst_idx_to_tile_idx, const float huber_delta, float* out_cost)
{
    const int cur_match_idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (cur_match_idx >= matches_num)
        return;

    float cost;
    float dist = compute_cost_L2(cur_match_idx, matches_src, matches_dst, tiles_params, match_src_idx_to_tile_idx, match_dst_idx_to_tile_idx);
    
    // Apply huber cost
    if (dist <= huber_delta)
    {
        cost = 0.5 * dist * dist;
    }
    else
    {
        cost = huber_delta * dist - (0.5 * huber_delta * huber_delta);
    }

    out_cost[cur_match_idx] = cost;
}

__global__ void compute_cost(const float2 *matches_src, const float2 *matches_dst, const int matches_num, const float3* tiles_params, const int* match_src_idx_to_tile_idx, const int* match_dst_idx_to_tile_idx, float* out_cost)
{
    const int cur_match_idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (cur_match_idx >= matches_num)
        return;

    out_cost[cur_match_idx] = compute_cost_L2(cur_match_idx, matches_src, matches_dst, tiles_params, match_src_idx_to_tile_idx, match_dst_idx_to_tile_idx);
}

__global__ void transform_points(const float2 *matches1, const int matches_num, const float3* tiles_params, const int* match_idx_to_tile_idx, float2* out)
{
    const int cur_match_idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (cur_match_idx >= matches_num)
        return;

    float2 pt_transformed = rigid_transform(matches1[cur_match_idx], tiles_params[match_idx_to_tile_idx[cur_match_idx]]);
    out[cur_match_idx] = pt_transformed;
}

__global__ void grad_f_contrib_huber(const float2 *matches_src, const float2 *matches_dst, const int matches_num, const float3 *tiles_params, const int* match_src_idx_to_tile_idx, const int* match_dst_idx_to_tile_idx, const float huber_delta, const float* cur_cost, float3 *out_grad_f)
{
    const int cur_match_idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (cur_match_idx >= matches_num)
        return;

    const float2 src_pt = matches_src[cur_match_idx];
    const float2 dst_pt = matches_dst[cur_match_idx];
    const float3 src_params = tiles_params[match_src_idx_to_tile_idx[cur_match_idx]];
    const float3 dst_params = tiles_params[match_dst_idx_to_tile_idx[cur_match_idx]];

    // Compute deltaX and deltaY of the src and dst points after transformations
    const float2 src_pt_transformed = rigid_transform(src_pt, src_params);
    const float2 dst_pt_transformed = rigid_transform(dst_pt, dst_params);

    // TODO - need to verify whether dst-src is better than src-dst
    float2 delta_xy;
    delta_xy.x = src_pt_transformed.x - dst_pt_transformed.x;
    delta_xy.y = src_pt_transformed.y - dst_pt_transformed.y;

    float3 out_grad_f_src_contrib;
    float3 out_grad_f_dst_contrib;

    float grad_f_multiplier = 1.0;
    if (cur_cost[cur_match_idx] > huber_delta)
        grad_f_multiplier = huber_delta / cur_cost[cur_match_idx];

    // Compute contribution to theta, Tx, Ty of src pt tile
    out_grad_f_src_contrib.x = delta_xy.x * (-src_pt.x * src_params.x - src_pt.y) + delta_xy.y * (src_pt.x - src_pt.y * src_params.x);
    out_grad_f_src_contrib.y = delta_xy.x;
    out_grad_f_src_contrib.z = delta_xy.y;

    // Compute contribution to theta, Tx, Ty of dst pt tile
    out_grad_f_dst_contrib.x = delta_xy.x * (dst_pt.x * dst_params.x + dst_pt.y) + delta_xy.y * (-dst_pt.x + dst_pt.y * dst_params.x);
    out_grad_f_dst_contrib.y = -delta_xy.x;
    out_grad_f_dst_contrib.z = -delta_xy.y;

//    out_grad_f_src_contrib_all[cur_match_idx] = out_grad_f_src_contrib.x;
//    out_grad_f_dst_contrib_all[cur_match_idx] = out_grad_f_dst_contrib.x;

    // Update the values of the grad_f parameters that correspond to the match's src and dst tiles
    atomicAdd(&out_grad_f[match_src_idx_to_tile_idx[cur_match_idx]].x, out_grad_f_src_contrib.x * grad_f_multiplier);
    atomicAdd(&out_grad_f[match_src_idx_to_tile_idx[cur_match_idx]].y, out_grad_f_src_contrib.y * grad_f_multiplier);
    atomicAdd(&out_grad_f[match_src_idx_to_tile_idx[cur_match_idx]].z, out_grad_f_src_contrib.z * grad_f_multiplier);
    atomicAdd(&out_grad_f[match_dst_idx_to_tile_idx[cur_match_idx]].x, out_grad_f_dst_contrib.x * grad_f_multiplier);
    atomicAdd(&out_grad_f[match_dst_idx_to_tile_idx[cur_match_idx]].y, out_grad_f_dst_contrib.y * grad_f_multiplier);
    atomicAdd(&out_grad_f[match_dst_idx_to_tile_idx[cur_match_idx]].z, out_grad_f_dst_contrib.z * grad_f_multiplier);
}



__global__ void grad_f_contrib(const float2 *matches_src, const float2 *matches_dst, const int matches_num, const float3 *tiles_params, const int* match_src_idx_to_tile_idx, const int* match_dst_idx_to_tile_idx, float3 *out_grad_f)//, float *out_grad_f_src_contrib_all, float *out_grad_f_dst_contrib_all)//, const float* cur_cost, float3 *out_grad_f_src_contrib, float3 *out_grad_f_dst_contrib)
{
    const int cur_match_idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (cur_match_idx >= matches_num)
        return;

    const float2 src_pt = matches_src[cur_match_idx];
    const float2 dst_pt = matches_dst[cur_match_idx];
    const float3 src_params = tiles_params[match_src_idx_to_tile_idx[cur_match_idx]];
    const float3 dst_params = tiles_params[match_dst_idx_to_tile_idx[cur_match_idx]];

    // Compute deltaX and deltaY of the src and dst points after transformations
    const float2 src_pt_transformed = rigid_transform(src_pt, src_params);
    const float2 dst_pt_transformed = rigid_transform(dst_pt, dst_params);

    // TODO - need to verify whether dst-src is better than src-dst
    float2 delta_xy;
    delta_xy.x = src_pt_transformed.x - dst_pt_transformed.x;
    delta_xy.y = src_pt_transformed.y - dst_pt_transformed.y;

    float3 out_grad_f_src_contrib;
    float3 out_grad_f_dst_contrib;

//    float grad_f_multiplier = 1.0;
//    if (cur_cost[TODO] <= huber_delta)
//        grad_f_multiplier = huber_delta / cur_cost[TODO];

    // Compute contribution to theta, Tx, Ty of src pt tile
    out_grad_f_src_contrib.x = delta_xy.x * (-src_pt.x * src_params.x - src_pt.y) + delta_xy.y * (src_pt.x - src_pt.y * src_params.x);
    out_grad_f_src_contrib.y = delta_xy.x;
    out_grad_f_src_contrib.z = delta_xy.y;

    // Compute contribution to theta, Tx, Ty of dst pt tile
    out_grad_f_dst_contrib.x = delta_xy.x * (dst_pt.x * dst_params.x + dst_pt.y) + delta_xy.y * (-dst_pt.x + dst_pt.y * dst_params.x);
    out_grad_f_dst_contrib.y = -delta_xy.x;
    out_grad_f_dst_contrib.z = -delta_xy.y;

//    out_grad_f_src_contrib_all[cur_match_idx] = out_grad_f_src_contrib.x;
//    out_grad_f_dst_contrib_all[cur_match_idx] = out_grad_f_dst_contrib.x;

    // Update the values of the grad_f parameters that correspond to the match's src and dst tiles
    atomicAdd(&out_grad_f[match_src_idx_to_tile_idx[cur_match_idx]].x, out_grad_f_src_contrib.x);
    atomicAdd(&out_grad_f[match_src_idx_to_tile_idx[cur_match_idx]].y, out_grad_f_src_contrib.y);
    atomicAdd(&out_grad_f[match_src_idx_to_tile_idx[cur_match_idx]].z, out_grad_f_src_contrib.z);
    atomicAdd(&out_grad_f[match_dst_idx_to_tile_idx[cur_match_idx]].x, out_grad_f_dst_contrib.x);
    atomicAdd(&out_grad_f[match_dst_idx_to_tile_idx[cur_match_idx]].y, out_grad_f_dst_contrib.y);
    atomicAdd(&out_grad_f[match_dst_idx_to_tile_idx[cur_match_idx]].z, out_grad_f_dst_contrib.z);
}

__device__ void zero_grad_f(float3 *grad_f, const int tiles_num)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < tiles_num)
    {
        grad_f[idx].x = 0;
        grad_f[idx].y = 0;
        grad_f[idx].z = 0;
    }
}

__global__ void compute_new_params(const float3* cur_params, float3* next_params, size_t tiles_num, float3* grad_f, float gamma, float3* diff_params)
{
    const int p_idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (p_idx >= tiles_num)
        return;

//     diff_params[p_idx].x = gamma * grad_f[p_idx].x;
//     diff_params[p_idx].y = gamma * grad_f[p_idx].y;
//     diff_params[p_idx].z = gamma * grad_f[p_idx].z;
// 
//     next_params[p_idx].x = cur_params[p_idx].x - diff_params[p_idx].x;
//     next_params[p_idx].x = cur_params[p_idx].y - diff_params[p_idx].y;
//     next_params[p_idx].x = cur_params[p_idx].z - diff_params[p_idx].z;
    next_params[p_idx].x = cur_params[p_idx].x - gamma * grad_f[p_idx].x;
    next_params[p_idx].y = cur_params[p_idx].y - gamma * grad_f[p_idx].y;
    next_params[p_idx].z = cur_params[p_idx].z - gamma * grad_f[p_idx].z;

    diff_params[p_idx].x = abs(cur_params[p_idx].x - next_params[p_idx].x);
    diff_params[p_idx].y = abs(cur_params[p_idx].y - next_params[p_idx].y);
    diff_params[p_idx].z = abs(cur_params[p_idx].z - next_params[p_idx].z);


    // reset the gradient
    zero_grad_f(grad_f, tiles_num);
}




