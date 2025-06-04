
#include "op_flash_attn.hpp"

#include "type_traits.hpp"
#include "util.hpp"
#include "vec_ops.hpp"

namespace {

// TODO: use a more efficient conversion
inline float f16_to_f32(const npu_device_fp16_t src) {
    return reinterpret_cast<const __fp16 &>(src);
}

void flash_attn_impl(hexagon::tensor * out, const hexagon::tensor * q, const hexagon::tensor * k,
                     const hexagon::tensor * v, const hexagon::tensor * mask, hexagon::compute_params * params) {
    static_assert(3 <= hexagon::kMaxParamsCount, "flash_attn op params count exceeds max params count");

    float       scale         = out->get_op_param<float>(0);
    const float max_bias      = out->get_op_param<float>(1);
    const float logit_softcap = out->get_op_param<float>(2);

    if (logit_softcap != 0) {
        scale /= logit_softcap;
    }

    // broadcast factors
    const int64_t rk2 = q->get_ne(2) / k->get_ne(2);
    const int64_t rk3 = q->get_ne(3) / k->get_ne(3);
    const int64_t rv2 = q->get_ne(2) / v->get_ne(2);
    const int64_t rv3 = q->get_ne(3) / v->get_ne(3);

    const uint32_t n_head      = q->get_ne(2);
    const uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(n_head));

    const float m0 = powf(2.0f, -(max_bias) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    const auto q_to_vec_dot = hexagon::get_type_traits(NPU_DATA_TYPE_F16).from_float;  // TODO: support more k types
    const auto kq_vec_dot   = hexagon::get_type_traits(k->get_type()).vec_dot;
    const auto v_to_float   = hexagon::get_type_traits(v->get_type()).to_float;

    const int64_t total_rows = q->get_ne(1) * q->get_ne(2) * q->get_ne(3);       // total number of rows in Q
    const auto    start_end_row =
        hexagon::get_thread_work_slice(total_rows, params->tidx, params->tcnt);  // work slice for this thread

    const auto DK          = k->get_ne(0);
    const auto DV          = v->get_ne(0);
    const auto row_bytes_k = DK * hexagon::get_type_traits(k->get_type()).type_size;

    size_t total_cache_size = sizeof(float) * (DK + 2 * DV) + 16;  // CACHE_LINE_SIZE_F32 == 16
    auto * cache_ptr        = params->get_vtcm_cache(total_cache_size);
    if (!cache_ptr) {
        DEVICE_LOG_ERROR("Failed to allocate VTCM cache for flash_attn: %zu bytes\n", total_cache_size);
        return;
    }

    // loop over n_batch and n_head
    const auto rows_per_batch     = q->get_ne(2) * q->get_ne(1);
    const auto out_rows_per_batch = out->get_ne(2) * out->get_ne(1);
    const bool is_v_f16 =
        v->get_type() == NPU_DATA_TYPE_F16;  // check if V is in FP16 format, otherwise it is in FP32 format
    auto * dst_ptr = reinterpret_cast<uint8_t *>(out->get_write_buffer());
    if (!dst_ptr) {
        DEVICE_LOG_ERROR("flash_attn_impl: dst_ptr is not writable, tensor: %p, type: %s\n", (void *) out,
                         hexagon::get_type_name(out->get_type()));
        return;
    }

    DEVICE_SCOPED_OP_PERFORMANCE_TRACKER_WITH_MULTI_SUB_PROC(out, params->tidx, flash_attn);
    const auto *    q_ptr    = q->get_read_buffer();
    const auto *    k_ptr    = k->get_read_buffer();
    const auto *    v_ptr    = v->get_read_buffer();
    const uint8_t * mask_ptr = mask ? mask->get_read_buffer() : nullptr;
    for (auto ir = start_end_row.first; ir < start_end_row.second; ++ir) {
        // q indices
        const auto iq3 = ir / rows_per_batch;
        const auto iq2 = (ir - iq3 * rows_per_batch) / q->get_ne(1);
        const auto iq1 = (ir - iq3 * rows_per_batch - iq2 * q->get_ne(1));

        const uint32_t h = iq2;  // head index
        const float    slope =
            (max_bias > 0.0f) ? h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2 * (h - n_head_log2) + 1) : 1.0f;

        float S = 0.0f;                                                     // sum
        float M = -INFINITY;                                                // maximum KQ value

        float * VKQ32 = reinterpret_cast<float *>(cache_ptr);               // FP32 VKQ accumulator
        float * V32   = VKQ32 + DV;                                         // (temporary) FP32 V buffer
        auto *  VKQ16 = reinterpret_cast<npu_device_fp16_t *>(VKQ32 + DV);  // (temporary) FP16 VKQ accumulator
        auto *  Q_q   = reinterpret_cast<npu_device_fp16_t *>(
            VKQ32 + 2 * DV);  // (temporary) buffer for Q converted to quantized/FP16

        if (is_v_f16) {
            memset(VKQ16, 0, DV * sizeof(npu_device_fp16_t));
        } else {
            memset(VKQ32, 0, DV * sizeof(float));
        }

        const npu_device_fp16_t * mp =
            mask_ptr ? reinterpret_cast<const npu_device_fp16_t *>(mask_ptr + iq1 * mask->get_nb(1)) : nullptr;

        // k indices
        const int ik3 = iq3 / rk3;
        const int ik2 = iq2 / rk2;

        // v indices
        const int iv3 = iq3 / rv3;
        const int iv2 = iq2 / rv2;

        const float * pq =
            reinterpret_cast<const float *>(q_ptr + (iq1 * q->get_nb(1) + iq2 * q->get_nb(2) + iq3 * q->get_nb(3)));
        q_to_vec_dot(pq, Q_q, DK, params->f16_to_f32_table);

        // online softmax / attention
        // loop over n_kv and n_head_kv
        // ref: https://arxiv.org/pdf/2112.05682.pdf
        for (int64_t ic = 0; ic < k->get_ne(1); ++ic) {
            DEVICE_SCOPED_OP_PERFORMANCE_TRACKER_ADD_ONE_SUB_PROC(flash_attn, 0, loop);
            float mv = 0.0f;
            if (mp) {
                mv = slope * f16_to_f32(mp[ic]);
            }

            if (mv == -INFINITY) {
                continue;
            }

            float s = 0.f;
            {
                DEVICE_SCOPED_OP_PERFORMANCE_TRACKER_ADD_ONE_SUB_PROC(flash_attn, 1, kq_dot);
                const auto * k_data = k_ptr + (ic * k->get_nb(1) + ik2 * k->get_nb(2) + ik3 * k->get_nb(3));
                if (ic < k->get_ne(1) - 1) {
                    hexagon::l2fetch_row(k_data + k->get_nb(1), row_bytes_k);
                }

                s = kq_vec_dot(k_data, Q_q, DK);  // KQ value
                s = s * scale;                    // scale KQ value
                if (logit_softcap != 0.0f) {
                    s = logit_softcap * tanhf(s);
                }

                s += mv;  // apply mask
            }

            const float Mold = M;

            float ms = 1.0f;  // upon new higher max val, scale VKQ and KQ sum with this value
            float vs = 1.0f;  // post-softmax KQ value, expf(s - M)

            const auto * v_data = v_ptr + (ic * v->get_nb(1) + iv2 * v->get_nb(2) + iv3 * v->get_nb(3));
            if (is_v_f16) {
                if (s > M) {
                    // s is new maximum, ms < 1.0f, vs == expf(s - s) == 1.0f
                    M  = s;
                    ms = expf(Mold - M);

                    // V = V*expf(Mold - M)
                    hexagon::vec_scale_f16(VKQ16, ms, VKQ16, DV);
                } else {
                    // no new maximum, ms == 1.0f, vs != 1.0f
                    vs = expf(s - M);
                }

                // V += v*expf(s - M)
                DEVICE_SCOPED_OP_PERFORMANCE_TRACKER_ADD_ONE_SUB_PROC(flash_attn, 2, mad);
                hexagon::vec_mad_f16(reinterpret_cast<const npu_device_fp16_t *>(v_data), vs, VKQ16, DV);
            } else {
                if (s > M) {
                    // s is new maximum, ms < 1.0f, vs == expf(s - s) == 1.0f
                    M  = s;
                    ms = expf(Mold - M);

                    // V = V*expf(Mold - M)
                    hexagon::vec_scale_f32(VKQ32, ms, VKQ32, DV);
                } else {
                    // no new maximum, ms == 1.0f, vs != 1.0f
                    vs = expf(s - M);
                }

                // V += v*expf(s - M)
                DEVICE_SCOPED_OP_PERFORMANCE_TRACKER_ADD_ONE_SUB_PROC(flash_attn, 2, mad);
                if (v_to_float) {
                    v_to_float(v_data, V32, DV, params->f16_to_f32_table);
                    hexagon::vec_mad_f32(V32, vs, VKQ32, DV);
                } else {
                    // V is F32
                    hexagon::vec_mad_f32(reinterpret_cast<const float *>(v_data), vs, VKQ32, DV);
                }
            }

            S = S * ms + vs;  // scale and increment sum with partial sum
        }

        if (is_v_f16) {
            // TODO: use a more efficient conversion
            for (int64_t d = 0; d < DV; ++d) {
                VKQ32[d] = f16_to_f32(VKQ16[d]);
            }
        }

        // V /= S
        const float S_inv = 1.0f / S;
        hexagon::vec_scale_f32(VKQ32, S_inv, VKQ32, DV);

        // dst indices
        const int i1 = iq1;
        const int i2 = iq2;
        const int i3 = iq3;

        // permute(0, 2, 1, 3)
        memcpy(dst_ptr + (i3 * out_rows_per_batch + i2 + i1 * out->get_ne(1)) * out->get_nb(1), VKQ32, out->get_nb(1));
    }
}

}  // namespace

namespace hexagon {

bool flash_attn_f32(tensor * out, compute_params * params) {
    if (!out || !params) {
        DEVICE_LOG_DEBUG("invalid out or params\n");
        return false;
    }

    const auto * q    = out->get_src(0);
    const auto * k    = out->get_src(1);
    const auto * v    = out->get_src(2);
    const auto * mask = out->get_src(3);
    if (!q || !k || !v || !mask) {
        DEVICE_LOG_DEBUG("invalid src tensors: q: %p, k: %p, v: %p, mask: %p\n", (void *) q, (void *) k, (void *) v,
                         (void *) mask);
        return false;
    }

    flash_attn_impl(out, q, k, v, mask, params);
    return true;
}

bool is_flash_attn_supported(npu_device_tensor_op op, const npu_device_tensor_spec * dst,
                             const npu_device_tensor_spec * srcs, size_t src_len) {
    if (op != NPU_OP_FLASH_ATTN) {
        DEVICE_LOG_DEBUG("op is not NPU_OP_FLASH_ATTN: %d\n", op);
        return false;
    }

    if (!dst || !srcs || src_len < 4) {
        DEVICE_LOG_DEBUG("[%s]invalid dst or srcs\n", op_get_name(op));
        return false;
    }

    if (dst->type != NPU_DATA_TYPE_F32) {
        DEVICE_LOG_DEBUG("[%s]dst type is not F32: %s\n", op_get_name(op), get_type_name(dst->type));
        return false;
    }

    const auto * q = &srcs[0];
    if (q->type != NPU_DATA_TYPE_F32) {
        DEVICE_LOG_DEBUG("[%s]q type is not F32: %s\n", op_get_name(op), get_type_name(q->type));
        return false;
    }

    const auto * k = &srcs[1];
    if (k->type != NPU_DATA_TYPE_F16) {  // TODO: support more k types
        DEVICE_LOG_DEBUG("[%s]k type is not F16: %s\n", op_get_name(op), get_type_name(k->type));
        return false;
    }

    const auto * v = &srcs[2];
    if (v->type != NPU_DATA_TYPE_F16) {  // TODO: support more v types
        DEVICE_LOG_DEBUG("[%s]v type is not F16: %s\n", op_get_name(op), get_type_name(v->type));
        return false;
    }

    const auto * mask = &srcs[3];
    if (mask->type != NPU_DATA_TYPE_F16) {
        DEVICE_LOG_DEBUG("[%s]mask type is not F16: %s\n", op_get_name(op), get_type_name(mask->type));
        return false;
    }

    // TODO: check tensor shapes
    return true;
}

}  // namespace hexagon
