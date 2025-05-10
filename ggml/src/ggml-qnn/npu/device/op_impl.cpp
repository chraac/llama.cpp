

#include "op_impl.hpp"

#include <hexagon_types.h>
#include <HTP/core/intrinsics.h>

#include "op_mul_mat.hpp"
#include "quants.hpp"

namespace {

template <HVX_Vector (*_OpIntrinsic)(HVX_Vector, HVX_Vector), typename _TyData>
inline void vec_op_impl(const _TyData * src0, const _TyData * src1, size_t count, _TyData * dst) {
    constexpr const size_t kElementsPerVector = hexagon::kBytesPerVector / sizeof(_TyData);

    HVX_Vector * iptr0     = ((HVX_Vector *) src0);
    HVX_Vector * iptr0_end = ((HVX_Vector *) src0) + (count / kElementsPerVector);
    HVX_Vector * iptr1     = ((HVX_Vector *) src1);
    HVX_Vector * optr      = ((HVX_Vector *) dst);
    HVX_Vector   prev0     = *iptr0++;
    HVX_Vector   prev1     = *iptr1++;

    // TODO: prefetch or just use VTCM?
    while (iptr0 < iptr0_end) {
        HVX_Vector curr0 = *iptr0++;
        HVX_Vector curr1 = *iptr1++;
        HVX_Vector s0    = Q6_V_valign_VVR(curr0, prev0, (size_t) src0);
        HVX_Vector s1    = Q6_V_valign_VVR(curr1, prev1, (size_t) src1);
        *optr++          = _OpIntrinsic(s0, s1);
        prev0            = curr0;
        prev1            = curr1;
    }

    if ((iptr0_end - ((HVX_Vector *) src0)) > 0) {
        // handle the last vector
        // see also:
        //   https://github.com/UbiquitousLearning/mllm/blob/babf4410352ce8730824c87699c025a0d4ce3a6f/src/backends/qnn/LLaMAOpPackageHtp/LLaMAPackage/src/ops/LLaMAMul.cpp#L147
        //   or qualcomm sdk libs\qhl_hvx\src\qhblas_hvx\qhblas_hvx_aw_vector_add_ah.c
        bool       iptr0_aligned = hexagon::is_addr_aligned(iptr0);
        HVX_Vector curr0         = iptr0_aligned ? prev0 : *iptr0;
        iptr0                    = iptr0_aligned ? iptr0 : iptr0 + 1;
        bool       iptr1_aligned = hexagon::is_addr_aligned(iptr1);
        HVX_Vector curr1         = iptr1_aligned ? prev1 : *iptr1;
        iptr1                    = iptr1_aligned ? iptr1 : iptr1 + 1;
        HVX_Vector s0            = Q6_V_valign_VVR(curr0, prev0, (size_t) src0);
        HVX_Vector s1            = Q6_V_valign_VVR(curr1, prev1, (size_t) src1);
        *optr++                  = _OpIntrinsic(s0, s1);
        prev0                    = curr0;
        prev1                    = curr1;
    }

    const size_t leftover       = count % kElementsPerVector;
    const size_t leftover_bytes = leftover * sizeof(_TyData);
    if (leftover > 0) {
        // handle the leftover elements
        HVX_Vector curr0 =
            (leftover_bytes + hexagon::unaligned_bytes(iptr0) > hexagon::kBytesPerVector) ? *iptr0 : prev0;
        curr0 = Q6_V_valign_VVR(curr0, prev0, (size_t) src0);

        HVX_Vector curr1 =
            (leftover_bytes + hexagon::unaligned_bytes(iptr1) > hexagon::kBytesPerVector) ? *iptr1 : prev1;
        curr1 = Q6_V_valign_VVR(curr1, prev1, (size_t) src1);

        q6op_vstu_variable_ARV(optr, leftover_bytes, _OpIntrinsic(curr0, curr1));
    }
}

template <HVX_Vector (*_OpIntrinsic)(HVX_Vector, HVX_Vector)>
inline void vec_op_f32_f32(const float * src0, const float * src1, size_t count, float * dst) {
    vec_op_impl<_OpIntrinsic, float>(src0, src1, count, dst);
}

inline HVX_Vector vadd_f32_f32(HVX_Vector a, HVX_Vector b) {
    return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b));
}

inline HVX_Vector vsub_f32_f32(HVX_Vector a, HVX_Vector b) {
    return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(a, b));
}

inline HVX_Vector vmul_f32_f32(HVX_Vector a, HVX_Vector b) {
    return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b));
}

template <HVX_Vector (*_OpIntrinsic)(HVX_Vector, HVX_Vector)>
inline void vec_op_f16_f16(const npu_device_fp16_t * src0, const npu_device_fp16_t * src1, size_t count,
                           npu_device_fp16_t * dst) {
    vec_op_impl<_OpIntrinsic, npu_device_fp16_t>(src0, src1, count, dst);
}

inline HVX_Vector vadd_f16_f16(HVX_Vector a, HVX_Vector b) {
    // TODO: fix this since qf16 has less precision than fp16
    return Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(a, b));
}

inline HVX_Vector vsub_f16_f16(HVX_Vector a, HVX_Vector b) {
    // TODO: fix this since qf16 has less precision than fp16
    return Q6_Vhf_equals_Vqf16(Q6_Vqf16_vsub_VhfVhf(a, b));
}

inline HVX_Vector vmul_f16_f16(HVX_Vector a, HVX_Vector b) {
    return Q6_Vhf_equals_Wqf32(Q6_Wqf32_vmpy_VhfVhf(a, b));
}

template <typename T> struct get_data_type {};

template <typename _TyData> struct get_data_type<void (*)(const _TyData *, const _TyData *, size_t, _TyData *)> {
    using type = _TyData;
};

template <auto _RowFunc> bool element_wise_op(hexagon::tensor * out, hexagon::compute_params * params) {
    using data_type                           = typename get_data_type<decltype(_RowFunc)>::type;
    constexpr const size_t kElementsPerVector = hexagon::kBytesPerVector / sizeof(data_type);

    if (!out) {
        return false;
    }

    static_assert(DEVICE_TENSOR_MAX_DIMS == 4, "element_wise_op requires max dims 4");
    auto * src0 = out->get_src(0);
    auto * src1 = out->get_src(1);
    if (!src0 || !src1) {
        return true;  // skip if no src
    }

    if (src0->get_ne(0) != src1->get_ne(0)) {
        // TODO: handle this case
        DEVICE_LOG_ERROR("src0[0] and src1[0] not match: %ld vs %ld\n", (long) src0->get_ne(0), (long) src1->get_ne(0));
        return false;
    }

    const auto * src0_ptr      = reinterpret_cast<const uint8_t *>(src0->get_data());
    const auto * src1_ptr      = reinterpret_cast<const uint8_t *>(src1->get_data());
    auto *       dst_ptr       = reinterpret_cast<uint8_t *>(out->get_data());
    auto         total_rows    = out->get_ne(3) * out->get_ne(2) * out->get_ne(1);
    const auto   rows_per_cube = out->get_ne(2) * out->get_ne(1);
    const auto   start_end     = hexagon::get_thread_work_slice(total_rows, params->tidx, params->tcnt);

    if (start_end.first >= start_end.second) {
        return true;
    }

    uint8_t * src1_plane_cache_ptr  = nullptr;
    size_t    src1_plane_cache_size = 0;
    if (src0->get_ne(1) / src1->get_ne(1) > 1) {
        // TODO: should we cache a cube instead of a plane?
        src1_plane_cache_size = src1->get_nb(1) * src1->get_ne(1);
        src1_plane_cache_ptr  = params->get_cache(src1_plane_cache_size, false);
        DEVICE_LOG_DEBUG("element_wise_op vtcm_mem allocated, size: %zu\n", src1_plane_cache_size);

        const auto i03 = start_end.first / rows_per_cube;
        const auto i02 = start_end.first / out->get_ne(1) - i03 * out->get_ne(2);
        const auto i13 = i03 % src1->get_ne(3);
        const auto i12 = i02 % src1->get_ne(2);

        if (src1_plane_cache_ptr) {
            auto * src1_plane = src1_ptr + i13 * src1->get_nb(3) + i12 * src1->get_nb(2);
            memcpy(src1_plane_cache_ptr, src1_plane, src1_plane_cache_size);
        }
    }

    for (int64_t ir = start_end.first; ir < start_end.second; ++ir) {
        const auto i03 = ir / rows_per_cube;
        const auto i02 = ir / out->get_ne(1) - i03 * out->get_ne(2);
        const auto i01 = ir % out->get_ne(1);  // TODO: should we use divide instead of mod?
        const auto i13 = i03 % src1->get_ne(3);
        const auto i12 = i02 % src1->get_ne(2);
        const auto i11 = i01 % src1->get_ne(1);

        auto * src1_plane = src1_ptr + i13 * src1->get_nb(3) + i12 * src1->get_nb(2);
        if (src1_plane_cache_ptr) {
            if (i01 == 0) {
                memcpy(src1_plane_cache_ptr, src1_plane, src1_plane_cache_size);
            }

            src1_plane = src1_plane_cache_ptr;
        }

        auto * src0_row = src0_ptr + i03 * src0->get_nb(3) + i02 * src0->get_nb(2) + i01 * src0->get_nb(1);
        auto * src1_row = src1_plane + i11 * src1->get_nb(1);
        auto * dst_row  = dst_ptr + i03 * out->get_nb(3) + i02 * out->get_nb(2) + i01 * out->get_nb(1);
        if (ir + 1 < start_end.second) {
            int32_t l2fetch_vectors = Q6_R_min_RR(src0->get_nb(1) / kElementsPerVector, hexagon::kL2FetchAheadVectors);
            // TODO: should we use small kL2FetchAheadVectors?
            hexagon::l2fetch(src0_row + src0->get_nb(1), hexagon::kBytesPerVector, hexagon::kBytesPerVector,
                             l2fetch_vectors, 0);

            l2fetch_vectors = Q6_R_min_RR(src1->get_nb(1) / kElementsPerVector, hexagon::kL2FetchAheadVectors);
            hexagon::l2fetch(src1_row + src1->get_nb(1), hexagon::kBytesPerVector, hexagon::kBytesPerVector,
                             l2fetch_vectors, 0);
        }

        _RowFunc(reinterpret_cast<const data_type *>(src0_row), reinterpret_cast<const data_type *>(src1_row),
                 static_cast<size_t>(out->get_ne(0)), reinterpret_cast<data_type *>(dst_row));
    }

    return true;
}

bool is_element_wise_op_supported(const npu_device_tensor_spec & src0, const npu_device_tensor_spec & src1,
                                  const npu_device_tensor_spec & dst, npu_device_tensor_op op) {
    if (op != NPU_OP_ADD && op != NPU_OP_SUB && op != NPU_OP_MUL) {
        DEVICE_LOG_DEBUG("[%s]unsupported\n", hexagon::op_get_name(op));
        return false;
    }

    if (dst.type != src0.type || dst.type != src1.type) {
        DEVICE_LOG_DEBUG("[%s]src0.type and dst.type mismatch: %s vs %s\n", hexagon::op_get_name(op),
                         hexagon::get_type_name(src0.type), hexagon::get_type_name(dst.type));
        return false;
    }

    if (dst.type != NPU_DATA_TYPE_F32 && dst.type != NPU_DATA_TYPE_F16) {
        DEVICE_LOG_DEBUG("[%s]unsupported data type: %s\n", hexagon::op_get_name(op), hexagon::get_type_name(dst.type));
        return false;
    }

    // TODO: fix FP16 add/sub
    if (dst.type == NPU_DATA_TYPE_F16 && op != NPU_OP_MUL) {
        DEVICE_LOG_DEBUG("[%s]unsupported data type: %s\n", hexagon::op_get_name(op), hexagon::get_type_name(dst.type));
        return false;
    }

    if (src0.ne[0] != src1.ne[0]) {
        DEVICE_LOG_DEBUG("[%s]src0.ne[0] and src1.ne[0] not match: %ld vs %ld\n", hexagon::op_get_name(op),
                         (long) src0.ne[0], (long) src1.ne[0]);
        return false;
    }

    for (size_t i = 0; i < DEVICE_TENSOR_MAX_DIMS; ++i) {
        if (src0.ne[i] != dst.ne[i]) {
            DEVICE_LOG_DEBUG("[%s]src0.ne[%zu] and dst.ne[%zu] not match: %lld vs %lld\n", hexagon::op_get_name(op), i,
                             i, (long long) src0.ne[i], (long long) dst.ne[i]);
            return false;
        }
    }

    return true;
}

struct op_capabilities {
    npu_device_tensor_op               op;
    hexagon::op_is_supported_func_type is_supported;
    hexagon::compute_func_type         compute_funcs[NPU_DATA_TYPE_COUNT];
};

constexpr const op_capabilities kOpCapabilities[] = {
    {
     NPU_OP_MUL_MAT, hexagon::is_mul_mat_supported,
     {
            hexagon::mul_mat_f32,  // NPU_DATA_TYPE_F32
            nullptr,               // NPU_DATA_TYPE_F16
        }, },
    { NPU_OP_ADD,
     is_element_wise_op_supported, {
          element_wise_op<vec_op_f32_f32<vadd_f32_f32>>,  // NPU_DATA_TYPE_F32
          element_wise_op<vec_op_f16_f16<vadd_f16_f16>>,  // NPU_DATA_TYPE_F16
      } },
    { NPU_OP_SUB,
     is_element_wise_op_supported, {
          element_wise_op<vec_op_f32_f32<vsub_f32_f32>>,  // NPU_DATA_TYPE_F32
          element_wise_op<vec_op_f16_f16<vsub_f16_f16>>,  // NPU_DATA_TYPE_F16
      } },
    { NPU_OP_MUL,
     is_element_wise_op_supported, {
          element_wise_op<vec_op_f32_f32<vmul_f32_f32>>,  // NPU_DATA_TYPE_F32
          element_wise_op<vec_op_f16_f16<vmul_f16_f16>>,  // NPU_DATA_TYPE_F16
      } },
};

static_assert(kOpCapabilities[NPU_OP_MUL_MAT].compute_funcs[NPU_DATA_TYPE_F32] == hexagon::mul_mat_f32,
              "kOpArray[NPU_OP_MUL_MAT] != mul_mat_f32");

static_assert(std::size(kOpCapabilities) == NPU_OP_COUNT);
static_assert(kOpCapabilities[NPU_OP_MUL_MAT].op == NPU_OP_MUL_MAT, "kOpArray[NPU_OP_MUL_MAT].op != NPU_OP_MUL_MAT");
static_assert(kOpCapabilities[NPU_OP_MUL].op == NPU_OP_MUL, "kOpArray[NPU_OP_MUL].op != NPU_OP_MUL");

hexagon::compute_func_type get_compute_func_impl(npu_device_tensor_op op, npu_device_tensor_data_type type) {
    if (op >= NPU_OP_COUNT) {
        return nullptr;
    }

    return kOpCapabilities[op].compute_funcs[type];
}

}  // namespace

namespace hexagon {

compute_func_type get_compute_func(tensor * dst) {
    return get_compute_func_impl(dst->get_op(), dst->get_type());
}

bool support_op(const npu_device_tensor_spec & src0, const npu_device_tensor_spec & src1,
                const npu_device_tensor_spec & dst, npu_device_tensor_op op) {
    if (get_compute_func_impl(op, dst.type) == nullptr) {
        DEVICE_LOG_ERROR("[%s]unsupported, get_compute_func failed\n", op_get_name(op));
        return false;
    }

    auto is_supported_func = kOpCapabilities[op].is_supported;
    if (!is_supported_func || !is_supported_func(src0, src1, dst, op)) {
        DEVICE_LOG_ERROR("[%s]unsupported, is_supported_func failed\n", op_get_name(op));
        return false;
    }

    return true;
}

}  // namespace hexagon
