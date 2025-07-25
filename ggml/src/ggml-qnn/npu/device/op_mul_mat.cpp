#include "op_mul_mat.hpp"

#include "thread_pool.hpp"  // TODO: remove this dependency
#include "type_traits.hpp"
#include "vec_ops.hpp"

namespace {

template <typename _T> struct get_data_type {};

template <typename _TData0, typename _TData1>
struct get_data_type<HVX_Vector (*)(const _TData0 *, const _TData1 *, size_t)> {
    using data_type0 = _TData0;
    using data_type1 = _TData1;
};

template <typename _TRet> struct convert_vector {};

template <> struct convert_vector<float> {
    static float convert(HVX_Vector vec) { return hexagon::get_flt0_from_fltv(Q6_Vsf_equals_Vqf32(vec)); }
};

template <> struct convert_vector<npu_device_fp16_t> {
    static float convert(HVX_Vector vec) {
        HVX_Vector vect = Q6_Vhf_equals_Vqf16(vec);
        uint16_t   i    = (vect[0] & 0xffff);
        return reinterpret_cast<__fp16 &>(i);
    }
};

template <auto _DotFunc, bool _ShouldCacheSrc0>
void mul_mat_impl(hexagon::tensor *         src0,
                  hexagon::tensor *         src1,
                  hexagon::tensor *         dst,
                  hexagon::compute_params * params) {
    using data_type0 = typename get_data_type<decltype(_DotFunc)>::data_type0;
    using data_type1 = typename get_data_type<decltype(_DotFunc)>::data_type1;

    const auto src0_actual_row_size = hexagon::get_dequantized_row_size(src0);
    auto *     dequantize_row_func  = hexagon::get_type_traits(src0->get_type()).to_float;
    if (_ShouldCacheSrc0 && dequantize_row_func == nullptr) {
        DEVICE_LOG_ERROR("Unsupported quantized src0 type: %d, dequantize_row_func is null\n", src0->get_type());
        return;
    }

    const auto r02          = src1->get_ne(2) / src0->get_ne(2);
    const auto r03          = src1->get_ne(3) / src0->get_ne(3);
    const auto total_planes = dst->get_ne(3) * dst->get_ne(2);

    auto start_end_plane   = std::pair<int64_t, int64_t>{ 0, total_planes };
    auto start_end_row     = std::pair<int64_t, int64_t>{ 0, dst->get_ne(1) };
    auto start_end_element = std::pair<int64_t, int64_t>{ 0, dst->get_ne(0) };

    if (total_planes >= params->get_thread_count()) {
        start_end_plane = params->get_work_slice(total_planes);
    } else if (dst->get_ne(0) >= params->get_thread_count()) {
        start_end_element = params->get_work_slice(dst->get_ne(0));
    } else {
        start_end_row = params->get_work_slice(dst->get_ne(1));
    }

    if (start_end_plane.second <= start_end_plane.first || start_end_row.second <= start_end_row.first ||
        start_end_element.second <= start_end_element.first) {
        DEVICE_LOG_DEBUG(
            "mul_mat_impl: no work to do, start_end_plane: (%ld, %ld), start_end_row: (%ld, %ld), "
            "start_end_element: (%ld, %ld)\n",
            start_end_plane.first,
            start_end_plane.second,
            start_end_row.first,
            start_end_row.second,
            start_end_element.first,
            start_end_element.second);
        return;
    }

    // cache the src0 plane in VTCM
    size_t          src0_plane_slice_row_count = start_end_element.second - start_end_element.first;
    size_t          src0_plane_cache_size      = 0;
    uint8_t *       src0_plane_cache_ptr       = nullptr;
    const uint8_t * last_cached_plane_ptr      = nullptr;
    if constexpr (_ShouldCacheSrc0) {
        src0_plane_slice_row_count =
            std::min(params->get_vtcm_quota_size() / src0_actual_row_size, src0_plane_slice_row_count);
        src0_plane_cache_size = src0_actual_row_size * src0_plane_slice_row_count;
        src0_plane_cache_ptr  = params->get_vtcm_cache(src0_plane_cache_size);
        if (src0_plane_cache_ptr == nullptr) {
            DEVICE_LOG_ERROR(
                "mul_mat_impl: failed to get VTCM cache for src0, size: %zu, src0_plane_slice_row_count: %zu, "
                "src0_actual_row_size: %zu, will fallback to mem cache\n",
                src0_plane_cache_size,
                src0_plane_slice_row_count,
                src0_actual_row_size);
            return;
        }
    }

    DEVICE_LOG_DEBUG(
        "mul_mat_impl src0_actual_row_size: %zu, src0_plane_slice_row_count: %zu, is_quantized: %d, vtcm_mem: "
        "%p(%zu)\n",
        src0_actual_row_size,
        src0_plane_slice_row_count,
        _ShouldCacheSrc0,
        (void *) src0_plane_cache_ptr,
        src0_plane_cache_size);

    const size_t valid_row0_bytes = src0->get_ne(0) * sizeof(data_type0);
    const size_t valid_row1_bytes =
        src0->get_ne(0) * sizeof(data_type1);  // src0 and src1 should have the same element count in the 1st dimension
    DEVICE_SCOPED_OP_PERFORMANCE_TRACKER_WITH_MULTI_SUB_PROC(dst, params->get_thread_index(), mul_mat);

    uint8_t * dst_ptr = dst->get_write_buffer();
    if (!dst_ptr) {
        DEVICE_LOG_ERROR("mul_mat_impl: dst_ptr is not writable, tensor: %p, type: %s\n",
                         (void *) dst,
                         hexagon::get_type_name(dst->get_type()));
        return;
    }

    constexpr bool  should_fetch_src0_row = !_ShouldCacheSrc0;
    const uint8_t * src0_ptr              = src0->get_read_buffer();
    const uint8_t * src1_ptr              = src1->get_read_buffer();
    for (int64_t ip = start_end_plane.first; ip < start_end_plane.second; ip++) {
        const auto   i3         = ip / dst->get_ne(2);
        const auto   i2         = ip - i3 * dst->get_ne(2);
        const auto * src1_plane = src1_ptr + i3 * src1->get_nb(3) + i2 * src1->get_nb(2);
        auto *       dst_plane  = dst_ptr + i3 * dst->get_nb(3) + i2 * dst->get_nb(2);
        for (int64_t col_idx = start_end_element.first; col_idx < start_end_element.second;
             col_idx += src0_plane_slice_row_count) {
            const uint8_t * src0_plane =
                src0_ptr + i3 / r03 * src0->get_nb(3) + i2 / r02 * src0->get_nb(2) + col_idx * src0->get_nb(1);
            hexagon::l2fetch_row(src0_plane, src0->get_nb(1));

            const int64_t actual_row_count =
                std::min<int64_t>(src0_plane_slice_row_count,
                                  start_end_element.second - col_idx);  // number of rows in this slice
            if constexpr (_ShouldCacheSrc0) {
                if (last_cached_plane_ptr != src0_plane) {
                    DEVICE_SCOPED_OP_PERFORMANCE_TRACKER_ADD_ONE_SUB_PROC(mul_mat, 0, dequant);

                    for (int64_t ir = 0; ir < actual_row_count; ir++) {
                        auto * src0_row = src0_plane + ir * src0->get_nb(1);
                        if (ir + 1 < actual_row_count) {
                            hexagon::l2fetch_row(src0_row + src0->get_nb(1), src0->get_nb(1));
                        }

                        auto * cached_row_ptr = src0_plane_cache_ptr + ir * src0_actual_row_size;
                        dequantize_row_func(src0_row,
                                            reinterpret_cast<hexagon::dequant_output_type *>(cached_row_ptr),
                                            src0->get_ne(0));
                    }

                    last_cached_plane_ptr = src0_plane;
                }

                src0_plane = src0_plane_cache_ptr;
            }

            if (start_end_row.second > start_end_row.first) {
                hexagon::l2fetch_row(src1_plane + start_end_row.first * src1->get_nb(1), valid_row1_bytes);
            }

            for (int64_t i1 = start_end_row.first; i1 < start_end_row.second; i1++) {
                DEVICE_SCOPED_OP_PERFORMANCE_TRACKER_ADD_ONE_SUB_PROC(mul_mat, 1, vec_dot);
                auto *  src1_row = src1_plane + i1 * src1->get_nb(1);
                auto *  dst_row  = reinterpret_cast<float *>(dst_plane + i1 * dst->get_nb(1)) + col_idx;
                int64_t i0       = 0;
                for (; i0 + 1 < actual_row_count; i0 += 2) {
                    auto * src0_row = src0_plane + i0 * src0_actual_row_size;
                    if constexpr (should_fetch_src0_row) {
                        hexagon::l2fetch_row(src0_row + src0_actual_row_size, valid_row0_bytes);
                    }

                    // TODO: figure dst how to handle a entire row
                    auto res0 = _DotFunc(reinterpret_cast<const data_type0 *>(src0_row),
                                         reinterpret_cast<const data_type1 *>(src1_row),
                                         (size_t) src0->get_ne(0));

                    if (should_fetch_src0_row && i0 + 2 < actual_row_count) {
                        hexagon::l2fetch_row(src0_row + src0_actual_row_size + src0_actual_row_size, valid_row0_bytes);
                    }

                    // TODO: figure dst how to handle a entire row
                    auto res1 = _DotFunc(reinterpret_cast<const data_type0 *>(src0_row + src0_actual_row_size),
                                         reinterpret_cast<const data_type1 *>(src1_row),
                                         (size_t) src0->get_ne(0));

                    {
                        DEVICE_SCOPED_OP_PERFORMANCE_TRACKER_ADD_ONE_SUB_PROC(mul_mat, 2, store);
                        dst_row[i0]     = convert_vector<data_type1>::convert(res0);
                        dst_row[i0 + 1] = convert_vector<data_type1>::convert(res1);
                    }
                }

                if (ip + 1 < start_end_plane.second) {
                    hexagon::l2fetch_row(src1_row + src1->get_nb(1), valid_row1_bytes);
                }

                if (i0 < actual_row_count) {
                    auto * src0_row = src0_plane + i0 * src0_actual_row_size;
                    auto   res      = _DotFunc(reinterpret_cast<const data_type0 *>(src0_row),
                                        reinterpret_cast<const data_type1 *>(src1_row),
                                        (size_t) src0->get_ne(0));
                    DEVICE_SCOPED_OP_PERFORMANCE_TRACKER_ADD_ONE_SUB_PROC(mul_mat, 2, store);
                    dst_row[i0] = convert_vector<data_type1>::convert(res);
                }
            }
        }
    }

    dst->release_write_buffer();  // mark the output tensor as modified
}

template <auto _DotFunc, bool _ShouldCacheSrc0>
void mul_mat_gemv_impl(hexagon::tensor *         src0,
                       hexagon::tensor *         src1,
                       hexagon::tensor *         dst,
                       hexagon::compute_params * params) {
    using data_type0 = typename get_data_type<decltype(_DotFunc)>::data_type0;
    using data_type1 = typename get_data_type<decltype(_DotFunc)>::data_type1;

    const auto src0_actual_row_size = hexagon::get_dequantized_row_size(src0);
    auto *     dequantize_row_func  = hexagon::get_type_traits(src0->get_type()).to_float;
    if (_ShouldCacheSrc0 && dequantize_row_func == nullptr) {
        DEVICE_LOG_ERROR("Unsupported quantized src0 type: %d, dequantize_row_func is null\n", src0->get_type());
        return;
    }

    auto start_end_element = std::pair<int64_t, int64_t>{ 0, dst->get_ne(0) };
    if (dst->get_ne(0) >= params->get_thread_count()) {
        start_end_element = params->get_work_slice(dst->get_ne(0));
    } else {
        DEVICE_LOG_ERROR("Unsupported src1 tensor shape for gemv: %s, ne: %ldx%ldx%ldx%ld\n",
                         hexagon::get_type_name(src1->get_type()),
                         src1->get_ne(0),
                         src1->get_ne(1),
                         src1->get_ne(2),
                         src1->get_ne(3));
        return;
    }

    if (start_end_element.second <= start_end_element.first) {
        DEVICE_LOG_DEBUG(
            "mul_mat_impl: no work to do, start_end_plane: [0, 1), start_end_row: [0, 1), "
            "start_end_element: [%ld, %ld)\n",
            start_end_element.first,
            start_end_element.second);
        return;
    }

    // cache the src0 plane in VTCM
    size_t     src0_plane_slice_row_count = start_end_element.second - start_end_element.first;
    size_t     src0_plane_cache_size      = 0;
    uint8_t *  src0_plane_cache_ptr       = nullptr;
    const auto src1_actual_row_size       = hexagon::get_aligned_size(src1->get_nb(1));
    uint8_t *  src1_row_cache_ptr         = nullptr;
    if constexpr (_ShouldCacheSrc0) {
        src0_plane_slice_row_count = std::min(
            (params->get_vtcm_quota_size() - src1_actual_row_size) / src0_actual_row_size, src0_plane_slice_row_count);
        src0_plane_cache_size = src0_actual_row_size * src0_plane_slice_row_count;
        src0_plane_cache_ptr  = params->get_vtcm_cache(src0_plane_cache_size + src1_actual_row_size);
        if (src0_plane_cache_ptr == nullptr) {
            DEVICE_LOG_ERROR(
                "mul_mat_impl: failed to get VTCM cache for src0, size: %zu, src0_plane_slice_row_count: %zu, "
                "src0_actual_row_size: %zu, will fallback to mem cache\n",
                src0_plane_cache_size,
                src0_plane_slice_row_count,
                src0_actual_row_size);
            return;
        }

        src1_row_cache_ptr = src0_plane_cache_ptr + src0_plane_cache_size;
    } else {
        src1_row_cache_ptr = params->get_vtcm_cache(src1_actual_row_size);
        if (src1_row_cache_ptr == nullptr) {
            DEVICE_LOG_ERROR("mul_mat_impl: failed to get VTCM cache for src1, size: %zu\n", src1_actual_row_size);
            return;
        }
    }

    DEVICE_LOG_DEBUG(
        "mul_mat_impl src0_actual_row_size: %zu, src0_plane_slice_row_count: %zu, is_quantized: %d, vtcm_mem: "
        "%p(%zu)\n",
        src0_actual_row_size,
        src0_plane_slice_row_count,
        _ShouldCacheSrc0,
        (void *) src0_plane_cache_ptr,
        src0_plane_cache_size);

    const size_t valid_row0_bytes = src0->get_ne(0) * sizeof(data_type0);
    DEVICE_SCOPED_OP_PERFORMANCE_TRACKER_WITH_MULTI_SUB_PROC(dst, params->get_thread_index(), mul_mat);

    uint8_t * dst_ptr = dst->get_write_buffer();
    if (!dst_ptr) {
        DEVICE_LOG_ERROR("mul_mat_impl: dst_ptr is not writable, tensor: %p, type: %s\n",
                         (void *) dst,
                         hexagon::get_type_name(dst->get_type()));
        return;
    }

    constexpr bool  should_fetch_src0_row = !_ShouldCacheSrc0;
    const uint8_t * src0_ptr              = src0->get_read_buffer();
    const uint8_t * src1_ptr              = src1->get_read_buffer();

    {
        if constexpr (std::is_same_v<data_type1, float>) {
            hexagon::vec_cpy_f32(reinterpret_cast<const data_type1 *>(src1_ptr),
                                 reinterpret_cast<data_type1 *>(src1_row_cache_ptr),
                                 src1->get_ne(0));
        } else {
            hexagon::vec_cpy_f16(reinterpret_cast<const data_type1 *>(src1_ptr),
                                 reinterpret_cast<data_type1 *>(src1_row_cache_ptr),
                                 src1->get_ne(0));
        }

        src1_ptr = src1_row_cache_ptr;
    }

    {
        for (int64_t col_idx = start_end_element.first; col_idx < start_end_element.second;
             col_idx += src0_plane_slice_row_count) {
            const uint8_t * src0_plane = src0_ptr + col_idx * src0->get_nb(1);
            hexagon::l2fetch_row(src0_plane, src0->get_nb(1));

            const int64_t actual_row_count =
                std::min<int64_t>(src0_plane_slice_row_count,
                                  start_end_element.second - col_idx);  // number of rows in this slice
            if constexpr (_ShouldCacheSrc0) {
                DEVICE_SCOPED_OP_PERFORMANCE_TRACKER_ADD_ONE_SUB_PROC(mul_mat, 0, dequant);

                for (int64_t ir = 0; ir < actual_row_count; ir++) {
                    auto * src0_row = src0_plane + ir * src0->get_nb(1);
                    if (ir + 1 < actual_row_count) {
                        hexagon::l2fetch_row(src0_row + src0->get_nb(1), src0->get_nb(1));
                    }

                    auto * cached_row_ptr = src0_plane_cache_ptr + ir * src0_actual_row_size;
                    dequantize_row_func(
                        src0_row, reinterpret_cast<hexagon::dequant_output_type *>(cached_row_ptr), src0->get_ne(0));
                }

                src0_plane = src0_plane_cache_ptr;
            }

            {
                DEVICE_SCOPED_OP_PERFORMANCE_TRACKER_ADD_ONE_SUB_PROC(mul_mat, 1, vec_dot);
                auto *  dst_row = reinterpret_cast<float *>(dst_ptr) + col_idx;
                int64_t i0      = 0;
                for (; i0 + 1 < actual_row_count; i0 += 2) {
                    auto * src0_row = src0_plane + i0 * src0_actual_row_size;
                    if constexpr (should_fetch_src0_row) {
                        hexagon::l2fetch_row(src0_row + src0_actual_row_size, valid_row0_bytes);
                    }

                    // TODO: figure dst how to handle a entire row
                    auto res0 = _DotFunc(reinterpret_cast<const data_type0 *>(src0_row),
                                         reinterpret_cast<const data_type1 *>(src1_ptr),
                                         (size_t) src0->get_ne(0));

                    if (should_fetch_src0_row && i0 + 2 < actual_row_count) {
                        hexagon::l2fetch_row(src0_row + src0_actual_row_size + src0_actual_row_size, valid_row0_bytes);
                    }

                    // TODO: figure dst how to handle a entire row
                    auto res1 = _DotFunc(reinterpret_cast<const data_type0 *>(src0_row + src0_actual_row_size),
                                         reinterpret_cast<const data_type1 *>(src1_ptr),
                                         (size_t) src0->get_ne(0));

                    {
                        DEVICE_SCOPED_OP_PERFORMANCE_TRACKER_ADD_ONE_SUB_PROC(mul_mat, 2, store);
                        dst_row[i0]     = convert_vector<data_type1>::convert(res0);
                        dst_row[i0 + 1] = convert_vector<data_type1>::convert(res1);
                    }
                }

                if (i0 < actual_row_count) {
                    auto * src0_row = src0_plane + i0 * src0_actual_row_size;
                    auto   res      = _DotFunc(reinterpret_cast<const data_type0 *>(src0_row),
                                        reinterpret_cast<const data_type1 *>(src1_ptr),
                                        (size_t) src0->get_ne(0));
                    DEVICE_SCOPED_OP_PERFORMANCE_TRACKER_ADD_ONE_SUB_PROC(mul_mat, 2, store);
                    dst_row[i0] = convert_vector<data_type1>::convert(res);
                }
            }
        }
    }

    dst->release_write_buffer();  // mark the output tensor as modified
}

bool is_row_size_cacheable(const npu_device_tensor_spec & src) {
    const auto & type_traits = hexagon::get_type_traits(src.type);
    if (type_traits.to_float == nullptr) {
        DEVICE_LOG_DEBUG("[MUL_MAT]src.type(%s) cannot be cached, to_float is null\n",
                         hexagon::get_type_name(src.type));
        return false;
    }

    const size_t type_size = type_traits.is_quantized ? sizeof(hexagon::dequant_output_type) : type_traits.type_size;
    const auto   vtcm_thread_quota_size = hexagon::default_thread_pool::get_per_thread_vtcm_quota();
    if (src.ne[0] * type_size > vtcm_thread_quota_size) {
        DEVICE_LOG_DEBUG("[MUL_MAT]src.type(%s) ne[0] is too large: %ld, vtcm_thread_quota_size: %zu\n",
                         hexagon::get_type_name(src.type),
                         (long) src.ne[0],
                         vtcm_thread_quota_size);
        return false;
    }

    return true;
}

bool is_quantized_mul_mat_supported(const npu_device_tensor_spec & src0, const npu_device_tensor_spec & src1) {
    if (src1.type != NPU_DATA_TYPE_F32 && src1.type != NPU_DATA_TYPE_F16) {
        DEVICE_LOG_DEBUG("[MUL_MAT]src0.type(%s) and src1.type(%s) mismatch and src1 is not F32\n",
                         hexagon::get_type_name(src0.type),
                         hexagon::get_type_name(src1.type));
        return false;
    }

    const auto type_traits = hexagon::get_type_traits(src0.type);
    if (!type_traits.is_quantized || type_traits.to_float == nullptr) {
        DEVICE_LOG_DEBUG("[MUL_MAT]src0.type(%s) and src1.type(%s) mismatch and src0 is not quantized\n",
                         hexagon::get_type_name(src0.type),
                         hexagon::get_type_name(src1.type));
        return false;
    }

    if (src0.ne[0] % type_traits.blck_size) {
        DEVICE_LOG_DEBUG(
            "[MUL_MAT]src0.type(%s) ne[0] is not aligned: %ld\n", hexagon::get_type_name(src0.type), (long) src0.ne[0]);
        return false;
    }

    if (!is_row_size_cacheable(src0)) {
        return false;
    }

    DEVICE_LOG_DEBUG("[MUL_MAT]supported quantized src0.type(%s) and src1.type(%s)\n",
                     hexagon::get_type_name(src0.type),
                     hexagon::get_type_name(src1.type));
    return true;
}

bool is_mul_mat_f16_f32_src_tensors_aligned(hexagon::tensor * src0,
                                            hexagon::tensor * src1,
                                            bool              is_src0_cached,
                                            bool              is_src1_cached) {
    const auto * src1_ptr = is_src1_cached ? nullptr : src1->get_read_buffer_as<float>();
    const auto * src0_ptr = is_src0_cached ? nullptr : src0->get_read_buffer_as<npu_device_fp16_t>();

    if (!hexagon::is_f16_f32_dot_product_aligned(src0_ptr, src1_ptr, src0->get_ne(0))) {
        DEVICE_LOG_DEBUG("[MUL_MAT]src_tensors_unaligned: ne[0]: %ld\n", (long) src0->get_ne(0));
        return false;
    }

    DEVICE_LOG_DEBUG("[MUL_MAT]src_tensors_aligned: ne[0]: %ld\n", (long) src0->get_ne(0));
    return true;
}

bool is_mul_mat_f16_f16_src_tensors_aligned(hexagon::tensor * src0, hexagon::tensor * src1, bool is_src0_quantized) {
    const auto * src1_ptr = src1->get_read_buffer_as<npu_device_fp16_t>();
    const auto * src0_ptr = is_src0_quantized ? nullptr : src0->get_read_buffer_as<npu_device_fp16_t>();

    if (!hexagon::is_f16_f16_dot_product_aligned(src0_ptr, src1_ptr, src0->get_ne(0))) {
        DEVICE_LOG_DEBUG("[MUL_MAT]src_tensors_unaligned: ne[0]: %ld\n", (long) src0->get_ne(0));
        return false;
    }

    if (!is_src0_quantized && !hexagon::is_size_aligned(src0->get_nb(1))) {
        DEVICE_LOG_DEBUG("[MUL_MAT]src0 tensor nb[1] is not aligned: %zu\n", src0->get_nb(1));
        return false;
    }

    if (!hexagon::is_size_aligned(src1->get_nb(1))) {
        DEVICE_LOG_DEBUG("[MUL_MAT]src1 tensor nb[1] is not aligned: %zu\n", src1->get_nb(1));
        return false;
    }

    DEVICE_LOG_DEBUG("[MUL_MAT]src_tensors_aligned: ne[0]: %ld\n", (long) src0->get_ne(0));
    return true;
}

bool is_mul_mat_f32_f32_src_tensors_aligned(hexagon::tensor * src0, hexagon::tensor * src1) {
    const auto * src1_ptr = src1->get_read_buffer_as<float>();
    const auto * src0_ptr = src0->get_read_buffer_as<float>();

    if (!hexagon::is_f32_f32_dot_product_aligned(src0_ptr, src1_ptr, src0->get_ne(0))) {
        DEVICE_LOG_DEBUG("[MUL_MAT]src_tensors_unaligned: ne[0]: %ld\n", (long) src0->get_ne(0));
        return false;
    }

    if (!hexagon::is_size_aligned(src0->get_nb(1))) {
        DEVICE_LOG_DEBUG("[MUL_MAT]src0 tensor nb[1] is not aligned: %zu\n", src0->get_nb(1));
        return false;
    }

    if (!hexagon::is_size_aligned(src1->get_nb(1))) {
        DEVICE_LOG_DEBUG("[MUL_MAT]src1 tensor nb[1] is not aligned: %zu\n", src1->get_nb(1));
        return false;
    }

    DEVICE_LOG_DEBUG("[MUL_MAT]src_tensors_aligned: ne[0]: %ld\n", (long) src0->get_ne(0));
    return true;
}

typedef void (*mul_mat_func_type)(hexagon::tensor *         src0,
                                  hexagon::tensor *         src1,
                                  hexagon::tensor *         dst,
                                  hexagon::compute_params * params);

constexpr const size_t kMulMatGemvBaseIndex = 2;

constexpr const mul_mat_func_type kMulMatF32F32CachedFuncs[4] = {
    // quantized and non-quantized
    mul_mat_impl<hexagon::vec_dot_product_vqf32_f32_f32, true>,               // F32 * F32 quantized unaligned
    mul_mat_impl<hexagon::vec_dot_product_aligned_vqf32_f32_f32, true>,       // F32 * F32 quantized aligned
    mul_mat_gemv_impl<hexagon::vec_dot_product_aligned_vqf32_f32_f32, true>,  // F32 * F32 quantized gemv
    mul_mat_gemv_impl<hexagon::vec_dot_product_aligned_vqf32_f32_f32, true>,  // F32 * F32 quantized gemv
};

constexpr const mul_mat_func_type kMulMatF32F32Funcs[4] = {
    // quantized and non-quantized
    mul_mat_impl<hexagon::vec_dot_product_vqf32_f32_f32, false>,               // F32 * F32 quantized unaligned
    mul_mat_impl<hexagon::vec_dot_product_aligned_vqf32_f32_f32, false>,       // F32 * F32 quantized aligned
    mul_mat_gemv_impl<hexagon::vec_dot_product_vqf32_f32_f32, false>,          // F32 * F32 quantized gemv
    mul_mat_gemv_impl<hexagon::vec_dot_product_aligned_vqf32_f32_f32, false>,  // F32 * F32 quantized gemv
};

constexpr const mul_mat_func_type kMulMatF16F32Funcs[4] = {
    // quantized and non-quantized
    mul_mat_impl<hexagon::vec_dot_product_vqf32_f16_f32, true>,               // F32 * F32 quantized unaligned
    mul_mat_impl<hexagon::vec_dot_product_aligned_vqf32_f16_f32, true>,       // F32 * F32 quantized aligned
    mul_mat_gemv_impl<hexagon::vec_dot_product_vqf32_f16_f32, true>,          // F32 * F32 quantized unaligned
    mul_mat_gemv_impl<hexagon::vec_dot_product_aligned_vqf32_f16_f32, true>,  // F32 * F32 quantized aligned
};

constexpr const mul_mat_func_type kMulMatF16CachedFuncs[4] = {
    mul_mat_impl<hexagon::vec_dot_product_vqf16_f16_f16, true>,               // F16 * F16 quantized unaligned
    mul_mat_impl<hexagon::vec_dot_product_aligned_vqf16_f16_f16, true>,       // F16 * F16 quantized aligned
    mul_mat_gemv_impl<hexagon::vec_dot_product_aligned_vqf16_f16_f16, true>,  // F16 * F16 quantized gemv
    mul_mat_gemv_impl<hexagon::vec_dot_product_aligned_vqf16_f16_f16, true>,  // F16 * F16 quantized gemv
};

constexpr const mul_mat_func_type kMulMatF16Funcs[4] = {
    mul_mat_impl<hexagon::vec_dot_product_vqf16_f16_f16, false>,               // F16 * F16 quantized unaligned
    mul_mat_impl<hexagon::vec_dot_product_aligned_vqf16_f16_f16, false>,       // F16 * F16 quantized aligned
    mul_mat_gemv_impl<hexagon::vec_dot_product_vqf16_f16_f16, false>,          // F16 * F16 quantized gemv
    mul_mat_gemv_impl<hexagon::vec_dot_product_aligned_vqf16_f16_f16, false>,  // F16 * F16 quantized gemv
};

}  // namespace

namespace hexagon {

bool mul_mat_f32(hexagon::tensor * out, compute_params * params) {
    static_assert(DEVICE_TENSOR_MAX_DIMS == 4, "mul_mat_f32 requires max dims 4");
    static_assert(std::is_same<hexagon::dequant_output_type, float>::value ||
                      std::is_same<hexagon::dequant_output_type, npu_device_fp16_t>::value,
                  "dequant_output_type must be float or npu_device_fp16_t");

    if (!out) {
        return false;
    }

    auto * src0 = out->get_src(0);
    auto * src1 = out->get_src(1);
    if (!src0 || !src1) {
        return true;  // skip if no src
    }

    const bool is_src0_quantized = is_quantized_type(src0->get_type());
    const bool should_cache_src0 = is_src0_quantized || src1->get_ne(1) > 1;
    const bool is_gemv           = src1->get_ne(1) == 1 && src1->get_ne(2) == 1 && src1->get_ne(3) == 1;
    const auto base_index        = is_gemv ? kMulMatGemvBaseIndex : 0;
    switch (src1->get_type()) {
        case NPU_DATA_TYPE_F32:
            if (is_src0_quantized || src0->get_type() == NPU_DATA_TYPE_F16) {
                kMulMatF16F32Funcs[is_mul_mat_f16_f32_src_tensors_aligned(src0, src1, is_src0_quantized, is_gemv) +
                                   base_index](src0, src1, out, params);
            } else if (should_cache_src0) {
                kMulMatF32F32CachedFuncs[is_mul_mat_f32_f32_src_tensors_aligned(src0, src1) + base_index](
                    src0, src1, out, params);
            } else {
                kMulMatF32F32Funcs[is_mul_mat_f32_f32_src_tensors_aligned(src0, src1) + base_index](
                    src0, src1, out, params);
            }
            return true;
        case NPU_DATA_TYPE_F16:
            if (should_cache_src0) {
                kMulMatF16CachedFuncs[is_mul_mat_f16_f16_src_tensors_aligned(src0, src1, is_src0_quantized) +
                                      base_index](src0, src1, out, params);
            } else {
                kMulMatF16Funcs[is_mul_mat_f16_f16_src_tensors_aligned(src0, src1, is_src0_quantized) + base_index](
                    src0, src1, out, params);
            }
            return true;
        default:
            break;
    }

    DEVICE_LOG_ERROR("Unsupported src1 tensor type: %s\n", get_type_name(src1->get_type()));
    return false;
}

bool is_mul_mat_supported(npu_device_tensor_op           op,
                          const npu_device_tensor_spec * dst,
                          const npu_device_tensor_spec * srcs,
                          size_t                         src_len) {
    if (op != NPU_OP_MUL_MAT) {
        DEVICE_LOG_DEBUG("op is not MUL_MAT: %d\n", op);
        return false;
    }

    if (!dst || !srcs || src_len < 2) {
        DEVICE_LOG_DEBUG("[%s]invalid dst or srcs\n", hexagon::op_get_name(op));
        return false;
    }

    if (dst->type != NPU_DATA_TYPE_F32) {
        DEVICE_LOG_DEBUG("[%s]dst type is not F32: %s\n", op_get_name(op), get_type_name(dst->type));
        return false;
    }

    const auto & src0 = srcs[0];
    const auto & src1 = srcs[1];
    if (src0.type != src1.type) {
        if (src1.type == NPU_DATA_TYPE_F32 && src0.type == NPU_DATA_TYPE_F16) {
            DEVICE_LOG_DEBUG("[%s]src0.type(%s) and src1.type(%s) mismatch, but src0 is F16 and src1 is F32\n",
                             op_get_name(op),
                             get_type_name(src0.type),
                             get_type_name(src1.type));
            return true;  // F16 * F32 is supported
        }

#ifdef GGML_HEXAGON_ENABLE_QUANTIZED_TENSORS
        if (!is_quantized_mul_mat_supported(src0, src1)) {
            return false;
        }
#else
        DEVICE_LOG_DEBUG("[%s]src0.type(%s) and src1.type(%s) mismatch and quantized tensors are not supported\n",
                         op_get_name(op),
                         get_type_name(src0.type),
                         get_type_name(src1.type));
        return false;
#endif
    }

    if (src0.ne[0] != src1.ne[0] || src0.ne[1] != dst->ne[0]) {
        DEVICE_LOG_DEBUG("[%s]src0 and src1 cannot multiply: %ldx%ld vs %ldx%ld\n",
                         op_get_name(op),
                         (long) src0.ne[0],
                         (long) src0.ne[1],
                         (long) src1.ne[0],
                         (long) src1.ne[1]);
        return false;
    }

    if (src1.ne[1] != dst->ne[1] || src1.ne[2] != dst->ne[2] || src1.ne[3] != dst->ne[3]) {
        DEVICE_LOG_DEBUG("[%s]src1 and dst dimensions not match: %ldx%ld vs %ldx%ld\n",
                         op_get_name(op),
                         (long) src1.ne[2],
                         (long) src1.ne[3],
                         (long) dst->ne[2],
                         (long) dst->ne[3]);
        return false;
    }

    if (src1.ne[2] % src0.ne[2] || src1.ne[3] % src0.ne[3]) {
        DEVICE_LOG_DEBUG("[%s]src0 cannot broadcast to src1: %ldx%ld vs %ldx%ld\n",
                         op_get_name(op),
                         (long) src0.ne[2],
                         (long) src0.ne[3],
                         (long) src1.ne[2],
                         (long) src1.ne[3]);
        return false;
    }

    return true;
}

}  // namespace hexagon
