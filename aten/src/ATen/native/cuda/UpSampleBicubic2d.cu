#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/native/cuda/UpSample.cuh>

namespace at {
namespace native {
namespace {

template <typename scalar_t, typename accscalar_t>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void upsample_bicubic2d_out_frame(
    const int64_t num_elements,
    const accscalar_t height_scale,
    const accscalar_t width_scale,
    const bool align_corners,
    const PackedTensorAccessor<scalar_t, 4> idata,
    PackedTensorAccessor<scalar_t, 4> odata) {
  int64_t index = threadIdx.x + blockIdx.x * blockDim.x;

  const int64_t batchsize = idata.size(0);
  const int64_t channels = idata.size(1);
  const int64_t input_height = idata.size(2);
  const int64_t input_width = idata.size(3);
  const int64_t output_height = odata.size(2);
  const int64_t output_width = odata.size(3);

  if (index >= num_elements) {
    return;
  }

  // Special case: input and output are the same size, just copy
  const int64_t output_x = index % output_width;
  const int64_t output_y = index / output_width;

  if (input_height == output_height && input_width == output_width) {
    for (int n = 0; n < batchsize; n++) {
      for (int c = 0; c < channels; c++) {
        const scalar_t val = idata[n][c][output_y][output_x];
        odata[n][c][output_x][output_y] = val;
      }
    }
    return;
  }

  // Interpolation kernel
  accscalar_t real_x = area_pixel_compute_source_index(
      width_scale, output_x, align_corners, /*cubic=*/true);
  int64_t in_x = floorf(real_x);
  accscalar_t t_x = real_x - in_x;

  accscalar_t real_y = area_pixel_compute_source_index(
      height_scale, output_y, align_corners, /*cubic=*/true);
  int64_t in_y = floorf(real_y);
  accscalar_t t_y = real_y - in_y;

  for (int n = 0; n < batchsize; n++) {
    for (int c = 0; c < channels; c++) {
      accscalar_t coefficients[4];

      for (int k = 0; k < 4; k++) {
        coefficients[k] = cubic_interp1d(
            upsampling_get_value_bounded<scalar_t>(
                idata, c, n, input_width, input_height, in_x - 1, in_y - 1 + k),
            upsampling_get_value_bounded<scalar_t>(
                idata, c, n, input_width, input_height, in_x + 0, in_y - 1 + k),
            upsampling_get_value_bounded<scalar_t>(
                idata, c, n, input_width, input_height, in_x + 1, in_y - 1 + k),
            upsampling_get_value_bounded<scalar_t>(
                idata, c, n, input_width, input_height, in_x + 2, in_y - 1 + k),
            t_x);
      }

      odata[n][c][output_y][output_x] = static_cast<scalar_t>(cubic_interp1d(
          coefficients[0],
          coefficients[1],
          coefficients[2],
          coefficients[3],
          t_y));
    }
  }
}

// Backward (adjoint) operation 1 <- 2 (accumulates)
template <typename scalar_t, typename accscalar_t>
#ifdef __HIP_PLATFORM_HCC__
C10_LAUNCH_BOUNDS_1(1024)
#endif
__global__ void upsample_bicubic2d_backward_out_frame(
    const int64_t num_elements,
    const accscalar_t height_scale,
    const accscalar_t width_scale,
    const bool align_corners,
    PackedTensorAccessor<scalar_t, 4> idata,
    const PackedTensorAccessor<scalar_t, 4> odata) {

  int64_t index = threadIdx.x + blockIdx.x * blockDim.x;

  const int64_t batchsize = idata.size(0);
  const int64_t channels = idata.size(1);
  const int64_t input_height = idata.size(2);
  const int64_t input_width = idata.size(3);
  const int64_t output_height = odata.size(2);
  const int64_t output_width = odata.size(3);

  if (index >= num_elements) {
    return;
  }

  const int64_t output_x = index % output_width;
  const int64_t output_y = index / output_width;
  // special case: output_xust copy
  if (input_height == output_height && input_width == output_width) {
    for (int n = 0; n < batchsize; n++) {
      for (int c = 0; c < channels; ++c) {
        const scalar_t val = odata[n][c][output_y][output_x];
        idata[n][c][output_y][output_x] += val;
      }
    }
    return;
  }

  accscalar_t real_x = area_pixel_compute_source_index(
      width_scale, output_x, align_corners, /*cubic=*/true);
  int64_t input_x = floorf(real_x);
  accscalar_t t_x = real_x - input_x;

  accscalar_t real_y = area_pixel_compute_source_index(
      height_scale, output_y, align_corners, /*cubic=*/true);
  int64_t input_y = floorf(real_y);
  accscalar_t t_y = real_y - input_y;

  accscalar_t x_coeffs[4];
  accscalar_t y_coeffs[4];

  get_cubic_upsampling_coefficients(x_coeffs, t_x);
  get_cubic_upsampling_coefficients(y_coeffs, t_y);

  for (int n = 0; n < batchsize; n++) {
    for (int c = 0; c < channels; ++c) {
      scalar_t out_value = odata[n][c][output_y][output_x];
      for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
          upsampling_increment_value_bounded<scalar_t, accscalar_t>(
              idata,
              c,
              n,
              input_width,
              input_height,
              input_x - 1 + j,
              input_y - 1 + i,
              out_value * y_coeffs[i] * x_coeffs[j]);
        }
      }
    }
  }
}

static void upsample_bicubic2d_out_cuda_template(
    Tensor& output,
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners) {
  TensorArg input_arg{input, "input", 1}, output_arg{output, "output", 2};
  checkAllSameGPU("upsample_bicubic2d_out", {input_arg, output_arg});

  AT_CHECK(
      output_size.size() == 2,
      "It is expected output_size equals to 2, but got size ",
      output_size.size());

  int64_t output_height = output_size[0];
  int64_t output_width = output_size[1];

  int64_t nbatch = input.size(0);
  int64_t channels = input.size(1);
  int64_t input_height = input.size(2);
  int64_t input_width = input.size(3);

  upsample_2d_shape_check(
      input,
      Tensor(),
      nbatch,
      channels,
      input_height,
      input_width,
      output_height,
      output_width);

  output.resize_({input.size(0), input.size(1), output_height, output_width});
  output.zero_();

  AT_ASSERT(
      input_height > 0 && input_width > 0 && output_height > 0 &&
      output_width > 0);

  const int64_t num_output_elements = output_height * output_width;
  const int64_t max_threads =
      at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock;

  // Launch kernel
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "upsample_bicubic2d_out_frame", [&] {
        using accscalar_t = at::acc_type<scalar_t, true>;

        auto idata = input.packed_accessor<scalar_t, 4>();
        auto odata = output.packed_accessor<scalar_t, 4>();

        // Get scaling factors
        const accscalar_t rheight = area_pixel_compute_scale<accscalar_t>(
            input_height, output_height, align_corners);
        const accscalar_t rwidth = area_pixel_compute_scale<accscalar_t>(
            input_width, output_width, align_corners);

        upsample_bicubic2d_out_frame<scalar_t, accscalar_t>
            <<<(num_output_elements + max_threads - 1) / max_threads,
               max_threads,
               0,
               stream>>>(
                num_output_elements,
                rheight,
                rwidth,
                align_corners,
                idata,
                odata);
      });

  AT_CUDA_CHECK(cudaGetLastError());
}

static void upsample_bicubic2d_backward_out_cuda_template(
    Tensor& grad_input,
    const Tensor& grad_output_,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners) {
  TensorArg grad_input_arg{grad_input, "grad_input", 1},
      grad_output_arg{grad_output_, "grad_output_", 2};
  checkAllSameGPU(
      "upsample_bicubic2d_backward_out_cuda",
      {grad_output_arg, grad_input_arg});

  AT_CHECK(
      output_size.size() == 2,
      "It is expected output_size equals to 2, but got size ",
      output_size.size());

  AT_CHECK(
      input_size.size() == 4,
      "It is expected input_size equals to 4, but got size ",
      input_size.size());

  int64_t output_height = output_size[0];
  int64_t output_width = output_size[1];

  int64_t nbatch = input_size[0];
  int64_t channels = input_size[1];
  int64_t input_height = input_size[2];
  int64_t input_width = input_size[3];

  upsample_2d_shape_check(
      Tensor(),
      grad_output_,
      nbatch,
      channels,
      input_height,
      input_width,
      output_height,
      output_width);

  Tensor grad_output = grad_output_.contiguous();

  grad_input.resize_({nbatch, channels, input_height, input_width});
  grad_input.zero_();

  const int64_t num_kernels = output_height * output_width;
  const int64_t num_threads =
      at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_output.scalar_type(), "upsample_bicubic2d_backward_out_frame", [&] {
        using accscalar_t = at::acc_type<scalar_t, true>;

        auto idata = grad_input.packed_accessor<scalar_t, 4>();
        auto odata = grad_output.packed_accessor<scalar_t, 4>();

        const accscalar_t rheight = area_pixel_compute_scale<accscalar_t>(
            input_height, output_height, align_corners);
        const accscalar_t rwidth = area_pixel_compute_scale<accscalar_t>(
            input_width, output_width, align_corners);

        upsample_bicubic2d_backward_out_frame<scalar_t, accscalar_t>
            <<<(num_kernels + num_threads - 1) / num_threads,
               num_threads,
               0,
               stream>>>(
                num_kernels, rheight, rwidth, align_corners, idata, odata);
      });

  AT_CUDA_CHECK(cudaGetLastError());
}

} // namespace

Tensor& upsample_bicubic2d_out_cuda(
    Tensor& output,
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners) {
  upsample_bicubic2d_out_cuda_template(
      output, input, output_size, align_corners);
  return output;
}

Tensor upsample_bicubic2d_cuda(
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners) {
  Tensor output = at::empty({0}, input.options());
  upsample_bicubic2d_out_cuda_template(
      output, input, output_size, align_corners);
  return output;
}

Tensor& upsample_bicubic2d_backward_out_cuda(
    Tensor& grad_input,
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners) {
  upsample_bicubic2d_backward_out_cuda_template(
      grad_input, grad_output, output_size, input_size, align_corners);
  return grad_input;
}

Tensor upsample_bicubic2d_backward_cuda(
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners) {
  Tensor grad_input = at::zeros(input_size, grad_output.options());
  upsample_bicubic2d_backward_out_cuda_template(
      grad_input, grad_output, output_size, input_size, align_corners);
  return grad_input;
}

} // namespace native
} // namespace at
