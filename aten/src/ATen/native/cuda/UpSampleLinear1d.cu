// Adapted from interp.cpp from Caffe util by Pauline Luc
// Originally developed by George Papandreou
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
#ifdef __HIP_PLATFORM_HCC__
C10_LAUNCH_BOUNDS_1(1024)
#endif
__global__ void upsample_linear1d_out_frame(
    const int64_t n,
    const accscalar_t rwidth,
    const bool align_corners,
    const PackedTensorAccessor<scalar_t, 3> idata,
    PackedTensorAccessor<scalar_t, 3> odata) {
  int64_t index = threadIdx.x + blockIdx.x * blockDim.x;

  const int64_t batchsize = idata.size(0);
  const int64_t channels = idata.size(1);
  const int64_t width1 = idata.size(2);
  const int64_t width2 = odata.size(2);

  if (index < n) {
    const int64_t w2 = index % width2;
    // special case: just copy
    if (width1 == width2) {
      const int64_t w1 = w2;
      for (int n = 0; n < batchsize; n++) {
        for (int c = 0; c < channels; ++c) {
          const scalar_t val = idata[n][c][w1];
          odata[n][c][w2] = val;
        }
      }
      return;
    }
    //
    const accscalar_t w1r = area_pixel_compute_source_index<accscalar_t>(
        rwidth, w2, align_corners, /*cubic=*/false);
    const int64_t w1 = w1r;
    const int64_t w1p = (w1 < width1 - 1) ? 1 : 0;
    const accscalar_t w1lambda = w1r - w1;
    const accscalar_t w0lambda = static_cast<accscalar_t>(1) - w1lambda;
    //
    for (int n = 0; n < batchsize; n++) {
      for (int c = 0; c < channels; ++c) {
        const accscalar_t val =
            w0lambda * idata[n][c][w1] + w1lambda * idata[n][c][w1 + w1p];
        odata[n][c][w2] = static_cast<scalar_t>(val);
      }
    }
  }
}

// Backward (adjoint) operation 1 <- 2 (accumulates)
template <typename scalar_t, typename accscalar_t>
#ifdef __HIP_PLATFORM_HCC__
C10_LAUNCH_BOUNDS_1(1024)
#endif
__global__ void upsample_linear1d_out_frame_backward(
    const int64_t n,
    const accscalar_t rwidth,
    const bool align_corners,
    PackedTensorAccessor<scalar_t, 3> idata,
    const PackedTensorAccessor<scalar_t, 3> odata) {
  int64_t index = threadIdx.x + blockIdx.x * blockDim.x;

  const int64_t batchsize = idata.size(0);
  const int64_t channels = idata.size(1);
  const int64_t width1 = idata.size(2);
  const int64_t width2 = odata.size(2);

  if (index < n) {
    const int64_t w2 = index % width2;
    // special case: just copy
    if (width1 == width2) {
      const int64_t w1 = w2;
      for (int n = 0; n < batchsize; n++) {
        for (int c = 0; c < channels; ++c) {
          const scalar_t val = odata[n][c][w1];
          idata[n][c][w2] += val;
        }
      }
      return;
    }
    //
    const accscalar_t w1r = area_pixel_compute_source_index<accscalar_t>(
        rwidth, w2, align_corners, /*cubic=*/false);
    const int64_t w1 = w1r;
    const int64_t w1p = (w1 < width1 - 1) ? 1 : 0;
    const accscalar_t w1lambda = w1r - w1;
    const accscalar_t w0lambda = static_cast<accscalar_t>(1) - w1lambda;
    //
    for (int n = 0; n < batchsize; n++) {
      for (int c = 0; c < channels; ++c) {
        const scalar_t d2val = odata[n][c][w2];
        atomicAdd(&idata[n][c][w1], static_cast<scalar_t>(w0lambda * d2val));
        atomicAdd(
            &idata[n][c][w1 + w1p], static_cast<scalar_t>(w1lambda * d2val));
      }
    }
  }
}

static void upsample_linear1d_out_cuda_template(
    Tensor& output,
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners) {
  checkAllSameGPU("upsample_linear1d_out_cuda", {input, output});

  AT_CHECK(
      output_size.size() == 1,
      "It is expected output_size equals to 1, but got size ",
      output_size.size());

  int64_t output_width = output_size[0];

  int64_t nbatch = input.size(0);
  int64_t channels = input.size(1);
  int64_t input_width = input.size(2);

  upsample_1d_shape_check(
      input, Tensor(), nbatch, channels, input_width, output_width);

  output.resize_({input.size(0), input.size(1), output_width});
  output.zero_();

  auto idata = input.packed_accessor<scalar_t, 3>();
  auto odata = output.packed_accessor<scalar_t, 3>();

  AT_ASSERT(input_width > 0 && output_width > 0);

  const accscalar_t rwidth = area_pixel_compute_scale<accscalar_t>(
      input_width, output_width, align_corners);
  const int64_t num_kernels = output_width;
  const int64_t num_threads =
      at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "upsample_linear1d_out_frame", [&] {
        using accscalar_t = at::acc_type<scalar_t, true>;

        upsample_linear1d_out_frame<scalar_t, accscalar_t>
            <<<(num_kernels + num_threads - 1) / num_threads,
               num_threads,
               0,
               stream>>>(num_kernels, rwidth, align_corners, idata, odata);
      });

  AT_CHECK(
      cudaGetLastError() == cudaSuccess,
      "Failed with error code ",
      cudaGetLastError());
}

static void upsample_linear1d_backward_out_cuda_template(
    Tensor& grad_input,
    const Tensor& grad_output_,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners) {
  checkAllSameGPU(
      "upsample_linear1d_backward_out_cuda", {grad_output_, grad_input});

  AT_CHECK(
      output_size.size() == 1,
      "It is expected output_size equals to 1, but got size ",
      output_size.size());

  AT_CHECK(
      input_size.size() == 3,
      "It is expected input_size equals to 3, but got size ",
      input_size.size());

  int64_t output_width = output_size[0];

  int64_t nbatch = input_size[0];
  int64_t channels = input_size[1];
  int64_t input_width = input_size[2];

  upsample_1d_shape_check(
      Tensor(), grad_output_, nbatch, channels, input_width, output_width);

  Tensor grad_output = grad_output_.contiguous();

  grad_input.resize_({nbatch, channels, input_width});
  grad_input.zero_();

  auto idata = grad_input.packed_accessor<scalar_t, 3>();
  auto odata = grad_output.packed_accessor<scalar_t, 3>();

  const int64_t num_kernels = output_width;
  const int64_t num_threads =
      at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_output.scalar_type(), "upsample_linear1d_out_frame_backward", [&] {
        using accscalar_t = at::acc_type<scalar_t, true>;

        const accscalar_t rwidth = area_pixel_compute_scale<accscalar_t>(
            input_width, output_width, align_corners);

        upsample_linear1d_out_frame_backward<scalar_t, accscalar_t>
            <<<(num_kernels + num_threads - 1) / num_threads,
               num_threads,
               0,
               stream>>>(num_kernels, rwidth, align_corners, idata, odata);
      });

  AT_CHECK(
      cudaGetLastError() == cudaSuccess,
      "Failed with error code ",
      cudaGetLastError());
}

} // namespace

Tensor& upsample_linear1d_out_cuda(
    Tensor& output,
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners) {
  upsample_linear1d_out_cuda_template(
      output, input, output_size, align_corners);
  return output;
}

Tensor upsample_linear1d_cuda(
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners) {
  Tensor output = at::empty({0}, input.options());
  upsample_linear1d_out_cuda_template(
      output, input, output_size, align_corners);
  return output;
}

Tensor& upsample_linear1d_backward_out_cuda(
    Tensor& grad_input,
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners) {
  upsample_linear1d_backward_out_cuda_template(
      grad_input, grad_output, output_size, input_size, align_corners);
  return grand_input;
}

Tensor upsample_linear1d_backward_cuda(
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners) {
  Tensor grad_input = at::zeros(input_size, grad_output.options());
  upsample_linear1d_backward_out_cuda_template(
      grad_input, grad_output, output_size, input_size, align_corners);
  return grad_input;
}

} // namespace native
} // namespace at
