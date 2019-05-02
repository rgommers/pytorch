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
C10_LAUNCH_BOUNDS_1(1024)
__global__ void upsample_trilinear3d_out_frame(
    const int64_t n,
    const accscalar_t rdepth,
    const accscalar_t rheight,
    const accscalar_t rwidth,
    const bool align_corners,
    const PackedTensorAccessor<scalar_t, 5> idata,
    PackedTensorAccessor<scalar_t, 5> odata) {
  int64_t index = threadIdx.x + blockIdx.x * blockDim.x;

  const int64_t batchsize = idata.size(0);
  const int64_t channels = idata.size(1);
  const int64_t depth1 = idata.size(2);
  const int64_t height1 = idata.size(3);
  const int64_t width1 = idata.size(4);
  const int64_t depth2 = odata.size(2);
  const int64_t height2 = odata.size(3);
  const int64_t width2 = odata.size(4);

  if (index < n) {
    const int64_t w2 = (index % (height2 * width2)) % width2; // 0:width2-1
    const int64_t h2 = (index % (height2 * width2)) / width2; // 0:height2-1
    const int64_t t2 = index / (height2 * width2); // 0:depth2-1
    // special case: just copy
    if (depth1 == depth2 && height1 == height2 && width1 == width2) {
      const int64_t t1 = t2;
      const int64_t h1 = h2;
      const int64_t w1 = w2;

      for (int n = 0; n < batchsize; n++) {
        for (int c = 0; c < channels; ++c) {
          const scalar_t val = idata[n][c][t1][h1][w1];
          odata[n][c][t2][h2][w2] = val;
        }
      }
      return;
    }
    //
    const accscalar_t t1r = linear_upsampling_compute_source_index<accscalar_t>(
        rdepth, t2, align_corners);
    const int64_t t1 = t1r;
    const int64_t t1p = (t1 < depth1 - 1) ? 1 : 0;
    const accscalar_t t1lambda = t1r - t1;
    const accscalar_t t0lambda = static_cast<accscalar_t>(1) - t1lambda;
    //
    const accscalar_t h1r = linear_upsampling_compute_source_index<accscalar_t>(
        rheight, h2, align_corners);
    const int64_t h1 = h1r;
    const int64_t h1p = (h1 < height1 - 1) ? 1 : 0;
    const accscalar_t h1lambda = h1r - h1;
    const accscalar_t h0lambda = static_cast<accscalar_t>(1) - h1lambda;
    //
    const accscalar_t w1r = linear_upsampling_compute_source_index<accscalar_t>(
        rwidth, w2, align_corners);
    const int64_t w1 = w1r;
    const int64_t w1p = (w1 < width1 - 1) ? 1 : 0;
    const accscalar_t w1lambda = w1r - w1;
    const accscalar_t w0lambda = static_cast<accscalar_t>(1) - w1lambda;
    //
    for (int n = 0; n < batchsize; n++) {
      for (int c = 0; c < channels; ++c) {
        const accscalar_t val = t0lambda *
                (h0lambda *
                     (w0lambda * idata[n][c][t1][h1][w1] +
                      w1lambda * idata[n][c][t1][h1][w1 + w1p]) +
                 h1lambda *
                     (w0lambda * idata[n][c][t1][h1 + h1p][w1] +
                      w1lambda * idata[n][c][t1][h1 + h1p][w1 + w1p])) +
            t1lambda *
                (h0lambda *
                     (w0lambda * idata[n][c][t1 + t1p][h1][w1] +
                      w1lambda * idata[n][c][t1 + t1p][h1][w1 + w1p]) +
                 h1lambda *
                     (w0lambda * idata[n][c][t1 + t1p][h1 + h1p][w1] +
                      w1lambda * idata[n][c][t1 + t1p][h1 + h1p][w1 + w1p]));
        odata[n][c][t2][h2][w2] = static_cast<scalar_t>(val);
      }
    }
  }
}

// Backward (adjoint) operation 1 <- 2 (accumulates)
template <typename scalar_t, typename accscalar_t>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void upsample_trilinear3d_backward_out_frame(
    const int64_t n,
    const accscalar_t rdepth,
    const accscalar_t rheight,
    const accscalar_t rwidth,
    const bool align_corners,
    PackedTensorAccessor<scalar_t, 5> idata,
    const PackedTensorAccessor<scalar_t, 5> odata) {
  int64_t index = threadIdx.x + blockIdx.x * blockDim.x;

  const int64_t batchsize = idata.size(0);
  const int64_t channels = idata.size(1);
  const int64_t depth1 = idata.size(2);
  const int64_t height1 = idata.size(3);
  const int64_t width1 = idata.size(4);
  const int64_t depth2 = odata.size(2);
  const int64_t height2 = odata.size(3);
  const int64_t width2 = odata.size(4);

  if (index < n) {
    const int64_t w2 = (index % (height2 * width2)) % width2; // 0:width2-1
    const int64_t h2 = (index % (height2 * width2)) / width2; // 0:height2-1
    const int64_t t2 = index / (height2 * width2); // 0:depth2-1
    // special case: just copy
    if (depth1 == depth2 && height1 == height2 && width1 == width2) {
      const int64_t t1 = t2;
      const int64_t h1 = h2;
      const int64_t w1 = w2;

      for (int n = 0; n < batchsize; n++) {
        for (int c = 0; c < channels; ++c) {
          const scalar_t val = odata[n][c][t1][h1][w1];
          idata[n][c][t2][h2][w2] += val;
        }
      }
      return;
    }
    //
    const accscalar_t t1r = linear_upsampling_compute_source_index<accscalar_t>(
        rdepth, t2, align_corners);
    const int64_t t1 = t1r;
    const int64_t t1p = (t1 < depth1 - 1) ? 1 : 0;
    const accscalar_t t1lambda = t1r - t1;
    const accscalar_t t0lambda = static_cast<accscalar_t>(1) - t1lambda;
    //
    const accscalar_t h1r = linear_upsampling_compute_source_index<accscalar_t>(
        rheight, h2, align_corners);
    const int64_t h1 = h1r;
    const int64_t h1p = (h1 < height1 - 1) ? 1 : 0;
    const accscalar_t h1lambda = h1r - h1;
    const accscalar_t h0lambda = static_cast<accscalar_t>(1) - h1lambda;
    //
    const accscalar_t w1r = linear_upsampling_compute_source_index<accscalar_t>(
        rwidth, w2, align_corners);
    const int64_t w1 = w1r;
    const int64_t w1p = (w1 < width1 - 1) ? 1 : 0;
    const accscalar_t w1lambda = w1r - w1;
    const accscalar_t w0lambda = static_cast<accscalar_t>(1) - w1lambda;
    //
    for (int n = 0; n < batchsize; n++) {
      for (int c = 0; c < channels; ++c) {
        const scalar_t d2val = odata[n][c][t2][h2][w2];
        atomicAdd(
            &idata[n][c][t1][h1][w1],
            static_cast<scalar_t>(t0lambda * h0lambda * w0lambda * d2val));
        atomicAdd(
            &idata[n][c][t1][h1][w1 + w1p],
            static_cast<scalar_t>(t0lambda * h0lambda * w1lambda * d2val));
        atomicAdd(
            &idata[n][c][t1][h1 + h1p][w1],
            static_cast<scalar_t>(t0lambda * h1lambda * w0lambda * d2val));
        atomicAdd(
            &idata[n][c][t1][h1 + h1p][w1 + w1p],
            static_cast<scalar_t>(t0lambda * h1lambda * w1lambda * d2val));
        atomicAdd(
            &idata[n][c][t1 + t1p][h1][w1],
            static_cast<scalar_t>(t1lambda * h0lambda * w0lambda * d2val));
        atomicAdd(
            &idata[n][c][t1 + t1p][h1][w1 + w1p],
            static_cast<scalar_t>(t1lambda * h0lambda * w1lambda * d2val));
        atomicAdd(
            &idata[n][c][t1 + t1p][h1 + h1p][w1],
            static_cast<scalar_t>(t1lambda * h1lambda * w0lambda * d2val));
        atomicAdd(
            &idata[n][c][t1 + t1p][h1 + h1p][w1 + w1p],
            static_cast<scalar_t>(t1lambda * h1lambda * w1lambda * d2val));
      }
    }
  }
}

static void upsample_trilinear3d_out_cuda_template(
    Tensor& output,
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners) {
  TensorArg input_arg{input, "input", 1}, output_arg{output, "output", 2};

  checkAllSameGPU("upsample_trilinear3d_out_cuda", {input_arg, output_arg});

  AT_CHECK(
      output_size.size() == 3,
      "It is expected output_size equals to 3, but got size ",
      output_size.size());

  int64_t output_depth = output_size[0];
  int64_t output_height = output_size[1];
  int64_t output_width = output_size[2];

  int64_t nbatch = input.size(0);
  int64_t channels = input.size(1);
  int64_t input_depth = input.size(2);
  int64_t input_height = input.size(3);
  int64_t input_width = input.size(4);

  upsample_3d_shape_check(
      input,
      Tensor(),
      nbatch,
      channels,
      input_depth,
      input_height,
      input_width,
      output_depth,
      output_height,
      output_width);

  output.resize_({input.size(0),
                  input.size(1),
                  output_depth,
                  output_height,
                  output_width});
  output.zero_();

  AT_ASSERT(
      input_depth > 0 && input_height > 0 && input_width > 0 &&
      output_depth > 0 && output_height > 0 && output_width > 0);

  const int64_t num_kernels = output_depth * output_height * output_width;
  const int64_t num_threads =
      at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "upsample_trilinear3d_out_frame", [&] {
        using accscalar_t = at::acc_type<scalar_t, true>;

        auto idata = input.packed_accessor<scalar_t, 5>();
        auto odata = output.packed_accessor<scalar_t, 5>();

        const accscalar_t rdepth = linear_upsampling_compute_scale<accscalar_t>(
            input_depth, output_depth, align_corners);
        const accscalar_t rheight =
            linear_upsampling_compute_scale<accscalar_t>(
                input_height, output_height, align_corners);
        const accscalar_t rwidth = linear_upsampling_compute_scale<accscalar_t>(
            input_width, output_width, align_corners);

        upsample_trilinear3d_out_frame<scalar_t, accscalar_t>
            <<<(num_kernels + num_threads - 1) / num_threads,
               num_threads,
               0,
               stream>>>(
                num_kernels,
                rdepth,
                rheight,
                rwidth,
                align_corners,
                idata,
                odata);
      });

  AT_CHECK(
      cudaGetLastError() == cudaSuccess,
      "Failed with error code ",
      cudaGetLastError());
}

static void upsample_trilinear3d_backward_out_cuda_template(
    Tensor& grad_input,
    const Tensor& grad_output_,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners) {
  TensorArg grad_input_arg{grad_input, "grad_input", 1},
      grad_output_arg{grad_output_, "grad_output_", 2};

  checkAllSameGPU(
      "upsample_trilinear3d_backward_out_cuda",
      {grad_output_arg, grad_input_arg});

  AT_CHECK(
      output_size.size() == 3,
      "It is expected output_size equals to 3, but got size ",
      output_size.size());

  AT_CHECK(
      input_size.size() == 5,
      "It is expected input_size equals to 5, but got size ",
      input_size.size());

  int64_t output_depth = output_size[0];
  int64_t output_height = output_size[1];
  int64_t output_width = output_size[2];

  int64_t nbatch = input_size[0];
  int64_t channels = input_size[1];
  int64_t input_depth = input_size[2];
  int64_t input_height = input_size[3];
  int64_t input_width = input_size[4];

  upsample_3d_shape_check(
      Tensor(),
      grad_output_,
      nbatch,
      channels,
      input_depth,
      input_height,
      input_width,
      output_depth,
      output_height,
      output_width);
  Tensor grad_output = grad_output_.contiguous();

  grad_input.resize_(
      {nbatch, channels, input_depth, input_height, input_width});
  grad_input.zero_();

  const int64_t num_kernels = output_depth * output_height * output_width;
  const int64_t num_threads =
      at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_output.scalar_type(),
      "upsample_trilinear3d_backward_out_frame",
      [&] {
        using accscalar_t = at::acc_type<scalar_t, true>;

        auto idata = grad_input.packed_accessor<scalar_t, 5>();
        auto odata = grad_output.packed_accessor<scalar_t, 5>();

        const accscalar_t rdepth = linear_upsampling_compute_scale<accscalar_t>(
            input_depth, output_depth, align_corners);
        const accscalar_t rheight =
            linear_upsampling_compute_scale<accscalar_t>(
                input_height, output_height, align_corners);
        const accscalar_t rwidth = linear_upsampling_compute_scale<accscalar_t>(
            input_width, output_width, align_corners);

        upsample_trilinear3d_backward_out_frame<scalar_t, accscalar_t>
            <<<(num_kernels + num_threads - 1) / num_threads,
               num_threads,
               0,
               stream>>>(
                num_kernels,
                rdepth,
                rheight,
                rwidth,
                align_corners,
                idata,
                odata);
      });

  AT_CHECK(
      cudaGetLastError() == cudaSuccess,
      "Failed with error code ",
      cudaGetLastError());
}

} // namespace

Tensor& upsample_trilinear3d_out_cuda(
    Tensor& output,
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners) {
  upsample_trilinear3d_out_cuda_template(
      output, input, output_size, align_corners);
  return output;
}

Tensor upsample_trilinear3d_cuda(
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners) {
  Tensor output = at::empty({0}, input.options());
  upsample_trilinear3d_out_cuda_template(
      output, input, output_size, align_corners);
  return output;
}

Tensor& upsample_trilinear3d_backward_out_cuda(
    Tensor& grad_input,
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners) {
  upsample_trilinear3d_backward_out_cuda_template(
      grad_input, grad_output, output_size, input_size, align_corners);
  return grad_input;
}

Tensor upsample_trilinear3d_backward_cuda(
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners) {
  Tensor grad_input = at::zeros(input_size, grad_output.options());
  upsample_trilinear3d_backward_out_cuda_template(
      grad_input, grad_output, output_size, input_size, align_corners);
  return grad_input;
}

} // namespace native
} // namespace at
