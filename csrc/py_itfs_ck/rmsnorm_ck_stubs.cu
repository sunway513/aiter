// CK-free stubs: forward CK rmsnorm functions to HIP rms_norm
// Used when CK headers are not available (gfx1250 CK-free builds)
#include <torch/all.h>
#include <optional>
void rms_norm(torch::Tensor& out, torch::Tensor& input, torch::Tensor& weight, double epsilon);
torch::Tensor rmsnorm2d(torch::Tensor& input, torch::Tensor& weight, double epsilon, int) {
    auto out = torch::empty_like(input); rms_norm(out, input, weight, epsilon); return out;
}
void rmsnorm2d_with_add(torch::Tensor& out, torch::Tensor& input, torch::Tensor& residual_in, torch::Tensor& residual_out, torch::Tensor& weight, double epsilon, int) {
    residual_out.copy_(input + residual_in); rms_norm(out, residual_out, weight, epsilon);
}
void rmsnorm2d_with_smoothquant(torch::Tensor& out, torch::Tensor& input, torch::Tensor& xscale, torch::Tensor& yscale, torch::Tensor& weight, double epsilon, int) {
    rms_norm(out, input, weight, epsilon);
}
void rmsnorm2d_with_add_smoothquant(torch::Tensor& out, torch::Tensor& input, torch::Tensor& residual_in, torch::Tensor& residual_out, torch::Tensor& xscale, torch::Tensor& yscale, torch::Tensor& weight, double epsilon, std::optional<torch::Tensor>, int) {
    residual_out.copy_(input + residual_in); rms_norm(out, residual_out, weight, epsilon);
}
void rmsnorm2d_with_dynamicquant(torch::Tensor& out, torch::Tensor& input, torch::Tensor& yscale, torch::Tensor& weight, double epsilon, int) {
    rms_norm(out, input, weight, epsilon);
}
void rmsnorm2d_with_add_dynamicquant(torch::Tensor& out, torch::Tensor& input, torch::Tensor& residual_in, torch::Tensor& residual_out, torch::Tensor& yscale, torch::Tensor& weight, double epsilon, int) {
    residual_out.copy_(input + residual_in); rms_norm(out, residual_out, weight, epsilon);
}
