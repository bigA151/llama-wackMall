#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static double percentile(std::vector<double> values, double p) {
    std::sort(values.begin(), values.end());
    const size_t index = std::min(values.size() - 1,
            (size_t) std::ceil(p*(double) values.size()) - 1);
    return values[index];
}

int main(int argc, char ** argv) {
    const int64_t ne0 = argc > 1 ? std::atoll(argv[1]) : 2048;
    const int64_t n_expert = argc > 2 ? std::atoll(argv[2]) : 384;
    const int iterations = argc > 3 ? std::atoi(argv[3]) : 1000;
    const int warmup = argc > 4 ? std::atoi(argv[4]) : 50;
    if (ne0 <= 0 || n_expert <= 0 || iterations <= 0 || warmup < 0) {
        std::fprintf(stderr, "usage: %s [ne0=2048] [experts=384] [iterations=1000] [warmup=50]\n", argv[0]);
        return 1;
    }

    ggml_backend_load_all();
    ggml_backend_dev_t device = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t candidate = ggml_backend_dev_get(i);
        if (std::strncmp(ggml_backend_dev_name(candidate), "Vulkan", 6) == 0) {
            device = candidate;
            break;
        }
    }
    if (!device) {
        std::fprintf(stderr, "no Vulkan device found\n");
        return 2;
    }

    ggml_backend_t backend = ggml_backend_dev_init(device, "queue=auxiliary");
    if (!backend) {
        std::fprintf(stderr, "Vulkan device %s has no independent auxiliary compute queue\n",
                ggml_backend_dev_name(device));
        return 3;
    }

    ggml_init_params params = {
        /* .mem_size   = */ 1024*1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * gate = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, n_expert);
    ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne0);
    ggml_tensor * logits = ggml_mul_mat(ctx, gate, x);
    ggml_set_name(gate, "effective_gate");
    ggml_set_name(x, "router_input");
    ggml_set_name(logits, "logits");

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(graph, logits);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        std::fprintf(stderr, "failed to allocate Vulkan tensors\n");
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 4;
    }

    std::mt19937 rng(1);
    std::uniform_real_distribution<float> distribution(-0.125f, 0.125f);
    std::vector<float> gate_data((size_t) ne0*(size_t) n_expert);
    std::vector<float> x_data((size_t) ne0);
    std::vector<float> logits_data((size_t) n_expert);
    for (float & value : gate_data) {
        value = distribution(rng);
    }
    for (float & value : x_data) {
        value = distribution(rng);
    }
    ggml_backend_tensor_set(gate, gate_data.data(), 0, gate_data.size()*sizeof(float));

    ggml_backend_event_t event = ggml_backend_event_new(device);
    if (!event) {
        std::fprintf(stderr, "Vulkan event creation failed\n");
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 5;
    }

    std::vector<double> samples_us;
    samples_us.reserve((size_t) iterations);
    for (int iteration = -warmup; iteration < iterations; iteration++) {
        x_data[0] += 1e-7f;
        const auto start = std::chrono::steady_clock::now();
        ggml_backend_tensor_set_async(backend, x, x_data.data(), 0, x_data.size()*sizeof(float));
        const ggml_status status = ggml_backend_graph_compute_async(backend, graph);
        if (status != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "Vulkan graph submission failed: %d\n", (int) status);
            ggml_backend_event_free(event);
            ggml_backend_buffer_free(buffer);
            ggml_free(ctx);
            ggml_backend_free(backend);
            return 6;
        }
        ggml_backend_tensor_get_async(backend, logits, logits_data.data(), 0,
                logits_data.size()*sizeof(float));
        ggml_backend_event_record(event, backend);
        ggml_backend_event_synchronize(event);
        const auto end = std::chrono::steady_clock::now();
        if (iteration >= 0) {
            samples_us.push_back(std::chrono::duration<double, std::micro>(end - start).count());
        }
    }

    double max_abs_error = 0.0;
    for (int64_t expert = 0; expert < n_expert; expert++) {
        double expected = 0.0;
        const float * row = gate_data.data() + (size_t) expert*(size_t) ne0;
        for (int64_t i = 0; i < ne0; i++) {
            expected += (double) row[i]*(double) x_data[(size_t) i];
        }
        max_abs_error = std::max(max_abs_error,
                std::abs(expected - (double) logits_data[(size_t) expert]));
    }

    double sum = 0.0;
    for (double sample : samples_us) {
        sum += sample;
    }
    std::printf("device=%s ne0=%lld experts=%lld iterations=%d weight_mib=%.3f\n",
            ggml_backend_dev_description(device), (long long) ne0, (long long) n_expert,
            iterations, (double) gate_data.size()*sizeof(float)/(1024.0*1024.0));
    std::printf("roundtrip_us: mean=%.3f p50=%.3f p95=%.3f p99=%.3f max=%.3f\n",
            sum/(double) samples_us.size(), percentile(samples_us, 0.50),
            percentile(samples_us, 0.95), percentile(samples_us, 0.99),
            *std::max_element(samples_us.begin(), samples_us.end()));
    std::printf("max_abs_error=%.9g\n", max_abs_error);

    ggml_backend_event_free(event);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return max_abs_error <= 1e-3 ? 0 : 7;
}
