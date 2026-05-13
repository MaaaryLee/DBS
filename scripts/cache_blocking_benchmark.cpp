#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct LayerSpec {
  std::string name;
  int input_dim = 0;
  int output_dim = 0;
};

struct PresetSpec {
  std::string name;
  std::vector<LayerSpec> layers;
};

struct Options {
  std::string preset = "400x300";
  int row_tile = 8;
  int col_tile = 32;
  int warmup = 250;
  int repeats = 5000;
  uint32_t seed = 7;
  std::size_t flush_bytes = 0;
};

struct KernelStats {
  std::string kernel;
  double avg_ns = 0.0;
  double ns_per_mac = 0.0;
  double gmac_per_s = 0.0;
  double checksum = 0.0;
};

struct BenchmarkResult {
  std::string layer_name;
  int input_dim = 0;
  int output_dim = 0;
  int macs = 0;
  double blocked_max_abs_diff = 0.0;
  double packed_max_abs_diff = 0.0;
  KernelStats naive;
  KernelStats blocked;
  KernelStats packed;
};

void PrintUsage(const char* argv0) {
  std::cerr
      << "Usage: " << argv0 << " [--preset 96x96|400x300]"
      << " [--row-tile N] [--col-tile N] [--warmup N] [--repeats N]"
      << " [--seed N] [--flush-bytes N]\n";
}

bool ParseInt(std::string_view text, int* out) {
  char* end = nullptr;
  const long value = std::strtol(text.data(), &end, 10);
  if (end == nullptr || *end != '\0') {
    return false;
  }
  *out = static_cast<int>(value);
  return true;
}

bool ParseSize(std::string_view text, std::size_t* out) {
  char* end = nullptr;
  const unsigned long long value = std::strtoull(text.data(), &end, 10);
  if (end == nullptr || *end != '\0') {
    return false;
  }
  *out = static_cast<std::size_t>(value);
  return true;
}

bool ParseUint32(std::string_view text, uint32_t* out) {
  char* end = nullptr;
  const unsigned long value = std::strtoul(text.data(), &end, 10);
  if (end == nullptr || *end != '\0') {
    return false;
  }
  *out = static_cast<uint32_t>(value);
  return true;
}

bool ParseOptions(int argc, char** argv, Options* options) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto require_value = [&](const char* name) -> const char* {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << name << "\n";
        return nullptr;
      }
      ++i;
      return argv[i];
    };

    if (arg == "--preset") {
      const char* value = require_value("--preset");
      if (!value) return false;
      options->preset = value;
    } else if (arg == "--row-tile") {
      const char* value = require_value("--row-tile");
      if (!value || !ParseInt(value, &options->row_tile)) return false;
    } else if (arg == "--col-tile") {
      const char* value = require_value("--col-tile");
      if (!value || !ParseInt(value, &options->col_tile)) return false;
    } else if (arg == "--warmup") {
      const char* value = require_value("--warmup");
      if (!value || !ParseInt(value, &options->warmup)) return false;
    } else if (arg == "--repeats") {
      const char* value = require_value("--repeats");
      if (!value || !ParseInt(value, &options->repeats)) return false;
    } else if (arg == "--seed") {
      const char* value = require_value("--seed");
      if (!value || !ParseUint32(value, &options->seed)) return false;
    } else if (arg == "--flush-bytes") {
      const char* value = require_value("--flush-bytes");
      if (!value || !ParseSize(value, &options->flush_bytes)) return false;
    } else if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      std::exit(0);
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      return false;
    }
  }

  if (options->row_tile <= 0 || options->col_tile <= 0) {
    std::cerr << "row_tile and col_tile must be positive.\n";
    return false;
  }
  if (options->warmup < 0 || options->repeats <= 0) {
    std::cerr << "warmup must be >= 0 and repeats must be > 0.\n";
    return false;
  }
  return true;
}

PresetSpec BuildPreset(const std::string& name) {
  if (name == "96x96") {
    return PresetSpec{
        "96x96",
        {
            {"fc1", 6, 96},
            {"fc2", 96, 96},
            {"fc3", 96, 2},
        },
    };
  }
  if (name == "400x300") {
    return PresetSpec{
        "400x300",
        {
            {"fc1", 6, 400},
            {"fc2", 400, 300},
            {"fc3", 300, 2},
        },
    };
  }
  std::cerr << "Unsupported preset: " << name << "\n";
  std::exit(2);
}

void FillRandomVector(std::mt19937& rng, std::vector<float>* values) {
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (float& value : *values) {
    value = dist(rng);
  }
}

void MatVecNaive(
    const float* weights,
    const float* input,
    const float* bias,
    float* output,
    int input_dim,
    int output_dim) {
  for (int row = 0; row < output_dim; ++row) {
    float acc = bias[row];
    const float* weight_row = weights + static_cast<std::size_t>(row) * input_dim;
    for (int col = 0; col < input_dim; ++col) {
      acc += weight_row[col] * input[col];
    }
    output[row] = acc;
  }
}

void MatVecBlockedRowMajor(
    const float* weights,
    const float* input,
    const float* bias,
    float* output,
    int input_dim,
    int output_dim,
    int row_tile,
    int col_tile,
    float* scratch) {
  for (int row_base = 0; row_base < output_dim; row_base += row_tile) {
    const int row_stop = std::min(row_base + row_tile, output_dim);
    const int local_rows = row_stop - row_base;

    for (int local_row = 0; local_row < local_rows; ++local_row) {
      scratch[local_row] = bias[row_base + local_row];
    }

    for (int col_base = 0; col_base < input_dim; col_base += col_tile) {
      const int col_stop = std::min(col_base + col_tile, input_dim);
      for (int local_row = 0; local_row < local_rows; ++local_row) {
        const float* weight_row =
            weights + static_cast<std::size_t>(row_base + local_row) * input_dim + col_base;
        float acc = scratch[local_row];
        for (int col = col_base; col < col_stop; ++col) {
          acc += *weight_row++ * input[col];
        }
        scratch[local_row] = acc;
      }
    }

    for (int local_row = 0; local_row < local_rows; ++local_row) {
      output[row_base + local_row] = scratch[local_row];
    }
  }
}

std::vector<float> PackWeightsRowTiles(
    const float* weights,
    int input_dim,
    int output_dim,
    int row_tile,
    int col_tile) {
  std::vector<float> packed;
  packed.reserve(static_cast<std::size_t>(input_dim) * output_dim);

  for (int row_base = 0; row_base < output_dim; row_base += row_tile) {
    const int row_stop = std::min(row_base + row_tile, output_dim);
    for (int col_base = 0; col_base < input_dim; col_base += col_tile) {
      const int col_stop = std::min(col_base + col_tile, input_dim);
      for (int row = row_base; row < row_stop; ++row) {
        const float* weight_row =
            weights + static_cast<std::size_t>(row) * input_dim + col_base;
        for (int col = col_base; col < col_stop; ++col) {
          packed.push_back(*weight_row++);
        }
      }
    }
  }

  return packed;
}

void MatVecBlockedPacked(
    const float* packed_weights,
    const float* input,
    const float* bias,
    float* output,
    int input_dim,
    int output_dim,
    int row_tile,
    int col_tile,
    float* scratch) {
  std::size_t packed_index = 0;

  for (int row_base = 0; row_base < output_dim; row_base += row_tile) {
    const int row_stop = std::min(row_base + row_tile, output_dim);
    const int local_rows = row_stop - row_base;

    for (int local_row = 0; local_row < local_rows; ++local_row) {
      scratch[local_row] = bias[row_base + local_row];
    }

    for (int col_base = 0; col_base < input_dim; col_base += col_tile) {
      const int col_stop = std::min(col_base + col_tile, input_dim);
      for (int local_row = 0; local_row < local_rows; ++local_row) {
        float acc = scratch[local_row];
        for (int col = col_base; col < col_stop; ++col) {
          acc += packed_weights[packed_index++] * input[col];
        }
        scratch[local_row] = acc;
      }
    }

    for (int local_row = 0; local_row < local_rows; ++local_row) {
      output[row_base + local_row] = scratch[local_row];
    }
  }
}

void TouchBuffer(std::vector<uint8_t>* flush_buffer) {
  if (flush_buffer->empty()) {
    return;
  }
  volatile uint8_t sink = 0;
  constexpr std::size_t kStride = 64;
  for (std::size_t index = 0; index < flush_buffer->size(); index += kStride) {
    sink ^= (*flush_buffer)[index];
  }
  if (sink == 255) {
    std::cerr << "";
  }
}

double MaxAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
  double max_abs_diff = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    max_abs_diff = std::max(max_abs_diff, std::fabs(static_cast<double>(a[i] - b[i])));
  }
  return max_abs_diff;
}

template <typename Function>
KernelStats RunKernelBenchmark(
    std::string kernel_name,
    Function fn,
    std::vector<uint8_t>* flush_buffer,
    std::vector<float>* output,
    int macs,
    int warmup,
    int repeats) {
  for (int i = 0; i < warmup; ++i) {
    TouchBuffer(flush_buffer);
    fn();
  }

  double checksum = 0.0;
  double total_ns = 0.0;
  if (flush_buffer->empty()) {
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeats; ++i) {
      fn();
      checksum += (*output)[i % output->size()];
    }
    const auto stop = std::chrono::steady_clock::now();
    total_ns =
        std::chrono::duration_cast<std::chrono::duration<double, std::nano>>(stop - start).count();
  } else {
    for (int i = 0; i < repeats; ++i) {
      TouchBuffer(flush_buffer);
      const auto start = std::chrono::steady_clock::now();
      fn();
      const auto stop = std::chrono::steady_clock::now();
      total_ns +=
          std::chrono::duration_cast<std::chrono::duration<double, std::nano>>(stop - start).count();
      checksum += (*output)[i % output->size()];
    }
  }

  const double avg_ns = total_ns / static_cast<double>(repeats);
  const double ns_per_mac = avg_ns / static_cast<double>(macs);
  const double gmac_per_s = static_cast<double>(macs) / avg_ns;

  return KernelStats{
      std::move(kernel_name),
      avg_ns,
      ns_per_mac,
      gmac_per_s,
      checksum,
  };
}

BenchmarkResult BenchmarkLayer(
    const LayerSpec& layer,
    const Options& options,
    std::mt19937* rng,
    std::vector<uint8_t>* flush_buffer) {
  std::vector<float> weights(static_cast<std::size_t>(layer.input_dim) * layer.output_dim);
  std::vector<float> input(layer.input_dim);
  std::vector<float> bias(layer.output_dim);
  std::vector<float> output_naive(layer.output_dim, 0.0f);
  std::vector<float> output_blocked(layer.output_dim, 0.0f);
  std::vector<float> output_packed(layer.output_dim, 0.0f);
  std::vector<float> scratch(std::max(1, options.row_tile), 0.0f);

  FillRandomVector(*rng, &weights);
  FillRandomVector(*rng, &input);
  FillRandomVector(*rng, &bias);

  const std::vector<float> packed_weights = PackWeightsRowTiles(
      weights.data(), layer.input_dim, layer.output_dim, options.row_tile, options.col_tile);

  MatVecNaive(weights.data(), input.data(), bias.data(), output_naive.data(), layer.input_dim, layer.output_dim);
  MatVecBlockedRowMajor(
      weights.data(),
      input.data(),
      bias.data(),
      output_blocked.data(),
      layer.input_dim,
      layer.output_dim,
      options.row_tile,
      options.col_tile,
      scratch.data());
  MatVecBlockedPacked(
      packed_weights.data(),
      input.data(),
      bias.data(),
      output_packed.data(),
      layer.input_dim,
      layer.output_dim,
      options.row_tile,
      options.col_tile,
      scratch.data());

  const int macs = layer.input_dim * layer.output_dim;
  auto naive_stats = RunKernelBenchmark(
      "naive",
      [&]() {
        MatVecNaive(
            weights.data(), input.data(), bias.data(), output_naive.data(), layer.input_dim, layer.output_dim);
      },
      flush_buffer,
      &output_naive,
      macs,
      options.warmup,
      options.repeats);
  auto blocked_stats = RunKernelBenchmark(
      "blocked_row_major",
      [&]() {
        MatVecBlockedRowMajor(
            weights.data(),
            input.data(),
            bias.data(),
            output_blocked.data(),
            layer.input_dim,
            layer.output_dim,
            options.row_tile,
            options.col_tile,
            scratch.data());
      },
      flush_buffer,
      &output_blocked,
      macs,
      options.warmup,
      options.repeats);
  auto packed_stats = RunKernelBenchmark(
      "blocked_packed",
      [&]() {
        MatVecBlockedPacked(
            packed_weights.data(),
            input.data(),
            bias.data(),
            output_packed.data(),
            layer.input_dim,
            layer.output_dim,
            options.row_tile,
            options.col_tile,
            scratch.data());
      },
      flush_buffer,
      &output_packed,
      macs,
      options.warmup,
      options.repeats);

  return BenchmarkResult{
      layer.name,
      layer.input_dim,
      layer.output_dim,
      macs,
      MaxAbsDiff(output_naive, output_blocked),
      MaxAbsDiff(output_naive, output_packed),
      std::move(naive_stats),
      std::move(blocked_stats),
      std::move(packed_stats),
  };
}

void PrintKernelLine(
    const std::string& layer_name,
    int input_dim,
    int output_dim,
    int macs,
    const KernelStats& stats,
    double max_abs_diff) {
  std::cout << "RESULT"
            << " layer=" << layer_name
            << " input_dim=" << input_dim
            << " output_dim=" << output_dim
            << " macs=" << macs
            << " kernel=" << stats.kernel
            << " avg_ns=" << std::fixed << std::setprecision(3) << stats.avg_ns
            << " ns_per_mac=" << std::setprecision(6) << stats.ns_per_mac
            << " gmac_per_s=" << std::setprecision(6) << stats.gmac_per_s
            << " checksum=" << std::setprecision(6) << stats.checksum
            << " max_abs_diff=" << std::setprecision(6) << max_abs_diff
            << "\n";
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (!ParseOptions(argc, argv, &options)) {
    PrintUsage(argv[0]);
    return 1;
  }

  const PresetSpec preset = BuildPreset(options.preset);
  std::mt19937 rng(options.seed);
  std::vector<uint8_t> flush_buffer(options.flush_bytes, 0);

  std::cout << "CONFIG"
            << " preset=" << preset.name
            << " row_tile=" << options.row_tile
            << " col_tile=" << options.col_tile
            << " warmup=" << options.warmup
            << " repeats=" << options.repeats
            << " seed=" << options.seed
            << " flush_bytes=" << options.flush_bytes
            << "\n";

  double total_naive_ns = 0.0;
  double total_blocked_ns = 0.0;
  double total_packed_ns = 0.0;
  int total_macs = 0;

  for (const LayerSpec& layer : preset.layers) {
    const BenchmarkResult result = BenchmarkLayer(layer, options, &rng, &flush_buffer);
    total_naive_ns += result.naive.avg_ns;
    total_blocked_ns += result.blocked.avg_ns;
    total_packed_ns += result.packed.avg_ns;
    total_macs += result.macs;

    PrintKernelLine(
        result.layer_name,
        result.input_dim,
        result.output_dim,
        result.macs,
        result.naive,
        0.0);
    PrintKernelLine(
        result.layer_name,
        result.input_dim,
        result.output_dim,
        result.macs,
        result.blocked,
        result.blocked_max_abs_diff);
    PrintKernelLine(
        result.layer_name,
        result.input_dim,
        result.output_dim,
        result.macs,
        result.packed,
        result.packed_max_abs_diff);
  }

  const auto print_total = [&](std::string_view kernel, double total_avg_ns) {
    std::cout << "TOTAL"
              << " kernel=" << kernel
              << " total_avg_ns=" << std::fixed << std::setprecision(3) << total_avg_ns
              << " total_ns_per_mac=" << std::setprecision(6)
              << (total_avg_ns / static_cast<double>(total_macs))
              << " total_gmac_per_s=" << std::setprecision(6)
              << (static_cast<double>(total_macs) / total_avg_ns)
              << "\n";
  };

  print_total("naive", total_naive_ns);
  print_total("blocked_row_major", total_blocked_ns);
  print_total("blocked_packed", total_packed_ns);
  return 0;
}
