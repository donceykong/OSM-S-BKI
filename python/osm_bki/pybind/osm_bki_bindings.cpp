/**
 * Python bindings for OSM-S-BKI (composite_bki_cpp).
 * Exposes run_pipeline and PySemanticBKI for use by basic_usage.py and other scripts.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>
#include <fstream>
#include <unordered_map>
#include <cmath>
#include <algorithm>

// OSM-S-BKI core (relative to cpp/osm_bki)
#include "continuous_bki.hpp"
#include "file_io.hpp"

namespace py = pybind11;

namespace {

using namespace continuous_bki;

// Write refined labels to a binary file (uint32 per label)
bool writeLabelsBin(const std::string& path, const std::vector<uint32_t>& labels) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) return false;
    file.write(reinterpret_cast<const char*>(labels.data()),
               static_cast<std::streamsize>(labels.size() * sizeof(uint32_t)));
    return true;
}

// Compute accuracy and mIoU between predicted and ground truth labels
void compute_metrics(const uint32_t* refined, const uint32_t* gt, size_t n,
                    double& accuracy, double& miou) {
    size_t correct = 0;
    for (size_t i = 0; i < n; ++i)
        if (refined[i] == gt[i]) ++correct;
    accuracy = n > 0 ? static_cast<double>(correct) / static_cast<double>(n) : 0.0;

    std::unordered_map<uint32_t, int64_t> tp, fp, fn;
    for (size_t i = 0; i < n; ++i) {
        uint32_t r = refined[i], g = gt[i];
        if (r == g) {
            tp[r] += 1;
        } else {
            fp[r] += 1;
            fn[g] += 1;
        }
    }
    std::unordered_map<uint32_t, int64_t> all_classes;
    for (const auto& kv : tp) all_classes[kv.first];
    for (const auto& kv : fp) all_classes[kv.first];
    for (const auto& kv : fn) all_classes[kv.first];

    double sum_iou = 0.0;
    int num_classes = 0;
    for (const auto& kv : all_classes) {
        uint32_t c = kv.first;
        int64_t tp_c = tp.count(c) ? tp[c] : 0;
        int64_t fp_c = fp.count(c) ? fp[c] : 0;
        int64_t fn_c = fn.count(c) ? fn[c] : 0;
        int64_t denom = tp_c + fp_c + fn_c;
        if (denom > 0) {
            sum_iou += static_cast<double>(tp_c) / static_cast<double>(denom);
            ++num_classes;
        }
    }
    miou = num_classes > 0 ? sum_iou / static_cast<double>(num_classes) : 0.0;
}

// Build ContinuousBKI with defaults used by the Python API
ContinuousBKI make_bki(const Config& config, const OSMData& osm_data,
                       float l_scale, float sigma_0, float prior_delta, int num_threads) {
    const float resolution = 0.1f;
    const float height_sigma = 0.3f;
    const bool use_semantic_kernel = true;
    const bool use_spatial_kernel = true;
    const float alpha0 = 1.0f;
    const bool seed_osm_prior = true;
    const float osm_prior_strength = 1.0f;
    const bool osm_fallback_in_infer = true;
    const float lambda_min = 0.8f;
    const float lambda_max = 0.99f;

    return ContinuousBKI(config, osm_data,
                          resolution, l_scale, sigma_0, prior_delta, height_sigma,
                          use_semantic_kernel, use_spatial_kernel,
                          num_threads, alpha0,
                          seed_osm_prior, osm_prior_strength, osm_fallback_in_infer,
                          lambda_min, lambda_max);
}

}  // namespace

// --- run_pipeline: load scan + labels, run BKI, optionally evaluate and save
py::array_t<uint32_t> run_pipeline(
    const std::string& lidar_path,
    const std::string& label_path,
    const std::string& osm_path,
    const std::string& config_path,
    py::object ground_truth_path,  // None or str
    const std::string& output_path,
    float l_scale,
    float sigma_0,
    float prior_delta,
    float alpha_0,
    int num_threads) {

    Config config = loadConfigFromYAML(config_path);
    OSMData osm_data = loadOSM(osm_path, config);

    std::vector<Point3D> points;
    std::vector<uint32_t> labels;
    if (!readPointCloudBin(lidar_path, points) || !readLabelBin(label_path, labels)) {
        throw std::runtime_error("Failed to load lidar or labels: " + lidar_path + " / " + label_path);
    }
    if (points.size() != labels.size()) {
        throw std::runtime_error("Point and label count mismatch");
    }

    ContinuousBKI bki = make_bki(config, osm_data, l_scale, sigma_0, prior_delta, num_threads);
    bki.update(labels, points);
    std::vector<uint32_t> refined = bki.infer(points);

    if (!output_path.empty()) {
        if (!writeLabelsBin(output_path, refined)) {
            throw std::runtime_error("Failed to write output: " + output_path);
        }
    }

    bool has_gt = !ground_truth_path.is_none();
    if (has_gt) {
        std::string gt_path = py::cast<std::string>(ground_truth_path);
        std::vector<uint32_t> gt;
        if (readLabelBin(gt_path, gt) && gt.size() == refined.size()) {
            double acc, miou;
            compute_metrics(refined.data(), gt.data(), refined.size(), acc, miou);
            std::cout << "Accuracy: " << (acc * 100.0) << "%, mIoU: " << (miou * 100.0) << "%" << std::endl;
        }
    }

    py::array_t<uint32_t> out(static_cast<py::ssize_t>(refined.size()));
    py::buffer_info buf = out.request();
    uint32_t* ptr = static_cast<uint32_t*>(buf.ptr);
    std::copy(refined.begin(), refined.end(), ptr);
    return out;
}

// --- PySemanticBKI: wrapper class for script API
class PySemanticBKI {
public:
    PySemanticBKI(const std::string& osm_path,
                  const std::string& config_path,
                  float l_scale = 3.0f,
                  float sigma_0 = 1.0f,
                  float prior_delta = 5.0f,
                  int num_threads = -1)
        : config_(loadConfigFromYAML(config_path)),
          osm_data_(loadOSM(osm_path, config_)),
          bki_(make_bki(config_, osm_data_, l_scale, sigma_0, prior_delta, num_threads)) {}

    py::array_t<uint32_t> process_point_cloud(py::array_t<float> points,
                                              py::array_t<uint32_t> labels,
                                              float alpha_0 = 0.01f) {
        py::buffer_info pb = points.request();
        py::buffer_info lb = labels.request();
        if (pb.ndim != 2 || pb.shape[1] < 3)
            throw std::runtime_error("points must be (N, 3) or (N, 4) float32");
        if (lb.ndim != 1)
            throw std::runtime_error("labels must be (N,) uint32");
        size_t n = static_cast<size_t>(pb.shape[0]);
        if (lb.shape[0] != static_cast<py::ssize_t>(n))
            throw std::runtime_error("points and labels length mismatch");

        const float* px = static_cast<const float*>(pb.ptr);
        const uint32_t* lx = static_cast<const uint32_t*>(lb.ptr);
        std::vector<Point3D> pts(n);
        std::vector<uint32_t> lbls(n);
        for (size_t i = 0; i < n; ++i) {
            pts[i] = Point3D(px[i * pb.shape[1]], px[i * pb.shape[1] + 1], px[i * pb.shape[1] + 2]);
            lbls[i] = lx[i];
        }
        (void)alpha_0;  // BKI uses internal alpha0 from config; kept for API compatibility
        bki_.update(lbls, pts);
        std::vector<uint32_t> refined = bki_.infer(pts);

        py::array_t<uint32_t> out(static_cast<py::ssize_t>(refined.size()));
        py::buffer_info ob = out.request();
        uint32_t* out_ptr = static_cast<uint32_t*>(ob.ptr);
        std::copy(refined.begin(), refined.end(), out_ptr);
        return out;
    }

    py::dict evaluate_metrics(py::array_t<uint32_t> refined, py::array_t<uint32_t> gt) {
        py::buffer_info rb = refined.request();
        py::buffer_info gb = gt.request();
        if (rb.ndim != 1 || gb.ndim != 1 || rb.shape[0] != gb.shape[0]) {
            throw std::runtime_error("refined and gt must be 1D arrays of same length");
        }
        size_t n = static_cast<size_t>(rb.shape[0]);
        const uint32_t* rp = static_cast<const uint32_t*>(rb.ptr);
        const uint32_t* gp = static_cast<const uint32_t*>(gb.ptr);
        double accuracy, miou;
        compute_metrics(rp, gp, n, accuracy, miou);

        py::dict d;
        d["accuracy"] = accuracy;
        d["miou"] = miou;
        return d;
    }

private:
    Config config_;
    OSMData osm_data_;
    ContinuousBKI bki_;
};

PYBIND11_MODULE(composite_bki_cpp, m) {
    m.doc() = "OSM-S-BKI / Composite BKI C++ extension: semantic BKI with OSM priors";

    m.def("run_pipeline", &run_pipeline,
          py::arg("lidar_path"),
          py::arg("label_path"),
          py::arg("osm_path"),
          py::arg("config_path"),
          py::arg("ground_truth_path") = py::none(),
          py::arg("output_path") = "",
          py::arg("l_scale") = 3.0f,
          py::arg("sigma_0") = 1.0f,
          py::arg("prior_delta") = 5.0f,
          py::arg("alpha_0") = 0.01f,
          py::arg("num_threads") = -1,
          "Run full pipeline: load scan/labels, refine with BKI, optionally evaluate and save.");

    py::class_<PySemanticBKI>(m, "PySemanticBKI")
        .def(py::init<const std::string&, const std::string&, float, float, float, int>(),
             py::arg("osm_path"),
             py::arg("config_path"),
             py::arg("l_scale") = 3.0f,
             py::arg("sigma_0") = 1.0f,
             py::arg("prior_delta") = 5.0f,
             py::arg("num_threads") = -1)
        .def("process_point_cloud", &PySemanticBKI::process_point_cloud,
             py::arg("points"),
             py::arg("labels"),
             py::arg("alpha_0") = 0.01f,
             "Update BKI with points/labels and return refined labels.")
        .def("evaluate_metrics", &PySemanticBKI::evaluate_metrics,
             py::arg("refined"),
             py::arg("gt"),
             "Return dict with 'accuracy' and 'miou'.");
}
