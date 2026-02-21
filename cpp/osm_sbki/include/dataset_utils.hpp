#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace continuous_bki {

struct DatasetConfig {
    std::string dataset_name;
    std::string dataset_root_path;
    std::string sequence;
    int skip_frames = 0;
    std::string lidar_dir;
    std::string label_dir;
    std::string pose_path;
    std::string calibration_path;
    /** OSM file path relative to dataset_root_path (e.g. "kth.osm"). Empty if not set. */
    std::string osm_file;
    /** If true, use osm_origin_lat/lon for OSM parser origin (from init_latlon_day_06). */
    bool use_osm_origin_from_mcd = false;
    double osm_origin_lat = 0.0;
    double osm_origin_lon = 0.0;
    /** World origin for pose alignment (from init_rel_pos_day_06 [x,y,z]). If not set, (0,0,0) is used. */
    bool use_init_rel_pos = false;
    double init_rel_pos_x = 0.0;
    double init_rel_pos_y = 0.0;
    double init_rel_pos_z = 0.0;
};

struct ScanLabelPair {
    std::string scan_id;
    std::string scan_path;
    std::string label_path;
};

bool loadDatasetConfig(const std::string& config_path, DatasetConfig& config, std::string& error_msg);
bool collectScanLabelPairs(const DatasetConfig& config, std::vector<ScanLabelPair>& pairs, std::string& error_msg);
bool getScanLabelPair(const std::vector<ScanLabelPair>& pairs, size_t index, ScanLabelPair& pair, std::string& error_msg);

}  // namespace continuous_bki
