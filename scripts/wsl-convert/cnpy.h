// cnpy.h — minimal npz reader
#pragma once
#include <string>
#include <vector>
#include <map>
#include <cstdint>
#include <stdexcept>

namespace cnpy {

struct NpyArray {
    std::vector<char> data;
    std::vector<size_t> shape;
    char type_code;  // 'f'=float32, 'i'=int32, 'l'=int64
    size_t word_size;
    bool fortran_order;

    template<typename T> T* data_ptr() { return reinterpret_cast<T*>(data.data()); }
    template<typename T> const T* data_ptr() const { return reinterpret_cast<const T*>(data.data()); }
    size_t num_vals() const { size_t n=1; for(auto s:shape) n*=s; return n; }
};

NpyArray npz_load(const std::string& fname, const std::string& varname);
std::map<std::string, NpyArray> npz_load_all(const std::string& fname);

} // namespace cnpy
