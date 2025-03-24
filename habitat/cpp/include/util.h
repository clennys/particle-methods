#include <iomanip>
#include <sstream>
#include <vector>

namespace util {
template <typename T> std::vector<T> arange(T start, T stop, T step = 1) {
  std::vector<T> values;
  for (T value = start; value < stop; value += step)
    values.push_back(value);
  return values;
}
template <typename Base, typename T> inline bool instanceof(const T *ptr) {
  return dynamic_cast<const Base *>(ptr) != nullptr;
}
template<typename T>
std::string format_number(T num, int width = 3) {
  std::ostringstream oss;
  oss << std::setw(width) << std::setfill('0') << num;
  return oss.str();
}
} // namespace util
