
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include <set>
#include <tuple>
#include <unordered_set>
#include <vector>

namespace tensorrt_llm
{
namespace common
{

std::set<int> getLocalGroup(std::set<int> const& group);
std::tuple<bool, bool> setGroupTopology(std::set<int> group);
std::vector<bool> initGroupTopology(std::set<int> group);

} // namespace common
} // namespace tensorrt_llm
