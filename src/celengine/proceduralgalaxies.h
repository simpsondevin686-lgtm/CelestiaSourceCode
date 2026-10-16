// proceduralgalaxies.h
//
// Deterministic, on-demand generation of galaxies in intergalactic space.
// Generated galaxies are real celestia::engine::Galaxy objects using the
// standard GalacticForms, so they render identically to catalog galaxies
// (3D blobs, no billboards) and can be flown through.
//
// The generator is stateless: for a given integer cell coordinate the
// galaxies produced are always the same. A per-frame cache in Renderer
// holds the cells near the observer and evicts them as the observer
// travels, so memory stays bounded regardless of distance covered.
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 2
// of the License, or (at your option) any later version.

#pragma once

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "deepskyobj.h"
#include "galaxy.h"

namespace celestia::engine::procedural
{

constexpr double InnerRadiusMly   = 10.0;
constexpr double OuterRadiusMly   = 500.0;
constexpr double CellSizeMly      = 1.0;
constexpr double DensityPerMly3   = 0.3;
constexpr std::size_t MaxCachedCells = 4096;

struct CellCoord
{
    std::int32_t x, y, z;

    bool operator==(const CellCoord& o) const noexcept
    {
        return x == o.x && y == o.y && z == o.z;
    }
};

struct CellCoordHash
{
    std::size_t operator()(const CellCoord& c) const noexcept
    {
        std::uint64_t h = 1469598103934665603ull;
        auto mix = [&h](std::uint64_t v) { h ^= v; h *= 1099511628211ull; };
        mix(static_cast<std::uint32_t>(c.x));
        mix(static_cast<std::uint32_t>(c.y));
        mix(static_cast<std::uint32_t>(c.z));
        return static_cast<std::size_t>(h);
    }
};

struct SeededRng
{
    std::uint64_t state;
    explicit SeededRng(CellCoord c) noexcept
    {
        std::uint64_t h = 1469598103934665603ull;
        auto mix = [&h](std::uint64_t v) { h ^= v; h *= 1099511628211ull; };
        mix(static_cast<std::uint32_t>(c.x) + 0x9e3779b9u);
        mix(static_cast<std::uint32_t>(c.y) + 0x85ebca6bu);
        mix(static_cast<std::uint32_t>(c.z) + 0xc2b2ae35u);
        state = h ? h : 1ull;
    }
    std::uint64_t next() noexcept
    {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return state;
    }
    float uniform() noexcept
    {
        return static_cast<float>(next() >> 11) * (1.0f / 9007199254740992.0f);
    }
    float range(float lo, float hi) noexcept { return lo + (hi - lo) * uniform(); }
    float signedUnit() noexcept { return range(-1.0f, 1.0f); }
};

struct CellContent
{
    // Stored as base-class pointers because DSORenderer::process takes
    // const std::unique_ptr<DeepSkyObject>&. Galaxy upcasts implicitly;
    // the renderer's internal static_cast<const Galaxy*> works as usual.
    std::vector<std::unique_ptr<DeepSkyObject>> galaxies;
};

inline GalaxyType pickGalaxyType(SeededRng& rng) noexcept
{
    const float u = rng.uniform();
    if (u < 0.15f)
        return GalaxyType::Irr;
    if (u < 0.65f)
    {
        const int idx = static_cast<int>(rng.uniform() * 7.0f);
        return static_cast<GalaxyType>(static_cast<int>(GalaxyType::S0) + idx);
    }
    const int idx = static_cast<int>(rng.uniform() * 8.0f);
    return static_cast<GalaxyType>(static_cast<int>(GalaxyType::E0) + idx);
}

inline const char* galaxyTypeToStr(GalaxyType t) noexcept
{
    switch (t)
    {
    case GalaxyType::Irr: return "Irr";
    case GalaxyType::S0:  return "S0";
    case GalaxyType::Sa:  return "Sa";
    case GalaxyType::Sb:  return "Sb";
    case GalaxyType::Sc:  return "Sc";
    case GalaxyType::SBa: return "SBa";
    case GalaxyType::SBb: return "SBb";
    case GalaxyType::SBc: return "SBc";
    case GalaxyType::E0:  return "E0";
    case GalaxyType::E1:  return "E1";
    case GalaxyType::E2:  return "E2";
    case GalaxyType::E3:  return "E3";
    case GalaxyType::E4:  return "E4";
    case GalaxyType::E5:  return "E5";
    case GalaxyType::E6:  return "E6";
    case GalaxyType::E7:  return "E7";
    }
    return "Irr";
}

inline CellContent
generateCell(CellCoord cell)
{
    CellContent content;
    SeededRng rng(cell);

    const double volume = CellSizeMly * CellSizeMly * CellSizeMly;
    const double meanCount = DensityPerMly3 * volume;

    // Poisson sample (Knuth)
    const double L = std::exp(-meanCount);
    int k = 0;
    double p = 1.0;
    do { ++k; p *= static_cast<double>(rng.uniform()); } while (p > L && k < 32);
    const int count = k - 1;

    const Eigen::Vector3d cellOrigin(cell.x * CellSizeMly,
                                     cell.y * CellSizeMly,
                                     cell.z * CellSizeMly);

    for (int i = 0; i < count; ++i)
    {
        auto galaxy = std::make_unique<Galaxy>();

        galaxy->setPosition(cellOrigin + Eigen::Vector3d(
            rng.range(0.0f, static_cast<float>(CellSizeMly)),
            rng.range(0.0f, static_cast<float>(CellSizeMly)),
            rng.range(0.0f, static_cast<float>(CellSizeMly))));

        const float radiusKly = std::pow(rng.uniform(), 2.0f) * 200.0f + 5.0f;
        galaxy->setRadius(radiusKly * 1000.0f);

        const float baseMag = -21.0f + 3.0f * std::log10(radiusKly / 30.0f);
        galaxy->setAbsoluteMagnitude(baseMag + rng.range(-1.0f, 1.0f));

        Eigen::Vector3f axis(rng.signedUnit(), rng.signedUnit(), rng.signedUnit());
        if (axis.norm() < 1.0e-6f) axis = Eigen::Vector3f::UnitZ();
        axis.normalize();
        galaxy->setOrientation(Eigen::Quaternionf(
            Eigen::AngleAxisf(rng.range(0.0f, 6.2831853f), axis)));

        galaxy->setType(galaxyTypeToStr(pickGalaxyType(rng)));
        galaxy->setVisible(true);
        galaxy->setClickable(false);

        // Upcast to base pointer for the renderer's consume path.
        content.galaxies.push_back(std::move(galaxy));
    }

    return content;
}

} // namespace celestia::engine::procedural
