// lightamount.h
//
// Compute the fraction of a star's light that reaches the observer,
// accounting for every body that falls between them on the sky.
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 2
// of the License, or (at your option) any later version.

#pragma once

#include <algorithm>
#include <cmath>
#include <numbers>

#include <Eigen/Core>

#include <celengine/body.h>
#include <celengine/atmosphere.h>
#include <celengine/star.h>
#include <celengine/universe.h>
#include <celengine/univcoord.h>

namespace celestia::engine::lightamount
{

namespace detail
{

// Fraction of a star disc (angular radius starRad) hidden by a body disc
// (angular radius bodyRad) at angular separation sepRad. All angles in
// radians. Returns a value in [0, 1].
inline double
discOcclusionFraction(double starRad, double bodyRad, double sepRad)
{
    if (starRad <= 0.0 || bodyRad <= 0.0 || sepRad < 0.0)
        return 0.0;

    if (sepRad >= starRad + bodyRad)
        return 0.0;

    if (sepRad <= starRad - bodyRad)
        return (bodyRad * bodyRad) / (starRad * starRad);

    if (sepRad <= bodyRad - starRad)
        return 1.0;

    const double d2 = sepRad * sepRad;
    const double a2 = starRad * starRad;
    const double b2 = bodyRad * bodyRad;

    const double alpha = std::acos(std::clamp((d2 + a2 - b2) / (2.0 * sepRad * starRad), -1.0, 1.0));
    const double beta  = std::acos(std::clamp((d2 + b2 - a2) / (2.0 * sepRad * bodyRad), -1.0, 1.0));

    const double prod = (-sepRad + starRad + bodyRad)
                      * ( sepRad + starRad - bodyRad)
                      * ( sepRad - starRad + bodyRad)
                      * ( sepRad + starRad + bodyRad);
    const double tri = 0.5 * std::sqrt(std::max(prod, 0.0));

    const double overlapArea = a2 * alpha + b2 * beta - tri;
    return overlapArea / (std::numbers::pi * a2);
}

// Linear limb darkening coefficient (visible light, Sun-like star).
// The stellar disc is brighter at the center than at the limb because we
// see deeper (hotter) layers looking straight down. u = 0 gives uniform
// brightness; u = 0.6 is a typical value for a G-type star.
constexpr double LimbDarkeningCoefficient = 0.6;

// Surface brightness at normalized distance d from the star's disc center,
// relative to the disc-averaged brightness. Returns >1 near the center,
// <1 near the limb. d = sepRad / starRad, clamped to [0, 1].
//
// Derivation:
//   I(d)    = 1 - u*(1 - sqrt(1 - d^2))         surface brightness profile
//   I_avg   = 1 - u/3                            disc-averaged brightness
//   weight  = I(d) / I_avg
inline double
limbDarkeningWeight(double sepRad, double starRad)
{
    if (starRad <= 0.0)
        return 1.0;

    const double d  = std::min(sepRad / starRad, 1.0);
    const double mu = std::sqrt(std::max(0.0, 1.0 - d * d));
    const double I  = 1.0 - LimbDarkeningCoefficient * (1.0 - mu);
    const double Iavg = 1.0 - LimbDarkeningCoefficient / 3.0;

    return I / Iavg;
}

// Positions are supplied in the frame of the system's root star
// (the barycenter for multiple-star systems), all in km.
inline void
accumulateBodies(const PlanetarySystem* system,
                 const Eigen::Vector3d& observerInFrame,
                 const Eigen::Vector3d& starInFrame,
                 double starAngularRadius,
                 double jd,
                 double& totalOcclusion,
                 int depth = 0)
{
    if (system == nullptr || depth > 8)
        return;

    const Eigen::Vector3d starFromObserver = starInFrame - observerInFrame;
    const double distStar = starFromObserver.norm();
    if (distStar < 1.0e-6)
        return;

    const int n = system->getSystemSize();
    for (int i = 0; i < n; ++i)
    {
        const Body* body = system->getBody(i);
        if (body == nullptr || !body->isVisible())
            continue;

        // Astrocentric positions are relative to the frame root, which is
        // the same root we are using for observerInFrame/starInFrame.
        const Eigen::Vector3d bodyInFrame     = body->getAstrocentricPosition(jd);
        const Eigen::Vector3d bodyFromObserver = bodyInFrame - observerInFrame;
        const double distBody = bodyFromObserver.norm();
        if (distBody < 1.0e-6)
            continue;

        // Compute angular separation with atan2 rather than acos. When the
        // star and body are nearly aligned on the sky (e.g. an exoplanet
        // transiting its star, seen from light years away), cos(sep) is
        // 1 - O(1e-13) and acos loses ~6 digits of precision -- enough for
        // the result to flicker frame-to-frame. atan2(|cross|, dot) preserves
        // full precision even for arbitrarily small separations.
        const Eigen::Vector3d cross = starFromObserver.cross(bodyFromObserver);
        const double sinSep = cross.norm() / (distStar * distBody);
        const double cosSep = starFromObserver.dot(bodyFromObserver) / (distStar * distBody);
        const double sepRad = std::atan2(sinSep, cosSep);

        double bodyRadiusKm = static_cast<double>(body->getRadius());
        const BodyFeaturesManager* mgr = GetBodyFeaturesManager();
        if (const Atmosphere* atm = mgr->getAtmosphere(body); atm != nullptr)
        {
            const double atmHeight = std::max(static_cast<double>(atm->height),
                                              static_cast<double>(atm->cloudHeight));
            if (atmHeight > 0.0)
                bodyRadiusKm += atmHeight;
        }
        const double bodyAngularRadius = bodyRadiusKm / distBody;

        // Weight the occluded area by the local surface brightness. A body
        // transiting the disc center blocks ~1.25x more than its area would
        // suggest; one at the limb blocks ~0.5x. Accurate for small
        // occluders (exoplanets, moons); for a Moon-sized occluder spanning
        // the disc, we approximate with brightness at the body center.
        const double ldWeight = limbDarkeningWeight(sepRad, starAngularRadius);
        totalOcclusion += discOcclusionFraction(starAngularRadius, bodyAngularRadius, sepRad)
                        * ldWeight;

        if (const RingSystem* rings = mgr->getRings(body); rings != nullptr)
        {
            const double outerAng = static_cast<double>(rings->outerRadius) / distBody;
            const double innerAng = static_cast<double>(rings->innerRadius) / distBody;
            totalOcclusion += (discOcclusionFraction(starAngularRadius, outerAng, sepRad)
                            -  discOcclusionFraction(starAngularRadius, innerAng, sepRad))
                            * ldWeight;
        }

        // Recurse into the body's own satellites, not its parent system.
        // getSystem() returns the system the body belongs to (would recurse
        // forever); getSatellites() returns the body's own child system.
        if (const PlanetarySystem* moons = body->getSatellites(); moons != nullptr)
            accumulateBodies(moons, observerInFrame, starInFrame, starAngularRadius, jd, totalOcclusion, depth + 1);
    }
}

} // namespace detail

inline double
computeLightAmount(const Star& star,
                   const Universe& universe,
                   const UniversalCoord& observerPos,
                   double jd)
{
    const double starRadiusKm = static_cast<double>(star.getRadius());
    if (starRadiusKm <= 0.0)
        return 1.0;

    // In a multiple-star system, planets often orbit the barycenter rather
    // than any individual star. Those bodies live in the barycenter's
    // PlanetarySystem, not in the star's own system. Walk up to the barycenter
    // and use its system as the source of occluders.
    const Star* systemRoot = star.getOrbitBarycenter();
    if (systemRoot == nullptr)
        systemRoot = &star;

    const UniversalCoord rootPos = systemRoot->getPosition(jd);
    const UniversalCoord starPos = star.getPosition(jd);

    // Working frame: root (barycenter or the star itself) at the origin, km.
    const Eigen::Vector3d observerInFrame = observerPos.offsetFromKm(rootPos);
    const Eigen::Vector3d starInFrame     = starPos.offsetFromKm(rootPos);

    const Eigen::Vector3d starFromObserver = starInFrame - observerInFrame;
    const double distStar = starFromObserver.norm();
    if (distStar < 1.0e-6)
        return 1.0;

    const double starAngularRadius = starRadiusKm / distStar;

    double totalOcclusion = 0.0;
    const SolarSystem* sys = universe.getSolarSystem(systemRoot);
    if (sys != nullptr && sys->getPlanets() != nullptr)
    {
        detail::accumulateBodies(sys->getPlanets(),
                                 observerInFrame,
                                 starInFrame,
                                 starAngularRadius,
                                 jd,
                                 totalOcclusion);
    }

    totalOcclusion = std::clamp(totalOcclusion, 0.0, 1.0);
    return 1.0 - totalOcclusion;
}

} // namespace celestia::engine::lightamount
