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

inline void
accumulateBodies(const PlanetarySystem* system,
                 const Eigen::Vector3d& observerFromStar,
                 double starAngularRadius,
                 double jd,
                 double& totalOcclusion)
{
    if (system == nullptr)
        return;

    const int n = system->getSystemSize();
    for (int i = 0; i < n; ++i)
    {
        const Body* body = system->getBody(i);
        if (body == nullptr || !body->isVisible())
            continue;

        const Eigen::Vector3d bodyFromStar     = body->getAstrocentricPosition(jd);
        const Eigen::Vector3d bodyFromObserver = bodyFromStar - observerFromStar;
        const double distBody = bodyFromObserver.norm();
        if (distBody < 1.0e-6)
            continue;

        const Eigen::Vector3d starFromObserver = -observerFromStar;
        const double distStar = starFromObserver.norm();
        if (distStar < 1.0e-6)
            continue;

        const double cosSep = starFromObserver.dot(bodyFromObserver) / (distStar * distBody);
        const double sepRad = std::acos(std::clamp(cosSep, -1.0, 1.0));

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

        totalOcclusion += discOcclusionFraction(starAngularRadius, bodyAngularRadius, sepRad);

        if (const RingSystem* rings = mgr->getRings(body); rings != nullptr)
        {
            const double outerAng = static_cast<double>(rings->outerRadius) / distBody;
            const double innerAng = static_cast<double>(rings->innerRadius) / distBody;
            totalOcclusion += discOcclusionFraction(starAngularRadius, outerAng, sepRad)
                            - discOcclusionFraction(starAngularRadius, innerAng, sepRad);
        }

        accumulateBodies(body->getSystem(), observerFromStar, starAngularRadius, jd, totalOcclusion);
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

    const UniversalCoord starPos = star.getPosition(jd);

    const Eigen::Vector3d observerFromStar = observerPos.offsetFromKm(starPos);
    const double distStar = observerFromStar.norm();
    if (distStar < 1.0e-6)
        return 1.0;

    const double starAngularRadius = starRadiusKm / distStar;

    double totalOcclusion = 0.0;
    const SolarSystem* sys = universe.getSolarSystem(&star);
    if (sys != nullptr && sys->getPlanets() != nullptr)
    {
        detail::accumulateBodies(sys->getPlanets(),
                                 observerFromStar,
                                 starAngularRadius,
                                 jd,
                                 totalOcclusion);
    }

    totalOcclusion = std::clamp(totalOcclusion, 0.0, 1.0);
    return 1.0 - totalOcclusion;
}

} // namespace celestia::engine::lightamount
