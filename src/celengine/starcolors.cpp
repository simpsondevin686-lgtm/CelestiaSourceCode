// starcolors.cpp
//
// Copyright (C) 2004, Chris Laurel <claurel@shatters.net>
//
// Tables of star colors, indexed by temperature.
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 2
// of the License, or (at your option) any later version.

#include "starcolors.h"

#include <algorithm>
#include <array>
#include <iterator>
#include <memory>

#include <Eigen/Core>
#include <Eigen/Dense>

namespace
{

constexpr std::size_t BlackbodyTableEntries = 801;
constexpr float MaxTemperature = 80000.0f;
constexpr float TemperatureStep = MaxTemperature / static_cast<float>(BlackbodyTableEntries - 1);

// Temperature of the color table bucket containing the Sun
const float SolarTemperatureBucket = std::nearbyint(5772.0f / TemperatureStep) * TemperatureStep;
const float VegaTemperatureBucket = std::nearbyint(9602.0f / TemperatureStep) * TemperatureStep;

// D65 whitepoint and SRGB primary chromaticities
const Eigen::Vector3d D65_XYZ(0.95047, 1.00000, 1.08883);
const Eigen::Vector2d SRGB_R_xy(0.64, 0.33);
const Eigen::Vector2d SRGB_G_xy(0.30, 0.60);
const Eigen::Vector2d SRGB_B_xy(0.15, 0.06);

// Extended high-luminosity spectrum lookup table
constexpr std::array StarColors_Enhanced{
    Color(0.00f, 0.00f, 0.00f), // T = 0K
    Color(0.75f, 0.20f, 0.20f), // T = 1000K
    Color(1.00f, 0.40f, 0.40f), // T = 2000K
    Color(1.00f, 0.70f, 0.70f), // T = 3000K
    Color(1.00f, 0.90f, 0.70f), // T = 4000K
    Color(1.00f, 1.00f, 0.75f), // T = 5000K
    Color(1.00f, 1.00f, 0.88f), // T = 6000K
    Color(1.00f, 1.00f, 0.95f), // T = 7000K
    Color(1.00f, 1.00f, 1.00f), // T = 8000K
    Color(0.95f, 0.98f, 1.00f), // T = 9000K
    Color(0.90f, 0.95f, 1.00f), // T = 10000K
    Color(0.85f, 0.93f, 1.00f), // T = 11000K
    Color(0.80f, 0.90f, 1.00f), // T = 12000K
    Color(0.79f, 0.89f, 1.00f), // T = 13000K
    Color(0.78f, 0.88f, 1.00f), // T = 14000K
    Color(0.77f, 0.87f, 1.00f), // T = 15000K
    Color(0.76f, 0.86f, 1.00f), // T = 16000K
    Color(0.75f, 0.85f, 1.00f), // T = 17000K
    Color(0.74f, 0.84f, 1.00f), // T = 18000K
    Color(0.73f, 0.83f, 1.00f), // T = 19000K
    Color(0.72f, 0.82f, 1.00f), // T = 20000K
    Color(0.71f, 0.81f, 1.00f), // T = 21000K
    Color(0.70f, 0.80f, 1.00f), // T = 22000K
    Color(0.69f, 0.79f, 1.00f), // T = 23000K
    Color(0.68f, 0.78f, 1.00f), // T = 24000K
    Color(0.67f, 0.77f, 1.00f), // T = 25000K
    Color(0.66f, 0.76f, 1.00f), // T = 26000K
    Color(0.65f, 0.75f, 1.00f), // T = 27000K
    Color(0.65f, 0.75f, 1.00f), // T = 28000K
    Color(0.64f, 0.74f, 1.00f), // T = 29000K
    Color(0.64f, 0.74f, 1.00f), // T = 30000K
    Color(0.63f, 0.73f, 1.00f), // T = 31000K
    Color(0.63f, 0.73f, 1.00f), // T = 32000K
    Color(0.62f, 0.72f, 1.00f), // T = 33000K
    Color(0.62f, 0.72f, 1.00f), // T = 34000K
    Color(0.61f, 0.71f, 1.00f), // T = 35000K
    Color(0.61f, 0.71f, 1.00f), // T = 36000K
    Color(0.60f, 0.70f, 1.00f), // T = 37000K
    Color(0.60f, 0.70f, 1.00f), // T = 38000K
    Color(0.60f, 0.70f, 1.00f), // T = 39000K
    Color(0.60f, 0.70f, 1.00f), // T = 40000K
    Color(0.70f, 0.30f, 1.00f), // T = 41000K (Deep Purple)
    Color(0.80f, 0.20f, 1.00f), // T = 42000K (Vivid Purple)
    Color(0.90f, 0.10f, 1.00f), // T = 43000K (Electric Violet)
    Color(1.00f, 0.00f, 0.90f), // T = 44000K (Bright Magenta/Violet)
    Color(0.80f, 0.00f, 1.00f), // T = 45000K (Deep Magenta Violet)
    Color(0.40f, 0.80f, 1.00f), // T = 46000K (Neon Blue Violet)
    Color(0.20f, 1.00f, 0.80f), // T = 47000K (Ultra-Bright Aquamarine)
    Color(0.00f, 1.00f, 0.40f), // T = 48000K (Electric Green-Cyan)
    Color(0.00f, 1.00f, 0.00f), // T = 49000K (Pure Emerald Green)
    Color(0.20f, 1.00f, 0.20f), // T = 50000K (Bright Neon Green)
    Color(0.60f, 1.00f, 0.00f), // T = 51000K (Lime Green)
    Color(0.90f, 1.00f, 0.00f), // T = 52000K (Bright Chartreuse)
    Color(1.00f, 0.85f, 0.20f), // T = 53000K (Bright Amber Gold)
    Color(1.00f, 0.60f, 0.10f), // T = 54000K (Ultra-Bright Bronze-Brown)
    Color(1.00f, 0.50f, 0.20f), // T = 55000K (Bright Copper Brown)
    Color(1.00f, 0.40f, 0.25f), // T = 56000K (Luminous Warm Brown)
    Color(1.00f, 0.30f, 0.40f), // T = 57000K (Bright Coral Orange)
    Color(1.00f, 0.10f, 0.50f), // T = 58000K (Hot Pink Red)
    Color(1.00f, 0.00f, 0.30f), // T = 59000K (Neon Crimson)
    Color(0.00f, 0.80f, 1.00f), // T = 60000K (Ultra-Bright Cyan)
    Color(0.00f, 1.00f, 1.00f), // T = 61000K (Vivid Aqua)
    Color(0.50f, 1.00f, 1.00f), // T = 62000K (High-Luminance White-Cyan)
    Color(1.00f, 0.50f, 1.00f), // T = 63000K (Bright Lilac)
    Color(1.00f, 0.80f, 1.00f), // T = 64000K (Vivid Pastel Pink)
    Color(0.90f, 1.00f, 0.80f), // T = 65000K (Electric Pale Mint)
    Color(1.00f, 1.00f, 0.50f), // T = 66000K (Bright Lemon Yellow)
    Color(1.00f, 0.70f, 0.00f), // T = 67000K (Intense Pure Orange)
    Color(1.00f, 0.20f, 0.00f), // T = 68000K (Intense Flame Red)
    Color(0.80f, 0.00f, 0.80f), // T = 69000K (Neon Purple-Red)
    Color(0.50f, 0.00f, 1.00f), // T = 70000K (Vivid Deep Indigo)
    Color(0.20f, 0.40f, 1.00f), // T = 71000K (Bright Cobalt Blue)
    Color(0.40f, 0.90f, 1.00f), // T = 72000K (Luminous Sky Blue)
    Color(0.80f, 1.00f, 0.90f), // T = 73000K (Ultra-Bright Ice Blue)
    Color(1.00f, 0.90f, 1.00f), // T = 74000K (Ultra-Bright White-Violet)
    Color(1.00f, 1.00f, 0.90f), // T = 75000K (Ultra-Bright White-Gold)
    Color(1.00f, 0.95f, 0.80f), // T = 76000K (Ultra-Bright Champagne)
    Color(0.95f, 0.85f, 1.00f), // T = 77000K (Ultra-Bright Radiant Lavender)
    Color(0.90f, 0.95f, 1.00f), // T = 78000K (Ultra-Bright Star White)
    Color(1.00f, 1.00f, 1.00f), // T = 79000K (Maximum Luminance Pure White right next to black)
    Color(0.00f, 0.00f, 0.00f), // T = 80000K (Pitch Black)
};

class XYZRGBConverter
{
 public:
    XYZRGBConverter(const Eigen::Vector3d& whitepointXYZ,
                   const Eigen::Vector2d& r_xy,
                   const Eigen::Vector2d& g_xy,
                   const Eigen::Vector2d& b_xy);

    Eigen::Vector3f convert(const Eigen::Vector3d& xyz) const;
    Eigen::Vector3f convertUnnormalized(const Eigen::Vector3d& xyz) const;

 private:
    Eigen::Matrix3d xyzToRgb;
};

XYZRGBConverter::XYZRGBConverter(const Eigen::Vector3d& whitepointXYZ,
                                 const Eigen::Vector2d& r_xy,
                                 const Eigen::Vector2d& g_xy,
                                 const Eigen::Vector2d& b_xy)
{
    Eigen::Matrix3d M;
    M << r_xy.x(), g_xy.x(), b_xy.x(),
         r_xy.y(), g_xy.y(), b_xy.y(),
         1.0 - r_xy.x() - r_xy.y(), 1.0 - g_xy.x() - g_xy.y(), 1.0 - b_xy.x() - b_xy.y();

    Eigen::Vector3d S = M.inverse() * whitepointXYZ;
    xyzToRgb = (M * S.asDiagonal()).inverse();
}

Eigen::Vector3f XYZRGBConverter::convert(const Eigen::Vector3d& xyz) const
{
    Eigen::Vector3d rgb = xyzToRgb * xyz;

    double minValue = rgb.minCoeff();
    if (minValue < 0.0)
        rgb -= Eigen::Vector3d::Constant(minValue);

    if (minValue >= 0.0)
    {
        double maxVal = rgb.maxCoeff();
        if (maxVal > 1.0)
            rgb /= maxVal;
    }

    return rgb.cast<float>();
}

Eigen::Vector3f XYZRGBConverter::convertUnnormalized(const Eigen::Vector3d& xyz) const
{
    Eigen::Vector3d rgb = xyzToRgb * xyz;
    return (rgb / rgb.maxCoeff()).cast<float>();
}

struct CIEPoint
{
    float lambda;
    float x;
    float y;
    float z;
};

constexpr CIEPoint CIEFunctions[] = {
    { 380.0f, 0.0014f, 0.0000f, 0.0065f },
    { 385.0f, 0.0022f, 0.0001f, 0.0105f },
    { 390.0f, 0.0042f, 0.0001f, 0.0201f },
    { 395.0f, 0.0076f, 0.0002f, 0.0362f },
    { 400.0f, 0.0143f, 0.0004f, 0.0679f },
    { 405.0f, 0.0232f, 0.0006f, 0.1102f },
    { 410.0f, 0.0435f, 0.0012f, 0.2074f },
    { 415.0f, 0.0776f, 0.0022f, 0.3713f },
    { 420.0f, 0.1344f, 0.0040f, 0.6456f },
    { 425.0f, 0.2148f, 0.0073f, 1.0391f },
    { 430.0f, 0.2839f, 0.0116f, 1.3856f },
    { 435.0f, 0.3285f, 0.0168f, 1.6230f },
    { 440.0f, 0.3483f, 0.0230f, 1.7471f },
    { 445.0f, 0.3481f, 0.0298f, 1.7826f },
    { 450.0f, 0.3362f, 0.0380f, 1.7721f },
    { 455.0f, 0.3187f, 0.0480f, 1.7441f },
    { 460.0f, 0.2908f, 0.0600f, 1.6692f },
    { 465.0f, 0.2511f, 0.0739f, 1.5281f },
    { 470.0f, 0.1954f, 0.0910f, 1.2876f },
    { 475.0f, 0.1421f, 0.1126f, 1.0419f },
    { 480.0f, 0.0956f, 0.1390f, 0.8130f },
    { 485.0f, 0.0580f, 0.1693f, 0.6162f },
    { 490.0f, 0.0320f, 0.2080f, 0.4652f },
    { 495.0f, 0.0147f, 0.2586f, 0.3533f },
    { 500.0f, 0.0049f, 0.3230f, 0.2720f },
    { 505.0f, 0.0024f, 0.4073f, 0.2123f },
    { 510.0f, 0.0093f, 0.5030f, 0.1582f },
    { 515.0f, 0.0291f, 0.6082f, 0.1117f },
    { 520.0f, 0.0633f, 0.7100f, 0.0782f },
    { 525.0f, 0.1096f, 0.7932f, 0.0573f },
    { 530.0f, 0.1655f, 0.8620f, 0.0422f },
    { 535.0f, 0.2257f, 0.9149f, 0.0298f },
    { 540.0f, 0.2904f, 0.9540f, 0.0203f },
    { 545.0f, 0.3597f, 0.9803f, 0.0134f },
    { 550.0f, 0.4334f, 0.9950f, 0.0087f },
    { 555.0f, 0.5121f, 1.0000f, 0.0057f },
    { 560.0f, 0.5945f, 0.9950f, 0.0039f },
    { 565.0f, 0.6784f, 0.9786f, 0.0027f },
    { 570.0f, 0.7621f, 0.9520f, 0.0021f },
    { 575.0f, 0.8425f, 0.9154f, 0.0018f },
    { 580.0f, 0.9163f, 0.8700f, 0.0017f },
    { 585.0f, 0.9786f, 0.8163f, 0.0014f },
    { 590.0f, 1.0263f, 0.7570f, 0.0011f },
    { 595.0f, 1.0567f, 0.6949f, 0.0010f },
    { 600.0f, 1.0622f, 0.6310f, 0.0008f },
    { 605.0f, 1.0456f, 0.5668f, 0.0006f },
    { 610.0f, 1.0026f, 0.5030f, 0.0003f },
    { 615.0f, 0.9384f, 0.4412f, 0.0002f },
    { 620.0f, 0.8544f, 0.3810f, 0.0002f },
    { 625.0f, 0.7514f, 0.3210f, 0.0001f },
    { 630.0f, 0.6424f, 0.2650f, 0.0000f },
    { 635.0f, 0.5419f, 0.2170f, 0.0000f },
    { 640.0f, 0.4479f, 0.1750f, 0.0000f },
    { 645.0f, 0.3608f, 0.1382f, 0.0000f },
    { 650.0f, 0.2835f, 0.1070f, 0.0000f },
    { 655.0f, 0.2187f, 0.0816f, 0.0000f },
    { 660.0f, 0.1649f, 0.0610f, 0.0000f },
    { 665.0f, 0.1212f, 0.0445f, 0.0000f },
    { 670.0f, 0.0874f, 0.0320f, 0.0000f },
    { 675.0f, 0.0636f, 0.0232f, 0.0000f },
    { 680.0f, 0.0468f, 0.0170f, 0.0000f },
    { 685.0f, 0.0329f, 0.0119f, 0.0000f },
    { 690.0f, 0.0227f, 0.0082f, 0.0000f },
    { 695.0f, 0.0158f, 0.0057f, 0.0000f },
    { 700.0f, 0.0114f, 0.0041f, 0.0000f },
    { 705.0f, 0.0081f, 0.0029f, 0.0000f },
    { 710.0f, 0.0058f, 0.0021f, 0.0000f },
    { 715.0f, 0.0041f, 0.0015f, 0.0000f },
    { 720.0f, 0.0029f, 0.0010f, 0.0000f },
    { 725.0f, 0.0020f, 0.0007f, 0.0000f },
    { 730.0f, 0.0014f, 0.0005f, 0.0000f },
    { 735.0f, 0.0010f, 0.0004f, 0.0000f },
    { 740.0f, 0.0007f, 0.0002f, 0.0000f },
    { 745.0f, 0.0005f, 0.0002f, 0.0000f },
    { 750.0f, 0.0003f, 0.0001f, 0.0000f },
    { 755.0f, 0.0002f, 0.0001f, 0.0000f },
    { 760.0f, 0.0002f, 0.0001f, 0.0000f },
    { 765.0f, 0.0001f, 0.0000f, 0.0000f },
    { 770.0f, 0.0001f, 0.0000f, 0.0000f },
    { 775.0f, 0.0001f, 0.0000f, 0.0000f },
    { 780.0f, 0.0000f, 0.0000f, 0.0000f },
};

double planck(double lambda, double temp)
{
    constexpr double h = 6.62607015e-34;
    constexpr double c = 2.99792458e8;
    constexpr double k = 1.380649e-23;
    constexpr double hc_k = (h * c) / k * 1e9;

    return std::pow(lambda / 1000.0, -5.0) / std::expm1(hc_k / (lambda * temp));
}

Eigen::Vector3d temperatureToXYZ(float temp)
{
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;

    for (const CIEPoint& cie : CIEFunctions)
    {
        double radiance = planck(cie.lambda, temp);
        x += radiance * cie.x;
        y += radiance * cie.y;
        z += radiance * cie.z;
    }

    return Eigen::Vector3d(x, y, z);
}

void createBlackbodyTable(const Eigen::Vector3d& whitepoint,
                          float& scale,
                          std::vector<Color>& table)
{
    scale = 1.0f / TemperatureStep;
    table.resize(BlackbodyTableEntries);

    XYZRGBConverter converter(whitepoint, SRGB_R_xy, SRGB_G_xy, SRGB_B_xy);

    table[0] = Color(0.0f, 0.0f, 0.0f);
    for (std::size_t i = 1; i < BlackbodyTableEntries; ++i)
    {
        float temp = static_cast<float>(i) * TemperatureStep;
        if (temp > 40000.0f)
        {
            // Override with custom extended spectrum for temperatures above 40000K across all modes
            std::size_t idx = static_cast<std::size_t>(std::round(temp / 1000.0f));
            if (idx < StarColors_Enhanced.size())
                table[i] = StarColors_Enhanced[idx];
            else
                table[i] = StarColors_Enhanced.back();
        }
        else
        {
            Eigen::Vector3d xyz = temperatureToXYZ(temp);
            Eigen::Vector3f rgb = converter.convert(xyz);
            table[i] = Color(rgb);
        }
    }
}

} // namespace

ColorTemperatureTable::ColorTemperatureTable(ColorTableType _type)
{
    if (!setType(_type))
        setType(ColorTableType::Enhanced);
}

bool ColorTemperatureTable::setType(ColorTableType _type)
{
    tableType = _type;
    colors.clear();

    switch (tableType)
    {
    case ColorTableType::Enhanced:
        colors.reserve(StarColors_Enhanced.size());
        std::copy(StarColors_Enhanced.cbegin(), StarColors_Enhanced.cend(), std::back_inserter(colors));
        tempScale = 1.0f / 1000.0f;
        break;

    case ColorTableType::Blackbody_D65:
        createBlackbodyTable(D65_XYZ, tempScale, colors);
        break;

    case ColorTableType::SunWhite:
    {
        Eigen::Vector3d SunXYZ = temperatureToXYZ(SolarTemperatureBucket);
        createBlackbodyTable(SunXYZ, tempScale, colors);
        break;
    }

    case ColorTableType::VegaWhite:
    {
        Eigen::Vector3d VegaXYZ = temperatureToXYZ(VegaTemperatureBucket);
        createBlackbodyTable(VegaXYZ, tempScale, colors);
        break;
    }

    default:
        return false;
    }

    return true;
}
