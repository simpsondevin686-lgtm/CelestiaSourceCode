// cometrenderer.h
//
// Copyright (C) 2001-present, the Celestia Development Team
// Original version by Chris Laurel <claurel@gmail.com>
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 2
// of the License, or (at your option) any later version.

#include "cometrenderer.h"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <numbers>
#include <string_view>

#include <Eigen/Geometry>

#include <celastro/astro.h>
#include <celengine/body.h>
#include <celengine/glsupport.h>
#include <celengine/observer.h>
#include <celengine/render.h>
#include <celengine/shadermanager.h>
#include <celmath/mathlib.h>
#include <celmath/vecgl.h>
#include <celutil/indexlist.h>

using celestia::util::BuildIndexList;
using celestia::util::IndexListCapacity;
using ushort = unsigned short;

namespace celestia::render
{

namespace
{
constexpr int MaxCometTailPoints = 120;
constexpr int MaxCometTailSlices = 48;
constexpr int MaxVertices = MaxCometTailPoints*MaxCometTailSlices;
constexpr int MaxIndices = IndexListCapacity(MaxCometTailSlices, MaxCometTailPoints);

// Distance from the Sun at which comet tails will start to fade out
constexpr float CometTailAttenDistSol = astro::AUtoKilometers(5.0f);

// 5x7 bitmap font for ASCII 0x20..0x5F (space through underscore).
// Each char = 7 rows; in each row bit4=leftmost, bit0=rightmost.
// Lowercase letters are uppercased before lookup.
constexpr std::uint8_t kFont5x7[64][7] = {
    {0x00,0x00,0x00,0x00,0x00,0x00,0x00}, // ' '
    {0x04,0x04,0x04,0x04,0x00,0x00,0x04}, // '!'
    {0x0A,0x0A,0x0A,0x00,0x00,0x00,0x00}, // '"'
    {0x0A,0x0A,0x1F,0x0A,0x1F,0x0A,0x0A}, // '#'
    {0x04,0x0F,0x14,0x0E,0x05,0x1E,0x04}, // '$'
    {0x18,0x19,0x02,0x04,0x08,0x13,0x03}, // '%'
    {0x0C,0x12,0x14,0x08,0x15,0x12,0x0D}, // '&'
    {0x04,0x04,0x04,0x00,0x00,0x00,0x00}, // '\''
    {0x02,0x04,0x08,0x08,0x08,0x04,0x02}, // '('
    {0x08,0x04,0x02,0x02,0x02,0x04,0x08}, // ')'
    {0x00,0x0A,0x04,0x1F,0x04,0x0A,0x00}, // '*'
    {0x00,0x04,0x04,0x1F,0x04,0x04,0x00}, // '+'
    {0x00,0x00,0x00,0x00,0x00,0x04,0x04}, // ','
    {0x00,0x00,0x00,0x1F,0x00,0x00,0x00}, // '-'
    {0x00,0x00,0x00,0x00,0x00,0x00,0x04}, // '.'
    {0x00,0x01,0x02,0x04,0x08,0x10,0x00}, // '/'
    {0x0E,0x11,0x13,0x15,0x19,0x11,0x0E}, // '0'
    {0x04,0x0C,0x04,0x04,0x04,0x04,0x0E}, // '1'
    {0x0E,0x11,0x01,0x02,0x04,0x08,0x1F}, // '2'
    {0x1F,0x02,0x04,0x02,0x01,0x11,0x0E}, // '3'
    {0x02,0x06,0x0A,0x12,0x1F,0x02,0x02}, // '4'
    {0x1F,0x10,0x1E,0x01,0x01,0x11,0x0E}, // '5'
    {0x06,0x08,0x10,0x1E,0x11,0x11,0x0E}, // '6'
    {0x1F,0x01,0x02,0x04,0x08,0x08,0x08}, // '7'
    {0x0E,0x11,0x11,0x0E,0x11,0x11,0x0E}, // '8'
    {0x0E,0x11,0x11,0x0F,0x01,0x02,0x0C}, // '9'
    {0x00,0x00,0x04,0x00,0x04,0x00,0x00}, // ':'
    {0x00,0x00,0x04,0x00,0x04,0x04,0x08}, // ';'
    {0x02,0x04,0x08,0x10,0x08,0x04,0x02}, // '<'
    {0x00,0x00,0x1F,0x00,0x1F,0x00,0x00}, // '='
    {0x08,0x04,0x02,0x01,0x02,0x04,0x08}, // '>'
    {0x0E,0x11,0x01,0x02,0x04,0x00,0x04}, // '?'
    {0x0E,0x11,0x01,0x0D,0x15,0x15,0x0E}, // '@'
    {0x0E,0x11,0x11,0x1F,0x11,0x11,0x11}, // 'A'
    {0x1E,0x11,0x11,0x1E,0x11,0x11,0x1E}, // 'B'
    {0x0E,0x11,0x10,0x10,0x10,0x11,0x0E}, // 'C'
    {0x1C,0x12,0x11,0x11,0x11,0x12,0x1C}, // 'D'
    {0x1F,0x10,0x10,0x1E,0x10,0x10,0x1F}, // 'E'
    {0x1F,0x10,0x10,0x1E,0x10,0x10,0x10}, // 'F'
    {0x0E,0x11,0x10,0x17,0x11,0x11,0x0F}, // 'G'
    {0x11,0x11,0x11,0x1F,0x11,0x11,0x11}, // 'H'
    {0x0E,0x04,0x04,0x04,0x04,0x04,0x0E}, // 'I'
    {0x07,0x02,0x02,0x02,0x02,0x12,0x0C}, // 'J'
    {0x11,0x12,0x14,0x18,0x14,0x12,0x11}, // 'K'
    {0x10,0x10,0x10,0x10,0x10,0x10,0x1F}, // 'L'
    {0x11,0x1B,0x15,0x15,0x11,0x11,0x11}, // 'M'
    {0x11,0x19,0x15,0x13,0x11,0x11,0x11}, // 'N'
    {0x0E,0x11,0x11,0x11,0x11,0x11,0x0E}, // 'O'
    {0x1E,0x11,0x11,0x1E,0x10,0x10,0x10}, // 'P'
    {0x0E,0x11,0x11,0x11,0x15,0x12,0x0D}, // 'Q'
    {0x1E,0x11,0x11,0x1E,0x14,0x12,0x11}, // 'R'
    {0x0F,0x10,0x10,0x0E,0x01,0x01,0x1E}, // 'S'
    {0x1F,0x04,0x04,0x04,0x04,0x04,0x04}, // 'T'
    {0x11,0x11,0x11,0x11,0x11,0x11,0x0E}, // 'U'
    {0x11,0x11,0x11,0x11,0x11,0x0A,0x04}, // 'V'
    {0x11,0x11,0x11,0x15,0x15,0x15,0x0A}, // 'W'
    {0x11,0x11,0x0A,0x04,0x0A,0x11,0x11}, // 'X'
    {0x11,0x11,0x11,0x0A,0x04,0x04,0x04}, // 'Y'
    {0x1F,0x01,0x02,0x04,0x08,0x10,0x1F}, // 'Z'
    {0x0E,0x08,0x08,0x08,0x08,0x08,0x0E}, // '['
    {0x00,0x10,0x08,0x04,0x02,0x01,0x00}, // '\'
    {0x0E,0x02,0x02,0x02,0x02,0x02,0x0E}, // ']'
    {0x04,0x0A,0x11,0x00,0x00,0x00,0x00}, // '^'
    {0x00,0x00,0x00,0x00,0x00,0x00,0x1F}, // '_'
};

} // end unnamed namespace

CometRenderer::CometRenderer(Renderer &renderer) :
    m_renderer(renderer),
    m_vertices(std::make_unique<CometTailVertex[]>(MaxVertices)),
    m_indices(std::make_unique<GLushort[]>(MaxIndices))
{
}

CometRenderer::~CometRenderer() = default;

bool
CometRenderer::initGL()
{
    if (m_initialized)
        return true;

    m_prog = m_renderer.getShaderManager().getShader(StaticShader::Comet);
    if (m_prog == nullptr)
        return false;

    m_brightnessLoc = m_prog->attribIndex("in_Brightness");

    m_vo = gl::VertexObject(gl::VertexObject::Primitive::TriangleStrip);
    m_bo = gl::Buffer::create(gl::Buffer::TargetHint::Array,
                              static_cast<GLsizeiptr>(sizeof(CometTailVertex) * MaxVertices),
                              gl::Buffer::BufferUsage::StreamDraw);
    m_ibo = gl::Buffer::create(gl::Buffer::TargetHint::ElementArray,
                               static_cast<GLsizeiptr>(sizeof(GLushort) * MaxIndices),
                               gl::Buffer::BufferUsage::StreamDraw);

    m_vo
        .addVertexBuffer(
            m_bo,
            CelestiaGLProgram::VertexCoordAttributeIndex,
            3,
            gl::VertexObject::DataType::Float,
            false,
            sizeof(CometTailVertex),
            offsetof(CometTailVertex, point))
        .addVertexBuffer(
            m_bo,
            CelestiaGLProgram::NormalAttributeIndex,
            3,
            gl::VertexObject::DataType::Float,
            false,
            sizeof(CometTailVertex),
            offsetof(CometTailVertex, normal))
        .addVertexBuffer(
            m_bo,
            m_brightnessLoc,
            1,
            gl::VertexObject::DataType::Float,
            false,
            sizeof(CometTailVertex),
            offsetof(CometTailVertex, brightness))
        .setIndexBuffer(m_ibo, 0, gl::VertexObject::IndexType::UnsignedShort);

    m_voText = gl::VertexObject(gl::VertexObject::Primitive::Triangles);
    m_voText
        .addVertexBuffer(
            m_bo,
            CelestiaGLProgram::VertexCoordAttributeIndex,
            3,
            gl::VertexObject::DataType::Float,
            false,
            sizeof(CometTailVertex),
            offsetof(CometTailVertex, point))
        .addVertexBuffer(
            m_bo,
            CelestiaGLProgram::NormalAttributeIndex,
            3,
            gl::VertexObject::DataType::Float,
            false,
            sizeof(CometTailVertex),
            offsetof(CometTailVertex, normal))
        .addVertexBuffer(
            m_bo,
            m_brightnessLoc,
            1,
            gl::VertexObject::DataType::Float,
            false,
            sizeof(CometTailVertex),
            offsetof(CometTailVertex, brightness))
        .setIndexBuffer(m_ibo, 0, gl::VertexObject::IndexType::UnsignedShort);

    m_initialized = true;
    return true;
}

void
CometRenderer::render(const Body &body,
                      const Observer &observer,
                      const Eigen::Vector3f &pos,
                      float dustTailLength,
                      float discSizeInPixels,
                      const Matrices &m)
{
    if (m_prog == nullptr)
        return;

    double now = observer.getTime();

    Eigen::Vector3f cometPoints[MaxCometTailPoints];
#if 0
    Eigen::Vector3d pos0 = body.getOrbit(now)->positionAtTime(now);
    Eigen::Vector3d pos1 = body.getOrbit(now)->positionAtTime(now - 0.01);
    Eigen::Vector3d vd = pos1 - pos0;
#endif

    // Adjust the amount of triangles used for the comet tail based on
    // the screen size of the comet.
    float lod = std::clamp(discSizeInPixels / 1000.0f, 0.2f, 1.0f);
    auto nTailPoints = static_cast<int>(MaxCometTailPoints * lod);
    auto nTailSlices = static_cast<int>(MaxCometTailSlices * lod);

    float irradiance_max = 0.0f;
    // Find the sun with the largest irrradiance of light onto the comet
    // as function of the comet's position;
    // irradiance = sun's luminosity / square(distanceFromSun);
    Eigen::Vector3d sunPos(Eigen::Vector3d::Zero());
    for (const auto star : m_renderer.getNearStars())
    {
        if (star->getVisibility())
        {
            Eigen::Vector3d p = star->getPosition(now).offsetFromKm(observer.getPosition());
            float distanceFromSun = static_cast<float>((pos.cast<double>() - p).norm());
            float irradiance = star->getBolometricLuminosity() / math::square(distanceFromSun);

            if (irradiance > irradiance_max)
            {
                irradiance_max = irradiance;
                sunPos = p;
            }
        }
    }

    float fadeDistance = 1.0f / (CometTailAttenDistSol * std::sqrt(irradiance_max));

    // direction to sun with dominant light irradiance:
    Eigen::Vector3f sunDir = (pos.cast<double>() - sunPos).cast<float>().normalized();

    float dustTailRadius = dustTailLength * 0.1f;

    // TailWords override: if the body has non-empty TailWords, render
    // text geometry instead of the swept tube. Stock geometry path
    // below is unchanged when TailWords is empty.
    const auto& tailWords = GetBodyFeaturesManager()->getCometTailWords(&body);
    if (!tailWords.empty())
    {
        float textFade = 0.5f * (1.0f - std::tanh(fadeDistance - 1.0f / fadeDistance));
        renderTextTail(body, pos, sunDir, dustTailLength, textFade, m, tailWords);
        return;
    }

    Eigen::Vector3f origin = -sunDir * (body.getRadius() * 100);

    for (int i = 0; i < nTailPoints; i++)
    {
        float alpha = static_cast<float>(i) / static_cast<float>(nTailPoints);
        alpha = alpha * alpha;
        cometPoints[i] = origin + sunDir * (dustTailLength * alpha);
    }

    // We need three axes to define the coordinate system for rendering the
    // comet. The first axis is the sun-to-comet direction, and the other
    // two are chose orthogonal to each other and the primary axis.
    Eigen::Vector3f v = (cometPoints[1] - cometPoints[0]).normalized();
    Eigen::Quaternionf q = body.getEclipticToEquatorial(now).cast<float>();
    Eigen::Vector3f u = v.unitOrthogonal();
    Eigen::Vector3f w = u.cross(v);

    for (int i = 0; i < nTailPoints; i++)
    {
        float brightness = 1.0f - static_cast<float>(i) / static_cast<float>(nTailPoints - 1);
        Eigen::Vector3f v0, v1;
        float w0, w1;
        // Special case for the first vertex in the comet tail
        if (i == 0)
        {
            v0 = cometPoints[1] - cometPoints[0];
            v0.normalize();
            v1 = v0;
            w0 = 1.0f;
            w1 = 0.0f;
        }
        else
        {
            v0 = cometPoints[i] - cometPoints[i - 1];
            float sectionLength = v0.norm();
            v0.normalize();

            if (i == nTailPoints - 1)
            {
                v1 = v0;
            }
            else
            {
                v1 = (cometPoints[i + 1] - cometPoints[i]).normalized();
                q.setFromTwoVectors(v0, v1);
                Eigen::Matrix3f rot = q.toRotationMatrix();
                u = rot * u;
                v = rot * v;
                w = rot * w;
            }
            float dr = dustTailRadius / static_cast<float>(nTailPoints) / sectionLength;
            w0 = std::atan(dr);
            float d = std::sqrt(1.0f + w0 * w0);
            w1 = 1.0f / d;
            w0 = w0 / d;
        }

        float radius = static_cast<float>(i) / static_cast<float>(nTailPoints) * dustTailRadius;
        for (int j = 0; j < nTailSlices; j++)
        {
            float theta = 2.0f * std::numbers::pi_v<float> * static_cast<float>(j) / static_cast<float>(nTailSlices);
            float s, c;
            math::sincos(theta, s, c);
            CometTailVertex& vtx = m_vertices[i * nTailSlices + j];
            vtx.normal = u * (s * w1) + w * (c * w1) + v * w0;
            vtx.normal.normalize();
            s *= radius;
            c *= radius;

            vtx.point = cometPoints[i] + u * s + w * c;
            vtx.brightness = brightness;
        }
    }

    BuildIndexList(static_cast<ushort>(nTailPoints-1), static_cast<ushort>(nTailSlices), m_indices.get());

    // If fadeDistFromSun = x/x0 >= 1.0, comet tail starts fading,
    // i.e. fadeFactor quickly transits from 1 to 0.
    float fadeFactor = 0.5f * (1.0f - std::tanh(fadeDistance - 1.0f / fadeDistance));

    Renderer::PipelineState ps;
    ps.blending = true;
    ps.blendFunc = {GL_SRC_ALPHA, GL_ONE};
    ps.depthTest = true;
    m_renderer.setPipelineState(ps);

    m_prog->use();
    m_prog->setMVPMatrices(*m.projection, (*m.modelview) * math::translate(pos));
    m_prog->vec3Param("color") = GetBodyFeaturesManager()->getCometTailColor(&body).toVector3();
    m_prog->vec3Param("viewDir") = pos.normalized();
    m_prog->floatParam("fadeFactor") = fadeFactor;

    m_bo->invalidateData().setSubData(0, std::as_bytes(std::span(m_vertices.get(), MaxVertices)));
    m_ibo->invalidateData().setSubData(0, std::as_bytes(std::span(m_indices.get(), MaxIndices)));

    glDisable(GL_CULL_FACE);
    int count = IndexListCapacity(nTailSlices, nTailPoints);
    m_vo.draw(count);
    glEnable(GL_CULL_FACE);
}


void
CometRenderer::renderTextTail(const Body &body,
                              const Eigen::Vector3f &pos,
                              const Eigen::Vector3f &sunDir,
                              float dustTailLength,
                              float fadeFactor,
                              const Matrices &m,
                              std::string_view words)
{
    // viewDir == pos.normalized() is also what the stock render() sets as the
    // shader's "viewDir" uniform. Setting each vertex normal = viewDir makes
    // dot(viewDir, in_Normal) == 1, so shade = fadeFactor uniformly: the text
    // reads as a flat emissive panel with no per-vertex shading variation.
    Eigen::Vector3f viewDir = pos.normalized();

    // Project sunDir onto the sky plane so the text lies in a plane facing
    // the observer. Fall back to raw sunDir when the projection degenerates
    // (comet nearly aligned with view direction).
    Eigen::Vector3f sunDirSky = sunDir - viewDir * sunDir.dot(viewDir);
    float skyLen = sunDirSky.norm();
    if (skyLen < 1.0e-3f)
        sunDirSky = sunDir;
    else
        sunDirSky /= skyLen;
    Eigen::Vector3f upSky = viewDir.cross(sunDirSky).normalized();

    constexpr int kMaxChars = 32;
    int n = std::min<int>(static_cast<int>(words.size()), kMaxChars);
    if (n <= 0)
        return;

    // Letter width: total tail length divided across n characters.
    // At far viewing distances the whole tail-length worth of letters is
    // visible; close up, individual "pixels" dominate.
    float letterWidth  = dustTailLength / static_cast<float>(n);
    float letterHeight = letterWidth * 1.4f;
    float pxW = letterWidth  / 5.0f;
    float pxH = letterHeight / 7.0f;

    Eigen::Vector3f originLocal = Eigen::Vector3f::Zero();

    CometTailVertex* vptr = m_vertices.get();
    GLushort*        iptr = m_indices.get();
    int vtxUsed = 0;
    int idxUsed = 0;
    bool overflow = false;

    for (int c = 0; c < n && !overflow; c++)
    {
        char ch = words[static_cast<std::size_t>(n - 1 - c)];
        if (ch >= 'a' && ch <= 'z')
            ch = static_cast<char>(ch - 'a' + 'A');
        int glyphIdx = (ch >= 0x20 && ch < 0x60) ? (ch - 0x20) : ('?' - 0x20);

        Eigen::Vector3f charCenter = originLocal + sunDirSky * ((static_cast<float>(c) + 0.5f) * letterWidth);

        for (int row = 0; row < 7 && !overflow; row++)
        {
            std::uint8_t rowBits = kFont5x7[glyphIdx][row];
            if (rowBits == 0)
                continue;
            for (int col = 0; col < 5; col++)
            {
                if ((rowBits & (1u << (4 - col))) == 0)
                    continue;
                if (vtxUsed + 4 > MaxVertices || idxUsed + 6 > MaxIndices)
                {
                    overflow = true;
                    break;
                }

                float lx = -(static_cast<float>(col) - 2.0f) * pxW;
                float ly = (static_cast<float>(3 - row)) * pxH;
                Eigen::Vector3f qc = charCenter + sunDirSky * lx + upSky * ly;

                float hw = pxW * 0.5f;
                float hh = pxH * 0.5f;
                Eigen::Vector3f c00 = qc - sunDirSky * hw - upSky * hh;
                Eigen::Vector3f c10 = qc + sunDirSky * hw - upSky * hh;
                Eigen::Vector3f c11 = qc + sunDirSky * hw + upSky * hh;
                Eigen::Vector3f c01 = qc - sunDirSky * hw + upSky * hh;

                CometTailVertex* vbase = vptr + vtxUsed;
                vbase[0] = { c00, viewDir, 1.0f };
                vbase[1] = { c10, viewDir, 1.0f };
                vbase[2] = { c11, viewDir, 1.0f };
                vbase[3] = { c01, viewDir, 1.0f };

                GLushort base = static_cast<GLushort>(vtxUsed);
                GLushort* ibase = iptr + idxUsed;
                ibase[0] = base;
                ibase[1] = static_cast<GLushort>(base + 1);
                ibase[2] = static_cast<GLushort>(base + 2);
                ibase[3] = base;
                ibase[4] = static_cast<GLushort>(base + 2);
                ibase[5] = static_cast<GLushort>(base + 3);
                idxUsed += 6;
                vtxUsed += 4;
            }
        }
    }

    if (vtxUsed == 0)
        return;

    Renderer::PipelineState ps;
    ps.blending  = true;
    ps.blendFunc = {GL_SRC_ALPHA, GL_ONE};
    ps.depthTest = true;
    m_renderer.setPipelineState(ps);

    m_prog->use();
    m_prog->setMVPMatrices(*m.projection, (*m.modelview) * math::translate(pos));
    m_prog->vec3Param("color") = GetBodyFeaturesManager()->getCometTailColor(&body).toVector3();
    m_prog->vec3Param("viewDir") = viewDir;
    m_prog->floatParam("fadeFactor") = fadeFactor;

    m_bo->invalidateData().setSubData(0, std::as_bytes(std::span(vptr, vtxUsed)));
    m_ibo->invalidateData().setSubData(0, std::as_bytes(std::span(iptr, idxUsed)));

    glDisable(GL_CULL_FACE);
    m_voText.draw(idxUsed);
    glEnable(GL_CULL_FACE);
}

} // namespace celestia::render
