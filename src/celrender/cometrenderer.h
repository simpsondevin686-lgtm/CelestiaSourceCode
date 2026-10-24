// cometrenderer.h
//
// Copyright (C) 2001-present, the Celestia Development Team
// Original version by Chris Laurel <claurel@gmail.com>
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 2
// of the License, or (at your option) any later version.

#pragma once

#include <memory>
#include <string_view>

#include <Eigen/Core>

#include "gl/buffer.h"
#include "gl/vertexobject.h"

class Body;
class Observer;
class Renderer;
class CelestiaGLProgram;
struct Matrices;

namespace celestia::render
{

class CometRenderer
{
public:
    explicit CometRenderer(Renderer &renderer);
    ~CometRenderer();
    CometRenderer(const CometRenderer&) = delete;
    CometRenderer(CometRenderer&&) = delete;
    CometRenderer& operator=(const CometRenderer&) = delete;
    CometRenderer& operator=(CometRenderer&&) = delete;

    void render(const Body &body,
                const Observer &observer,
                const Eigen::Vector3f &pos,
                float dustTailLength,
                float discSizeInPixels,
                const Matrices &m);

    bool initGL();

private:
    void renderTextTail(const Body &body,
                        const Eigen::Vector3f &pos,
                        const Eigen::Vector3f &sunDir,
                        float dustTailLength,
                        float fadeFactor,
                        const Matrices &m,
                        std::string_view words);

private:

    struct CometTailVertex
    {
        Eigen::Vector3f point;
        Eigen::Vector3f normal;
        float brightness;
    };

    Renderer                          &m_renderer;
    CelestiaGLProgram                 *m_prog{ nullptr };
    int                                m_brightnessLoc{ -1 };
    bool                               m_initialized{ false };
    std::unique_ptr<CometTailVertex[]> m_vertices;
    std::unique_ptr<GLushort[]>        m_indices;
    gl::Buffer::SharedPtr              m_bo;
    gl::Buffer::SharedPtr              m_ibo;
    gl::VertexObject                   m_vo;
    gl::VertexObject                   m_voText;
};

} // namespace celestia::render
