/*
 * This source file is part of RmlUi, the HTML/CSS Interface Middleware
 *
 * For the latest information, see http://github.com/mikke89/RmlUi
 *
 * Copyright (c) 2008-2010 CodePoint Ltd, Shift Technology Ltd
 * Copyright (c) 2019-2023 The RmlUi Team, and contributors
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 */

#ifndef RMLUI_CORE_VERTEX_H
#define RMLUI_CORE_VERTEX_H

#include "Header.h"
#include "Types.h"

namespace Rml {

/**
    The element that makes up all geometry sent to the renderer.

    @author Peter Curry
 */

struct RMLUICORE_API Vertex {
	/// Two-dimensional position of the vertex (usually in pixels).
	Vector2f position;
	/// RGBA-ordered 8-bit/channel colour with premultiplied alpha.
	ColourbPremultiplied colour;
	/// Texture coordinate for any associated texture.
	Vector2f tex_coord;

	friend bool operator==(const Vertex& lhs, const Vertex& rhs)
	{
		return lhs.position == rhs.position && lhs.colour == rhs.colour && lhs.tex_coord == rhs.tex_coord;
	}
	friend bool operator!=(const Vertex& lhs, const Vertex& rhs) { return !(lhs == rhs); }
};

} // namespace Rml
#endif
